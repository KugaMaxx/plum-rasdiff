import io
import re
import argparse
import datasets
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


def convert_texture_to_heatmap(image, min_value, max_value, num_rows, num_cols):
    """
    Convert image dictionary from parquet to volume and back to image dict format.
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(image['bytes']))
    np_texture = np.array(image.convert('L'))
    
    # Get dimensions of the flipbook texture
    texture_height, texture_width = np_texture.shape
    
    # Calculate dimensions of each individual slice
    slice_height = texture_height // num_rows
    slice_width = texture_width // num_cols
    
    # Extract individual slices from the flipbook
    slices = []
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the position of this slice in the flipbook
            y_start = row * slice_height
            y_end = y_start + slice_height
            x_start = col * slice_width
            x_end = x_start + slice_width
            
            # Extract the slice
            slice_data = np_texture[y_start:y_end, x_start:x_end]
            slices.append(slice_data)
    
    # Stack slices to form 3D volume [height, width, depth]
    volume = np.stack(slices, axis=2)
    volume = volume.astype(np.float32) / 255.0
    volume = volume * (max_value - min_value) + min_value

    num_slices = volume.shape[2]
    min_value = min_value * num_slices
    max_value = max_value * num_slices
    
    volume = volume.sum(axis=2)
    volume = (volume - min_value) / (max_value - min_value) * 255.0
    volume = volume.astype(np.uint8)
    
    # Convert back to PIL Image and then to bytes
    result_image = Image.fromarray(volume, mode='L').convert('RGB')
    output_bytes = io.BytesIO()
    result_image.save(output_bytes, format='PNG')
    
    return {'bytes': output_bytes.getvalue()}


def convert_long_prompt_to_short(text):
    text = re.split(r'\s*;\s*', text.strip())
    text = [[float(num) for num in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)', dim)] for dim in text if dim]
    text = ", ".join(f"{int(t[-1]):03d}" for t in text)
    return text


def convert_long_prompt_to_vetors(text):
    """
    Convert long prompt to index vectors by sampling at equal intervals.
    """
    # Split by semicolon and parse each coordinate set
    parsed = re.split(r'\s*;\s*', text.strip())
    parsed = [[float(num) for num in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)', dim)] for dim in parsed if dim]

    return np.array(parsed)


def process_subdir(subdir, output_dir, split_type):
    """
    Process a single subdirectory for either training or validation.
    
    Args:
        subdir: Path to the subdirectory to process
        output_dir: Output directory path
        split_type: Either 'train' or 'validation'
    """
    try:
        parquet_file = subdir / (subdir.name + '.parquet')
        df = pd.read_parquet(parquet_file)

        df['index_vector'] = df.apply(
            lambda row: convert_long_prompt_to_vetors(
                row['text']
            ), 
            axis=1
        )

        df['text'] = df.apply(
            lambda row: convert_long_prompt_to_short(
                row['text']
            ), 
            axis=1
        )

        # Apply convert_to_stack_heatmap to each row
        df['image'] = df.apply(
            lambda row: convert_texture_to_heatmap(
                row['image'], 
                row['min_value'], 
                row['max_value'], 
                row['num_rows'], 
                row['num_cols']
            ), 
            axis=1
        )
        
        # Update min_value and max_value after converting to heatmap
        df['min_value'] = df['min_value'] * df['num_rows'] * df['num_cols']
        df['max_value'] = df['max_value'] * df['num_rows'] * df['num_cols']
        df.drop(columns=['num_rows', 'num_cols'], inplace=True)

        # # Extend index_vector to fixed length by rolling
        # tmp_vector = np.zeros((8, 72))
        # for i, vec in enumerate(df['index_vector']):
        #     df.at[df.index[i], 'index_vector'] = np.concatenate((tmp_vector, vec), axis=1)
        #     tmp_vector[:, :-1] = tmp_vector[:, 1:]
        #     tmp_vector[:, -1] = vec[:, 0] 

        # Convert DataFrame to list of dictionaries for datasets
        data_list = df.to_dict('records')
        
        sub_output_path = output_dir / split_type / subdir.name
        sub_output_path.mkdir(parents=True, exist_ok=True)
        
        dataset = datasets.Dataset.from_list(
            data_list,
            features=datasets.Features({
                'image': datasets.Image(),
                'case': datasets.Value("string"),
                'text': datasets.Value("string"),
                'min_value': datasets.Value("float32"),
                'max_value': datasets.Value("float32"),
                'index_vector': datasets.Array2D(shape=df['index_vector'].iloc[0].shape, dtype='float32'),
            }),
        )
        dataset.to_parquet(sub_output_path / f'{sub_output_path.stem}.parquet')
        return f"Success: {subdir.name}"
    except Exception as e:
        return f"Error processing {subdir.name}: {str(e)}"


def add_conditioning(subdir, train_path, is_train=True):
    """
    Add conditioning images to training or validation data.
    
    Args:
        subdir: Path to the subdirectory to process
        train_path: Path to the training data directory
        is_train: If True, use previous HRR as condition; if False, use same HRR
    """
    try:
        parquet_file = subdir / (subdir.name + '.parquet')
        df = pd.read_parquet(parquet_file)

        case, hrr = parquet_file.stem.split('h')
        case = case + 'h'
        hrr = int(hrr[:-2] + '00')

        # For training: use previous HRR (or same if HRR=100)
        # For validation: use same HRR
        condition_hrr = (hrr - 100 if hrr != 100 else hrr) if is_train else hrr
        condition_subdir = Path(case + f'{condition_hrr:04d}')
        conditioning_parquet_file = train_path / condition_subdir.name / (condition_subdir.name + '.parquet')
        conditioning_df = pd.read_parquet(conditioning_parquet_file)

        df['conditioning_case'] = conditioning_df['case']
        df['conditioning_image'] = conditioning_df['image']
        
        # Convert DataFrame to list of dictionaries for datasets
        data_list = df.to_dict('records')
        
        dataset = datasets.Dataset.from_list(
            data_list,
            features=datasets.Features({
                'image': datasets.Image(),
                'case': datasets.Value("string"),
                'text': datasets.Value("string"),
                'min_value': datasets.Value("float32"),
                'max_value': datasets.Value("float32"),
                'conditioning_case': datasets.Value("string"),
                'conditioning_image': datasets.Image(),
                'index_vector': datasets.Array2D(shape=df['index_vector'].iloc[0].shape, dtype='float32'),
            }),
        )
        dataset.to_parquet(parquet_file)
        return f"Success: {subdir.name}"
    except Exception as e:
        return f"Error processing {subdir.name}: {str(e)}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert texture to conditioning images for ControlNet training.')
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the processed dataset.",
    )
    parser.add_argument(
        "--max_workers",
        default=1,
        type=int,
        help="Maximum number of parallel workers.",
    )
    args = parser.parse_args()

    # Convert to Path objects
    args.dataset_dir = Path(args.dataset_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process training set
    train_path = args.dataset_dir / "train"

    print("Processing training set...")
    subdirs = sorted([d for d in train_path.iterdir() if d.is_dir()])

    process_func = partial(process_subdir, output_dir=args.output_dir, split_type="train")
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_func, subdir): subdir for subdir in subdirs}
        
        for future in tqdm(as_completed(futures), total=len(subdirs), desc="Training set"):
            result = future.result()
            if result.startswith("Error"):
                print(f"\n{result}")

    # Process validation set
    validation_path = args.dataset_dir / "validation"

    print("Processing validation set...")
    subdirs = sorted([d for d in validation_path.iterdir() if d.is_dir()])

    process_func = partial(process_subdir, output_dir=args.output_dir, split_type="validation")
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_func, subdir): subdir for subdir in subdirs}
        
        for future in tqdm(as_completed(futures), total=len(subdirs), desc="Validation set"):
            result = future.result()
            if result.startswith("Error"):
                print(f"\n{result}")

    print("Adding conditioning to train set...")
    train_path = args.output_dir / "train"
    subdirs = sorted([d for d in train_path.iterdir() if d.is_dir()])

    process_func = partial(add_conditioning, train_path=train_path, is_train=True)
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_func, subdir): subdir for subdir in subdirs}
        
        for future in tqdm(as_completed(futures), total=len(subdirs), desc="Adding conditioning (train)"):
            result = future.result()
            if result.startswith("Error"):
                print(f"\n{result}")


    print("Adding conditioning to validation set...")
    validation_path = args.output_dir / "validation"
    subdirs = sorted([d for d in validation_path.iterdir() if d.is_dir()])

    process_func = partial(add_conditioning, train_path=train_path, is_train=False)
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_func, subdir): subdir for subdir in subdirs}
        
        for future in tqdm(as_completed(futures), total=len(subdirs), desc="Adding conditioning (validation)"):
            result = future.result()
            if result.startswith("Error"):
                print(f"\n{result}")
