import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# ================= Configuration =================
# Source dataset path organized by disease category
SOURCE_DIR = Path("Final_Clean_Dataset/Videos")
# Output path for the split dataset
OUTPUT_DIR = Path("DELTA_Dataset_Splits")

# Split ratio settings
TEST_SIZE = 0.20
VAL_SIZE_OF_REMAINING = 0.125  # 12.5% of the remaining 80% corresponds to 10% of the full dataset

random_seed = 42
# ==========================================

def get_dataset_info(source_dir):
    """Get all patient directories and their disease labels."""
    data = []
    # Iterate over disease folders in the source directory
    for class_folder in sorted(source_dir.iterdir()):
        if class_folder.is_dir():
            class_name = class_folder.name
            # Iterate over patient folders or video files under each disease folder
            for patient_item in class_folder.iterdir():
                if patient_item.name.startswith('.'): continue  # Ignore hidden files
                data.append({
                    'patient_path': patient_item,
                    'class_name': class_name,
                    'patient_id': patient_item.name
                })
    return data

def main():
    data = get_dataset_info(SOURCE_DIR)
    
    # Extract features (paths) and labels (disease categories)
    X = [item['patient_path'] for item in data]
    y = [item['class_name'] for item in data]
    
    print(f"Total cases detected: {len(X)}")
    
    # First split: hold out Test set (20%)
    # stratify=y preserves disease-category proportions
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=random_seed, 
        stratify=y
    )
    
    # Second split: divide the remaining 80% into Train (70%) and Val (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=VAL_SIZE_OF_REMAINING, 
        random_state=random_seed, 
        stratify=y_temp
    )
    
    # Create output directory structure
    splits = {
        'Train': X_train,
        'Val': X_val,
        'Test': X_test
    }
    
    print("\nCopying data to the target folders with patient-level separation...")
    for split_name, patient_paths in splits.items():
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for p_path in patient_paths:
            class_name = p_path.parent.name
            target_class_dir = split_dir / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_class_dir / p_path.name
            
            # If patient data is a folder, copy all contained files together
            if p_path.is_dir():
                shutil.copytree(p_path, target_path, dirs_exist_ok=True)
            # If patient data is a single file, such as one video
            elif p_path.is_file():
                shutil.copy2(p_path, target_path)
                
    print(f"\nDataset split completed.")
    print(f"Train: {len(X_train)} cases")
    print(f"Val: {len(X_val)} cases")
    print(f"Test: {len(X_test)} cases")

if __name__ == "__main__":
    main()
