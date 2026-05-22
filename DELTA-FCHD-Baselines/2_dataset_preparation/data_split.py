import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# ================= 配置区 =================
# 原始数据集路径 (按病种分类的文件夹结构)
SOURCE_DIR = Path("Final_Clean_Dataset/Videos")
# 划分后存放的新路径
OUTPUT_DIR = Path("DELTA_Dataset_Splits")

# 比例设置
TEST_SIZE = 0.20
VAL_SIZE_OF_REMAINING = 0.125 # 剩下的 80% 中取 12.5% 就是整体的 10%

random_seed = 42
# ==========================================

def get_dataset_info(source_dir):
    """获取所有患者目录及其对应的病种标签"""
    data = []
    # 遍历源目录下的所有病种文件夹
    for class_folder in sorted(source_dir.iterdir()):
        if class_folder.is_dir():
            class_name = class_folder.name
            # 遍历病种文件夹下的所有患者文件夹/视频文件
            for patient_item in class_folder.iterdir():
                if patient_item.name.startswith('.'): continue # 忽略隐藏文件
                data.append({
                    'patient_path': patient_item,
                    'class_name': class_name,
                    'patient_id': patient_item.name
                })
    return data

def main():
    data = get_dataset_info(SOURCE_DIR)
    
    # 提取特征 (路径) 和 标签 (病种)
    X = [item['patient_path'] for item in data]
    y = [item['class_name'] for item in data]
    
    print(f"检测到总数据量: {len(X)} 例")
    
    # 第一次划分：分离出 测试集 Test (20%)
    # stratify=y 保证按病种比例划分
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=random_seed, 
        stratify=y
    )
    
    # 第二次划分：从剩下的 80% 中分离出 训练集 Train (70%) 和 验证集 Val (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=VAL_SIZE_OF_REMAINING, 
        random_state=random_seed, 
        stratify=y_temp
    )
    
    # 创建输出目录结构
    splits = {
        'Train': X_train,
        'Val': X_val,
        'Test': X_test
    }
    
    print("\n开始拷贝数据至目标文件夹 (以患者为单位隔离)...")
    for split_name, patient_paths in splits.items():
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for p_path in patient_paths:
            class_name = p_path.parent.name
            target_class_dir = split_dir / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_class_dir / p_path.name
            
            # 如果患者数据是文件夹，连同里面所有文件（视频+关键帧）一起拷贝
            if p_path.is_dir():
                shutil.copytree(p_path, target_path, dirs_exist_ok=True)
            # 如果患者数据是单个文件（如单一视频）
            elif p_path.is_file():
                shutil.copy2(p_path, target_path)
                
    print(f"\n划分完成！")
    print(f"Train (训练集): {len(X_train)} 例")
    print(f"Val   (验证集): {len(X_val)} 例")
    print(f"Test  (测试集): {len(X_test)} 例")

if __name__ == "__main__":
    main()