import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
# ================= 1. 配置区 (更新版) =================
DATASET_ROOT = Path("DELTA_Dataset_Splits")
BATCH_SIZE = 32
EPOCHS = 100              # 放宽最大训练轮数
LEARNING_RATE = 1e-4
PATIENCE = 20            # 早停容忍度：连续10轮不提升则停止
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义切面类别及其对应的标签ID
PLANE_CLASSES = {
    "Abdomen": 0,
    "4CH": 1,
    "LVOT": 2,
    "RVOT": 3,
    "3VT": 4
}
NUM_CLASSES = len(PLANE_CLASSES)

# ================= 2. 自定义数据集加载器 =================
class FetalPlaneDataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        """
        root_dir: 数据集根目录 (如 DELTA_Dataset_Splits)
        split: "Train", "Val" 或 "Test"
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        split_dir = root_dir / split
        
        # 遍历: split -> disease_class -> patient -> images
        for disease_folder in split_dir.iterdir():
            if not disease_folder.is_dir(): continue
            for patient_folder in disease_folder.iterdir():
                if not patient_folder.is_dir(): continue
                for file_path in patient_folder.iterdir():
                    if file_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        plane_name = file_path.stem
                        if plane_name in PLANE_CLASSES:
                            self.image_paths.append(file_path)
                            self.labels.append(PLANE_CLASSES[plane_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 将灰度或 RGBA 转为标准 RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ================= 3. 数据预处理管道 =================
# 图像标准化采用 ImageNet 标准
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),    # 数据增强
    transforms.RandomRotation(10),        # 轻微旋转增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= 4. 模型构建 =================
def build_resnet50(num_classes):
    # 加载预训练的 ResNet-50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 修改最后的全连接层以适应我们的5分类任务
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ================= 5. 评估函数 =================
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # 计算 Macro 级别的 Precision, Recall, F1-score
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return epoch_loss, acc, precision, recall, f1
# ================= 6. 主训练循环 (引入早停机制) =================
def main():
    print("正在加载数据集...")
    train_dataset = FetalPlaneDataset(DATASET_ROOT, split="Train", transform=train_transforms)
    val_dataset = FetalPlaneDataset(DATASET_ROOT, split="Val", transform=eval_transforms)
    test_dataset = FetalPlaneDataset(DATASET_ROOT, split="Test", transform=eval_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"数据加载完成: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    model = build_resnet50(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_f1 = 0.0
    epochs_no_improve = 0  # 连续未提升的轮数计数器
    
    print("\n--- 开始训练 ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        train_loss = running_loss / len(train_dataset)
        val_loss, val_acc, val_pre, val_rec, val_f1 = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - F1(Macro): {val_f1:.4f}")
        
        # 判断是否触发保存与早停计数
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_resnet50_plane_cls.pth")
            print("  --> 发现最佳模型，已保存！")
            epochs_no_improve = 0  # 重置早停计数器
        else:
            epochs_no_improve += 1
            print(f"  --> 早停计数: {epochs_no_improve}/{PATIENCE}")
            
        # 触发早停
        if epochs_no_improve >= PATIENCE:
            print(f"\n!!! 触发早停机制 (连续 {PATIENCE} 轮 F1 无提升)，提前停止训练 !!!")
            break

    print("\n--- 在测试集上进行最终验证 ---")
    # 加载早停机制保存的那个最佳权重
    model.load_state_dict(torch.load("best_resnet50_plane_cls.pth"))
    test_loss, test_acc, test_pre, test_rec, test_f1 = evaluate_model(model, test_loader, criterion)
    
    print("最终 Test 性能报告:")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_pre:.4f}")
    print(f"Recall   : {test_rec:.4f}")
    print(f"F1-score : {test_f1:.4f}")

if __name__ == "__main__":
    main()