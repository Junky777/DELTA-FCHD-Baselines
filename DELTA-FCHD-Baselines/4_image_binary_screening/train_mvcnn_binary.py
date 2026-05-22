import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 配置区 =================
DATASET_ROOT = Path("DELTA_Dataset_Splits")
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "00_Normal", "01_TOF", "02_DORV", "03_PTA", "04_TGA", 
    "05_AVSD", "06_SV", "07_HLHS", "08_HRHS", "09_AA", 
    "10_PS", "11_PLSVC", "12_RAA"
]

# ================= 2. 二分类数据集加载器 =================
class FetalBinaryDataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        self.transform = transform
        self.cases = []
        
        split_dir = root_dir / split
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists(): continue
            
            # 正常为 0，异常为 1
            label_id = 0 if class_name == "00_Normal" else 1
            
            for patient_folder in class_dir.iterdir():
                if patient_folder.is_dir():
                    imgs = list(patient_folder.glob("*.jpg")) + list(patient_folder.glob("*.png"))
                    if len(imgs) >= 5:
                        self.cases.append((patient_folder, label_id))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        patient_path, label = self.cases[idx]
        plane_names = ["Abdomen", "4CH", "LVOT", "RVOT", "3VT"]
        view_images = []
        
        for plane in plane_names:
            img_path = list(patient_path.glob(f"{plane}.*"))[0]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            view_images.append(img)
        
        return torch.stack(view_images), label

# ================= 3. 经典 MVCNN 架构 (Su et al., ICCV 2015) =================
class ClassicMVCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ClassicMVCNN, self).__init__()
        # 使用共享权重的 ResNet-50 作为特征提取主干
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_dim = base_model.fc.in_features # 2048
        
        # 移除全连接层，只保留特征提取部分
        self.features = nn.Sequential(*(list(base_model.children())[:-1]))
        
        # 标准分类头 (直接作用于池化后的特征)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # x 维度: (Batch, 5, 3, 224, 224)
        batch_size, num_views, c, h, w = x.shape
        
        # 将 Batch 和 Views 合并，统一进行特征提取
        x = x.view(batch_size * num_views, c, h, w)
        feats = self.features(x) # (Batch*5, 2048, 1, 1)
        
        # 恢复视图维度: (Batch, 5, 2048)
        feats = feats.view(batch_size, num_views, self.feature_dim)
        
        # 🌟 核心：View-Pooling (视角池化)
        # 跨 5 个视角取特征最大值，保留最显著的病理信号
        pooled_feats, _ = torch.max(feats, dim=1) # (Batch, 2048)
        
        out = self.classifier(pooled_feats)
        return out

# ================= 4. 绘图辅助函数 =================
def plot_results(y_true, y_pred, save_path="binary_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'], 
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Baseline 1: Classic MVCNN Binary Classification')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================= 5. 训练与测试主循环 =================
def main():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = FetalBinaryDataset(DATASET_ROOT, "Train", train_transform)
    val_ds = FetalBinaryDataset(DATASET_ROOT, "Val", eval_transform)
    test_ds = FetalBinaryDataset(DATASET_ROOT, "Test", eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ClassicMVCNN(num_classes=2).to(DEVICE)
    
    # 类别权重：正常 vs 异常 约为 4.6 : 1
    weights = torch.tensor([1.0, 4.6]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0.0
    epochs_no_improve = 0
    save_path = "best_mvcnn_classic_binary.pth"
    
    print("\n" + "="*50)
    print("开始训练正宗 MVCNN (Su et al.) 二分类模型")
    print("="*50)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        _, _, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {running_loss/len(train_ds):.4f} | Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= PATIENCE:
            print(f"--- 早停触发 ---")
            break

    # 最终测试
    print("\n加载最佳模型进行最终测试...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(test_labels, test_preds)
    p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro', zero_division=0)
    print(f"\n[测试结果] Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    
    plot_results(test_labels, test_preds)
    print("混淆矩阵已保存为 binary_confusion_matrix.png")

if __name__ == "__main__":
    main()