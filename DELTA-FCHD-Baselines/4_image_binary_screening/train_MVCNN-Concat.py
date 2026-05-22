import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F

# ================= 1. 配置与权重设置 =================
DATASET_ROOT = Path("DELTA_Dataset_Splits")
BATCH_SIZE = 16 # 多图输入，显存占用较高，建议 16
EPOCHS = 100
LEARNING_RATE = 1e-5 # 融合任务更复杂，建议使用更小的学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 13 分类映射
CLASS_NAMES = [
    "00_Normal", "01_TOF", "02_DORV", "03_PTA", "04_TGA", 
    "05_AVSD", "06_SV", "07_HLHS", "08_HRHS", "09_AA", 
    "10_PS", "11_PLSVC", "12_RAA"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# 根据您提供的数据量计算类别权重 (用于处理长尾分布)
# 正常:619, TOF:33, DORV:11, PTA:12, TGA:4, AVSD:9, SV:5, HLHS:6, HRHS:7, AA:17, PS:8, PLSVC:10, RAA:11
samples_per_class = np.array([619, 33, 11, 12, 4, 9, 5, 6, 7, 17, 8, 10, 11])
weights = 1.0 / samples_per_class
weights = weights / weights.sum() * len(CLASS_NAMES)
CLASS_WEIGHTS = torch.FloatTensor(weights).to(DEVICE)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # 可以传入我们之前算好的 CLASS_WEIGHTS

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ================= 2. 多视角数据集加载器 =================
class FetalMultiViewDataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        self.transform = transform
        self.cases = [] # 存放 (patient_path, label_id)
        
        split_dir = root_dir / split
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists(): continue
            
            label_id = CLASS_TO_IDX[class_name]
            for patient_folder in class_dir.iterdir():
                if patient_folder.is_dir():
                    # 检查是否 5 张图都齐全
                    imgs = list(patient_folder.glob("*.jpg")) + list(patient_folder.glob("*.png"))
                    if len(imgs) >= 5:
                        self.cases.append((patient_folder, label_id))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        patient_path, label = self.cases[idx]
        
        # 定义 5 个标准切面的文件名
        plane_names = ["Abdomen", "4CH", "LVOT", "RVOT", "3VT"]
        view_images = []
        
        for plane in plane_names:
            # 匹配文件夹下对应的切面图片
            img_path = list(patient_path.glob(f"{plane}.*"))[0]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            view_images.append(img)
        
        # 将 5 张图堆叠为 (5, 3, 224, 224)
        return torch.stack(view_images), label

# ================= 3. MVCNN-Concat 模型架构 =================
class MVCNNConcat(nn.Module):
    def __init__(self, num_classes, backbone_type="resnet50"):
        super(MVCNNConcat, self).__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(base_model.children())[:-1]))
        
        # --- 新增：给单个视图降维 (2048 -> 256) ---
        self.reduce_dim = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 分类头：5个切面 * 256维 = 1280维 (大大减轻分类器压力)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        batch_size, num_views, c, h, w = x.shape
        x = x.view(batch_size * num_views, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size * num_views, -1) # (batch*5, 2048)
        
        # 降维
        reduced_features = self.reduce_dim(features) # (batch*5, 256)
        reduced_features = reduced_features.view(batch_size, num_views, -1) # (batch, 5, 256)
        
        # 拼接并分类
        combined_features = reduced_features.view(batch_size, -1) # (batch, 1280)
        out = self.classifier(combined_features)
        return out

# ================= 4. 训练与评估逻辑 =================
def train_and_eval():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = FetalMultiViewDataset(DATASET_ROOT, "Train", transform)
    val_ds = FetalMultiViewDataset(DATASET_ROOT, "Val", transform)
    test_ds = FetalMultiViewDataset(DATASET_ROOT, "Test", transform)
    # --- 新增：计算训练集每个样本的采样权重 ---
    sample_weights = []
    for _, label_id in train_ds.cases:
        # 赋予少数类更高的被抽中概率
        sample_weights.append(weights[label_id]) 
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MVCNNConcat(num_classes=len(CLASS_NAMES)).to(DEVICE)
    # 使用加权交叉熵损失处理长尾数据
    # criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    criterion = FocalLoss(alpha=CLASS_WEIGHTS, gamma=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 每个 Epoch 评估一次验证集 (逻辑同前，此处省略具体 evaluate 函数)
        # 如果 val_f1 > best_f1，则保存模型并更新 best_f1
        print(f"Epoch {epoch+1}/{EPOCHS} 完成...")

    print("开始最终测试...")

    
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算整体指标
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print("\n" + "="*45)
    print("最终 Test 性能报告 (MVCNN-Concat):")
    print("="*45)
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro)   : {recall:.4f}")
    print(f"F1-score (Macro) : {f1_macro:.4f}")
    print("-" * 45)

    # 计算每个类别的具体 F1 分数
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    print("各疾病亚型 F1-score 详情:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  [{i:02d}] {class_name:<10} : {f1_per_class[i]:.4f}")
    print("="*45)

if __name__ == "__main__":
    train_and_eval()