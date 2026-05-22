import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models.video as video_models
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 配置区 =================
DATASET_ROOT = Path("DELTA_Dataset_Splits")
BATCH_SIZE = 16  # 视频极占显存，如 OOM 请降至 4
EPOCHS = 100
LEARNING_RATE = 5e-5 # 视频预训练模型建议使用更小的学习率微调
PATIENCE = 30
NUM_FRAMES = 32 # 从每个视频中均匀提取的帧数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "00_Normal", "01_TOF", "02_DORV", "03_PTA", "04_TGA", 
    "05_AVSD", "06_SV", "07_HLHS", "08_HRHS", "09_AA", 
    "10_PS", "11_PLSVC", "12_RAA"
]

# ================= 2. 视频数据集加载器 =================
class FetalVideoBinaryDataset(Dataset):
    def __init__(self, root_dir, split="Train", num_frames=16, transform=None):
        self.transform = transform
        self.num_frames = num_frames
        self.cases = []
        
        split_dir = root_dir / split
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists(): continue
            
            # 二分类逻辑：正常为 0，异常为 1
            label_id = 0 if class_name == "00_Normal" else 1
            
            for patient_folder in class_dir.iterdir():
                if patient_folder.is_dir():
                    # 寻找视频文件 (支持多种后缀)
                    video_files = list(patient_folder.glob("video.*"))
                    if video_files:
                        self.cases.append((video_files[0], label_id))

    def __len__(self):
        return len(self.cases)

    def _read_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 均匀采样索引
        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.zeros(self.num_frames, dtype=int) # 防御性编程

        curr_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if curr_frame in indices:
                # BGR 转 RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            curr_frame += 1
        cap.release()

        # 填补由于视频损坏或过短导致帧数不够的情况
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if len(frames) > 0 else np.zeros((224, 224, 3), dtype=np.uint8))
        
        # 截断多余的帧
        frames = frames[:self.num_frames]
        return frames

    def __getitem__(self, idx):
        video_path, label = self.cases[idx]
        frames = self._read_video(video_path) # list of numpy arrays (H, W, C)
        
        processed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame) # 返回 (C, H, W)
            processed_frames.append(frame)
        
        # 堆叠所有帧: (T, C, H, W) -> R(2+1)D 要求输入格式为 (C, T, H, W)
        video_tensor = torch.stack(processed_frames) 
        video_tensor = video_tensor.permute(1, 0, 2, 3) 
        
        return video_tensor, label

# ================= 3. R(2+1)D 权威视频分类架构 =================
def build_r2plus1d(num_classes=2):
    # 加载 Kinetics-400 数据集上的预训练权重
    weights = video_models.R2Plus1D_18_Weights.KINETICS400_V1
    model = video_models.r2plus1d_18(weights=weights)
    
    # 替换最后的全连接层以适应我们的二分类任务
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ================= 4. 绘图辅助函数 =================
def plot_results(y_true, y_pred, save_path="video_binary_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Normal', 'Abnormal'], 
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Baseline 4: R(2+1)D Video Binary Classification')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================= 5. 训练与评估主循环 =================
def main():
    # 注意：R(2+1)D 的标准输入大小通常是 112x112，但预训练模型能适应 224x224
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)), # 使用 112x112 可以大幅节约显存并加快训练
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                             std=[0.22803, 0.22145, 0.216989]) # Kinetics400 标准均值方差
    ])

    print("正在从视频文件中提取时空序列，这可能需要一些时间...")
    train_ds = FetalVideoBinaryDataset(DATASET_ROOT, "Train", NUM_FRAMES, transform)
    val_ds = FetalVideoBinaryDataset(DATASET_ROOT, "Val", NUM_FRAMES, transform)
    test_ds = FetalVideoBinaryDataset(DATASET_ROOT, "Test", NUM_FRAMES, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_r2plus1d(num_classes=2).to(DEVICE)
    weights = torch.tensor([1.0, 4.6]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0.0
    epochs_no_improve = 0
    save_path = "best_r2plus1d_binary.pth"
    
    print("\n" + "="*50)
    print("开始训练权威视频模型 R(2+1)D (Tran et al.)")
    print("="*50)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for videos, labels in train_loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * videos.size(0)

        # 验证集评估
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(DEVICE), labels.to(DEVICE)
                outputs = model(videos)
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

    print("\n加载 R(2+1)D 最佳模型进行最终测试...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            outputs = model(videos)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(test_labels, test_preds)
    p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro', zero_division=0)
    print(f"\n[R(2+1)D 视频测试结果] Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    
    plot_results(test_labels, test_preds)
    print("混淆矩阵已保存为 video_binary_confusion_matrix.png")

if __name__ == "__main__":
    main()