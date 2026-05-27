import os
# ================= 0. Compute-resource scheduling =================
# Force use of the first four GPUs; ensure they are available


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

# ================= 1. Configuration =================
DATASET_ROOT = Path("DELTA_Dataset_Splits")
# Four-GPU training with global batch size 16
# If OOM still occurs, reduce it to 8
BATCH_SIZE = 16 
EPOCHS = 100
LEARNING_RATE = 5e-5 
PATIENCE = 15
NUM_FRAMES = 32  # Use 32 frames to cover more of the cardiac cycle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "00_Normal", "01_TOF", "02_DORV", "03_SGA", "04_TGA", 
    "05_AVSD", "06_SV", "07_HLHS", "08_HRHS", "09_AA", 
    "10_PS", "11_PLSVC", "12_RAA"
]

# ================= 2. Video dataset loader =================
class FetalVideoBinaryDataset(Dataset):
    def __init__(self, root_dir, split="Train", num_frames=32, transform=None):
        self.transform = transform
        self.num_frames = num_frames
        self.cases = []
        
        split_dir = root_dir / split
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists(): continue
            label_id = 0 if class_name == "00_Normal" else 1
            for patient_folder in class_dir.iterdir():
                if patient_folder.is_dir():
                    video_files = list(patient_folder.glob("video.*"))
                    if video_files:
                        self.cases.append((video_files[0], label_id))

    def __len__(self):
        return len(self.cases)

    def _read_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.zeros(self.num_frames, dtype=int)

        curr_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if curr_frame in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            curr_frame += 1
        cap.release()

        while len(frames) < self.num_frames:
            frames.append(frames[-1] if len(frames) > 0 else np.zeros((224, 224, 3), dtype=np.uint8))
        frames = frames[:self.num_frames]
        return frames

    def __getitem__(self, idx):
        video_path, label = self.cases[idx]
        frames = self._read_video(video_path)
        
        processed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            processed_frames.append(frame)
        
        video_tensor = torch.stack(processed_frames) 
        video_tensor = video_tensor.permute(1, 0, 2, 3) 
        return video_tensor, label

# ================= 3. Video Swin Transformer architecture =================
def build_swin3d(num_classes=2):
    weights = video_models.Swin3D_S_Weights.KINETICS400_V1
    model = video_models.swin3d_s(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model

# ================= 4. Plotting helper function =================
def plot_results(y_true, y_pred, save_path="video_swin3d_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=['Normal', 'Abnormal'], 
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Baseline 5: Swin3D (32 Frames) Binary Classification')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================= 5. Training and evaluation loop =================
def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Extracting {NUM_FRAMES} frames from video files...")
    train_ds = FetalVideoBinaryDataset(DATASET_ROOT, "Train", NUM_FRAMES, transform)
    val_ds = FetalVideoBinaryDataset(DATASET_ROOT, "Val", NUM_FRAMES, transform)
    test_ds = FetalVideoBinaryDataset(DATASET_ROOT, "Test", NUM_FRAMES, transform)

    # Set num_workers according to CPU cores; 8 is used here
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # Build model
    model = build_swin3d(num_classes=2)
    
    # Enable multi-GPU parallelism with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPU(s); nn.DataParallel enabled.")
        model = nn.DataParallel(model)
        
    model = model.to(DEVICE)
    
    weights = torch.tensor([1.0, 4.6]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0.0
    epochs_no_improve = 0
    save_path = "best_swin3d_binary_4gpu.pth"
    
    print("\n" + "="*50)
    print("Start training Video Swin Transformer")
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

        # Validation-set evaluation
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
            
            # Remove the DataParallel-generated 'module.' prefix when saving weights
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
                
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= PATIENCE:
            print(f"--- Early stopping triggered ---")
            break

    print("\nLoading the best 32-frame Swin3D model for final testing...")
    
    # Build a clean model for testing without DataParallel and load weights
    test_model = build_swin3d(num_classes=2).to(DEVICE)
    test_model.load_state_dict(torch.load(save_path))
    test_model.eval()
    
    test_preds, test_labels = [], []
    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            outputs = test_model(videos)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(test_labels, test_preds)
    p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro', zero_division=0)
    print(f"\n[Swin3D 32-frame video test results] Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    
    plot_results(test_labels, test_preds)
    print("Confusion matrix saved as video_swin3d_confusion_matrix.png")

if __name__ == "__main__":
    main()
