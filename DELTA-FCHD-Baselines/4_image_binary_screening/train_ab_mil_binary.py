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

# ================= 1. Configuration =================
DATASET_ROOT = Path("DELTA_Dataset_Splits")
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "00_Normal", "01_TOF", "02_DORV", "03_SGA", "04_TGA", 
    "05_AVSD", "06_SV", "07_HLHS", "08_HRHS", "09_AA", 
    "10_PS", "11_PLSVC", "12_RAA"
]

# ================= 2. Binary-classification dataset loader =================
class FetalBinaryDataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        self.transform = transform
        self.cases = []
        split_dir = root_dir / split
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists(): continue
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

# ================= 3. Reference model: Attention-based MIL (Ilse et al., ICML 2018) =================
class AttentionMIL(nn.Module):
    def __init__(self, num_classes=2):
        super(AttentionMIL, self).__init__()
        # 1. Instance-level feature extractor
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_dim = base_model.fc.in_features # 2048
        self.features = nn.Sequential(*(list(base_model.children())[:-1]))
        
        # 2. MIL attention module as defined in Ilse et al.
        self.attention_V = nn.Linear(self.feature_dim, 256)
        self.attention_w = nn.Linear(256, 1)
        
        # 3. Bag-level classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # x: (Batch, 5, 3, 224, 224)
        batch_size, num_instances, c, h, w = x.shape
        
        # Extract features for each plane (instance)
        x = x.view(batch_size * num_instances, c, h, w)
        H = self.features(x) # (Batch*5, 2048, 1, 1)
        H = H.view(batch_size, num_instances, self.feature_dim) # (Batch, 5, 2048)
        
        # MIL core: compute attention weights for each instance
        # Formula from the paper: a_k = w^T * tanh(V * h_k)
        A = self.attention_V(H)      # (Batch, 5, 256)
        A = torch.tanh(A)            # (Batch, 5, 256)
        A = self.attention_w(A)      # (Batch, 5, 1)
        
        # Apply softmax normalization across instances (planes)
        A = F.softmax(A, dim=1)      # (Batch, 5, 1)
        
        # Aggregate the five features using attention weights
        # Formula: z = \sum_{k=1}^{K} a_k * h_k
        Z = torch.sum(H * A, dim=1)  # (Batch, 2048)
        
        # Pass the aggregated feature to the classifier
        out = self.classifier(Z)
        
        # Return both classification logits and attention weights
        # Attention weights can be used for interpretability analysis
        return out, A

# ================= 4. Plotting helper function =================
def plot_results(y_true, y_pred, save_path="binary_confusion_matrix_mil.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=['Normal', 'Abnormal'], 
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Baseline 3: Attention-based MIL Binary Classification')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================= 5. Training and testing loop =================
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

    model = AttentionMIL(num_classes=2).to(DEVICE)
    weights = torch.tensor([1.0, 4.6]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0.0
    epochs_no_improve = 0
    save_path = "best_ab_mil_binary.pth"
    
    print("\n" + "="*50)
    print("Start training the AB-MIL binary-classification model (Ilse et al.)")
    print("="*50)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            # The MIL model returns two values: logits and attention_weights
            outputs, _ = model(imgs) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs, _ = model(imgs)
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
            print(f"--- Early stopping triggered ---")
            break

    print("\nLoading the best MIL model for final testing...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs, _ = model(imgs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(test_labels, test_preds)
    p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro', zero_division=0)
    print(f"\n[AB-MIL test results] Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    
    plot_results(test_labels, test_preds)
    print("Confusion matrix saved as binary_confusion_matrix_mil.png")

if __name__ == "__main__":
    main()
