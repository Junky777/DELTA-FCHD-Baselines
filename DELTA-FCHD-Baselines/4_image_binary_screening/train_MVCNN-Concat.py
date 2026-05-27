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

# ================= 1. Configuration and class-weight settings =================
DATASET_ROOT = Path("DELTA_Dataset_Splits")
BATCH_SIZE = 16  # Multi-image input is memory intensive; 16 is recommended
EPOCHS = 100
LEARNING_RATE = 1e-5  # Use a smaller learning rate for the fusion task
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 13-class label mapping
CLASS_NAMES = [
    "00_Normal", "01_TOF", "02_DORV", "03_SGA", "04_TGA", 
    "05_AVSD", "06_SV", "07_HLHS", "08_HRHS", "09_AA", 
    "10_PS", "11_PLSVC", "12_RAA"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Compute class weights from the dataset counts for long-tail imbalance
# Normal:619, TOF:33, DORV:11, SGA:12, TGA:4, AVSD:9, SV:5, HLHS:6, HRHS:7, AA:17, PS:8, PLSVC:10, RAA:11
samples_per_class = np.array([619, 33, 11, 12, 4, 9, 5, 6, 7, 17, 8, 10, 11])
weights = 1.0 / samples_per_class
weights = weights / weights.sum() * len(CLASS_NAMES)
CLASS_WEIGHTS = torch.FloatTensor(weights).to(DEVICE)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Precomputed CLASS_WEIGHTS can be passed in

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ================= 2. Multi-view dataset loader =================
class FetalMultiViewDataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        self.transform = transform
        self.cases = []  # Store (patient_path, label_id)
        
        split_dir = root_dir / split
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists(): continue
            
            label_id = CLASS_TO_IDX[class_name]
            for patient_folder in class_dir.iterdir():
                if patient_folder.is_dir():
                    # Check whether all five images are present
                    imgs = list(patient_folder.glob("*.jpg")) + list(patient_folder.glob("*.png"))
                    if len(imgs) >= 5:
                        self.cases.append((patient_folder, label_id))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        patient_path, label = self.cases[idx]
        
        # Define filenames for the five standard planes
        plane_names = ["Abdomen", "4CH", "LVOT", "RVOT", "3VT"]
        view_images = []
        
        for plane in plane_names:
            # Match the corresponding plane image in the folder
            img_path = list(patient_path.glob(f"{plane}.*"))[0]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            view_images.append(img)
        
        # Stack the five images as (5, 3, 224, 224)
        return torch.stack(view_images), label

# ================= 3. MVCNN-Concat model architecture =================
class MVCNNConcat(nn.Module):
    def __init__(self, num_classes, backbone_type="resnet50"):
        super(MVCNNConcat, self).__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(base_model.children())[:-1]))
        
        # Added: reduce dimensionality for each view (2048 -> 256)
        self.reduce_dim = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier head: five planes times 256 dimensions = 1280 dimensions
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
        
        # Dimensionality reduction
        reduced_features = self.reduce_dim(features) # (batch*5, 256)
        reduced_features = reduced_features.view(batch_size, num_views, -1) # (batch, 5, 256)
        
        # Concatenate and classify
        combined_features = reduced_features.view(batch_size, -1) # (batch, 1280)
        out = self.classifier(combined_features)
        return out

# ================= 4. Training and evaluation logic =================
def train_and_eval():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = FetalMultiViewDataset(DATASET_ROOT, "Train", transform)
    val_ds = FetalMultiViewDataset(DATASET_ROOT, "Val", transform)
    test_ds = FetalMultiViewDataset(DATASET_ROOT, "Test", transform)
    # Compute sampling weights for each training sample
    sample_weights = []
    for _, label_id in train_ds.cases:
        # Assign higher sampling probability to minority classes
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
    # Use weighted cross-entropy for long-tail data
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

        # Evaluate on the validation set once per epoch
        # If val_f1 > best_f1, save the model and update best_f1
        print(f"Epoch {epoch+1}/{EPOCHS} completed...")

    print("Start final testing...")

    
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

    # Compute overall metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print("\n" + "="*45)
    print("Final Test performance report (MVCNN-Concat):")
    print("="*45)
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro)   : {recall:.4f}")
    print(f"F1-score (Macro) : {f1_macro:.4f}")
    print("-" * 45)

    # Compute F1-score for each class
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    print("F1-score by disease subtype:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  [{i:02d}] {class_name:<10} : {f1_per_class[i]:.4f}")
    print("="*45)

if __name__ == "__main__":
    train_and_eval()
