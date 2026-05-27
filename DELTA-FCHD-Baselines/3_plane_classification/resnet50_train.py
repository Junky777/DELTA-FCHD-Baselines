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
# ================= 1. Configuration (updated) =================
DATASET_ROOT = Path("DELTA_Dataset_Splits")
BATCH_SIZE = 32
EPOCHS = 100              # Increase the maximum number of training epochs
LEARNING_RATE = 1e-4
PATIENCE = 20            # Early-stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define plane categories and their label IDs
PLANE_CLASSES = {
    "Abdomen": 0,
    "4CH": 1,
    "LVOT": 2,
    "RVOT": 3,
    "3VT": 4
}
NUM_CLASSES = len(PLANE_CLASSES)

# ================= 2. Custom dataset loader =================
class FetalPlaneDataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        """
        root_dir: dataset root directory, e.g., DELTA_Dataset_Splits
        split: "Train", "Val", or "Test"
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        split_dir = root_dir / split
        
        # Traverse: split -> disease_class -> patient -> images
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
        
        # Convert grayscale or RGBA images to standard RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ================= 3. Data preprocessing pipeline =================
# Use ImageNet normalization statistics
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),    # Data augmentation
    transforms.RandomRotation(10),        # Minor rotation augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= 4. Model construction =================
def build_resnet50(num_classes):
    # Load pretrained ResNet-50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify the final fully connected layer for the five-class task
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ================= 5. Evaluation function =================
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
    
    # Compute macro precision, recall, and F1-score
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return epoch_loss, acc, precision, recall, f1
# ================= 6. Main training loop with early stopping =================
def main():
    print("Loading dataset...")
    train_dataset = FetalPlaneDataset(DATASET_ROOT, split="Train", transform=train_transforms)
    val_dataset = FetalPlaneDataset(DATASET_ROOT, split="Val", transform=eval_transforms)
    test_dataset = FetalPlaneDataset(DATASET_ROOT, split="Test", transform=eval_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    model = build_resnet50(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_f1 = 0.0
    epochs_no_improve = 0  # Counter for consecutive epochs without improvement
    
    print("\n--- Start training ---")
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
        
        # Check whether to save the model and update early-stopping counter
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_resnet50_plane_cls.pth")
            print("  --> New best model found and saved.")
            epochs_no_improve = 0  # Reset early-stopping counter
        else:
            epochs_no_improve += 1
            print(f"  --> Early-stopping counter: {epochs_no_improve}/{PATIENCE}")
            
        # Trigger early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\n!!! Early stopping triggered: no F1 improvement for {PATIENCE} consecutive epochs. Stopping training early. !!!")
            break

    print("\n--- Final evaluation on the test set ---")
    # Load the best weights saved by early stopping
    model.load_state_dict(torch.load("best_resnet50_plane_cls.pth"))
    test_loss, test_acc, test_pre, test_rec, test_f1 = evaluate_model(model, test_loader, criterion)
    
    print("Final Test performance report:")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_pre:.4f}")
    print(f"Recall   : {test_rec:.4f}")
    print(f"F1-score : {test_f1:.4f}")

if __name__ == "__main__":
    main()
