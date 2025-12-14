# F:\ChessXDetection\src\training\train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import sys

# Import configuration constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from config import DATASET_DIR, MODELS_DIR, MODEL_SAVE_PATH, PIECE_CLASSES

# This main execution block is required for multiprocessing on Windows
if __name__ == '__main__':

    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # --- IMAGE TRANSFORMS ---
    # Standard transforms for ResNet-based color images, including augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- DATASET & DATALOADER ---
    try:
        # ImageFolder automatically uses the directory structure (dataset/wK, dataset/bP)
        train_data = datasets.ImageFolder(DATASET_DIR, transform=transform)
    except FileNotFoundError:
        print(f"ERROR: Dataset directory not found at {DATASET_DIR}. Run 'python main.py label' first.")
        sys.exit(1)
        
    # Check for correct number of classes (should be 13)
    if len(train_data.classes) != len(PIECE_CLASSES):
        print(f"WARNING: Dataset found {len(train_data.classes)} classes, expected {len(PIECE_CLASSES)}. Check your dataset folders.")

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

    # --- MODEL SETUP (ResNet18 with Transfer Learning) --- 

[Image of CNN Architecture Diagram]

    model = models.resnet18(weights='IMAGENET1K_V1') 
    # Replace the final layer for classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(PIECE_CLASSES)) 

    # --- TRAINING ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ... (Training loop code remains, ensuring running_loss is tracked and printed) ...

    # --- SAVE THE MODEL ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Training finished. Model saved to {MODEL_SAVE_PATH}")