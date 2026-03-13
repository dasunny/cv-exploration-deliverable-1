#!/usr/bin/env python
import matplotlib
matplotlib.use("Agg")
# coding: utf-8

# In[1]:


# Load dataset cell
# Mount Google Drive
# from google.colab import drive # UNCOMMENT THIS IF USING COLAB
import os
from PIL import Image
import numpy as np
# NOTE: I'm using my school's HPC, so I don't need to mount Google Drive. You can uncomment the above lines if you're using Google Colab.

# drive.mount('/content/drive')

# Set dataset path Use your own PATH!!
# dataset_path = '/scratch/jjung43/OceanicAI_Dataset/data/classification_dataset'
dataset_path = '/scratch/dfrom001/classification_dataset'
# dataset_path = '/content/drive/MyDrive/CV-Classification/classification_dataset'



# Load labels
labels_file = f'{dataset_path}/labels.txt'
image_names = []
labels = []

with open(labels_file, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split()
        image_names.append(parts[0])
        labels.append(parts[1])

# Load image paths
image_folder = f'{dataset_path}/images'
image_paths = [os.path.join(image_folder, name) for name in image_names]

print(f"Total images: {len(image_paths)}")
print(f"Total labels: {len(labels)}")

# Display label distribution
import matplotlib.pyplot as plt
from collections import Counter

label_counts = Counter(labels)

plt.figure(figsize=(10, 6))
plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Label Distribution')
plt.xticks(rotation=45)
plt.tight_layout()




# In[5]:


# TODO: Build dataset with transform
# Hint: transforms.Resize, transforms.ToTensor, transforms.Normalize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# I composed the transforms to be in line with standard resnet input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.label_to_idx[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

dataset = ImageDataset(image_paths, labels, transform=transform)
print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.unique_labels}")
print(f"Label to index: {dataset.label_to_idx}")


# 

# # **Split Dataset**
# 0.8 training, 0.1 validation, 0.1 testing
# 
# Reasoning:
# We'll use 0.8 for training because we aren't augmenting our data yet. We can consider switching to 0.75 training and 0.15 validation if we want to use data augmentation.

# In[9]:


# TODO: Split dataset and create DataLoaders
# Hint: use torch.utils.data.random_split
# Hint: DataLoader needs batch_size and shuffle
from torch.utils.data import random_split, DataLoader
import torch

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")


# In[10]:


# TODO: Define the model (Done)
# Hint: torchvision.models has many pretrained models to choose from
# Hint: don't forget to modify the final layer to match number of classes
import torch
import torch.nn as nn
from torchvision import models

num_classes = len(dataset.unique_labels)

# Option 1: ResNet18 (lightweight, good for small datasets)
# Hint: pretrained weights can be loaded with weights=models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Hint: the final layer is model.fc, replace it with a new Linear layer
model.fc = nn.Linear(model.fc.in_features, num_classes)


# TODO: Move model to GPU if available (Done)
# Hint: use torch.device and model.to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Using device: {device}")
print(f"Number of classes: {num_classes}")
print(f"Model: {model.__class__.__name__}")


# In[11]:


# TODO: Define loss function (Done)
# Hint: CrossEntropyLoss is standard for multi-class classification
# Hint: for class imbalance, use weight parameter in CrossEntropyLoss
criterion = torch.nn.CrossEntropyLoss()


# TODO: Define optimizer (Done)
# Hint: lr (learning rate) controls how big each update step is
# Hint: too high = unstable training, too low = slow convergence

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# TODO: Define learning rate scheduler (Done)
# Hint: scheduler adjusts learning rate during training
# Option 1: StepLR (reduce lr by gamma every step_size epochs)
# Hint: gamma=0.1 means lr is multiplied by 0.1 every step_size epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# TODO: Define the epoch number of training (Done)
num_epochs = 40

print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
print(f"Scheduler: {scheduler.__class__.__name__}")
print(f"Number of epochs: {num_epochs}")


# In[12]:


# TODO: Write the training loop
# Hint: each epoch consists of a training phase and a validation phase
# Hint: don't forget to switch between model.train() and model.eval()

# ============ Early stopping and checkpointing parameters ============
# best_val_acc = 0.0 <-- initialize best validation accuracy to 0, but I think we should be checking for val_loss?
best_val_loss = float('inf')     # Initialize best validation loss to infinity
patience = 20                     # Number of epochs to wait for improvement before stopping
epochs_no_improve = 0
checkpoint_dir = 'model_checkpoints' # Path to save the best model checkpoint
save_path = os.path.join(checkpoint_dir, 'best_resnet18.pth')
os.makedirs(checkpoint_dir, exist_ok=True)
# ======================================================================
from tqdm import tqdm

for epoch in range(num_epochs):

    # Training phase
    # model.train() enables dropout and batch normalization
    model.train()
    train_loss = 0.0
    train_correct = 0

    # Hint: wrap dataloader with tqdm to show progress bar
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")
    for images, labels in train_bar:

        images, labels = images.to(device), labels.to(device) #  move data to device
        optimizer.zero_grad() # zero the gradients before each batch

        # ================ forward pass ================
        outputs = model(images)
        loss = criterion(outputs, labels)
        # ==============================================

        # ====== backward pass and update weights ======
        loss.backward()
        optimizer.step()
        # ==============================================

        # =============== Log Performance ===============
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}") # update progress bar with current loss
        # ===============================================


    # =================== Validation phase ===================
    model.eval()
    val_loss = 0.0
    val_correct = 0

    # Hint: no gradient computation needed during validation
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val")
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    # step the scheduler after each epoch
    scheduler.step()

    # metrics for the epoch
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_acc = train_correct / len(train_dataset)
    val_acc = val_correct / len(val_dataset)
    
    # Print epoch results
    print(f"Epoch [{epoch+1}/{num_epochs}]"
          f" | Train Loss: {avg_train_loss:.4f}"
          f" | Train Acc: {train_acc:.4f}"
          f" | Val Loss: {avg_val_loss:.4f}"
          f" | Val Acc: {val_acc:.4f}")

    # --- Early Stopping & Model Checkpointing (based on validation loss) ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Save the model state dictionary (the weights)
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved to {save_path} with Val Loss: {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1
        print(f"No improvement in Val Loss for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= patience:
        print(f"Early stopping... Training stopped after {epoch+1} epochs.")
        break


# In[15]:


# Test Model Performance with the weights that achieved the best val accuracy

model.load_state_dict(torch.load(save_path))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')


# In[ ]:


# ============ Part 3: Hyperparameter Tuning ============

import time, shutil

def fresh_resnet18(num_classes, device):
    # Load pretrained ResNet18 and modify the final layer for our number of classes
    # Called at the start of each experiemnt to ensure fair comparison
    # (each run starts from the same pretrained weights, not influenced by previous runs)
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(device)

def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler=None, num_epochs=40, patience=20, save_name="best_model.pth", device=None):
    # Reusable training loop used across all Part 3 experiments
    # Monitors val loss for early stopping
    # saves best checkpoint during training, loads it before test evaluation
    # returns histories and metrics so each experiemnt can plot and compare results
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_path = os.path.join(checkpoint_dir, save_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_acc_history = []
    val_acc_history = []  
    total_train_time = 0.0

    for epoch in range(num_epochs):
        #--------------- Training phase ----------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        epoch_start = time.time()   # track time per epoch for experiment 3.2

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_time = time.time() - epoch_start
        total_train_time += epoch_train_time

        #--------------- Validation phase ----------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        if scheduler is not None:
            scheduler.step()

        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / len(train_loader.dataset)
        epoch_val_acc = val_correct / len(val_loader.dataset)

        train_acc_history.append(epoch_train_acc)
        val_acc_history.append(epoch_val_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}]"
            f" | Train Loss: {avg_train_loss:.4f}"
            f" | Train Acc: {epoch_train_acc:.4f}"
            f" | Val Loss: {avg_val_loss:.4f}"
            f" | Val Acc: {epoch_val_acc:.4f}"
            f" | Epoch Train Time: {epoch_train_time:.2f}s"
        )

        #--- Early Stopping & Checkpointing based on Val Loss ---
        # Save checkpoint if val loss improved, otherwwise incremement 'patience'
        # Training stops when no imrpovement for 'patience' consecutive epochs
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in Val Loss for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= patience:
            print(f"Early stopping... Training stopped after {epoch+1} epochs.")
            break

    #--- Load best checkpoint and evaluate on test set ---
    model.load_state_dict(torch.load(save_path))    # loads best checkpoint
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    print(f"Test Accuracy (best checkpoint): {test_acc*100:.2f}%\n")

    return train_acc_history, val_acc_history, test_acc, epoch+1, total_train_time




# In[ ]:


# 3.1 -- Learning Rate

# We test 4 LRs spanning 3 orders of magnitutde to see how sensitive the model is to this hyperparameter
# All other settings are fixed so LR is the only variable
print("=" * 60)
print("Tuning Learning Rate")
print("=" * 60)

learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
lr_results = {}         # stores results for plotting and table after all runs finish

for lr in learning_rates:
    print(f"\n--- LR = {lr} ---")
    model_lr = fresh_resnet18(num_classes, device)
    optimizer_lr = torch.optim.Adam(model_lr.parameters(), lr=lr)
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer_lr, step_size=5, gamma=0.5)
    
    train_hist, val_hist, test_acc, epochs_run, _ = train_and_evaluate(
        model_lr, train_loader, val_loader, test_loader, criterion, optimizer_lr, scheduler_lr, num_epochs=40, patience=20, save_name=f"best_resnet18_lr_{lr}.pth", device=device
    )
    lr_results[lr] = (train_hist, val_hist, test_acc, epochs_run)


# In[ ]:


# 3.1 -- Plot and Results
# Plot all 4 LR runs on the same figure for visual comparison
# best_lr is used automatically in 3.2 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for lr, (train_hist, val_hist, test_acc, _) in lr_results.items():
    x = range(1, len(train_hist) + 1)
    axes[0].plot(x, train_hist, label=f"lr={lr}")
    axes[1].plot(x, val_hist,   label=f"lr={lr}")

for ax, title in zip(axes, ["Training Accuracy", "Validation Accuracy"]):
    ax.set_title(f"{title} — Different LRs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("exp3_1_lr_curves.png", dpi=150)


# Results table
print(f"\n{'Learning Rate':<20} {'Test Accuracy':<18} {'Epochs Run'}")
print("-" * 50)
best_lr, best_lr_acc = None, 0.0
for lr, (_, _, test_acc, epochs_run) in lr_results.items():
    print(f"{lr:<20} {test_acc*100:.2f}%             {epochs_run}")
    if test_acc > best_lr_acc:
        best_lr_acc, best_lr = test_acc, lr

print(f"\n→ Best LR for Experiment 3.2: {best_lr}  ({best_lr_acc*100:.2f}%)")


# In[ ]:


# 3.2 -- Batch Size
# rebuild DataLoaders for each batch size, not just existing loaders
# Reuse the same underlying datasets (same split) so only batch size varies
# avg_time_per_epoch is for comparing speed fairly
# (total time would be unfair since different batch sizes could trigger early stopping at different epochs)
print("=" * 60)
print("Tuning Batch Size")
print("=" * 60)

batch_sizes = [16, 32, 64, 128]
bs_results = {}
for bs in batch_sizes:
    print(f"\n--- Batch Size = {bs} ---")
    train_loader_bs = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6, pin_memory=True)
    val_loader_bs = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=6, pin_memory=True)
    test_loader_bs = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=6, pin_memory=True)

    model_bs = fresh_resnet18(num_classes, device)
    optimizer_bs = torch.optim.Adam(model_bs.parameters(), lr=best_lr)
    scheduler_bs = torch.optim.lr_scheduler.StepLR(optimizer_bs, step_size=5, gamma=0.5)

    train_hist, val_hist, test_acc, epochs_run, total_time = train_and_evaluate(
        model_bs, train_loader_bs, val_loader_bs, test_loader_bs,
        criterion, optimizer_bs, scheduler_bs,
        num_epochs=40, patience=20,
        save_name=f"best_resnet18_bs_{bs}.pth",
        device=device
    )
    avg_time_per_epoch = total_time / epochs_run    # average time per epoch for fair speed comparison (fairer than total time since different runs)
    bs_results[bs] = (test_acc, avg_time_per_epoch, epochs_run)

# 3.2 -- Plot and Results

# Results table
print(f"\n{'Batch Size':<15} {'Test Accuracy':<18} {'Avg Time/Epoch (s)':<22} {'Epochs Run'}")
print("-" * 65)
best_bs, best_bs_acc = None, 0.0
for bs, (test_acc, avg_time, epochs_run) in bs_results.items():
    print(f"{bs:<15} {test_acc*100:.2f}%             {avg_time:.2f}s                  {epochs_run}")
    if test_acc > best_bs_acc:
        best_bs_acc, best_bs = test_acc, bs

print(f"\n→ Best Batch Size for Experiment 3.3: {best_bs}  ({best_bs_acc*100:.2f}%)")

# Rebuild loaders with best batch size for 3.3
best_train_loader = DataLoader(train_dataset, batch_size=best_bs, shuffle=True,  num_workers=6, pin_memory=True)
best_val_loader   = DataLoader(val_dataset,   batch_size=best_bs, shuffle=False, num_workers=6, pin_memory=True)
best_test_loader  = DataLoader(test_dataset,  batch_size=best_bs, shuffle=False, num_workers=6, pin_memory=True)


# In[ ]:


# Experiment 3.3 -- Optimizer


print("=" * 60)
print(f"Experiment 3.3 — Optimizer  (LR={best_lr}, BS={best_bs})")
print("=" * 60)

# Same LR given to both SGD and Adam for fair comparison. Note Adam is generally less sensitive to LR than SGD, but we want to give both optimizers the best chance to perform well.
optimizers_cfg = {
    "SGD":  lambda p: torch.optim.SGD(p,  lr=best_lr, momentum=0.9),    # SGD with momentum=0.9 is classic baseline
    "Adam": lambda p: torch.optim.Adam(p, lr=best_lr),                  # Adam adapts learning rates per parameter, often faster convergence
}
opt_results = {}

# Copy best model to a consistent filename so Part 4 always knows where to find it, regardless of which optimizer won
for opt_name, opt_factory in optimizers_cfg.items():
    print(f"\n--- Optimizer = {opt_name} ---")
    model_opt     = fresh_resnet18(num_classes, device)
    optimizer_opt = opt_factory(model_opt.parameters())
    scheduler_opt = torch.optim.lr_scheduler.StepLR(optimizer_opt, step_size=5, gamma=0.5)

    train_hist, val_hist, test_acc, epochs_run, total_time = train_and_evaluate(
        model_opt, best_train_loader, best_val_loader, best_test_loader, criterion, optimizer_opt, scheduler_opt, num_epochs=40, patience=20, save_name=f"best_resnet18_opt_{opt_name}.pth", device=device
    )
    opt_results[opt_name] = (train_hist, val_hist, test_acc, epochs_run, total_time)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for opt_name, (train_hist, val_hist, _, _, _) in opt_results.items():
    x = range(1, len(train_hist) + 1)
    axes[0].plot(x, train_hist, label=opt_name)
    axes[1].plot(x, val_hist,   label=opt_name)

for ax, title in zip(axes, ["Training Accuracy", "Validation Accuracy"]):
    ax.set_title(f"{title} — SGD vs Adam")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("exp3_3_optimizer_curves.png", dpi=150)


# Results table
print(f"\n{'Optimizer':<12} {'Test Accuracy':<18} {'Epochs Run':<15} {'Total Train Time (s)'}")
print("-" * 60)
for opt_name, (_, _, test_acc, epochs_run, total_time) in opt_results.items():
    print(f"{opt_name:<12} {test_acc*100:.2f}%             {epochs_run:<15} {total_time:.1f}s")

# Save best model for Part 4
# Copy best optimizer's checkpoint to a consistent name for part 4
best_opt_name = max(opt_results, key=lambda k: opt_results[k][2])
shutil.copy(
    os.path.join(checkpoint_dir, f"best_resnet18_opt_{best_opt_name}.pth"),
    os.path.join(checkpoint_dir, "best_part3.pth"),
)
print(f"\n→ Best model ({best_opt_name}) copied to best_part3.pth for Part 4.")
