import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 0. seed
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False 
        torch.backends.cudnn.benchmark = True

# ==========================================
# 1.  Dataset
# ==========================================
class MaizeNitrogenDataset(Dataset):
    def __init__(self, csv_file, split='train', transform=None, img_dir=''):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.loc[idx, 'filepath'])
        image = Image.open(img_path).convert('RGB')
        label = self.df.loc[idx, 'label_index']
        if self.transform:
            image = self.transform(image)
        return image, label


def get_efficientnet_b0(num_classes, dropout_rate=0.3):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model

# ==========================================
# 2. Visualisation
# ==========================================
def evaluate_and_save_metrics(model, dataloader, device, exp_name, classes=['N0', 'N75', 'NFull']):
    model.eval()
    all_preds = []
    all_labels = []
    
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="[正在生成评估报告]"):
            images, labels = images.to(device), labels.to(device)
            
            use_amp = device.type == 'cuda'
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 1. CSV
    report_dict = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    csv_path = f'metrics_report_{exp_name}.csv'
    report_df.to_csv(csv_path)
    print(f"saved: {csv_path}")

    # 2. PNG
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 14})
    plt.title(f'Confusion Matrix ({exp_name})', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = f'confusion_matrix_{exp_name}.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")

def plot_training_curves(history, exp_name):
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(12, 5))

    # Accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'b--', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'b-', linewidth=2, label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'r--', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    curve_path = f'training_curves_{exp_name}.png'
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Accuracy curve saved: {curve_path}")

# ==========================================
# 4. main
# ==========================================
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    set_seed(42)

    # ========================================================
    # Parameters
    # ========================================================
    AUG_MODE = 'mild'   #  'mild' or 'medium' 
    
    CONFIG = {
        'csv_file': 'split.csv',
        'img_dir': './',
        'batch_size': 16,
        'num_epochs': 20, 
        'num_classes': 3,
        'learning_rate': 3e-4,
        'weight_decay': 1e-2,
        'dropout_rate': 0.4,
        'label_smoothing': 0.1,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    # ========================================================

    print(f"Environment: {CONFIG['device']} | AUG_MODE: [{AUG_MODE.upper()}]")

    # AUG_MODE
    transforms_dict = {
        'mild': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'medium': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    train_transform = transforms_dict[AUG_MODE]
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # data set
    df_temp = pd.read_csv(CONFIG['csv_file'])
    val_split_name = 'val' if 'val' in df_temp['split'].unique() else 'test'
    
    train_dataset = MaizeNitrogenDataset(CONFIG['csv_file'], split='train', transform=train_transform, img_dir=CONFIG['img_dir'])
    val_dataset = MaizeNitrogenDataset(CONFIG['csv_file'], split=val_split_name, transform=val_transform, img_dir=CONFIG['img_dir'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Initialisation
    model = get_efficientnet_b0(CONFIG['num_classes'], CONFIG['dropout_rate']).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    
    
    use_amp = CONFIG['device'].type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if hasattr(torch.cuda.amp, 'GradScaler') else None

    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_path = f'best_model_{AUG_MODE}.pth'

    for epoch in range(CONFIG['num_epochs']):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]")
        for images, labels in pbar:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=CONFIG['device'].type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        
        # --- Val ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                with torch.autocast(device_type=CONFIG['device'].type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        # Record and print the current information
        epoch_val_acc = val_correct / val_total
        history['train_loss'].append(train_loss / train_total)
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"-> Val Acc: {epoch_val_acc:.4f}", end="")
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(" (New model saved)")
        else:
            print()

    print("\n" + "="*50)
    print(f"Highest accuracy rate: {best_val_acc:.4f}")
    
    #   Loss and Accuracy 
    plot_training_curves(history, exp_name=AUG_MODE)

    
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    evaluate_and_save_metrics(model, val_loader, CONFIG['device'], exp_name=AUG_MODE)
    
    print("="*50)
    print("All completed!")