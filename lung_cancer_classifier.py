import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np
from tqdm import tqdm

class LungCancerDataset(Dataset):
    """Custom Dataset class for Lung Cancer CT Scan Images"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Class mapping
        self.class_to_idx = {
            'adenocarcinoma': 0,
            'large.cell.carcinoma': 1,
            'squamous.cell.carcinoma': 2
        }
        
        # Load images and labels
        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
        
        print(f"Found {len(self.image_paths)} images")
        print("Class distribution:")
        for class_name, idx in self.class_to_idx.items():
            count = sum(1 for label in self.labels if label == idx)
            print(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class LungCancerClassifier(nn.Module):
    """CNN Model for Lung Cancer Classification using Transfer Learning"""
    
    def __init__(self, num_classes=3):
        super(LungCancerClassifier, self).__init__()
        # Using ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
            
        # Replace the final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
    def train_epoch(self, dataloader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, dataloader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels, all_probabilities
    
    def train(self, train_loader, val_loader, criterion, optimizer, scheduler):
        best_val_acc = 0.0
        
        for epoch in range(self.config['num_epochs']):
            print(f'\nEpoch [{epoch+1}/{self.config["num_epochs"]}]')
            print('-' * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels, val_probs = self.validate_epoch(
                val_loader, criterion
            )
            
            # Update learning rate
            if scheduler:
                scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_lung_cancer_model.pth')
                print(f'New best model saved! Validation Accuracy: {val_acc:.2f}%')
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Best Val Acc: {best_val_acc:.2f}%')
        
        return self.history

def create_data_loaders(data_dir, batch_size=32, image_size=224):
    """Create data loaders with augmentation"""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = LungCancerDataset(data_dir, transform=train_transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply val transform to validation set
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, dataset.class_to_idx

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(true_labels, predictions, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def plot_roc_curve(true_labels, probabilities, class_names):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    
    for i, class_name in enumerate(class_names):
        # Binarize labels for current class
        binary_true = (np.array(true_labels) == i).astype(int)
        class_probs = np.array(probabilities)[:, i]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(binary_true, class_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    
    # Configuration
    config = {
        'data_dir': 'ct-scan-images-of-lung-cancer',  # Update this path
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 30,
        'num_classes': 3,
        'image_size': 224
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, class_to_idx = create_data_loaders(
        config['data_dir'], 
        config['batch_size'], 
        config['image_size']
    )
    
    # Initialize model
    print("Initializing model...")
    model = LungCancerClassifier(num_classes=config['num_classes']).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Initialize trainer
    trainer = ModelTrainer(model, device, config)
    
    # Start training
    print("Starting training...")
    history = trainer.train(train_loader, val_loader, criterion, optimizer, scheduler)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_lung_cancer_model.pth'))
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    final_val_loss, final_val_acc, final_preds, final_labels, final_probs = trainer.validate_epoch(
        val_loader, criterion
    )
    
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    # Class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i].replace('.', ' ').title() for i in range(len(class_to_idx))]
    
    # Generate plots and metrics
    print("\nGenerating evaluation plots...")
    
    # Training history
    plot_training_history(history)
    
    # Confusion matrix
    cm = plot_confusion_matrix(final_labels, final_preds, class_names)
    
    # ROC curves
    plot_roc_curve(final_labels, final_probs, class_names)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds, target_names=class_names))
    
    # Calculate additional metrics
    accuracy = np.mean(np.array(final_preds) == np.array(final_labels))
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(final_labels) == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(np.array(final_preds)[class_mask] == i)
            print(f"  {class_name}: {class_accuracy:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 'final_lung_cancer_model.pth')
    print(f"\nBest model saved with validation accuracy: {final_val_acc:.2f}%")

if __name__ == "__main__":
    main()