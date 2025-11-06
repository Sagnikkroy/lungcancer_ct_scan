import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')

print("FINAL LUNG CANCER CLASSIFIER - WITH GPU SUPPORT")

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
            'large cell carcinoma': 1,
            'squamous cell carcinoma': 2,
            'normal cases': 3,
            'benign cases': 4
        }
        
        # Load images and labels
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)
        
        print(f"Found {len(self.image_paths)} images")
        print("Class distribution:")
        for class_name, idx in self.class_to_idx.items():
            count = sum(1 for label in self.labels if label == idx)
            print(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return dummy image if there's an error
            return torch.zeros(3, 128, 128), 0

class LungCancerClassifier(nn.Module):
    """CNN Model for Lung Cancer Classification"""
    
    def __init__(self, num_classes=5):
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
        
        for batch_idx, (images, labels) in enumerate(dataloader):
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
            
            if batch_idx % 20 == 0:
                print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
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
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, train_loader, val_loader, criterion, optimizer, scheduler):
        best_val_acc = 0.0
        
        for epoch in range(self.config['num_epochs']):
            print(f'\nEpoch [{epoch+1}/{self.config["num_epochs"]}]')
            print('-' * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(val_loader, criterion)
            
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

def create_data_loaders(data_dir, batch_size=16, image_size=128):
    """Create data loaders"""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
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
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(true_labels, predictions, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    return cm

def main():
    """Main training function"""
    
    # Configuration
    config = {
        'data_dir': 'D:\Lung Cancer Dataset',  # Your dataset path
        'batch_size': 16,
        'learning_rate': 0.001,
        'num_epochs': 15,
        'num_classes': 5,
        'image_size': 128
    }
    
    # Try GPU, fallback to CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("🎯 Using GPU for training!")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU (GPU not available)")
    
    print(f"Device: {device}")
    
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Initialize trainer
    trainer = ModelTrainer(model, device, config)
    
    # Start training
    print("Starting training...")
    history = trainer.train(train_loader, val_loader, criterion, optimizer, scheduler)
    
    # Load best model for evaluation
    if os.path.exists('best_lung_cancer_model.pth'):
        model.load_state_dict(torch.load('best_lung_cancer_model.pth'))
        print("Loaded best model for evaluation.")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    final_val_loss, final_val_acc, final_preds, final_labels = trainer.validate_epoch(
        val_loader, criterion
    )
    
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    # Class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i].title() for i in range(len(class_to_idx))]
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    plot_training_history(history)
    plot_confusion_matrix(final_labels, final_preds, class_names)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds, target_names=class_names, digits=4))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'config': config
    }, 'final_lung_cancer_model.pth')
    
    print(f"\n✅ Training completed successfully!")
    print(f"📊 Best validation accuracy: {final_val_acc:.2f}%")
    print("💾 Models saved: 'best_lung_cancer_model.pth' and 'final_lung_cancer_model.pth'")
    print("📈 Plots saved: 'training_history.png' and 'confusion_matrix.png'")

if __name__ == "__main__":
    main()