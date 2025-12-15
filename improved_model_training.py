import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import random
import matplotlib.pyplot as plt

class ImprovedFoodClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ImprovedFoodClassifier, self).__init__()
        
        # Use ResNet18 - reliable and efficient for small datasets
        if pretrained:
            try:
                from torchvision.models import ResNet18_Weights, resnet18
                self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except:
                self.backbone = resnet18(weights=None)
        else:
            from torchvision.models import resnet18
            self.backbone = resnet18(weights=None)
        
        # Replace classifier with enhanced regularization
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class AdvancedDataAugmentation:
    """Advanced data augmentation for small datasets"""
    
    @staticmethod
    def get_train_transforms():
        return transforms.Compose([
            transforms.Resize((256, 256)),  # Higher resolution for better features
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=30),  # More rotation
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # Note: GaussianBlur doesn't have probability parameter
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class SyntheticDataAugmentation:
    """Create synthetic variations of images to increase dataset size"""
    
    @staticmethod
    def create_variations(image, num_variations=3):
        variations = []
        
        for i in range(num_variations):
            # Slight variations
            var_transform = transforms.Compose([
                transforms.ColorJitter(brightness=random.uniform(0.1, 0.3),
                                     contrast=random.uniform(0.1, 0.3),
                                     saturation=random.uniform(0.1, 0.3)),
                transforms.RandomRotation(degrees=(-15, 15)),
                transforms.RandomHorizontalFlip(p=0.5)
            ])
            variations.append(var_transform(image))
        
        return variations

class ImprovedFoodDataset(Dataset):
    def __init__(self, data_dir, transform=None, augment_synthetic=True):
        self.data_dir = data_dir
        self.transform = transform
        self.augment_synthetic = augment_synthetic
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.samples = []
        
        # Load metadata
        self.food_metadata = self._load_food_metadata()
        self._build_dataset()
        
        # Expand dataset with synthetic augmentations
        if augment_synthetic:
            self._add_synthetic_data()
    
    def _load_food_metadata(self):
        metadata = {}
        unified_path = os.path.join(self.data_dir, 'unified_foods_metadata.json')
        if os.path.exists(unified_path):
            with open(unified_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    name = self.normalize_name(item['name'])
                    metadata[name] = item
        return metadata
    
    def normalize_name(self, name):
        import re
        name = name.lower()
        name = re.sub(r'[_\s-]+', '_', name)
        name = re.sub(r'[^\w]', '', name)
        return name
    
    def _build_dataset(self):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    file_path = os.path.join(root, file)
                    class_name = self.normalize_name(file.split('.')[0])
                    
                    # Enhanced matching logic
                    matched_class = None
                    for metadata_class in self.food_metadata.keys():
                        if class_name == metadata_class or class_name in metadata_class or metadata_class in class_name:
                            matched_class = metadata_class
                            break
                    
                    if matched_class is None:
                        matched_class = class_name
                    
                    if matched_class not in self.class_to_idx:
                        idx = len(self.class_to_idx)
                        self.class_to_idx[matched_class] = idx
                        self.idx_to_class[idx] = matched_class
                    
                    self.samples.append((file_path, self.class_to_idx[matched_class]))
    
    def _add_synthetic_data(self):
        """Add synthetic variations to increase dataset size"""
        synthetic_samples = []
        
        for img_path, label in self.samples:
            try:
                image = Image.open(img_path).convert('RGB')
                variations = SyntheticDataAugmentation.create_variations(image, num_variations=2)
                
                for i, variation in enumerate(variations):
                    # Create synthetic file path
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    synthetic_path = f"{img_path}_synthetic_{i}"
                    synthetic_samples.append((variation, label, synthetic_path))
            except Exception as e:
                print(f"Warning: Could not create synthetic data for {img_path}: {e}")
        
        # Add synthetic samples to dataset
        for variation, label, path in synthetic_samples:
            self.samples.append((variation, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        if isinstance(item[0], Image.Image):
            # Synthetic data (already a PIL image)
            image = item[0]
        else:
            # Regular file path
            img_path, label = item
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
                label = 0
        
        if self.transform:
            image = self.transform(image)
        
        if isinstance(item[0], Image.Image):
            return image, item[1]  # Return synthetic sample with its label
        else:
            return image, label
    
    def get_class_info(self, idx):
        class_name = self.idx_to_class[idx]
        if class_name in self.food_metadata:
            return self.food_metadata[class_name]
        return None

class ImprovedModelTrainer:
    def __init__(self, num_classes, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
    
    def build_model(self):
        self.model = ImprovedFoodClassifier(self.num_classes).to(self.device)
        
        # Use label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Use lower learning rate with weight decay for regularization
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=0.0005, 
                                   weight_decay=1e-3,
                                   betas=(0.9, 0.999))
        
        # Use ReduceLROnPlateau for better learning rate scheduling
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                         mode='max', 
                                         factor=0.5, 
                                         patience=5)
        
        print(f"Improved model built with {self.num_classes} classes on device {self.device}")
        return self.model
    
    def train(self, train_loader, val_loader, epochs=50, save_path='models/improved_food_classifier.pth'):
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if batch_idx % 5 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # Validation phase
            val_loss, val_accuracy = self.validate(val_loader)
            
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = 100 * correct / total
            
            train_losses.append(epoch_train_loss)
            val_losses.append(val_loss)
            train_accs.append(epoch_train_acc)
            val_accs.append(val_accuracy)
            
            print(f'Epoch [{epoch+1}/{epochs}]:')
            print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            # Step scheduler
            self.scheduler.step(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_accuracy,
                    'num_classes': self.num_classes,
                    'train_history': {
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_accs': train_accs,
                        'val_accs': val_accs
                    }
                }, save_path)
                print(f'  ✅ Best model saved! Accuracy: {val_accuracy:.2f}%')
        
        print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return val_loss, accuracy

def create_improved_data_loaders(data_dir, batch_size=16, val_split=0.25):
    """Create improved data loaders with advanced augmentation"""
    
    # Use advanced transforms
    train_transform = AdvancedDataAugmentation.get_train_transforms()
    val_transform = AdvancedDataAugmentation.get_val_transforms()
    
    # Create dataset with synthetic augmentation
    full_dataset = ImprovedFoodDataset(data_dir, transform=val_transform, augment_synthetic=True)
    
    print(f"Dataset with synthetic augmentation: {len(full_dataset)} samples")
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply appropriate transforms
    train_dataset.dataset.transform = train_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, full_dataset

def plot_training_results(history, save_path='models/improved_training_history.png'):
    """Plot training results with enhanced visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot losses
    ax1.plot(epochs, history['train_losses'], label='Training Loss', color='blue', alpha=0.7)
    ax1.plot(epochs, history['val_losses'], label='Validation Loss', color='red', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, history['train_accs'], label='Training Accuracy', color='green', alpha=0.7)
    ax2.plot(epochs, history['val_accs'], label='Validation Accuracy', color='orange', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot loss difference (overfitting indicator)
    loss_diff = [train - val for train, val in zip(history['train_losses'], history['val_losses'])]
    ax3.plot(epochs, loss_diff, label='Loss Difference (Train - Val)', color='purple')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.set_title('Overfitting Indicator')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot accuracy difference
    acc_diff = [train - val for train, val in zip(history['train_accs'], history['val_accs'])]
    ax4.plot(epochs, acc_diff, label='Accuracy Difference (Train - Val)', color='brown')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Difference (%)')
    ax4.set_title('Overfitting Indicator (Accuracy)')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training results saved to {save_path}")

def train_improved_model(data_dir='data_set_images', epochs=50, batch_size=16):
    """Train improved model with better generalization"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create improved data loaders
    print("Creating improved data loaders with synthetic augmentation...")
    train_loader, val_loader, dataset = create_improved_data_loaders(
        data_dir, batch_size=batch_size, val_split=0.25
    )
    
    print(f"Enhanced dataset info:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Number of classes: {len(dataset.class_to_idx)}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    
    # Create improved model
    num_classes = len(dataset.class_to_idx)
    trainer = ImprovedModelTrainer(num_classes, device)
    trainer.build_model()
    
    # Train the model
    print(f"\n🚀 Starting improved training for {epochs} epochs...")
    history = trainer.train(
        train_loader, 
        val_loader, 
        epochs=epochs, 
        save_path='models/improved_food_classifier.pth'
    )
    
    # Plot results
    plot_training_results(history)
    
    # Save enhanced model info
    model_info = {
        'num_classes': num_classes,
        'class_to_idx': dataset.class_to_idx,
        'idx_to_class': dataset.idx_to_class,
        'enhanced_features': [
            'Synthetic data augmentation',
            'Advanced image transformations', 
            'Label smoothing',
            'Gradient clipping',
            'Learning rate scheduling',
            'Enhanced regularization'
        ],
        'class_info': {}
    }
    
    for idx in range(num_classes):
        class_info = dataset.get_class_info(idx)
        class_name = dataset.idx_to_class[idx]
        model_info['class_info'][class_name] = class_info
    
    info_path = 'models/improved_food_classifier_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Enhanced model saved: models/improved_food_classifier.pth")
    print(f"📁 Model info saved: {info_path}")
    
    return trainer, dataset

if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("=" * 60)
    print("🍽️  IMPROVED FOOD CLASSIFICATION TRAINING")
    print("=" * 60)
    print("✨ Enhanced features:")
    print("   • Synthetic data augmentation")
    print("   • Advanced image transformations")
    print("   • Better regularization")
    print("   • Improved learning rate scheduling")
    print("   • Enhanced model architecture")
    print("=" * 60)
    
    # Train improved model
    trainer, dataset = train_improved_model(epochs=50, batch_size=12)
    
    print("\n🎉 Improved training completed successfully!")
    print("🔍 Your model should now generalize much better!")