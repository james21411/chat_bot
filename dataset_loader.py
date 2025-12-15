import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class FoodDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.samples = []
        
        # Collect all food categories and their metadata
        self.food_metadata = self._load_food_metadata()
        
        # Build image dataset
        self._build_dataset()
    
    def _load_food_metadata(self):
        """Load food metadata from JSON files - prioritize unified metadata"""
        metadata = {}
        
        # Try to load unified metadata first (best source)
        unified_path = os.path.join(self.data_dir, 'unified_foods_metadata.json')
        if os.path.exists(unified_path):
            with open(unified_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # Normalize name for consistent matching
                    name = self.normalize_name(item['name'])
                    metadata[name] = item
            print(f"✅ Loaded {len(data)} items from unified metadata")
            return metadata
        
        # Fallback to original files if unified not found
        json_files = ['mets.json', 'recette.json']
        
        for json_file in json_files:
            json_path = os.path.join(self.data_dir, json_file)
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        name = self.normalize_name(item['name'])
                        metadata[name] = item
        
        # Also check IMAGE folder
        image_json_path = os.path.join(self.data_dir, 'IMAGE/recette.json')
        if os.path.exists(image_json_path):
            with open(image_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    name = self.normalize_name(item['name'])
                    if name not in metadata:  # Don't override existing
                        metadata[name] = item
        
        print(f"⚠️  Using fallback metadata: {len(metadata)} items")
        return metadata
    
    def normalize_name(self, name):
        """Normalize food name for consistent matching"""
        import re
        name = name.lower()
        name = re.sub(r'[_\s-]+', '_', name)
        name = re.sub(r'[^\w]', '', name)
        # Remove trailing numbers (eru1 -> eru, koki2 -> koki)
        name = re.sub(r'\d+$', '', name)
        return name
    
    def _build_dataset(self):
        """Build the dataset from images and metadata"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # Only process files in the root directory (avoid subfolders)
        for file in os.listdir(self.data_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(self.data_dir, file)

                # Extract class name from filename and normalize it
                class_name = self.normalize_name(file.split('.')[0])

                # Try to match with metadata using better matching
                matched_class = None
                for metadata_class in self.food_metadata.keys():
                    # Direct match
                    if class_name == metadata_class:
                        matched_class = metadata_class
                        break
                    # Partial match (either direction)
                    elif class_name in metadata_class or metadata_class in class_name:
                        matched_class = metadata_class
                        break

                # Special handling for known mappings
                if matched_class is None:
                    # Check for common alternative names
                    if 'banku' in class_name and 'tilapia' in class_name:
                        matched_class = 'banku_et_tilapia'
                    elif class_name in ['ugali', 'sadza', 'pap']:
                        matched_class = class_name
                    else:
                        matched_class = class_name

                print(f"🎯 Matched: {file} -> {matched_class}")

                # Add to class mapping
                if matched_class not in self.class_to_idx:
                    idx = len(self.class_to_idx)
                    self.class_to_idx[matched_class] = idx
                    self.idx_to_class[idx] = matched_class

                self.samples.append((file_path, self.class_to_idx[matched_class]))
        
        print(f"Dataset built: {len(self.samples)} images, {len(self.class_to_idx)} classes")
        print(f"Classes: {list(self.class_to_idx.keys())}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_info(self, idx):
        """Get detailed information about a class"""
        idx_int = int(idx)  # Ensure Python int
        class_name = self.idx_to_class.get(idx_int, f"Unknown_Class_{idx_int}")
        if class_name in self.food_metadata:
            return self.food_metadata[class_name]
        return None

def get_transforms():
    """Get image transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(data_dir, batch_size=32, val_split=0.2):
    """Create training and validation data loaders"""
    train_transform, val_transform = get_transforms()
    
    # Create full dataset
    full_dataset = FoodDataset(data_dir, transform=val_transform)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, full_dataset

if __name__ == "__main__":
    # Test dataset loading
    data_dir = "data_set_images"
    train_loader, val_loader, dataset = create_data_loaders(data_dir, batch_size=8)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Total images: {len(dataset)}")
    print(f"Number of classes: {len(dataset.class_to_idx)}")
    
    # Test a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break