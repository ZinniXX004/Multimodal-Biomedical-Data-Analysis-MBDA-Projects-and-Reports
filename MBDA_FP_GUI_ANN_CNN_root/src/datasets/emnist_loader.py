import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def fix_orientation(img):
    return img.permute(0, 2, 1)

def adjust_label(y):
    """Shifts EMNIST Letters labels from 1-26 to 0-25."""
    return y - 1

def load_emnist_data(data_dir='./data', batch_size_train=128, batch_size_test=512, use_augmentation=False):
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if dataset already exists
    dataset_path = os.path.join(data_dir, 'EMNIST')
    if os.path.exists(dataset_path):
        print(f"📂 Found existing dataset at '{dataset_path}'. Loading locally (skip download)...")
    else:
        print("🌐 Dataset not found locally. Downloading from the internet... (This might take a while)")

    _mean = (0.1307,) if use_augmentation else (0.1722,)
    _std  = (0.3081,) if use_augmentation else (0.3309,)

    # Base transforms
    transform_list_test = [
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
        transforms.Lambda(fix_orientation)
    ]
    
    transform_list_train = transform_list_test.copy()
    if use_augmentation:
        # Add light augmentation for CNN
        transform_list_train.append(transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)))

    transform_train = transforms.Compose(transform_list_train)
    transform_test = transforms.Compose(transform_list_test)

    # download=True acts as "download if not present"
    train_data = datasets.EMNIST(
        root=data_dir, split='letters', train=True, 
        download=True, transform=transform_train, target_transform=adjust_label
    )
    test_data = datasets.EMNIST(
        root=data_dir, split='letters', train=False, 
        download=True, transform=transform_test, target_transform=adjust_label
    )

    # Set num_workers to 2 for faster loading, and pin_memory if CUDA is available
    train_loader = DataLoader(
        train_data, batch_size=batch_size_train, shuffle=True, 
        num_workers=2, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size_test, shuffle=False, 
        num_workers=2, pin_memory=torch.cuda.is_available()
    )

    alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    return train_loader, test_loader, alphabet