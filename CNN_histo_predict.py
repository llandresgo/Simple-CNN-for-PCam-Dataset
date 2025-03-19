import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

class PCamDataset(Dataset):
    """
    Custom Dataset for loading the microscopic histopathology images within the PCam dataset
    """
    def __init__(self, csv_file, transform=None, num_samples=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            transform (callable, optional): Optional transform to be applied on a sample
            num_samples (int, optional): Number of samples to load. If None, loads all samples
        """
        self.annotations = pd.read_csv(csv_file)
        if num_samples is not None:
            self.annotations = self.annotations.head(num_samples)
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image path and label
        img_path = self.annotations.iloc[idx, 0]
        label = self.annotations.iloc[idx, 1]
        
        # Load and convert image
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        # Convert label to float
        label = torch.tensor(label, dtype=torch.float)
            
        return image, label

cnn_model_histo = torch.load('cnn_model_histo_entire.pth')
cnn_model_histo.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = PCamDataset(
    csv_file='data/test_labels.csv',
    transform=test_transform
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False
)



test_pred_probs = []
test_pred_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = cnn_model_histo(images)
        
        test_pred_probs.extend(outputs.cpu().numpy())
        
        pred_labels = torch.round(outputs)
        test_pred_labels.extend(pred_labels.cpu().numpy())


test_pred_probs = np.array(test_pred_probs)
test_pred_labels = np.array(test_pred_labels)

print("Third image predicted probability:", test_pred_probs[2])
print("Third image predicted label:", test_pred_labels[2])