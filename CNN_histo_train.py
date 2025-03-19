import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)


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
    

train_tansform=transforms.Compose([
    transforms.Resize((96,96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.20, contrast=0.20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

train_dataset=PCamDataset(csv_file='data/train_labels.csv',transform=train_tansform)

train_dataloader=DataLoader(
    dataset=train_dataset,
    shuffle=True, 
    batch_size=8)

val_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_dataset = PCamDataset(
    csv_file='data/validation_labels.csv',
    transform=val_test_transform
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False
)

test_dataset = PCamDataset(
    csv_file='data/test_labels.csv',
    transform=val_test_transform
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False
)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)).squeeze(1)
        return x
        
cnn_model_histo = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model_histo.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(cnn_model_histo.parameters(), lr=0.0005)

train_losses = []
val_losses = []

num_epochs = 5
for epoch in range(num_epochs):
    total_train_loss = 0
    total_val_loss = 0

    # Training
    cnn_model_histo.train()
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn_model_histo(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
    # Validation
    cnn_model_histo.eval()
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = cnn_model_histo(images)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

    
    # Calculate average losses
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    
    print(f"Epoch [{epoch+1}/{num_epochs}]: Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")

torch.save(cnn_model_histo, 'cnn_model_histo_entire.pth')
print("Model saved successfully as cnn_model_histo_entire.pth.")