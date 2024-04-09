import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
from torchsampler import ImbalancedDatasetSampler

import random

import os
import json

## Set random seeds
torch.manual_seed(42)
random.seed(42)

## Calculate mean and std
dataset = datasets.ImageFolder(root='Datasets/dataset1', transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset)

# Initialize variables to accumulate sum and sum of squares for mean and std calculation
mean = 0.0
std = 0.0
num_samples = 0

# Calculate mean and std over the dataset
for data in dataloader:
    images, _ = data
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    num_samples += batch_samples

mean /= num_samples
std /= num_samples

# Save tensors since test_dataset1.py also uses these values for reversing normalisation for visualisation
torch.save({'mean': mean, 'std': std}, 'Cache/dataset1_mean+std.pth')

## Custom transform
class RandomPadOneSide:
    def __init__(self, padding_range, padding_value=0):
        self.padding_range = padding_range  # Tuple (min_padding, max_padding)
        self.padding_value = padding_value

    def __call__(self, img):
        # Randomly choose one side to pad
        pad_side = random.choice(['left', 'right', 'top', 'bottom'])
        
        # Randomly select the amount of padding within the specified range
        pad_amount = random.randint(self.padding_range[0], self.padding_range[1])
        
        # Initialize padding values
        left_pad, right_pad, top_pad, bottom_pad = 0, 0, 0, 0
        
        # Apply padding to the chosen side
        if pad_side == 'left':
            left_pad = pad_amount
        elif pad_side == 'right':
            right_pad = pad_amount
        elif pad_side == 'top':
            top_pad = pad_amount
        elif pad_side == 'bottom':
            bottom_pad = pad_amount
        
        # Apply padding to the image
        padding = transforms.Pad((left_pad, top_pad, right_pad, bottom_pad), fill=self.padding_value)
        padded_img = padding(img)
        
        return padded_img


## Define dataset
transform = transforms.Compose([
    RandomPadOneSide(padding_range=(0, 200), padding_value=0),
    transforms.RandomCrop(448),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.transforms.Normalize(mean=mean, std=std),
])

dataset = datasets.ImageFolder(root='Datasets/dataset1', transform=None)

# Define split ratios
train_ratio = 0.64
val_ratio   = 0.16
test_ratio  = 0.20

# Calculate the actual number of samples for each split
total_dataset_size = len(dataset)
train_size = int(train_ratio * total_dataset_size)
val_size   = int(val_ratio   * total_dataset_size)
test_size  = total_dataset_size - train_size - val_size


# Use random_split to create the splits
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

## Integer Indexed Dataset
class IntegerIndexedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_labels(self):
        return [item[1] for item in self.dataset]


# Wrap the original dataset with the custom IntegerIndexedDataset
int_train_dataset = IntegerIndexedDataset(train_dataset, transform)
int_val_dataset   = IntegerIndexedDataset(val_dataset,   transform)
int_test_dataset  = IntegerIndexedDataset(test_dataset,  transform)

# Create dataloaders for each split
batch_size = 5

train_dataloader = DataLoader(
        int_train_dataset,
        batch_size=batch_size,
        sampler=ImbalancedDatasetSampler(int_train_dataset),
        pin_memory=True
)

val_dataloader = DataLoader(
        int_val_dataset,
        batch_size=batch_size,
        sampler=ImbalancedDatasetSampler(int_val_dataset),
        pin_memory=True
)

test_dataloader = DataLoader(
        int_test_dataset,
        batch_size=batch_size,
        sampler=ImbalancedDatasetSampler(int_test_dataset),
        pin_memory=True
)

# Save test_dataloader for future python files to load
torch.save(test_dataloader, 'Cache/dataset1_test_dataloader.pth')

## Define model

# VGG19
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# Add a MaxPooling layer at the top of the model
new_features = torch.nn.Sequential(
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    *model.features
)

# Replace the original features with the new_features
model.features = new_features

# Reduce output classes to 1
num_classes = 1
model.classifier[-1] = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[-1].in_features, num_classes),
        torch.nn.Sigmoid()
)


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f'Using {device} for inference')

# Define loss function and optimizer
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Reset validation loss and validation accuracy values for new training loop
if os.path.exists('Cache/training_valloss_valaccuracy.json'):
    os.rename('Cache/training_valloss_valaccuracy.json', 'Cache/training_valloss_valaccuracy.json.old')

## Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.float().to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

    # Validation
    model.eval()

    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = model(inputs).squeeze()

            outputs = outputs.cpu()
            labels = labels.cpu()

            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            predictions = torch.round(outputs).cpu().int()

            total_samples += labels.size(0)
            
            correct_predictions += (predictions == labels).sum().item()

    # Calculate validation loss and accuracy
    val_loss /= len(val_dataloader)
    val_accuracy = correct_predictions / total_samples

    # Print current epoch's results
    print(f'Epoch [{epoch+1}/{num_epochs}] - Val loss: {val_loss:.4f} - Val accuracy: {val_accuracy:.4f}')

    # Write to file for visualise_training.py
    with open('Cache/training_valloss_valaccuracy.json', 'a') as file:
        file.write(json.dumps({'val_loss': val_loss, 'val_accuracy': val_accuracy}) + "\n")

    # Save model weights
    torch.save(model.state_dict(), 'Model_Weights/vgg19.pth')
