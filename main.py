import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import copy
import scipy.io
from PIL import Image

# Load image labels
labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]

# Load train, val, test indices
setid = scipy.io.loadmat('setid.mat')
train_ids = setid['trnid'][0] - 1  # MATLAB indexing starts at 1, so subtract 1
val_ids = setid['valid'][0] - 1
test_ids = setid['tstid'][0] - 1

# Create a dictionary to hold the dataset splits
dataset = {'train': train_ids, 'val': val_ids, 'test': test_ids}

# Create image file paths
image_paths = sorted([os.path.join('jpg', f'image_{i:05d}.jpg') for i in range(1, len(labels) + 1)])

# Ensure the directory structure is in place
for phase in ['train', 'val', 'test']:
    for class_index in range(1, 103):
        os.makedirs(os.path.join('flowers_data', phase, str(class_index)), exist_ok=True)

# Copy images to appropriate directories
for phase in ['train', 'val', 'test']:
    for idx in dataset[phase]:
        img = Image.open(image_paths[idx])
        label = labels[idx]
        img.save(os.path.join('flowers_data', phase, str(label), os.path.basename(image_paths[idx])))

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = 'flowers_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


try:
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0) for x in ['train', 'val']}
except Exception as e:
    print(f"Error initializing DataLoader: {e}")
    raise

# Add a function to test the DataLoader
def test_dataloader(dataloader, max_batches=5):
    try:
        for i, (inputs, labels) in enumerate(dataloader):
            print(f"Batch {i+1}: inputs shape: {inputs.shape}, labels shape: {labels.shape}")
            if i >= max_batches - 1:
                break
        print("DataLoader test completed successfully.")
    except Exception as e:
        print(f"Error in DataLoader: {e}")
        raise

# Test the DataLoader before training
print("Testing train DataLoader:")
test_dataloader(dataloaders['train'])
print("\nTesting val DataLoader:")
test_dataloader(dataloaders['val'])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()  # Initialize 'since' here

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            try:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            except Exception as e:
                print(f"Error during {phase} phase: {e}")
                raise

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

torch.save(model_ft.state_dict(), 'resnet50_flowers.pth')