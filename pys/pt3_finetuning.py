import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = 'datasets/dog_breed'
all_labels_df = pd.read_csv(os.path.join(ROOT, 'labels.csv'))
# 提取标签中狗的种类，建立映射
breeds = all_labels_df.breed.unique()
breed2idx = dict((breed, idx) for idx, breed in enumerate(breeds))
idx2breed = dict((idx, breed) for idx, breed in enumerate(breeds))
print('There are in total %d breeds of dogs' %(len(breeds)))

all_labels_df['label_idx'] = [breed2idx[b] for b in all_labels_df.breed]

# 定义数据集
class DogDataset(Dataset):
    def __init__(self, labels_df, img_path, transform=None):
        self.labels_df = labels_df
        self.img_path = img_path
        self.transform = transform
    
    def __len__(self):
        return self.labels_df.shape[0]
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.img_path, self.labels_df.id[idx]) + '.jpg'
        img = Image.open(image_name)
        label = self.labels_df.label_idx[idx]

        if self.transform:
            img = self.transform(img)
        
        return img, label

IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])
valid_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

dataset_names=['train', 'valid']
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
train_split_idx, valid_split_idx = next(iter(stratified_split.split(all_labels_df.id, all_labels_df.breed)))
train_df = all_labels_df.iloc[train_split_idx].reset_index()
valid_df = all_labels_df.iloc[valid_split_idx].reset_index()
print('Number of samples in training set: %d' %(len(train_df)))
print('Number of samples in validation set: %d' %(len(valid_df)))

trainset = DogDataset(train_df, os.path.join(ROOT, 'train'), transform=train_transform)
validset = DogDataset(valid_df, os.path.join(ROOT, 'train'), transform=valid_transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False)

'''
imageset = {'train':trainset, 'valid':validset}
image_dataloader = {x:DataLoader(
    imageset[x], batch_size=BATCH_SIZE, shuffle=True)
    for x in dataset_names}
dataset_size = {x:len(imageset[x]) for x in dataset_names}
'''

model = torchvision.models.resnet50(pretrained=True)
# 冻结参数
for params in model.parameters():
    params.requires_grad = False
num_fc_in = model.fc.in_features
# new fc layer
model.fc = nn.Linear(num_fc_in, len(breeds))
model = model.to(DEVICE)
print(model)

metric = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

def train(model, trainloader, epoch):
    model.train()
    for data in trainloader:
        images, labels = data
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        preds = model(images)
        loss = metric(preds, labels)
        loss.backward()
        optimizer.step()
    print('Train epoch: %d \t Loss: %.6f' %(epoch, loss.item()))

def test(model, test_loader):
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            test_loss += metric(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            correct += (preds==labels).sum().item()
    test_loss /= len(test_loader.dataset)
    print('[Validation set] average loss: %.4f, accuracy: %d / %d, %.1f' % (test_loss, correct, len(validset), 100.*correct/len(validset)))

for epoch in range(1, 11):
    train(model, trainloader, epoch=epoch)
    test(model, validloader)