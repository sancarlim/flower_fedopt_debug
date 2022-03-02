#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : utils.py
# Modified   : 01.03.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

from collections import OrderedDict
import numpy as np 
import os 
from typing import List 
import random 
from PIL import Image 
import torch
import torchvision 
import torch.nn as nn 
from torch import optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader 
from efficientnet_pytorch import EfficientNet 
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import wandb


training_transforms = transforms.Compose([#Microscope(),
                                        #AdvancedHairAugmentation(),
                                        transforms.RandomRotation(30),
                                        #transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        #transforms.ColorJitter(brightness=32. / 255.,saturation=0.5,hue=0.01),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 

testing_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Creating seeds to make results reproducible
def seed_everything(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed = 2022
seed_everything(seed)


def get_parameters(net, EXCLUDE_LIST) -> List[np.ndarray]:
        parameters = []
        for i, (name, tensor) in enumerate(net.state_dict().items()):
            # print(f"  [layer {i}] {name}, {type(tensor)}, {tensor.shape}, {tensor.dtype}")

            # Check if this tensor should be included or not
            exclude = False
            for forbidden_ending in EXCLUDE_LIST:
                if forbidden_ending in name:
                    exclude = True
            if exclude:
                continue

            # Convert torch.Tensor to NumPy.ndarray
            parameters.append(tensor.cpu().numpy())

        return parameters


def set_parameters(net, parameters, EXCLUDE_LIST):
        keys = []
        for name in net.state_dict().keys():
            # Check if this tensor should be included or not
            exclude = False
            for forbidden_ending in EXCLUDE_LIST:
                if forbidden_ending in name:
                    exclude = True
            if exclude:
                continue

            # Add to list of included keys
            keys.append(name)

        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)



class Net(nn.Module):
    def __init__(self, arch, return_feats=False):
        super(Net, self).__init__()
        self.arch = arch
        self.return_feats = return_feats
        if 'fgdf' in str(arch.__class__):
            self.arch.fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        if 'EfficientNet' in str(arch.__class__):   
            self.arch._fc = nn.Linear(in_features=self.arch._fc.in_features, out_features=500, bias=True)
            #self.dropout1 = nn.Dropout(0.2)
        else:   
            self.arch.fc = nn.Linear(in_features=arch.fc.in_features, out_features=500, bias=True)
            
        self.output = nn.Linear(500, 1)
        
    def forward(self, images): 
        x = images
        features = self.arch(x)
        output = self.output(features)
        if self.return_feats:
            return features
        return output


def load_model(model = 'efficientnet-b2', device="cuda"):
    if "efficientnet" in model:
        arch = EfficientNet.from_pretrained(model)
    elif model == "googlenet":
        arch = torchvision.models.googlenet(pretrained=True)
    else:
        arch = torchvision.models.resnet50(pretrained=True)
        
    model = Net(arch=arch).to(device)

    return model


def load_isic_data(path='./data'):
    # ISIC Dataset

    df = pd.read_csv(os.path.join(path, 'train_debug.csv'))   

    train_split, valid_split = train_test_split (df, stratify=df.target, test_size = 0.20, random_state=42) 
    
    train_df=pd.DataFrame(train_split)
    validation_df=pd.DataFrame(valid_split) 
    
    training_dataset = CustomDataset(df = train_df, transforms = training_transforms ) 
    testing_dataset = CustomDataset(df = validation_df, transforms = testing_transforms ) 

    num_examples = {"trainset" : len(training_dataset), "testset" : len(testing_dataset)} 
    
    return training_dataset, testing_dataset, num_examples


def load_partition(trainset, testset, num_examples, idx, num_partitions = 5):
    """Load 1/num_partitions of the training and test data to simulate a partition."""
    assert idx in range(num_partitions) 
    n_train = int(num_examples["trainset"] / num_partitions)
    n_test = int(num_examples["testset"] / num_partitions)

    train_partition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_partition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )

    num_examples = {"trainset" : len(train_partition), "testset" : len(test_partition)} 

    return (train_partition, test_partition, num_examples)


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms= None):
        self.df = df
        self.transforms = transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        img_path = self.df.iloc[index]['image_name'] 
        images =Image.open(img_path)

        if self.transforms:
            images = self.transforms(images)
            
        labels = self.df.iloc[index]['target']

        return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
        

def train(model, train_loader, validate_loader, num_examples,partition, nowandb, device="cuda",  log_interval = 100, epochs = 10, es_patience = 3):
    print('Starts training...')

    best_val = 0
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2) 

    for e in range(epochs):
        correct = 0
        running_loss = 0
        model.train()
        
        for i, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device), labels.to(device) 

            optimizer.zero_grad()
            
            output = model(images) 
            loss = criterion(output, labels.view(-1,1))  
            loss.backward()
            optimizer.step()
            
            # Training loss
            running_loss += loss.item()

            # Number of correct training predictions and training accuracy
            train_preds = torch.round(torch.sigmoid(output))
                
            correct += (train_preds.cpu() == labels.cpu().unsqueeze(1)).sum().item()
            
            if i % log_interval == 0 and not nowandb: 
                wandb.log({f'Client{partition}/training_loss': loss, 'epoch':e})
                            
        train_acc = correct / num_examples["trainset"]

        val_loss, val_auc_score, val_accuracy, val_f1 = val(model, validate_loader, criterion, partition, nowandb)
            
        print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
            "Training Accuracy: {:.3f}..".format(train_acc),
            "Validation Loss: {:.3f}.. ".format(val_loss/len(validate_loader)),
            "Validation Accuracy: {:.3f}".format(val_accuracy),
            "Validation AUC Score: {:.3f}".format(val_auc_score),
            "Validation F1 Score: {:.3f}".format(val_f1))
            
        if not nowandb:
            wandb.log({f'Client{partition}/Training acc': train_acc, f'Client{partition}/training_loss': running_loss/len(train_loader), 'epoch':e})

        scheduler.step(val_auc_score)


    del train_loader, validate_loader, images 
    return model


def val(model, validate_loader, criterion, partition, nowandb, device="cuda"):          
    model.eval()
    preds=[]            
    all_labels=[] 
    # Turning off gradients for validation, saves memory and computations
    with torch.no_grad():
        
        val_loss = 0 
    
        for val_images, val_labels in validate_loader:
        
            val_images, val_labels = val_images.to(device), val_labels.to(device) 
            val_output = model(val_images)
            val_loss += (criterion(val_output, val_labels.view(-1,1))).item() 
            val_pred = torch.sigmoid(val_output)
            
            preds.append(val_pred.cpu())
            all_labels.append(val_labels.cpu())
        pred=np.vstack(preds).ravel()
        pred2 = torch.tensor(pred)
        val_gt = np.concatenate(all_labels)
        val_gt2 = torch.tensor(val_gt)
            
        val_accuracy = accuracy_score(val_gt2, torch.round(pred2))
        val_auc_score = roc_auc_score(val_gt, pred)
        val_f1_score = f1_score(val_gt, np.round(pred))

        if not nowandb:
            name = f'Client{partition}' if partition != -1 else 'Server'
            wandb.log({f'{name}/Validation AUC Score': val_auc_score, f'{name}/Validation Acc': val_accuracy,
                             f'{name}/Validation Loss': val_loss/len(validate_loader)})


        return val_loss/len(validate_loader), val_auc_score, val_accuracy, val_f1_score



def val_mp_server(arch, parameters, device, EXCLUDE_LIST, return_dict):          
    # Create model
    model = load_model(arch)
    model.to(device)
    # Set model parameters, train model, return updated model parameters 
    if parameters is not None:
        set_parameters(model, parameters, EXCLUDE_LIST)
    # Load data
    testset = load_isic_by_patient(partition=-1)
    test_loader = DataLoader(testset, batch_size=32, num_workers=4, worker_init_fn=seed_worker, shuffle = False)   
    preds=[]            
    all_labels=[]
    criterion = nn.BCEWithLogitsLoss()
    # Turning off gradients for validation, saves memory and computations
    with torch.no_grad():
        
        val_loss = 0 
    
        for val_images, val_labels in test_loader:
        
            val_images, val_labels = val_images.to(device), val_labels.to(device)
        
            val_output = model(val_images)
            val_loss += (criterion(val_output, val_labels.view(-1,1))).item() 
            val_pred = torch.sigmoid(val_output)
            
            preds.append(val_pred.cpu())
            all_labels.append(val_labels.cpu())
        pred=np.vstack(preds).ravel()
        pred2 = torch.tensor(pred)
        val_gt = np.concatenate(all_labels)
        val_gt2 = torch.tensor(val_gt)
            
        val_accuracy = accuracy_score(val_gt2, torch.round(pred2))
        val_auc_score = roc_auc_score(val_gt, pred)  

        return_dict['loss'] = val_loss/len(test_loader)
        return_dict['auc_score'] = val_auc_score
        return_dict['accuracy'] = val_accuracy 
        return_dict['num_examples'] = {"testset" : len(testset)}