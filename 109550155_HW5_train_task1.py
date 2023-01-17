
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import csv
import cv2
import numpy as np
import random
import os

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from timm.scheduler import CosineLRScheduler

WEIGHT_PATH = "swin_weight"
TRAIN_PATH = "captcha-hacker/train"
TEST_PATH = "captcha-hacker/test"
device = "cuda"
# try device = "cuda" 
# and change your settings/accelerator to GPU if you want it to run faster



#print(type(word))
word = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
        's','t','u','v','w','x','y','z']
class Task1Dataset(Dataset):
    def __init__(self, data, root, return_filename=False):
        self.data = [sample for sample in data if sample[0].startswith("task1")]
        self.return_filename = return_filename
        self.root = root
    
    def __getitem__(self, index):
        filename, label = self.data[index]
        img = cv2.imread(f"{self.root}/{filename}")
        img = cv2.resize(img, (224, 224))
        img=torch.FloatTensor(img).permute(2, 0, 1)
        label=torch.tensor([word.index(label[0])])
        if self.return_filename:
            return torch.FloatTensor((img - 128) / 128), filename
        else:
            return torch.FloatTensor((img - 128) / 128), label

    def __len__(self):
        return len(self.data)


train_data = []
val_data = []

with open(f'{TRAIN_PATH}/annotations.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        train_data.append(row)

train_ds = Task1Dataset(train_data, root=TRAIN_PATH)
train_dl = DataLoader(train_ds, batch_size=15, num_workers=4, drop_last=True, shuffle=True)




#print(timm.list_models('swin*',pretrained=True))
class Model(nn.Module):
    def __init__(self,model_name='swin_tiny_patch4_window7_224',pretrained=True,num_classes=0):
        super().__init__()
        self.layers = timm.create_model(model_name=model_name,pretrained=pretrained,num_classes=num_classes,drop_path_rate = 0.2)
        self.digit1 = nn.Linear(768, 10)
    def forward(self, x):
        out=self.layers.forward_features(x)
        #print(out.shape)
        out=out.mean(dim=1)
        #out=out[:,0,:]
        out1 = self.digit1(out)
        return out1




model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,weight_decay = 0.001)
loss_fn = nn.CrossEntropyLoss()
Epoch=15
scheduler=CosineLRScheduler(optimizer,t_initial = Epoch,k_decay=2, lr_min = 5e-7, warmup_t= 5, warmup_lr_init = 1e-6)
record_best=0
for epoch in range(Epoch):
    Train_loss=0
    cnt=0
    print(f"Epoch [{epoch}]")
    model.train()
    for image, label in train_dl:
        #image=image.resize(224,224)
        image = image.to(device)
        label = label.to(device)     
        pred1 = model(image)
        #print(pred1.shape)
        #print(type(pred1))
        #print(label[:,0].shape)
        #print(type(label[:,0]))
        loss = loss_fn(pred1, label[:, 0])
        Train_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        cnt+=1
    sample_count=0
    correct_count=0
    for image, label in train_dl:
        image = image.to(device)
        label = label.to(device)
        pred1= model(image)
        loss = loss_fn(pred1,label[:,0])
        pred1 = torch.argmax(pred1, dim=1)
        val_loss=loss
        sample_count += len(image)
        for i in range(len(pred1)):
            #print(label[i,0])
            #print(pred1[i])
            if label[i,0] == pred1[i]:
                #print(label[i,0])
                #print(label[i,1])
                #print(pred1[i])
                #print(pred2[i])
                correct_count+=1
    if record_best< correct_count / sample_count:
        record_best=correct_count / sample_count
    print(correct_count)
    print(sample_count)
    print("accuracy (ALL):", correct_count / sample_count)
    print("T_Loss:", Train_loss / cnt)
    print("Best:",record_best)
    

torch.save(model.state_dict(), f'{WEIGHT_PATH}/save_swin1.pth')
#model.load_state_dict(torch.load("save2.pth",map_location='cpu'))

