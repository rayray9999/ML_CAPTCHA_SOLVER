
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




word = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
        's','t','u','v','w','x','y','z']
class Task3Dataset(Dataset):
    def __init__(self, data, root, return_filename=False):
        self.data = [sample for sample in data if sample[0].startswith("task3")]
        self.return_filename = return_filename
        self.root = root
    
    def __getitem__(self, index):
        filename, label = self.data[index]
        img = cv2.imread(f"{self.root}/{filename}")
        img = cv2.resize(img, (224, 224))
        label=torch.tensor([word.index(label[0]),word.index(label[1]),word.index(label[2]),word.index(label[3])])
        img=torch.FloatTensor(img).permute(2, 0, 1)
        if self.return_filename:
            return torch.FloatTensor((img - 128) / 128), filename
        else:
            return torch.FloatTensor((img - 128) / 128), label

    def __len__(self):
        return len(self.data)


train_data = []

with open(f'{TRAIN_PATH}/annotations.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        train_data.append(row)
       

train_ds = Task3Dataset(train_data, root=TRAIN_PATH)
train_dl = DataLoader(train_ds, batch_size=15, num_workers=4, drop_last=True, shuffle=True)



#print(timm.list_models('swin*',pretrained=True))
class Model(nn.Module):
    def __init__(self,model_name='swin_tiny_patch4_window7_224',pretrained=True,num_classes=0):
        super().__init__()
        self.layers = timm.create_model(model_name=model_name,pretrained=pretrained,num_classes=num_classes, drop_path_rate = 0.2)
        self.Digit1 = nn.Linear(768, 36)
        self.Digit2 = nn.Linear(768, 36)
        self.Digit3 = nn.Linear(768, 36)
        self.Digit4 = nn.Linear(768, 36)
    def forward(self, x):
        out=self.layers.forward_features(x)
        #print(out.shape)
        out=out.mean(dim=1)
        #out=out[:,0,:]
        out1 = self.Digit1(out)
        out2 = self.Digit2(out)
        out3 = self.Digit3(out)
        out4 = self.Digit4(out)
        return out1, out2, out3, out4



pre_task_weight="save_swin2.pth"
model = Model().to(device)
model_dict=model.state_dict()
pretrained_dict=torch.load(f"{WEIGHT_PATH}/{pre_task_weight}",map_location='cpu')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5,weight_decay = 0.001)
loss_fn = nn.CrossEntropyLoss()
Epoch=100
scheduler=CosineLRScheduler(optimizer,t_initial = Epoch,k_decay=2, lr_min = 1e-7, warmup_t= 5, warmup_lr_init = 1e-6)
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
        pred1,pred2,pred3,pred4 = model(image)
        #print(pred1.shape)
        #print(type(pred1))
        #print(label[:,0].shape)
        #print(type(label[:,0]))
        loss = loss_fn(pred1, label[:, 0])
        loss += loss_fn(pred2,label[:,1])
        loss += loss_fn(pred3,label[:,2])
        loss += loss_fn(pred4,label[:,3])
        Train_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        cnt+=1
    sample_count = 0
    correct_count = 0
    model.eval()
    for image, label in train_dl:
        image = image.to(device)
        label = label.to(device)
        pred1,pred2,pred3,pred4 = model(image)
        loss = loss_fn(pred1, label[:, 0])
        loss += loss_fn(pred2,label[:,1])
        loss += loss_fn(pred3,label[:,2])
        loss += loss_fn(pred4,label[:,3])
        pred1 = torch.argmax(pred1, dim=1)
        pred2 = torch.argmax(pred2, dim=1)
        pred3 = torch.argmax(pred3, dim=1)
        pred4 = torch.argmax(pred4, dim=1)
        val_loss=loss
        sample_count += len(image)
        for i in range(len(pred1)):
            #print(label[i,0])
            #print(pred1[i])
            if label[i,0] == pred1[i] and label[i,1] == pred2[i] and label[i,2] == pred3[i] and label[i,3] == pred4[i]:
                correct_count+=1
    if record_best< correct_count / sample_count:
        record_best=correct_count / sample_count
    print(correct_count)
    print(sample_count)
    print("accuracy (ALL):", correct_count / sample_count)
    print("T_Loss:", Train_loss / cnt)
    print("Best:",record_best)

torch.save(model.state_dict(), f'{WEIGHT_PATH}/save_swin3.pth')
#model.load_state_dict(torch.load("save2.pth",map_location='cpu'))






    