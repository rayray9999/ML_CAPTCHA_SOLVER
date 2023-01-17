
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

WEIGHT_PATH = "swin_weight" #the model weight path
TRAIN_PATH = "captcha-hacker/train" #no use here
TEST_PATH = "captcha-hacker/test"#the test dataset path
device = "cuda"

word = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
        's','t','u','v','w','x','y','z']
class TaskDataset(Dataset):
    def __init__(self, data, root, return_filename=False,task_name=None):
        self.data = [sample for sample in data if sample[0].startswith(task_name)]
        self.return_filename = return_filename
        self.root = root
    
    def __getitem__(self, index):
        filename, label = self.data[index]
        img = cv2.imread(f"{self.root}/{filename}")
        img = cv2.resize(img, (224, 224))
        #label=torch.tensor([word.index(label[0]),word.index(label[1])])
        img=torch.FloatTensor(img).permute(2, 0, 1)
        if self.return_filename:
            return torch.FloatTensor((img - 128) / 128), filename
        else:
            return torch.FloatTensor((img - 128) / 128), label

    def __len__(self):
        return len(self.data)






#print(timm.list_models('swin*',pretrained=True))
class Model1(nn.Module):#task1
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
class Model2(nn.Module):#task2
    def __init__(self,model_name='swin_tiny_patch4_window7_224',pretrained=True,num_classes=0):
        super().__init__()
        self.layers = timm.create_model(model_name=model_name,pretrained=pretrained,num_classes=num_classes,drop_path_rate = 0.2)
        self.Digit1 = nn.Linear(768, 36)
        self.Digit2 = nn.Linear(768, 36)
    def forward(self, x):
        out=self.layers.forward_features(x)
        #print(out.shape)
        out=out.mean(dim=1)
        #out=out[:,0,:]
        out1 = self.Digit1(out)
        out2 = self.Digit2(out)
        return out1, out2
class Model3(nn.Module):#task3
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
#open a new submission.csv
if os.path.exists('submission.csv'):
    csv_writer = csv.writer(open('submission.csv', 'w', newline=''))
    csv_writer.writerow(["filename", "label"])

model=None
task123=["task1","task2","task3"] #task name
Weight123=["save_swin1.pth","save_swin2.pth","save_swin3.pth"] #weight name
for i in range(3):
    print("NOW is runing task:",(i+1))
    test_data = []
    #get task image&label
    with open(f'{TEST_PATH}/../sample_submission.csv', newline='') as csvfile:
        for row in csv.reader(csvfile, delimiter=','):
            test_data.append(row)

    test_ds = TaskDataset(test_data, root=TEST_PATH, return_filename=True,task_name=task123[i])
    test_dl = DataLoader(test_ds, batch_size=50, num_workers=4, drop_last=False, shuffle=False)


    
    if i==0:
        model = Model1().to(device)
    elif i==1:
        model = Model2().to(device)
    else:
        model = Model3().to(device)
    model.load_state_dict(torch.load(f"{WEIGHT_PATH}/{Weight123[i]}",map_location='cpu'))
    model.eval()
    for image, filenames in test_dl:#write in submission.csv
        image = image.to(device)
        if i==0:
            pred = model(image)
            pred = torch.argmax(pred, dim=1)
            for j in range(len(filenames)):
                csv_writer.writerow([filenames[j], str(pred[j].item())])
        elif i==1:
            pred1,pred2 = model(image)
            pred1 = torch.argmax(pred1, dim=1)
            pred2 = torch.argmax(pred2, dim=1)
            for j in range(len(filenames)):
                csv_writer.writerow([filenames[j], str(word[pred1[j].item()]+word[pred2[j].item()])])
        else:
            pred1,pred2,pred3,pred4 = model(image)
            pred1 = torch.argmax(pred1, dim=1)
            pred2 = torch.argmax(pred2, dim=1)
            pred3 = torch.argmax(pred3, dim=1)
            pred4 = torch.argmax(pred4, dim=1)
            for j in range(len(filenames)):
                csv_writer.writerow([filenames[j], str(word[pred1[j].item()]+word[pred2[j].item()]+word[pred3[j].item()]+word[pred4[j].item()])])


