import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torchvision import transforms

from torch import optim

import numpy as np
import cv2
import glob
import csv

import matplotlib.pyplot as plt

class cursordataset(Dataset):
    def __init__(self,img,position):
            self.img = torch.stack(img)
            self.position = torch.stack(position)
    def __getitem__(self, index):
        return self.img[index],self.position[index]

    def __len__(self):
        return len(self.img)

class cursor_model(nn.Module):
    def __init__(self, device):
        super(cursor_model,self).__init__()
        self.relu = nn.Tanh()
        self.pool = nn.MaxPool2d(3, stride=3)
        #self.dropout = nn.Dropout()

        self.conv1 = nn.Conv2d(3,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv3 = nn.Conv2d(64,16,3)

        self.fc1 = nn.Linear(836352, 100)
        self.fc2 = nn.Linear(100, 2)

        self.flatten = nn.Flatten()

    def forward(self, x):
        #print(x.shape)#10(batch) * 3 * 1080 * 1920
        #x = self.pool(x)
        #print(x.shape)#torch.Size([5, 3, 360, 640])
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool(x)
        #print
        x = self.flatten(x)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(epoch,train_data,device,lr):
    loss_list = []
    epoch_list = []
    print(device)
    #model = cursor_model(device).to(device)
    model = cursor_model(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for i in range(epoch):
        model.train()
        loss_train = 0
        for j, xy in enumerate(train_data):
            img = xy[0] #学習データ
            position = xy[1] #正解データ(label))

            img = img.to(device)
            position = position.to(device)
            model = model.to(device)
            loss = criterion(model(img), position)
            loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_train /= j+1

        loss_list.append(loss_train)
        epoch_list.append(i)

        if i % 1 == 0 and i != 0:
            print("Epoch:", i, "Loss_Train:", loss_train)
        
    return model, loss_list, epoch_list

def dataloader(img,position,batch_size):

    train_dataset = cursordataset(img,position)
    train_data,test_data = torch.utils.data.random_split(train_dataset,[900,100])
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_data = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_data,test_data

def img_load():
    path_list = glob.glob("./picture_train/*.png")
    img_list = []
    for i in range(len(path_list)):
        
        if i % 3 ==0:
            image = np.array(cv2.imread(path_list[0]))/255

        if i % 3 ==1:
            image = np.array(cv2.imread(path_list[1]))/255
         
        if i % 3 ==2:
            image = np.array(cv2.imread(path_list[2]))/255

        #image = np.array(cv2.imread(path_list[i]))/255
        image = np.reshape(image,(3,600,800))
        image = torch.from_numpy(image.astype(np.float64)).float()
        img_list.append(image)

    return img_list

def position_load():
    position_list = []
    with open("./cursor_position_for_2-3.csv","r") as r:
        data = r.readlines()
    for i in range(len(data)):

        if i % 3 ==0:
            x,y = data[0].replace("\n","").split(",")
            
        if i % 3 ==1:
            x,y = data[1].replace("\n","").split(",")

        if i % 3 ==2:
            x,y = data[2].replace("\n","").split(",")
            
        #x,y = data[i].replace("\n","").split(",")
        x = float(x)/800
        y = float(y)/600
        position_list.append(torch.tensor([x,y],dtype=torch.float64).float())
    return position_list

def draw_loss(loss_list,epoch_list):
        plt.plot(epoch_list,loss_list)
        plt.savefig("./loss.png")
    
def predict(model,test_data,device):
    model.eval()
    model.to(device)
    print("------------predict----------------")
    for j, xy in enumerate(test_data):
        img = xy[0].to(device)
        label = xy[1].to(device)

        output = model(img)
        x = output[0][0].item()*800
        y = output[0][1].item()*600
        label_x = label[0][0].item()*800
        label_y = label[0][1].item()*600

        print("xy",x,y,"  label",label_x,label_y)
        if j ==4:
            break

    return output

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = 10
    lr = 0.001
    batch_size = 1

    img_data = img_load()
    position_data = position_load()
    train_data,test_data = dataloader(img_data,position_data,batch_size)
    print("----------finish dataloading-----------")
    model,loss_list,epoch_list = train(epoch,train_data,device,lr)
    print("----------finish training----------------")
    draw_loss(loss_list,epoch_list)
    predict(model,test_data,device)
    torch.save(model.state_dict(), './model/model_{}.pth'.format(epoch))
