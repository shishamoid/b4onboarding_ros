import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision

from torch import optim

import numpy as np
import cv2
import glob

class cursor_model(nn.Module):
    def __init__(self, device):
        super(cursor_model,self).__init__()
        self.device = device
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=3)

        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(16,32,3)
        self.conv3 = nn.Conv2d(32,16,3)
        #self.conv3 = nn.Conv2d(16,,3)

        self.fc1 = nn.Linear(16*  78 * 141, 10)
        self.fc2 = nn.Linear(10, 2)

        self.flatten = nn.Flatten()

    def forward(self, x):
        #print("dev",x.device)
        #x = x.to(self.device)
        #print("dev",x.device)

        #print("0",x.shape) 
        #input torch.Size([10, 2160, 3840])
        x = self.pool(x)#torch.Size([10, 720, 1280])
        #print("1",x.shape)
        #x.to(self.device)
        x = self.conv1(x)
        
        #print("2",x.shape)
        x = self.relu(x)
        #print("3",x.shape)
        x = self.conv2(x)
        #print("4",x.shape)
        x = self.relu(x)
        #print("5",x.shape)
        x = self.pool(x)
        #print("6",x.shape)
        x = self.conv3(x)
        #print("7",x.shape)
        x = self.relu(x)
        #print("8",x.shape)
        x = self.pool(x)
        #print("9",x.shape)
        #x.view(4*78,4*141)
        x = self.flatten(x)
        #print("9.5",x.shape)
        x = self.fc1(x)
        #print("10",x.shape)
        x = self.relu(x)
        #print("11",x.shape)
        x = self.fc2(x)
        #print("output",x.shape)
        #print
        return x

def train(epoch,train_data,device):
    loss_list = []
    epoch_list = []

    model = cursor_model(device).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.015)
    
    for i in range(epoch):
        model.train()
        loss_train = 0
        for j, xy in enumerate(train_data):
            img = xy[0] #学習データ
            #print(img.shape)
            #print(img.reshape(3,2160,3840))
            position = xy[1] #正解データ(label))#torch.Size([10, 2, 1])
            #print("ラベル",position.shape
            #print(position)

            img = img.to(device)
            position = position.to(device)
            #print(type(img))

            loss = criterion(model(img), position)
            #print(loss.item())

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

def dataloader(img,position):
    img = torch.stack(img_data)
    position = torch.stack(position)

    train_dataset = TensorDataset(img,position)
    train_data,test_data = torch.utils.data.random_split(train_dataset,[45,5])
    train_data = DataLoader(train_data, batch_size=5, shuffle=False)
    test_data = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_data,test_data

def img_load():
    path_list = glob.glob("./picture/*.png")
    img_list = []
    for i in range(len(path_list)):
        
        image = np.array(cv2.imread(path_list[i]))/255
        image = np.reshape(image,(3,2160,3840))
        image = torch.from_numpy(image.astype(np.float32)).float()
        
        #image = torch.tensor(cv2.imread(path_list[i]),dtype=torch.float32).float()

        img_list.append(image)

    return img_list

def position_load():
    position_list = []
    with open("./cursor_position_for_2-3.txt","r") as r:
        data = r.readlines()
    for i in range(len(data)):
        x,y = data[i].split("|")
        x = int(x.replace(" ","").replace("x座標:","").replace("\n",""))/3840
        y = int(y.replace(" ","").replace("y座標:","").replace("\n",""))/2160
        position_list.append(torch.tensor([x,y],dtype=torch.float64).float())
        #position_list.append(torch.tensor([[x],[y]],dtype=torch.float64).float())
    return position_list

def predict(model,test_data,device):
    model.eval()
    model.to(device)
    #print(test_data)
    for j, xy in enumerate(test_data):
        img = xy[0].to(device)
        label = xy[1].to(device)

        output = model(img)
        #print(output.shape)
        #print(output)
        x = output[0][0]*3840
        y = output[0][1]*2160
        label_x = label[0][0]*3840
        label_y = label[0][1]*2160

        print("xy",x,y)
        print("label",label_x,label_y)
        if j ==1:
            break
        #print("item", output[0].data())

    #output[0]

    return output

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_data = img_load()
    position_data = position_load()
    train_data,test_data = dataloader(img_data,position_data)
    #print(len(test_data))
    print("----------finish dataloading-----------")
    model,loss_list,epoch_list = train(20,train_data,device)
    predict(model,test_data,device)

    
    
