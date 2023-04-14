# import os
# from keras.preprocessing import image
# import matplotlib.pyplot as plt 
# import numpy as np
# from keras.utils.np_utils import to_categorical
# import random,shutil
# from keras.models import Sequential
# from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
# from keras.models import load_model


# def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

#     return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

# BS= 32
# TS=(24,24)
# train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
# valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
# SPE= len(train_batch.classes)//BS
# VS = len(valid_batch.classes)//BS
# print(SPE,VS)


# # img,labels= next(train_batch)
# # print(img.shape)

# model = Sequential([
#     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
#     MaxPooling2D(pool_size=(1,1)),
#     Conv2D(32,(3,3),activation='relu'),
#     MaxPooling2D(pool_size=(1,1)),
# #32 convolution filters used each of size 3x3
# #again
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(1,1)),

# #64 convolution filters used each of size 3x3
# #choose the best features via pooling
    
# #randomly turn neurons on and off to improve convergence
#     Dropout(0.25),
# #flatten since too many dimensions, we only want a classification output
#     Flatten(),
# #fully connected to get all relevant data
#     Dense(128, activation='relu'),
# #one more dropout for convergence' sake :) 
#     Dropout(0.5),
# #output a softmax to squash the matrix into output probabilities
#     Dense(2, activation='softmax')
# ])

# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)

# model.save('models/cnnCat2.h5', overwrite=True)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.Resize((24, 24)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.ImageFolder(root='C:/Users/Public/dataset_new/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = datasets.ImageFolder(root='C:/Users/Public/dataset_new/test', transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*18*18, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = self.dropout1(x)
        x = x.view(-1, 64*18*18)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.softmax(x, dim=0)
        return output

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(15):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(inputs.shape)
        # print(outputs.shape)
        # print(labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

torch.save(model.state_dict(), 'models/cnnCat21.pt')
