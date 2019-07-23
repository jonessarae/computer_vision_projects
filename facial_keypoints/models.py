## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values 
        ## 2 for each of the 68 keypoint (x, y) pairs
        
        # input is 96x96x1
        self.conv1 = nn.Conv2d(1, 32, 5)
        #(W-F)/S + 1 = (96-4)/1 + 1 = 92/2 = 46       
        self.conv2 = nn.Conv2d(32, 64, 3)
        #(46-3)/1 + 1 = 44/2 = 22
        self.conv3 = nn.Conv2d(64, 128, 3)      
        #(22-3)/1 + 1 = 20/2 = 10
        self.conv4 = nn.Conv2d(128, 256, 1)
        #(10-1)/1 + 1 = 10/2 = 5
        
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers 
        ## (such as dropout or batch normalization) to avoid overfitting
        
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # linear layer 
        self.fc1 = nn.Linear(256 * 5 * 5, 1280)
        # xavier initialization
        I.xavier_uniform_(self.fc1.weight)
        
        # linear layer
        self.fc2 = nn.Linear(1280, 640)
        # xavier initialization
        I.xavier_uniform_(self.fc2.weight)
        
        # linear layer
        self.fc3 = nn.Linear(640, 136)
        # xavier initialization
        I.xavier_uniform_(self.fc3.weight)
        
        # dropout layer
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # convolutional layers with relu activation, maxpooling, and dropout
        
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))
        
        x = self.pool(F.relu(self.conv4(x)))

        x = self.dropout(x)
        
        # flatten image input
        x = x.view(-1, 256 * 5 * 5)
        
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
