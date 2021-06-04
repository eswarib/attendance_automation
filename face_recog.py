#Trying to learn to recognise faces
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from sklearn.metrics import accuracy_score
import face_recognition
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import PIL.Image

import os
import argparse

from torch.nn import Module, Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from skimage.io import imread
from skimage.transform import resize

from tqdm import tqdm

def get_data_from_images() :
    thisdir = './trainset'
    print(thisdir)
    files = []
    train_data_inp = []
    train_data_tar = []
    train_data_set = []
                        
    size = (800,600)
    # r=root, d=directories, f = files
    for r, d, f in os.walk(thisdir):
        for file in f:
            if file.endswith(".jpg"):
                #print(os.path.join(r, file))
                file_name = os.path.join(r, file)
                files.append(file_name)
            
                print("Getting data from file : ",file_name)
                image = face_recognition.load_image_file(file_name);
                face_encodings = face_recognition.face_encodings(image);
                target = r.split("/")[-1]
                print('TARGET : ', target)
                
                im = imread(file_name)
                im_resized = resize(im,output_shape=(600,800,3), mode='constant', anti_aliasing=True)
                im_resized /= 255.0
                #im_resized.save(file_name)

                #transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
                transform1=transforms.ToTensor()
                transform2=transforms.ToPILImage()
                #image_tensor = transform1(im_resized)
                #print(f"Lets print the tensor shape and values of transformed image : {image_tensor.shape} \n {image_tensor}")

                #plt.imshow(transform2(image_tensor))
                #if target already does not exist
                if(train_ylabel.get(target) == None):
                    train_ylabel[target] = len(train_ylabel)
                    train_ylabel_reverse[len(train_ylabel_reverse)]=target

                train_data_set.append((im_resized,target))
                train_data_inp.append(im_resized)
                train_data_tar.append(train_ylabel[target])
                print('train_data_tar : ',train_data_tar)
                
 
    print('Type of train_data_set : ', type(train_data_set))
    
    print("shape of train_data_set : ",train_data_set[0][0].shape)
    print("shape of train_data_inp : ",train_data_inp[0].shape)
                                      
    return train_data_set,np.array(train_data_inp),np.array(train_data_tar)



def rescale_images(directory, size):
  for img in os.listdir(directory):
    im = Image.open(directory+img)
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(directory+img)

import torch.nn.functional as func 
    
class recognition1(nn.Module):  
    def __init__(self,input_layer,hidden_layer1,hidden_layer2,output_layer):  
        super().__init__()  
        self.linear1=nn.Linear(input_layer,hidden_layer1)  
        self.linear2=nn.Linear(hidden_layer1,hidden_layer2)  
        self.linear3=nn.Linear(hidden_layer2,output_layer)  
    def forward(self,x):  
        x=func.relu(self.linear1(x))  
        x=func.relu(self.linear2(x))  
        x=self.linear3(x)  
        return x      

class Net(Module):
    out_features = 40
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            #Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            Conv2d(3,10, kernel_size=3, stride=1, padding=1),
            #BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            
            # Defining another 2D convolution layer
            #Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            Conv2d(10, Net.out_features, kernel_size=3, stride=1, padding=1),
            #BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            #Linear(8 * 150 * 200, 2)
            Linear(Net.out_features * 150 * 200, Net.out_features)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        print('shape od x after convolution : ',x.shape)
        
        x = x.view(x.size(0), -1)
        print('shape od x after changing view : ',x.shape)
        x = self.linear_layers(x)
        print('shape od x after linear layer : ',x.shape)
        return x


def cnn1(training_dataset) :
    training_loader=torch.utils.data.DataLoader(dataset=training_dataset,batch_size=100,shuffle=False)
    dataiter=iter(training_loader)  
    images,labels=dataiter.next()
    fig=plt.figure(figsize=(25,4)) 
    
    transform2=transforms.ToPILImage()
    for idx in np.arange(20):      
        ax=fig.add_subplot(2,10,idx+1) 
        
        plt.imshow(transform2(images[idx]))   
        #ax.set_title([labels[idx].item()])
        ax.set_title([labels[idx]])
        
    model=recognition1(1440000,125,65,20)  
    criteron=nn.CrossEntropyLoss()  
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)  
    epochs=12  
    loss_history=[]  
    correct_history=[]  
    for e in range(epochs):  
        loss=0.0  
        correct=0.0  
        for input,labels in training_loader:
            print(f"input : {input.shape} input view shape : {input.view(input.shape[0],-1)} ")
            inputs=input.view(input.shape[0],-1)
            print(f"Printing the shape of input for model : {inputs.shape}")
            outputs=model(inputs)  
            loss1=criteron(outputs,labels)  
            optimizer.zero_grad()  
            loss1.backward()  
            optimizer.step()  
            _,preds=torch.max(outputs,1)  
            loss+=loss1.item()  
            correct+=torch.sum(preds==labels.data)  
        else:  
            epoch_loss=loss/len(training_loader)  
            epoch_acc=correct.float()/len(training_loader)  
            loss_history.append(epoch_loss)  
            correct_history.append(epoch_acc)  
            print('training_loss:{:.4f},{:.4f}'.format(epoch_loss,epoch_acc.item())) 

    plt.show()
                                      
def train(train_x,train_y):
    
    
    # defining the model
    model = Net()
    model = model.double()
    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=0.0001)
    # defining the loss function
    criterion = CrossEntropyLoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print(model)
    
    # batch size of the model
    batch_size = 10

    # number of epochs to train the model
    n_epochs = 7

    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        
        permutation = torch.randperm(train_x.size()[0])
        
        training_loss = []
        for i in tqdm(range(0,train_x.size()[0], batch_size)):

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_x[indices].double(), train_y[indices]
            batch_y = batch_y.type(torch.LongTensor)
        
            
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            optimizer.zero_grad()
            # in case you wanted a semi-full example
            outputs = model(batch_x.double())
            
            loss = criterion(outputs,batch_y)
            
            print('loss calculated for this batch : ',loss)

            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        training_loss = np.average(training_loss)
        print('epoch: \t', epoch, '\t training loss: \t', training_loss)

        
def test(val_x):
    sample_submission = {}
    model = Net()
    model = model.double()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # generating predictions for test set
    with torch.no_grad():
        output = model(val_x)
        
    print('shape of val_x : ',val_x.shape)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    
    sample_submission["label"] = predictions
    sample = pd.DataFrame(sample_submission)
    
    print(sample.head(10))
    for i in range(0,val_x.shape[0]):
        print(predictions[i],train_ylabel_reverse[predictions[i]])
        #lets plot the image also
        
        transform2 = transforms.ToPILImage()
        x = (val_x[i] * 255)
        #print('val_x image ',val_x[i])
        #print('type of image : ',type(val_x[i][:]))
        #print('shape of output : ',output[i][:].shape)
        x = transform2(x)
        #print('shape of x after transform : ',x.shape)
    
    plt.imshow(x)
    plt.show()
        
if __name__ == '__main__':
    
    #train_ylabel = {'0001_0000268':0,'0001_0000255' : 1,'0001_0000262' : 2,'0001_0000298' : 3,'0001_0000284' : 4,'0001_0000274' : 5,'0001_0000305' : 6,'0001_0000304' : 7,'0001_0000303' : 8,'0001_0000301' : 9,'0001_0000299' : 10,'0001_0000297' : 11,'0001_0000293' : 12,'0001_0000292' : 13,'0001_0000286' : 14,'0001_0000283' : 15,'0001_0000281' : 16,'0001_0000278' : 17,'0001_0000265':18,'0001_0000264' : 19}
    
    #train_ylabel_reverse = {0:'0001_0000268',1:'0001_0000255',2:'0001_0000262',3:'0001_0000298',4:'0001_0000284',5:'0001_0000274',6:'0001_0000305',7:'0001_0000304',8:'0001_0000303',9:'0001_0000301',10:'0001_0000299',11:'0001_0000297',12:'0001_0000293',13:'0001_0000292',14:'0001_0000286',15:'0001_0000283',16:'0001_0000281',17:'0001_0000278',18:'0001_0000265',19:'0001_0000264'}
    
    train_ylabel = {}
    train_ylabel_reverse = {}
    
    
    training_dataset,train_x,train_y = get_data_from_images()
    
    print('Printing train_ylabel and train_ylabel_reverse')
    print(f"{train_ylabel}\n\n{train_ylabel_reverse}")
    
    # create validation set
    #train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = 13, stratify=train_y)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = 13)
    print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)

    # converting training images into torch format
    train_x = train_x.reshape(train_x.shape[0], 3, 600, 800)
    train_x  = torch.from_numpy(train_x).double()

    # converting the target into torch format
    train_y = train_y.astype(int)
    train_y = torch.from_numpy(train_y).double()

    # shape of training data
    print('printing shapes arfter reshaping',train_x.shape, train_y.shape)
    
    # converting validation images into torch format
    val_x = val_x.reshape(val_x.shape[0], 3, 600, 800)
    val_x  = torch.from_numpy(val_x)

    # converting the target into torch format
    val_y = val_y.astype(int)
    val_y = torch.from_numpy(val_y)

    # shape of validation data
    print('shape of validation vectors : ',val_x.shape, val_y.shape)

    train(train_x,train_y)
    
    test(val_x)
    
