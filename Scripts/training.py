import PIL.ImageQt as img

from PIL import ImageOps


import numpy

import torch

from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


 

numAplhaOld = {0: 'a', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
numAplha = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10: 'A', 11: 'B', 12: 'c', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'i', 19: 'j', 20: 'k',
21: 'l', 22: 'm', 23: 'N', 24: 'o', 25: 'p', 26: 'Q', 27: 'R', 28: 's', 29: 'T', 30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z', 36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f',
  41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r' , 46: 't'} 



#most of this Training model is based off the tutorial site
#https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/?fbclid=IwAR1g08ycRkKWPpa_drQ4QVhHfT-wkBNBPQZ6B1Q1cmdNAcnTOOSSOj03duo
#my thanks to the website.

#model had 2 convolution layers and 1 Fully connected layer.

def testingButton(dataSet, testingSet, epochNumber, splitValue, saveName, progressList):
    
    

    def train(epoch):
        model.train()
        tr_loss = 0
        # getting the training set
        x_train, y_train = Variable(train_x), Variable(train_y)
        # getting the validation set
        x_val, y_val = Variable(val_x), Variable(val_y)
        # converting the data into GPU format
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        
        # prediction for training and validation set
        output_train = model(x_train)
        output_val = model(x_val)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        loss_val = criterion(output_val, y_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
        if epoch%2 == 0:
            # printing the validation loss
            print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)
            progressList[2] = loss_val



    arr1 = []
    arr2 = []
    for i in range(len(dataSet)):
        first,second = dataSet[i]
        #first.show()
        x = numpy.asarray(first)
        x = numpy.true_divide(x, 255)
        x = x.astype('float32')
        #print("here")
        #print(len(x))
        #print(x)
        arr1.append(x)
        arr2.append(second)

    print (len(arr1))
    imgArray = numpy.array(arr1)

    i = 0

    print(((len(arr1) - splitValue)/len(arr1)))
    labelArray = numpy.array(arr2)



    
    train_x, val_x, train_y, val_y = train_test_split(imgArray, labelArray, test_size = ((len(arr1) - splitValue)/len(arr1)))
    print(train_x.shape)
    print(val_x.shape)
    print(train_y.shape)
    print(val_y.shape)




    train_x = train_x.reshape(splitValue, 1, 28, 28)
    train_x  = torch.from_numpy(train_x)

    # converting the target into torch format
    train_y = train_y.astype("int64")
    train_y = torch.from_numpy(train_y)

    # shape of training data
    print(train_x.shape, train_y.shape)

    val_x = val_x.reshape(len(arr1)-splitValue, 1, 28, 28)
    val_x  = torch.from_numpy(val_x)

    # converting the target into torch format
    val_y = val_y.astype("int64")
    val_y = torch.from_numpy(val_y)

    # shape of validation data
    print(val_x.shape)
    print(val_y.shape)

    
    # defining the model
    model = Net()
    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=0.07)
    # defining the loss function
    criterion = CrossEntropyLoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        
    print(model)

    # defining the number of epochs
    n_epochs = epochNumber
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    # training the model
    for epoch in range(n_epochs):
        if (progressList[1] == False):
            print("canceled")
            return 0
        progressList[0] = epoch
        train(epoch)
        if (progressList[1] == False):
            print("canceled")
            return 0


    print("done")



#testing Set
    test_img = []
    testLabels = []
    for i in range(len(testingSet)):
        first,second = testingSet[i]
        x = numpy.asarray(first)
        x = numpy.true_divide(x, 255.0)
        x = x.astype('float32')
        test_img.append(x)
        testLabels.append(second)
    test_x = numpy.array(test_img)
    test_x = test_x.reshape(len(testingSet), 1, 28, 28)
    test_x  = torch.from_numpy(test_x)
    with torch.no_grad():
        output = model(test_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = numpy.argmax(prob, axis=1)

    print(predictions)

    sumRights = 0

    #Checking the accuracy of the trained model.
    for x in range(len(testingSet)):
        if (predictions[x] == testLabels[x]):
            sumRights += 1

    print (predictions[1])
    print (testLabels[1])

    acc = sumRights/len(testingSet)
    print("Accuracy of :" + str(round(acc*100,2)))
    
    progressList[3] = round(acc*100,2)
    progressList[1] = False
        

    torch.save(model.state_dict(), "./Model/" + saveName + "_Epoch-" + str(epochNumber) + "_Split-" + str(round(splitValue/len(arr1),2)) + "_accuracy"+ str(progressList[3]) + ".pth")

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            #47 output values
            Linear(4 * 7 * 7, 47)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x



#return flipped and inverted image at index
#index of data to know which image and the dataset itself 
def returnImage(index, dataSet):
    Image = dataSet[index][0]

    Image2 = ImageOps.flip(Image).rotate(-90)
    
    return img.toqpixmap(Image2)

#returns the label of the image.
def returnLabel(index, dataSet):
    label = dataSet[index][1]
    
    #return numAplha[label]
    return numAplha[label]

#returns the size of the dataset.
def returnSizeofTraining(dataSet):
    return len(dataSet)

#predict button clicked, the path of the model is passed, checks what the predicted is and return the Predicted character and Accuracy
def predict(loadPath):
    loader = Net()
    loadModel = torch.load(loadPath)
    loader.load_state_dict(loadModel)
    #loader.eval()

    #reading image as an array formatted for analysis
    image = imread("./Scripts/temp/Scaled2.png", as_gray=True)
    image /= 255
    
    image = image.astype('float32')
 
    im  = torch.from_numpy(image)

    im = im.reshape(1, 1, 28, 28)


    with torch.no_grad():
        output = loader(im)
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = numpy.argmax(prob, axis=1)
    
    sum = 0

    for x in range(len(numAplha)):
        print(str(numAplha[x]) + ":" + str(prob[0][x]))
        sum += prob[0][x]

    print("total = " + str(sum))

    print(numAplha[predictions[0]] + ": probability = " + str(prob[0][predictions[0]]/sum))

    return(numAplha[predictions[0]], str(round(((prob[0][predictions[0]]/sum)*100),2))  )

def checkModelLoadable(pathToModel):
    try:
        loader = Net()
        loadModel = torch.load(pathToModel)
        loader.load_state_dict(loadModel)   
        return True
    except:
        print("cannot Load")
        return False


