from fileinput import filename
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout , QTabWidget, QGridLayout, QScrollArea,QFormLayout,QLineEdit,\
 QGroupBox, QMainWindow, QStatusBar, QComboBox,QMessageBox, QFileDialog, QSlider, QApplication, QProgressBar

from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QTransform
from PyQt5.QtCore import QThreadPool, QThread, Qt, QPoint

import torchvision
import Scripts.training as train
import time
from threading import Thread
from functools import partial

import sys


from pathlib import Path
from PIL import ImageOps


class mainWindow(QMainWindow):     

    #Initialising all the variables used in funciton
    #aswell as initialising the GUI

    def __init__(self):
        super().__init__()
        
        #Initialising the gui

        self.setWindowTitle("Hand Writing recognition DNN")
        self.resize(540, 570)

        #initialising the Statusbar 

        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet('font-size: 18pt; font-family: Courier;')
        self.setStatusBar(self.statusBar)
        

        #initialising the variables used for the image viewer's index
        self.counter = 0
        self.charCounter =0
        self.upperRange = 0
        self.lowerRange = 100
        self.range = 100
        
        # variables for the image index when the images are filtered.
        self.scrollTransitionFlag = True
        self.filteredIndex = {0:0}

        self.cancelButtonPressed = False

        #adding tabs for the gui
        tabs = QTabWidget()
        tabs.addTab(self.downloadTabUI(), "Dataset")
        tabs.addTab(self.imageTabUI(), "Images")
        tabs.addTab(self.trainingTabUI(), "Training")
        tabs.addTab(self.drawTabUI(), "Draw")

        tabs.tabBarClicked.connect(self.tabClicked)
        
        self.drawEnabled = False
        
        self.setCentralWidget(tabs) 

        #for downloading the Dataset on a thread.
        self.downloadThread = QThreadPool()

        self.t = self.WorkerThread()

        #index 0 is counter, index 1 is to check if it's training index2 = loss, index 3 = accuracy
        #using a list as it acts as a pointer. for the training thread.
        self.trainingList = [0,False, 0,0]

        #try load data on startup, Autoloading when the dataset is already downloaded. 
        try:
            self.loadData()
            self.isDataLoaded = True
            print("dataset loaded")
            self.Text.setText("Testing:\n" + str(self.dataTesting) + "\n\n Training:\n"+ str(self.dataTraining))
            self.findDatasetLenth()
            
            #the dataset is not loaded.
        except:
            self.isDataLoaded = False
            print("data not there")
            self.Text.setText("Dataset Not Downloaded")


        # to initiate the training values.

        try:
          self.changeSliderValues()
        except:
            pass
        self.statusBarShowDataSetPresent()

#TABS

    def downloadTabUI(self):
        """Create the Download page UI."""

        self.Text = QLabel("default")

        downloadTab = QWidget()
        downloadButton = QPushButton("Download EMNIST Dataset")
        self.CancelDownload = QPushButton("Cancel Download")
        self.downloadLayout = QGridLayout()
        self.downloadLayout.setContentsMargins(100,100,100,0)
        self.downloadLayout.addWidget(downloadButton,2,1)
        self.downloadLayout.addWidget(self.CancelDownload,3,2)
        self.CancelDownload.hide()
        self.downloadLayout.addWidget(self.Text,4,1)

        downloadButton.clicked.connect(self.Download)
        self.CancelDownload.clicked.connect(self.cancelDownloadFunction)

        ##add progressbar
        
        

        downloadTab.setLayout(self.downloadLayout)
        return downloadTab


    def imageTabUI(self):
        """Create the image page UI."""
        imageTab = QWidget()

        outterLayout = QVBoxLayout()
        toplayout = QFormLayout()
        self.imageGrid = QGridLayout()
        self.bottomLayout = QGroupBox()


        filterButton = QPushButton('Apply filter and Search')
        self.filterBox = QLineEdit("")

        self.filterType = QComboBox()
        self.filterType.addItems(["Training", "Testing"])

        toplayout.addRow("filter by character:", self.filterBox)
        toplayout.addRow(self.filterType)
        toplayout.addRow(filterButton)

        filterButton.clicked.connect(self.findDatasetLenth)

        self.bottomLayout.setLayout(self.imageGrid)

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.bottomLayout)

        #when scrolling it checks if the index needs to be itterated.
        self.scrollArea.verticalScrollBar().valueChanged.connect(self.scrollFunction)

        outterLayout.addLayout(toplayout)
        outterLayout.addWidget(self.scrollArea)

        imageTab.setLayout(outterLayout)
        return imageTab

    def trainingTabUI(self):
        trainingTab = QWidget()
        self.saveFileName = QLineEdit("newModel")
        self.hiddenTextHolder = QLineEdit("")
        self.saveFileText = QLabel("./Model/" + self.saveFileName.text() + ".pth" + "\n\nSave model name")
        self.epochValue = QLabel("Epoch:")
        self.trainSplitValue = QLabel("Training Split")
        self.ephocSlider = QSlider(Qt.Orientation.Horizontal)
        self.trainingSplitSlider = QSlider(Qt.Orientation.Horizontal)
        
        

        
        self.trainButton = QPushButton("Train and Save")
        self.trainCancelButton = QPushButton("Cancel Training")
        self.trainingProgressBar = QProgressBar()
        self.trainingProgressText = QLabel("epoch...")

        self.trainCancelButton.hide()
        self.trainingProgressBar.hide()
        self.trainingProgressText.hide()


        self.epochValue.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trainSplitValue.setAlignment(Qt.AlignmentFlag.AlignCenter)

        

        self.ephocSlider.setMinimum(1)
        self.ephocSlider.setMaximum(100)
        self.ephocSlider.setValue(25)
        self.ephocSlider.setTickPosition(QSlider.TicksBelow)
        self.ephocSlider.setTickInterval(2)
        self.ephocSlider.setSingleStep(10)

        #need to figure this shit out
        self.trainingSplitSlider.setMinimum(1)
        self.trainingSplitSlider.setMaximum(112800)
        self.trainingSplitSlider.setValue(102800)
        self.trainingSplitSlider.setTickPosition(QSlider.TicksBelow)
        self.trainingSplitSlider.setTickInterval(10)
        self.ephocSlider.setSingleStep(10)

        self.ephocSlider.valueChanged.connect(self.changeSliderValues)
        self.trainingSplitSlider.valueChanged.connect(self.changeSliderValues)
        self.saveFileName.textChanged.connect(self.changeSliderValues)      




        trainingLayout = QVBoxLayout()
        trainingLayout.setContentsMargins(100,0,100,50)
        
        trainingLayout.addWidget(self.saveFileText,alignment=Qt.AlignmentFlag.AlignBottom)
        trainingLayout.addWidget(self.saveFileName)
        trainingLayout.addWidget(self.epochValue,alignment=Qt.AlignmentFlag.AlignBottom)
        trainingLayout.addWidget(self.ephocSlider)
        trainingLayout.addWidget(self.trainSplitValue,alignment=Qt.AlignmentFlag.AlignBottom)
        trainingLayout.addWidget(self.trainingSplitSlider)
        trainingLayout.addWidget(self.trainButton)

        trainingLayout.addWidget(self.trainCancelButton)
        trainingLayout.addWidget(self.trainingProgressBar)
        trainingLayout.addWidget(self.trainingProgressText)

        trainingTab.setLayout(trainingLayout)

       
        
        self.trainButton.clicked.connect(self.trainingTest)
        self.trainCancelButton.clicked.connect(self.stopTraining)
        self.hiddenTextHolder.textChanged.connect(self.change)


        return trainingTab




    def drawTabUI(self):

        drawTab = QWidget()
        self.drawing = False
        self.lastPoint = QPoint()

        self.image = QImage(500,500, QImage.Format_RGB32)
        self.image.fill(Qt.black)

        drawlayout = QVBoxLayout()

        self.labelDrawImage = QLabel()
        clearButton = QPushButton("clear")
        self.predictButton = QPushButton("Predict")
        self.selectModelButton = QPushButton("Open Model")
        clearButton.clicked.connect(self.clearImage)
        self.predictButton.clicked.connect(self.predictionStatusBar)
        self.selectModelButton.clicked.connect(self.selectModel)
        
        #Image for the drawn image set in teh temp folder.
        self.image.fill(Qt.black)
        self.image.save("./Scripts/temp/drawn.png","PNG")   
        img2 = QPixmap()
        QPixmap.load(img2,"./Scripts/temp/drawn.png", format="PNG")
        self.labelDrawImage.setPixmap(img2)

        drawlayout.addWidget(self.labelDrawImage)
        drawlayout.addWidget(clearButton)
        drawlayout.addWidget(self.selectModelButton)
        drawlayout.addWidget(self.predictButton)
        
       
        self.setLayout(drawlayout)

        self.show()


        drawTab.setLayout(drawlayout)
        return drawTab


    #funciton for the drawing, only active when tabbed in
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawEnabled:
            self.drawing = True
            self.lastPoint = event.pos()
            
        
    #funciton for the drawing, only active when tabbed in
    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing and self.drawEnabled:
            
            painter = QPainter(self.image)

            #painter.begin(self)
            painter.setPen(QPen(Qt.white, 30, Qt.SolidLine))
            #self.toggle = False
            
            painter.drawPoint(self.lastPoint)
            self.lastPoint = event.pos()


            self.update()
            
            self.image.save("./Scripts/temp/drawn.png","PNG")   

            im2 = self.image.scaled(28,28,1)
            im3 = im2.mirrored(False,True)
            transformer = QTransform()
            transformer.rotate(90)
            im4 = im3.transformed(transformer)
            
            im2.save("./Scripts/temp//Scaled.png","PNG") 
            im4.save("./Scripts/temp//Scaled2.png","PNG") 

       

            painter.end()
            
            img2 = QPixmap()
            QPixmap.load(img2,"./Scripts/temp/drawn.png", format="PNG")
            self.labelDrawImage.setPixmap(img2)
            

        
    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = True
            self.toggle = True

    #clears the current image, then saves and updates the drawing image.
    def clearImage(self):
        self.image.fill(Qt.black)
        self.image.save("./Scripts/temp/drawn.png","PNG")   
        
        img2 = QPixmap()
        QPixmap.load(img2,"./Scripts/temp/drawn.png", format="PNG")
        self.labelDrawImage.setPixmap(img2)



        


    


##Functions


    def clearImgGrid(self):
        for i in reversed(range(self.imageGrid.count())): 
            self.imageGrid.itemAt(i).widget().setParent(None)
        

    def buttonClick(self):
        self.clearImgGrid()
        if (self.filterBox.text() == ""):
            self.loadImages(char="all")
        else:
            self.loadImages(char=self.filterBox.text())

        #resets the sizing of the scroll and groupbox
        self.bottomLayout = QGroupBox()
        self.bottomLayout.setLayout(self.imageGrid)
        self.scrollArea.setWidget(self.bottomLayout)
        
        #showing which image data is indexed/
        self.statusBar.showMessage("Displaying:" + str(self.charCounter) + " images of " + str(len(self.filteredIndex) -1) + " of character:" + str(self.filterBox.text()))
        print("button")

    #function run when tabs are clicked. index input representing which tab is clicked. relevent statusbar info is updated
    def tabClicked(self, index):
        print(index)
        if (index == 0 or index == 2):
            if (self.isDataLoaded):
                self.statusBar.showMessage("EMNIST IS LOADED")
            else:
                self.statusBar.showMessage("EMNIST IS NOT DOWNLOADED")

        if (index == 1):
            if (self.isDataLoaded):
                self.statusBar.showMessage("Displaying:" + str(self.charCounter) + " images")
            else:
                self.statusBar.showMessage("EMNIST IS NOT DOWNLOADED")

        if (index == 3):
            if (self.isDataLoaded):
                self.drawEnabled = True
                self.statusBar.showMessage("EMNIST IS LOADED")
            else:
                self.statusBar.showMessage("EMNIST IS NOT DOWNLOADED")
        else:
            self.drawEnabled =False
            print("draw Diasbled")

    #when images are updated. it takes in the char in the filter Qedit box, 
    def loadImages(self, char):
        self.counter = self.upperRange*10
        self.charCounter = 0

        #for a range of 1000, assigns pixelmap to index image. to label and appends it to the image viewer.
        for i in range(self.upperRange,self.lowerRange,1):
            for j in range(10):

                label = QLabel()

                #check if filter actually has any value. if so, it indexes for all linearly and checks if it's for the training images or testing images.
                if(char == "all"):
                    if (self.filterType.currentText() == "Training"):
                        pixmap = QPixmap(train.returnImage(self.counter,self.dataTraining))
                    else:
                        pixmap = QPixmap(train.returnImage(self.counter,self.dataTesting))
                else:
                    if (self.filterType.currentText() == "Training"):
                        pixmap = QPixmap(train.returnImage(self.filteredIndex[self.counter],self.dataTraining))
                    else:
                        pixmap = QPixmap(train.returnImage(self.filteredIndex[self.counter],self.dataTesting))

                #resizing for good viewing
                pixmap2 = pixmap.scaled(39,39)
                label.setPixmap(pixmap2)

                #Putting Image into drawing Tab for predicion. on click it runs Predict image fucntion which will change the drawn image with the image indexed.
                label.mousePressEvent = partial(self.predictImage, index = self.counter, inChar=char)
                
                self.imageGrid.addWidget(label,i,j)

                print (self.counter)
                
                self.charCounter += 1
                self.counter += 1

                #checkpoint to break if the index is greater than the size of the array when there is No Filtered dtata
                if (self.filterType.currentText() == "Training"):
                    if (self.counter > train.returnSizeofTraining(dataSet=self.dataTraining)-1):
                        print("broke")
                        break
                    if (self.counter > len(self.filteredIndex)-2 and  (not(char == "all"))):
                        print("broke filtered")
                        break
                else:
                    if (self.counter > train.returnSizeofTraining(dataSet=self.dataTesting)-1):
                        print("broke")
                        break
                    if (self.counter > len(self.filteredIndex)-2 and  (not(char == "all"))):
                        print("broke filtered")
                        break

            #checkpoint to break if the index is greater than the size of the array when there IS Filtered dtata
            if (self.filterType.currentText() == "Training"):
                if (self.counter > len(self.filteredIndex)-2 and (not(char == "all"))):
                    print("broke filtere")
                    self.lowerRange = int((len(self.filteredIndex)-1)/10)
                    break
                if (self.counter > train.returnSizeofTraining(dataSet=self.dataTraining)-1):
                    print("broke")
                    break
            else:
                if (self.counter > len(self.filteredIndex)-2 and (not(char == "all"))):
                    print("broke filtere")
                    self.lowerRange = int((len(self.filteredIndex)-1)/10)
                    break
                if (self.counter > train.returnSizeofTraining(dataSet=self.dataTesting)-1):
                    print("broke")
                    break


    #This is the scroll function ran everytime it scrolls. if the srcoll reaches the max or minimum the images are cleared and updated with new index
    def scrollFunction(self):

        if (self.scrollArea.verticalScrollBar().value() == self.scrollArea.verticalScrollBar().maximum() and not(self.lowerRange * 10 >= len(self.filteredIndex)-1) ):
            self.scrollTransitionFlag = False
            self.upperRange = self.lowerRange -50

            self.lowerRange += 50
            #Button clicked is the Update image function.
            self.buttonClick()
            self.scrollArea.verticalScrollBar().setValue(int(self.scrollArea.verticalScrollBar().maximum()/2 - 10))
            self.scrollTransitionFlag = True

        if (self.scrollArea.verticalScrollBar().value() == 0 and self.scrollTransitionFlag and self.upperRange != 0):  
            self.scrollTransitionFlag = False
            self.lowerRange = self.upperRange +50
            self.upperRange -= 50
            if (self.upperRange < 0):
                self.upperRange = 0
            #Button clicked is the Update image function.
            self.buttonClick()
            self.scrollArea.verticalScrollBar().setValue(int(self.scrollArea.verticalScrollBar().maximum()/2))
            self.scrollTransitionFlag = True

        
        self.statusBar.showMessage("displaying:" + str(self.upperRange * 10) + " - " + str(self.lowerRange * 10) + " out of " + str(len(self.filteredIndex) -1))

    # This function is used when filter is needed. It makesa a dictionary to index filtered data. where the key is linear number representing filtered index. the key's value is the index of that filtered data in the whole dataset.
    def findDatasetLenth(self):

        if (self.isDataLoaded):
            self.counter = 0
            self.upperRange = 0
            self.lowerRange = 100
            index = 0
            charCounter = 0
            self.filteredIndex.clear()
            
            if (self.filterType.currentText() == "Training"):
                dataSize = len(self.dataTraining)
            else:
                dataSize = len(self.dataTesting)

            searchChar = self.filterBox.text()
            print(self.filterBox.text())
        
            if (self.filterType.currentText() == "Training"):
                while (index < dataSize):
                    while ((train.returnLabel(index,self.dataTraining) != searchChar) and (searchChar != "")):
                        index += 1
                        if(index == dataSize):
                            break

                    self.filteredIndex[charCounter] = index
                    index += 1
                    charCounter += 1
            else:
                while (index < dataSize):
                    while ((train.returnLabel(index,self.dataTesting) != searchChar) and (searchChar != "")):
                        index += 1
                        if(index == dataSize):
                            break

                    self.filteredIndex[charCounter] = index
                    index += 1
                    charCounter += 1


                

            
            print(charCounter - 1)
            print(len(self.filteredIndex))
            self.buttonClick()
            #return charCounter
        else:
            print("Data Not Downloaded")
            print('\a') # makes a sound
            
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("the DataSet is not downloaded")
            msg.setIcon(QMessageBox.Critical)
            msg.exec()

            #make a pop up


    def Download(self):
        #when download button is pressed. only allows one instance of the thread to exist. so that download button cannot download again in a parallel thraed.
        if (self.t.isRunning() == False):
            self.t.start()
            self.CancelDownload.show()
            time.sleep(2)#allow it to make the directories

            DD = Thread(target=self.TestPrint)
            DD.start()

            print("Left threads")
        else:
            print("Already Downloading")

    def cancelDownloadFunction(self):
        if (self.t.isRunning()):
            self.t.terminate()
            self.isDataLoaded = False
            self.cancelButtonPressed = True
            time.sleep(1)
            
            print("terminated")
            

    #this funciton is for the updating of the download progress. it's ran in a thread parallel to the download thread.
    def TestPrint(self):

        dots = 1
        self.cancelButtonPressed = False

        #will constantly the size of the file, 
        while True:


            Size1 = Path('./data/EMNIST/raw/gzip.zip').stat().st_size
            time.sleep(0.5)
            Size2 = Path('./data/EMNIST/raw/gzip.zip').stat().st_size

            #print(Size2)

            timeRemaining = ((561753746 - Size2)/((Size2+2) - (Size1+1)) * 0.5)

            #size of the ZIP file 561753746 bytes 150233088
            percentage = Path('./data/EMNIST/raw/gzip.zip').stat().st_size / 561753746

            if (percentage < 1):
                self.statusBar.showMessage(str(round(percentage*100, 2)) + "%" + "  time remanining: " + str(int(timeRemaining))) 
                self.Text.setText(str(round(percentage*100, 2)) + "%" + "  time remanining: " + str(int(timeRemaining))) 

            if (percentage == 1):
                self.statusBar.showMessage("Extracting" + str("."*dots)) 
                self.Text.setText("Extracting" + str("."*dots))                  
                dots += 1
                if (dots > 3):
                    dots = 1

            if (self.cancelButtonPressed):
                print("download canceled")
                self.statusBar.showMessage("canceled download")
                self.Text.setText("canceled download")
                self.CancelDownload.hide()
                
                break


            if ((self.t.isRunning() == False) and (self.cancelButtonPressed == False)):
                print("done Downloading and loading")
                self.loadData()
                self.statusBar.showMessage("EMNIST Loaded")
                self.Text.setText("Testing:\n" + str(self.dataTesting) + "\n\n Training:\n"+ str(self.dataTraining))
                self.CancelDownload.hide()
                #assigning variable to file location.
                break

    # the downloading thread that is a Qthread becuase it can be terminated.
    class WorkerThread(QThread):
        def run(self):
            torchvision.datasets.EMNIST(root="./data",split="balanced",train=False,download=True,transform=None)
            torchvision.datasets.EMNIST(root="./data",split="balanced",train=True,download=True,transform=None)
            
            #not sure if i need to do both?

    #loading function, ran with try at start. and also 
    def loadData(self):
            self.dataTraining = torchvision.datasets.EMNIST(root="./data",split="balanced",train=True,download=False,transform=None)
            self.dataTesting = torchvision.datasets.EMNIST(root="./data",split="balanced",train=False,download=False,transform=None)
            #isdataloaded checkmark used for preventing certaion operations that cannot work wihtout dataset.
            self.isDataLoaded = True

    #just to show if theres dataset.
    def statusBarShowDataSetPresent(self):
        if (self.isDataLoaded):
            self.statusBar.showMessage("EMNIST Dataset Loaded")
        else:
            self.statusBar.showMessage("EMNIST Dataset is not downloaded")



    #shows the result of the prediciton in drawing tab
    def predictionStatusBar(self):
        letter,prob = train.predict(self.selectModelButton.text())
        self.statusBar.showMessage(letter+" - " + prob + "%")

    #setting the png in the Drawing to the selected PNG.
    def predictImage(self,event,index, inChar):


        print("\a")
        msg = QMessageBox()
        msg.setWindowTitle("Alert")
        msg.setText("Image at Index: " + str(index) + " set in the Drawing Tab")
        msg.exec()

        if (inChar == "all"):
            if(self.filterType.currentText() == "Training"):
                img = QPixmap(train.returnImage(index,self.dataTraining))
            else:
                img = QPixmap(train.returnImage(index,self.dataTesting))
        else:
            if(self.filterType.currentText() == "Training"):
                img = QPixmap(train.returnImage(self.filteredIndex[index],self.dataTraining))
            else:
                img = QPixmap(train.returnImage(self.filteredIndex[index],self.dataTesting))


        #formatting the image to allow it to be predicted and to display it in the drawing canvas

        img2 = img.scaled(500,500)
        img2.save("./Scripts/temp/drawn.png","PNG")

        self.image.load("./Scripts/temp/drawn.png","PNG")

        im2 = self.image.scaled(28,28,1)
        im3 = im2.mirrored(False,True)
        transformer = QTransform()
        transformer.rotate(90)
        im4 = im3.transformed(transformer)
        
        im2.save("./Scripts/temp//Scaled.png","PNG") 
        im4.save("./Scripts/temp//Scaled2.png","PNG") 

        img2 = QPixmap()
        QPixmap.load(img2,"./Scripts/temp/drawn.png", format="PNG")
        self.labelDrawImage.setPixmap(img2)

        print(index)

    def selectModel(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
     
        #check if the file is .pth
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            print(filenames[0])
            train.checkModelLoadable(filenames[0])
            self.selectModelButton.setText(filenames[0])

    #update for the training tab, when values are changed text needs to be updated.
    def changeSliderValues(self):
        self.epochValue.setText("Number of Epochs:" +str(self.ephocSlider.value()))
        self.trainSplitValue.setText("training:"+str(self.trainingSplitSlider.value()) + "  validate:" + str(len(self.dataTraining) - self.trainingSplitSlider.value()))
        self.saveFileText.setText("./Model/" + self.saveFileName.text() + "_Epoch-" + str(self.ephocSlider.value()) + "_Split-" + str(round(self.trainingSplitSlider.value()/len(self.dataTraining),2)) + "_accuracy-NULL" + ".pth" + "\n\nSave model name")

    def trainingThreadUpdate(self):

        self.trainCancelButton.show()
        self.trainingProgressBar.show()
        self.trainingProgressText.show()

        self.ephocSlider.hide()
        self.trainingSplitSlider.hide()
        self.saveFileName.hide()
        self.trainButton.hide()

        self.trainingProgressText.setText("loading") 

        epoch = self.ephocSlider.value()

        while((self.trainingList[0] < epoch - 1) and self.trainingList[1] == True):
            time.sleep(1)   
            self.trainingProgressText.setText("epoch:" + str(self.trainingList[0]) + " of " + str(self.ephocSlider.value()) + " loss:" + str(self.trainingList[2])) 
            self.hiddenTextHolder.setText(self.trainingProgressText.text())
            print(self.trainingList[0])
            
        while(self.trainingList[1] == True):    
            time.sleep(1)   
            self.trainingProgressText.setText("Testing accuracy on Test Set") 

        self.trainingProgressText.setText("Accuracy of " + str(self.trainingList[3])) 
        self.trainCancelButton.setText("Return to Training")
        self.hiddenTextHolder.setText("wawawawawalkjlfkonbb;ionjbksdf")
            
            
    #starting the training, starts a thread to train and one to update the progress.
    def trainingTest(self):
        
        if (self.trainingList[1] == False):
            self.trainingList[1] = True
            trainingThread = Thread(target=train.testingButton,args=(self.dataTraining,self.dataTesting,self.ephocSlider.value(), self.trainingSplitSlider.value(),self.saveFileName.text(), self.trainingList))
            trainingThread.start()
            time.sleep(0.2)
            trainingUpdateThread = Thread(target=self.trainingThreadUpdate)
            trainingUpdateThread.start()
        else:
            print("already training \a")

    
    #canceling the training needs to revert the GUI back to it's original state
    def stopTraining(self):

        self.trainingList[1] = False
        if (self.trainCancelButton.text() == "Return to Training"):
            self.trainCancelButton.setText("Cancel")
            self.trainCancelButton.hide()
            self.trainingProgressBar.hide()
            self.trainingProgressText.hide()

            self.ephocSlider.show()
            self.trainingSplitSlider.show()
            self.saveFileName.show()
            self.trainButton.show()


        #wait for update to catch
        time.sleep(1.1)
        self.trainingProgressText.setText("Cancel")   
        #finish cancel thing


    #changing progress bar because of issues changing from thread
    def change(self):
        if (self.trainingList[1] == False):
            self.trainingProgressBar.setValue(100)
        else:
            self.trainingProgressBar.setValue(int((self.trainingList[0]/self.ephocSlider.value())*100))
