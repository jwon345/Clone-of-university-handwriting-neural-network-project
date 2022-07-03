from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QCheckBox, QTabWidget, QGridLayout, QScrollArea,QFormLayout,QLineEdit,\
 QGroupBox, QMainWindow, QStatusBar

from PyQt5 import QtGui,QtCore,QtWidgets
from PyQt5.QtGui import QPixmap, QTransform,QPainter,QPen
from PyQt5.QtCore import QThreadPool, QRunnable, QThread,Qt

import torchvision
import Scripts.training as train
import time


from threading import Thread
import os
import io
import sys
import subprocess

from pathlib import Path




class mainWindow(QMainWindow):     
    def __init__(self):
        super().__init__()
        # Create a top-level layout

        self.setWindowTitle("Hand Writing recognition DNN")
        self.resize(550, 400)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("NO DATA LOADED")

        self.counter = 0
        self.charCounter =0
        self.upperRange = 0
        self.lowerRange = 100
        self.range = 100
        
        self.scrollTransitionFlag = True

        tabs = QTabWidget()
        tabs.addTab(self.generalTabUI(), "Dataset")
        tabs.addTab(self.imageTabUI(), "Images")
        tabs.addTab(self.generalTabUI(), "Training")
        tabs.addTab(self.drawingTabUI(), "Draw")

        tabs.tabBarClicked.connect(self.tabClicked)

        
        self.setCentralWidget(tabs) 

        self.downloadThread = QThreadPool()


    def generalTabUI(self):
        """Create the General page UI."""

        self.Text = QLabel("emptyRN")

        generalTab = QWidget()
        donwloadButton = QPushButton("Download/Load -> Data")
        layout = QVBoxLayout()
        layout.addWidget(donwloadButton)
        layout.addWidget(self.Text)
        donwloadButton.clicked.connect(self.Download)

        generalTab.setLayout(layout)
        return generalTab

    
    def drawTabUI(self):

        drawTab = QWidget()
        self.drawing = False
        self.lastPoint = QPoint()

        self.image = QImage(550,540, QImage.Format_RGB32)
        self.image.fill(Qt.white)

        

        #self.setGeometry(100, 100, 22, 500)
        #self.setWindowTitle('Draw the character')
        
        drawlayout = QVBoxLayout()
        #self.resize(self.image.width(), self.image.height())

        self.label = QLabel()
        clearButton = QPushButton("clear")
        predictButton = QPushButton("Predict")
        clearButton.clicked.connect(self.clearImage)
        predictButton.clicked.connect(self.predictionStatusBar)
        
        self.image.fill(Qt.black)
        self.image.save("./Scripts/temp/drawn.png","PNG")   
        img2 = QPixmap()
        QPixmap.load(img2,"./Scripts/temp/drawn.png", format="PNG")
        self.label.setPixmap(img2)

        drawlayout.addWidget(self.label)
        drawlayout.addWidget(clearButton)
        drawlayout.addWidget(predictButton)
        
       
        self.setLayout(drawlayout)

        self.show()


        drawTab.setLayout(drawlayout)
        return drawTab
 



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

        

        #self.filerBox.text().valueChanged.connect(self.Test)
        
  #      self.label = QLabel()
  #      pixmap = QPixmap(train.returnImage(0))
  #      self.label.setPixmap(pixmap)
#        self.imageGrid.addWidget(self.label,0,0)

        filterButton.clicked.connect(self.findDatasetLenth)

        #self.loadImages(char="all")

                    

        #print(train.returnSizeofTraining())
        self.bottomLayout.setLayout(self.imageGrid)

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.bottomLayout)

        
        self.scrollArea.verticalScrollBar().valueChanged.connect(self.Test)

        outterLayout.addLayout(toplayout)
        outterLayout.addWidget(self.scrollArea)

        imageTab.setLayout(outterLayout)
        return imageTab

    def trainingTabUI(self):
        trainingTab = QWidget()
        text = QLabel("This is Training Tab")
        trainButton = QPushButton("testing Button")
        testButton = QPushButton("testing Button2")
        trainingLayout = QVBoxLayout()
        trainingLayout.addWidget(text)
        trainingLayout.addWidget(trainButton)
        trainingLayout.addWidget(testButton)
        trainingTab.setLayout(trainingLayout)

        #partial import to allow for parameter passing
        
        trainButton.clicked.connect(self.trainingTest)
        testButton.clicked.connect(train.predict)

        return trainingTab

    def clearImgGrid(self):
        for i in reversed(range(self.imageGrid.count())): 
            self.imageGrid.itemAt(i).widget().setParent(None)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawEnabled:
            self.drawing = True
            self.lastPoint = event.pos()
            
        

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing and self.drawEnabled:
            
            
            painter = QPainter(self.image)

            #painter.begin(self)
            painter.setPen(QPen(Qt.white, 60, Qt.SolidLine))
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
            self.label.setPixmap(img2)
            

        
    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = True
            self.toggle = True

    def clearImage(self):
        self.image.fill(Qt.black)
        self.image.save("./Scripts/temp/drawn.png","PNG")   
        
        img2 = QPixmap()
        QPixmap.load(img2,"./Scripts/temp/drawn.png", format="PNG")
        self.label.setPixmap(img2)

        

    def buttonClick(self):
        self.clearImgGrid()
        self.loadImages(char=self.filterBox.text())

        #resets the sizing of the scroll and groupbox
        self.bottomLayout = QGroupBox()
        self.bottomLayout.setLayout(self.imageGrid)
        self.scrollArea.setWidget(self.bottomLayout)

        #self.statusBar.showMessage("Test")
        self.statusBar.showMessage("Displaying:" + str(self.charCounter) + " images")
        print("button")

    def tabClicked(self, index):
        print(index)
        if (index == 1):
             self.statusBar.showMessage("Displaying:" + str(self.charCounter) + " images")

    def loadImages(self, char):
        self.counter = self.upperRange*10
        self.charCounter = 0
        for i in range(self.upperRange,self.lowerRange,1):
            for j in range(10):

                #filter for char
                while ((train.returnLabel(self.counter,self.dataTesting) != char) and (char != "all")):
                    self.counter += 1
                    if (self.counter  > train.returnSizeofTraining(dataSet=self.dataTesting)-1):
                        break
                if (self.counter > train.returnSizeofTraining(dataSet=self.dataTesting)-1):
                    print("broke")
                    break

                label = QLabel()
                pixmap = QPixmap(train.returnImage(self.counter,self.dataTesting))
                pixmap2 = pixmap.scaled(40,40)
                label.setPixmap(pixmap2)

                #print(str(i) +" + "+ str(j))
                print (self.counter)
                
                self.charCounter += 1
                self.counter += 1
                if (self.counter > train.returnSizeofTraining(dataSet=self.dataTesting)-1):
                    print("broke")
                    break
                #print (train.returnLabel(self.counter))




                self.statusBar.showMessage("loading:" + str(self.counter))
                
                self.imageGrid.addWidget(label,i,j)

            if (self.counter > train.returnSizeofTraining(dataSet=self.dataTesting)-1):
                print("broke")
                break

    def Test(self):
        #print("scroll- " + str(self.scrollArea.verticalScrollBar().value()))
        #print("scrollmax " + str(self.scrollArea.verticalScrollBar().maximum()))
        if (self.scrollArea.verticalScrollBar().value() == self.scrollArea.verticalScrollBar().maximum()):
            self.scrollTransitionFlag = False
            self.upperRange = self.lowerRange -50
            self.lowerRange += 50
            self.buttonClick()
            self.scrollArea.verticalScrollBar().setValue(int(self.scrollArea.verticalScrollBar().maximum()/2))
            self.scrollTransitionFlag = True

        if (self.scrollArea.verticalScrollBar().value() == 0 and self.scrollTransitionFlag and self.upperRange != 0):  
            self.scrollTransitionFlag = False
            self.lowerRange = self.upperRange +50
            self.upperRange -= 50
            self.buttonClick()
            self.scrollArea.verticalScrollBar().setValue(int(self.scrollArea.verticalScrollBar().maximum()/2))
            self.scrollTransitionFlag = True

        self.statusBar.showMessage("displaying:" + str(self.upperRange * 10) + " - " + str(self.lowerRange * 10) + " out of " ) 

    def settext(self,a):
        self.Text.setText("hihi")

    def Download(self):
        os.system("title " + "cmd")
        print(self.counter)

        #funcOfThread = HelloWorldTask(self.lowerRange)
        #self.downloadThread.start(funcOfThread)

        self.t = self.WorkerThread()
        self.t.start()
        

        time.sleep(2)#allow it to make the directories

        DD = Thread(target=self.TestPrint)
        DD.start()

        print("Left threads")

         
    def TestPrint(self):
        while True:
            time.sleep(0.1)
            #print(os.path.getsize("./data/raw/EMNIST/gzip.zip"), 'bytes')

            #size of the ZIP file 561753746 bytes
            percentage = Path('./data/EMNIST/raw/gzip.zip').stat().st_size / 561753746
            self.statusBar.showMessage(str(round(percentage*100, 2)) + "%") 


            if (self.t.isFinished()):
                print("done Downloading and loading")
                self.dataTraining = torchvision.datasets.EMNIST(root="./data",split="balanced",train=False,download=False,transform=None)
                self.dataTesting = torchvision.datasets.EMNIST(root="./data",split="balanced",train=True,download=False,transform=None)
                #assigning variable to file location.
                break

           

    class WorkerThread(QThread):
        def run(self):
            torchvision.datasets.EMNIST(root="./data",split="balanced",train=False,download=True,transform=None)
            torchvision.datasets.EMNIST(root="./data",split="balanced",train=True,download=True,transform=None)
            #not sure if i need to do both?
    
    
