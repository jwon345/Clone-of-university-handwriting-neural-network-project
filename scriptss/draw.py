import sys
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import  QApplication, QLabel,QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QTransform

class Draw(QWidget):

    def __init__(self):
        super().__init__()
        self.drawing = False
        self.lastPoint = QPoint()

        self.image = QImage(500,500, QImage.Format_RGB32)
        self.image.fill(Qt.white)

        

        #self.setGeometry(100, 100, 22, 500)
        self.setWindowTitle('Draw the character')
        
        layout = QVBoxLayout()
        self.resize(self.image.width(), self.image.height())

        self.label = QLabel("ff")
        clearButton = QPushButton("clear")
        clearButton.clicked.connect(self.clearImage)
        

        img2 = QPixmap()
        QPixmap.load(img2,"this.png", format="PNG")
        self.label.setPixmap(img2)

        layout.addWidget(self.label)
        layout.addWidget(clearButton)
        
       
        self.setLayout(layout)

        self.show()



    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
            
        

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            
            
            painter = QPainter(self.image)

            #painter.begin(self)
            painter.setPen(QPen(Qt.black, 50, Qt.SolidLine))
            #self.toggle = False
            
            painter.drawPoint(self.lastPoint)
            self.lastPoint = event.pos()


            self.update()
            
            self.image.save("./this.png","PNG")   

            im2 = self.image.scaled(28,28,1)

            im2.save("./Scaled.png","PNG") 


            painter.end()
            
            img2 = QPixmap()
            QPixmap.load(img2,"this.png", format="PNG")
            self.label.setPixmap(img2)
            

        
    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = True
            print("Hi")
            self.toggle = True

    def clearImage(self):
        self.image.fill(Qt.white)
        self.image.save("./this.png","PNG")   
        
        img2 = QPixmap()
        QPixmap.load(img2,"this.png", format="PNG")
        self.label.setPixmap(img2)




  #  def paintEvent(self, event):
   #     painter = QPainter(self.image)
        
  #      painter.drawImage(self.rect(), self.image, \
   #         self.image.rect())
    #    painter.end()

        


if __name__ == '__main__':

    app = QApplication(sys.argv)
    drawing = Draw()
    drawing.show()
    sys.exit(app.exec_())