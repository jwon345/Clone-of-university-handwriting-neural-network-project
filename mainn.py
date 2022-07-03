from cgitb import lookup
import multiprocessing
from time import sleep
from PyQt5.QtWidgets import QApplication

import scriptss.GUI as GUI

if __name__ == "__main__":


    app = QApplication([])

    widg = GUI.mainWindow()


    widg.show()

    app.exec()


