from PyQt5.QtWidgets import QApplication

import Scripts.GUI as GUI

#MAIN file that needs to be ran,.

#starts the GUI and runs until exited.

if __name__ == "__main__":


    app = QApplication([])

    widg = GUI.mainWindow()


    widg.show()

    app.exec()


