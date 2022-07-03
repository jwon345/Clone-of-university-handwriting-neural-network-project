# Versions 1.0 Release
team 31 project 1 

A deep learning project that makes use of the EMNIST dataset to predict user's hand drawn letter. 

a project from the university of auckland's Compsys  302 course.

the project is fully functional. with a model at 81epochs and a 0.89 data split. producing an accuracy of 76.21 percent on the test set.

V1.0 is fully functional. 
although it does 

# Install instructions including depedencies.

    Python 3 or newer needs to be installed.


    open cmd
    pip install pyqt5
    pip install torchvision
    pip install pillow
    pip install scikit-image
    pip install tpmd
    pip install sklearn
    
    ^The dependencies

    Clone the repo.

    run the main.py file.

# How to use

## Download Dataset
when the program is opened, The status bar will indicate if the Dataset needs to be Downloaded.

![IMAGE ALT TEXT](/ReadMeContents/NotDownload.png)

----
pressing the donwload button will initiate the Download.
It can be cancelled at any time

Note that the video has been paused during the download


![IMAGE TEXT](/ReadMeContents/download.gif)

## View the Images
viewing of the image is Dynamic. It will automatically change it's index as the user scrolls to the limits. Apply filter with empy search to view all.


![IMAGE TEXT](/ReadMeContents/showImage.gif)


viewing with filtered characters 

![IMAGE TEXT](/ReadMeContents/imageFilter.gif)

## Train the Dataset

the model will be saved at in the ./model File.
Epoch, Split and Accuracy will be saved into the file name.

Note the Recording has been Paused during training

![IMAGE TEXT](/ReadMeContents/training.gif)
## Load the DataSet and Predict

The model needs to be loaded, and predictions can be made on the from the canvas.
From the image tab. clicking a image will send it to the drawing tab and can then be predicted.

![IMAGE TEXT](/ReadMeContents/predicting.gif)

# Updates
    - v1d developing
    - v1.01d developing with progress
    - v1.02d Image viewer and Downloader working with missing features.
        - moved most things into the gui since it can be handled better.
        - download percentage is based on file size -> don't know if this is right? but works
    - v1.03d training pof, saved need to be tested
    - v1.04d main concepts completed. need to add smaller features.
    - v1.05d Training is completed but buggy.
    - v1Release - currently has issues but is functional.
