import PIL.ImageQt as img
from PIL import ImageOps





numAplhaOld = {0: 'a', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
numAplha = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j', 20: 'k',
 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z', 36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f',
  41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r' , 46: 't'} 





#    for i in range(0,10,1):
#        varr, another = mnist_test[i]
#        print (another)
#        varr.show()


#return flipped and inverted image at index
def returnImage(index, dataSet):
    Image = dataSet[index][0]

    Image2 = ImageOps.flip(Image).rotate(-90)
    
    return img.toqpixmap(Image2)

def returnLabel(index, dataSet):
    label = dataSet[index][1]
    
    #return numAplha[label]
    return numAplha[label]

def returnSizeofTraining(dataSet):
    return len(dataSet)

