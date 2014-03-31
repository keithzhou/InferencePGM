import Image
import os 
import numpy as np

PATHCOMBINED = "data/combined/"

class loadData:
  def __init__(self, limit = None, randomize=True):

    self.files = os.listdir(PATHCOMBINED)
    if randomize == True:
      self.files = np.random.permutation(self.files)
    if limit:
      self.files = self.files[:limit]

    self.dataArray = []
    for i in self.files:
      image = Image.open(PATHCOMBINED+i) 
      self.dataArray.append(1.0*np.array(image.getdata(), dtype=np.float64)/255.0)
    
    self.dataArray = np.array(self.dataArray)
#    self.dataImage = [Image.open(PATHCOMBINED + i) for i in self.files]
#    self.dataArray = [1.0*np.array(img.getdata(), dtype=np.float64)/255.0 for img in self.dataImage]

  #def getDataImage(self):
    #return self.dataImage

  def getDataArray(self):
    return self.dataArray
