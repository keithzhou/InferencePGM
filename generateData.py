from PIL import Image
import os
import numpy as np

PATHBG = "data/background/"
PATHFG = "data/foreground/"
PATHCOMBINED = "data/combined/"

locFG = {}
count = 1

def addNoise(image):
  # noise
  img = 1.0*np.array(image.getdata())
  img -= np.min(img)
  img *= 255/np.max(img)
  
  dr = max(img) - min(img)
  noise = np.random.normal(0,dr*.02,image.size[0]*image.size[1])
  img = img + noise
  img[img < 0] = 0
  img[img > 255] = 255
  return Image.fromarray(img.reshape(image.size[1],image.size[0]).astype(np.uint8))
  
for fileFG in os.listdir(PATHFG):
  imgFG = Image.open(PATHFG + fileFG)
  for fileBG in os.listdir(PATHBG):
    imgBG = Image.open(PATHBG + fileBG)

    # paste 
    if fileFG not in locFG:
      y = imgBG.size[1] - imgFG.size[1]
      x = int((imgBG.size[0] - imgFG.size[0])*np.random.random())
      locFG[fileFG] = (x,y)
    imgBG.paste(imgFG, (locFG[fileFG][0],locFG[fileFG][1]), imgFG)

    # to grayScale
    imgBG = imgBG.convert('L')
    numPixels = imgBG.size[0] * imgBG.size[1]
    scaleFactor = pow(5000.0 / numPixels, .5);
    print scaleFactor

    imgBG = imgBG.resize((int(imgBG.size[0] * scaleFactor), int(imgBG.size[1] * scaleFactor)), Image.BILINEAR)
    for i in range(10):
      image = addNoise(imgBG)
      image.save(PATHCOMBINED + str(count) + ".png")
      count += 1

    print "count:", count, "combine:", fileFG, "with:", fileBG
