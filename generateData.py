import Image
import os
import numpy as np

PATHBG = "data/background/"
PATHFG = "data/foreground/"
PATHCOMBINED = "data/combined/"

locFG = {}
count = 1
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

    # save
    imgBG = imgBG.convert('L')
    imgBG.save(PATHCOMBINED + str(count) + ".png")

    print "count:", count, "combine:", fileFG, "with:", fileBG
    count += 1
