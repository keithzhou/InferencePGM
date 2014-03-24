import Image
import numpy as np

class visualizer:
  def __init__(self):
    self.shape = (225,300) # height * width

  def array2image(self,datain):
    data = datain.reshape(self.shape)
    rescaled = (255.0 * data).astype(np.uint8)
    return Image.fromarray(rescaled)

  def visualize(self, alpha, mu, sigma):
    im = Image.new('L', (self.shape[1] * 3, self.shape[0] * len(alpha)))
    for i in range(len(alpha)):
        im.paste(self.array2image(alpha[i]),(self.shape[1]*0, self.shape[0]*i))
        im.paste(self.array2image(mu[i]),(self.shape[1]*1, self.shape[0]*i))
        im.paste(self.array2image(sigma[i]),(self.shape[1]*2, self.shape[0]*i))
    im.show()
