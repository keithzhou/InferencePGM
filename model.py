import numpy as np
import loadData
import visualizer

class model:
  def __init__(self,  num_class = 14, limit = None):
    self.num_class = num_class
    self.iteration = 0

    # data
    self.data = loadData.loadData(limit = limit).getDataArray()
    print "number of training data:", len(self.data)

    # model parameters
    # for each pixel in each class, there are: 
      # 1. pi representing probability of the class
      # 2. alpha representing probability of being foreground
      # 3. mu and sigma representing gaussian of observation vs fg/bg data
    self.param_pi = np.array([1.0/self.num_class] * self.num_class)
    self.param_alpha = np.zeros((self.num_class, len(self.data[0]))) + 0.5
    self.param_mu = np.random.random((self.num_class, len(self.data[0]))) 
    self.param_sigma = np.ones((self.num_class, len(self.data[0]))) * 10

    # visualizer
    self.visualizer = visualizer.visualizer()

  def log_gaussian(self,data,mu,sigma):
    return -1.0 * (data - mu) ** 2 / (2 * sigma**2) - np.log(np.sqrt(2*np.pi)*sigma)

  def step_E(self):
      raise NotImplemented

  def step_M(self): 
      raise NotImplemented
    
  def train(self):
    while 1:
      try:
        print "iteration:", self.iteration
        self.step_E()
        print "done step_E."
        self.step_M()
        print "done step_M."
        self.iteration += 1
      except KeyboardInterrupt:
        self.visualize()
        break
      
  def visualize(self):
      self.visualizer.visualize(self.param_alpha, self.param_mu, self.param_sigma)
