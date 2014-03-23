import numpy as np
import loadData
import visualizer

class ICM:
  def __init__(self,  num_class = 14):
    self.num_class = num_class;

    # data
    self.data = loadData.loadData().getDataArray()

    # model parameters
    # for each pixel in each class, there are: 
      # 1. pi representing probability of the class
      # 2. alpha representing probability of being foreground
      # 3. mu and sigma representing gaussian of observation vs fg/bg data
    self.param_pi = np.array([1.0/self.num_class] * self.num_class)
    self.param_alpha = np.ones((self.num_class, len(self.data[0])))  * 0.5
    self.param_mu = np.random.random((self.num_class, len(self.data[0]))) 
    self.param_sigma = np.ones((self.num_class, len(self.data[0]))) * 1.0

    # hidden variables
    # for each training case, there are:
      # 1. f and b representing FG/BG class
      # 2. m representing the mask
    self.h_f = np.random.randint(0,self.num_class,len(self.data))
    self.h_m = np.random.random_integers(0,1,(len(self.data), len(self.data[0])))
    self.h_b = np.random.randint(0,self.num_class,len(self.data))

  def gaussian(self, x, mu, sig):
      return 1.0/(pow(2*np.pi, .5) * sig) * np.exp(-(x-mu)*(x-mu) / (2 * sig * sig ))

  def step_E(self):
    for t in range(len(self.data)):
      # f
      maxF = - np.infty
      for f in range(self.num_class):
        g = self.gaussian(self.data[t],self.param_mu[f], self.param_sigma[f])
        current = self.param_pi[f] * np.sum(np.log(np.where(self.h_m[t] == 0 , 1 - self.param_alpha[f], self.param_alpha[f]*g))) # use log to avoid underflow
        if current > maxF:
          maxF = current
          self.h_f[t] = f

      # m
      pm1 = self.param_alpha[self.h_f[t]] * self.gaussian(self.data[t], self.param_mu[self.h_f[t]], self.param_sigma[self.h_f[t]])
      pm0 = (1 - self.param_alpha[self.h_f[t]]) * self.gaussian(self.data[t], self.param_mu[self.h_b[t]], self.param_sigma[self.h_b[t]])
      self.h_m[t] = np.where(pm1 > pm0, 1, 0)

      # b
      maxB = -np.infty
      for b in range(self.num_class):
        g = self.gaussian(self.data[t],self.param_mu[b], self.param_sigma[b])
        current = self.param_pi[b] * np.sum(np.log(np.where(self.h_m[t] ==  0, g, 1.0)))
        if current > maxB:
          maxB = current
          self.h_b[t] = b

  def step_M(self): 
    # pi
    for j in range(self.num_class):
      self.param_pi[j] = 1.0/(2 * len(self.h_f)) * ((self.h_f == j).sum() + (self.h_b == j).sum())

    # alpha
    for j in range(self.num_class):
      scale = (self.h_f == j).sum()
      scale = scale if scale != 0 else 1.0
      self.param_alpha[j] = np.sum(np.where(self.h_f == j, 1.0, 0.0)[:,None] * self.h_m, axis=0) / scale

    # mu
    for j in range(self.num_class):
      scale = ((self.h_f == j) | (self.h_b == j)).sum()
      scale = scale if scale != 0 else 1.0
      self.param_mu[j] = np.sum( np.where((self.h_f == j) | (self.h_b == j), 1.0, 0.0)[:,None] * self.data, axis=0) / scale

    # sigma
    for j in range(self.num_class):
      scale = ((self.h_f == j) | (self.h_b == j)).sum()
      scale = scale if scale != 0 else 1.0
      self.param_sigma[j] = np.sum( np.where((self.h_f == j) | (self.h_b == j), 1.0, 0.0)[:,None] * pow(self.data - self.param_mu[j],2), axis=0) / scale
    
  def do_EM(self, n_iteration = 10):
    for i in range(n_iteration):
      print "iteration:", i
      self.step_E()
      self.step_M()

if __name__ == "__main__":
  sut = ICM()
  sut.do_EM(5)

  print "launch visualizer"
  v = visualizer.visualizer()
  v.visualize(sut.param_alpha, sut.param_mu, sut.param_sigma)
