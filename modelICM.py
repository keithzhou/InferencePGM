import numpy as np
import loadData
import visualizer
import model

class modelICM(model.model):
  def __init__(self,  num_class = 14):
    model.model.__init__(self,num_class)

    # hidden variables
    # for each training case, there are:
      # 1. f and b representing FG/BG class
      # 2. m representing the mask
    self.h_f = np.random.randint(0,self.num_class,len(self.data))
    self.h_m = np.random.random_integers(0,1,(len(self.data), len(self.data[0])))
    self.h_b = np.random.randint(0,self.num_class,len(self.data))

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
    
if __name__ == "__main__":
  sut = modelICM()
  sut.do_EM(10)
