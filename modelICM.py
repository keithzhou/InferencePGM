import numpy as np
import loadData
import visualizer
import model
import matplotlib.mlab as mlab

class modelICM(model.model):
  def __init__(self,  num_class = 14):
    model.model.__init__(self,num_class)

    # hidden variables
    # for each training case, there are:
      # 1. f and b representing FG/BG class
      # 2. m representing the mask
    self.h_f = np.random.randint(0,self.num_class,self.data.shape[0])
    self.h_m = np.random.random_integers(0,1,self.data.shape) 
    self.h_b = np.random.randint(0,self.num_class,self.data.shape[0])

  def log_gaussian(self,data,mu,sigma):
    return -1.0 * (data - mu) ** 2 / (2 * sigma**2) - np.log(np.sqrt(2*np.pi)*sigma)

  def free_energy(self):
    result = 0.0
    for t in range(self.data.shape[0]):
      term1 = np.log(self.param_pi[self.h_f[t]]) + np.log(self.param_pi[self.h_b[t]])
      term2 = np.sum(self.h_m[t] * np.log(self.param_alpha[self.h_f[t]]) + (1.0 - self.h_m[t]) * np.log(1.0 - self.param_alpha[self.h_f[t]]),axis=-1)
      term3 = np.sum(self.h_m[t] * self.log_gaussian(self.data[t], self.param_mu[self.h_f[t]], self.param_sigma[self.h_b[t]]) + (1.0 - self.h_m[t]) * self.log_gaussian(self.data[t], self.param_mu[self.h_b[t]], self.param_sigma[self.h_b[t]]),axis=-1)
      result += (term1 + term2 + term3)
    return - 1.0 * result/350.0 * 300.0

  def step_E(self):
    print "free energy:", self.free_energy();
    #print "f:", self.h_f.shape, "b:", self.h_b.shape, "m:", self.h_m.shape, "pi:", self.param_pi.shape, "alpha:", self.param_alpha.shape, "mu:", self.param_mu.shape, "sigma:", self.param_alpha.shape

    for t in range(len(self.data)):
      # f
      maxF = - np.infty
      for f in range(self.num_class):
        current = np.log(self.param_pi[f]) + np.sum(np.log(self.param_alpha[f]) * self.h_m[t] + (1 - self.h_m[t]) * np.log(1.0-self.param_alpha[f]) + self.h_m[t] * self.log_gaussian(self.data[t],self.param_mu[f],self.param_sigma[f]),axis=-1)
        assert not np.isnan(current) 
        if current > maxF:
          maxF = current
          self.h_f[t] = f

      assert not np.isnan(self.h_f).any()

      # m
      pm1 = np.log(self.param_alpha[self.h_f[t]])  + self.log_gaussian(self.data[t], self.param_mu[self.h_f[t]], self.param_sigma[self.h_f[t]])
      pm0 = np.log(1 - self.param_alpha[self.h_f[t]]) + self.log_gaussian(self.data[t], self.param_mu[self.h_b[t]], self.param_sigma[self.h_b[t]])
      self.h_m[t] = np.where(pm1 > pm0, 1, 0)

      assert not np.isnan(self.h_m).any()

      # b
      maxB = -np.infty
      for b in range(self.num_class):
        current = np.log(self.param_pi[b]) + np.sum((1-self.h_m[t]) * self.log_gaussian(self.data[t],self.param_mu[b],self.param_sigma[b]),axis=-1)
        assert not np.isnan(current) 
        if current > maxB:
          maxB = current
          self.h_b[t] = b
      assert not np.isnan(self.h_b).any()

  def step_M(self): 
    # pi
    for j in range(self.num_class):
      self.param_pi[j] = 1.0/(2 * len(self.h_f)) * ((self.h_f == j).sum() + (self.h_b == j).sum())
    assert not np.isnan(self.param_pi).any()

    # alpha
    for j in range(self.num_class):
      #print "class:", j, "count h_f:", (self.h_f == j).sum() , "count h_b:", (self.h_b==j).sum(), "prob:", self.param_pi[j],self.param_alpha[j][:10]
      if (self.h_f==j).any():
        self.param_alpha[j] = np.mean(self.h_m[self.h_f==j],axis=0)
    assert not np.isnan(self.param_alpha).any()

    # mu
    for j in range(self.num_class):
      self.param_mu[j] = np.mean( self.data[(self.h_f == j) | (self.h_b == j)], axis=0) 
    assert not np.isnan(self.param_mu).any()

    # sigma
    for j in range(self.num_class):
      d = self.data[(self.h_f == j) | (self.h_b == j)]
      self.param_sigma[j] = np.sqrt(np.mean((d - self.param_mu[j]) ** 2,axis=0))
    assert not np.isnan(self.param_sigma).any()

#    self.param_pi[self.param_pi < 1e-6] = 1e-6
    self.param_alpha[self.param_alpha < 1e-10] = 1e-10
    self.param_alpha[self.param_alpha > 1.0-1e-10] = 1.0-1e-10
#    self.param_mu[self.param_mu < 1e-6] = 1e-6
    self.param_sigma[self.param_sigma < 1e-10] = 1e-10
    #print self.param_pi
    
if __name__ == "__main__":
  sut = modelICM()
  sut.train()
