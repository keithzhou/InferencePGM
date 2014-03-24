import numpy as np
import loadData
import model

class modelExactEM(model.model):
  def __init__(self,  num_class = 14, limit = None):
    model.model.__init__(self,num_class, limit)

    # hidden variables
    # for each training case, there are:
      # 1. distribution of f and b representing FG/BG class
      # 2. m representing the mask
    self.q_f = np.ones((len(self.data), self.num_class)) * 1.0 / self.num_class
    self.q_b = np.ones((len(self.data), self.num_class)) * 1.0 / self.num_class
    self.q_tbm = np.ones((len(self.data), self.num_class, len(self.data[0])))
    self.q_tfm = np.ones((len(self.data), self.num_class, len(self.data[0])))

  def step_E(self):
    for t in range(len(self.data)):
      print "\t%d/%d" % (t+1, len(self.data))
      # b,f
      q_bf = np.ones((self.num_class, self.num_class))
      for b in range(self.num_class):
        for f in range(self.num_class):
          q_bf[b,f] = np.log(self.param_pi[b]) +  np.log(self.param_pi[f]) + np.sum(np.log(self.param_alpha[f]*self.gaussian(self.data[t],self.param_mu[f],self.param_sigma[f]) + (1 - self.param_alpha[f]) * self.gaussian(self.data[t], self.param_mu[b], self.param_sigma[b]))) # log domain to avoid underflow
      q_bf = q_bf - np.max(q_bf) # raise exponent 
      q_bf = np.exp(q_bf) # back to probability domain
      q_bf = q_bf / np.sum(q_bf) # normalization
      self.q_f[t] = np.sum(q_bf, axis=0)
      self.q_b[t] = np.sum(q_bf, axis=1)

      # m
      q_bfm = np.ones((self.num_class, self.num_class, len(self.data[0])))
      for b in range(self.num_class):
        for f in range(self.num_class):
          q_m1 = self.param_alpha[f] * self.gaussian(self.data[t], self.param_mu[f], self.param_sigma[f])
          q_m0 = (1 - self.param_alpha[f]) * self.gaussian(self.data[t], self.param_mu[b], self.param_sigma[b])
          q_bfm[b,f] = q_m1 / (q_m1 + q_m0)
      self.q_tbm[t] = np.sum(q_bfm * q_bf[:,:,None],axis=1)
      self.q_tfm[t] = np.sum(q_bfm * q_bf[:,:,None],axis=0)

  def step_M(self):
    # pi
    for j in range(self.num_class):
      self.param_pi[j] = 1.0 / (2*len(self.data)) * (self.q_f[:,j].sum() + self.q_b[:,j].sum())

    # alpha
    for j in range(self.num_class):
      self.param_alpha[j] = np.sum(self.q_tfm[:,j,:], axis=0) / np.sum(self.q_f[:,j])

    # mu
    for j in range(self.num_class):
      self.param_mu[j] = np.sum((self.q_tfm[:,j,:] + 1 - self.q_tbm[:,j,:])*self.data, axis=0) / np.sum(self.q_tfm[:,j,:] + 1 - self.q_tbm[:,j,:], axis=0)

    # sigma
    for j in range(self.num_class):
      self.param_sigma[j] = np.sum( (self.q_tfm[:,j,:] + 1 - self.q_tbm[:,j,:]) * pow(self.data - self.param_mu[j], 2), axis=0) / np.sum(self.q_tfm[:,j,:] + 1 - self.q_tbm[:,j,:], axis=0)
    
if __name__ == "__main__":
  sut = modelExactEM(limit=5)
  sut.do_EM(1)
