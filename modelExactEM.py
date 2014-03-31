import numpy as np
import loadData
import model
import matplotlib.cm as cm
import matplotlib.mlab as mlab

class modelExactEM(model.model):
  def __init__(self,  num_class = 12, limit = None):
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
      # b,f
      q_bf = np.ones((self.num_class, self.num_class))
      for b in range(self.num_class):
        for f in range(self.num_class):
          toLog = self.param_alpha[f]*mlab.normpdf(self.data[t],self.param_mu[f],self.param_sigma[f]) + (1 - self.param_alpha[f]) * mlab.normpdf(self.data[t], self.param_mu[b], self.param_sigma[b])
          toLog[toLog < 1e-6] = 1e-6
          q_bf[b][f] = np.log(self.param_pi[b]) +  np.log(self.param_pi[f]) + np.sum(np.log(toLog)) # log domain to avoid underflow
      q_bf = q_bf - np.max(q_bf) # raise exponent 
      c = np.log(np.sum(np.exp(q_bf)))
      q_bf = q_bf - c
      q_bf[q_bf < np.log(1e-6)] = np.log(1e-6)
      q_bf = np.exp(q_bf) # back to probability domain
      self.q_f[t] = np.sum(q_bf, axis=0)
      self.q_b[t] = np.sum(q_bf, axis=1)

      # m
      q_bfm = np.ones((self.num_class, self.num_class, len(self.data[0])))
      for b in range(self.num_class):
        for f in range(self.num_class):
          q_m1 = self.param_alpha[f] * mlab.normpdf(self.data[t], self.param_mu[f], self.param_sigma[f])
          q_m0 = (1 - self.param_alpha[f]) * mlab.normpdf(self.data[t], self.param_mu[b], self.param_sigma[b])
          q_m1[q_m1 < 1e-6] = 1e-6
          q_m0[q_m0 < 1e-6] = 1e-6
          q_bfm[b,f] = q_m1 / (q_m1 + q_m0)

      q_bfm[q_bfm < 1e-6] = 1e-6

      self.q_tfm[t] = np.sum(q_bfm * q_bf[:,:,None],axis=0)
      self.q_tfm[t][self.q_tfm[t] < 1e-6] = 1e-6

      self.q_tbm[t] = np.sum(q_bfm * q_bf[:,:,None],axis=1)
      self.q_tbm[t][self.q_tbm[t] < 1e-6] = 1e-6

  def step_M(self):
    # pi
    self.param_pi = 1.0 / (2.0 * len(self.data)) * np.sum(self.q_f + self.q_b, axis=0)

    # alpha
    self.param_alpha = np.sum( self.q_tfm, axis=0) / np.sum(self.q_f, axis = 0)[:,np.newaxis]

    # mu
    self.param_mu = np.sum((self.q_tfm + 1 - self.q_tbm) * self.data[:,np.newaxis,:], axis=0) / np.sum(self.q_tfm + 1 - self.q_tbm, axis=0)

    # sigma
    self.param_sigma = np.sqrt(np.sum((self.q_tfm + 1 - self.q_tbm)* pow(self.data[:,np.newaxis,:] - self.param_mu, 2), axis=0) / np.sum((self.q_tfm + 1-self.q_tbm), axis=0))
    self.param_sigma[self.param_sigma < 1e-6] = 1e-6

    vf = np.sum(self.q_f,axis=0) / len(self.q_f)
    vb = np.sum(self.q_b,axis=0) / len(self.q_b)
    print "vf:", vf
    print "vb:", vb
    print "pi:", self.param_pi
    
if __name__ == "__main__":
  sut = modelExactEM(limit=None)
  sut.train()
