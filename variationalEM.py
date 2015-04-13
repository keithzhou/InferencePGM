import numpy as np
import loadData
import model
import matplotlib.cm as cm
import matplotlib.mlab as mlab

class modelVariationalEM(model.model):
  def __init__(self,  num_class = 12, limit = None):
    model.model.__init__(self,num_class, limit)

    # hidden variables
    # for each training case, there are:
      # 1. distribution of f and b representing FG/BG class
      # 2. m representing the mask
    self.q_f = np.ones((len(self.data), self.num_class)) * 1.0 / self.num_class
    self.q_b = np.ones((len(self.data), self.num_class)) * 1.0 / self.num_class
    self.q_tm1 = np.ones((len(self.data), len(self.data[0])))*0.5;
    self.q_tm0 = np.ones((len(self.data), len(self.data[0])))*0.5;

  def freeEnergy(self):
    return 0.0

  def step_E(self):

    for t in range(len(self.data)):
      # f
      q_ff = np.ones((self.num_class))
      for f in range(self.num_class):
        toLog = np.power(self.param_alpha[f] * mlab.normpdf(self.data[t],self.param_mu[f],self.param_sigma[f]) , self.q_tm1[t]) * np.power((1 - self.param_alpha[f]), self.q_tm0[t]);
        assert not (toLog < 0).any()
        toLog[toLog < 1e-6] = 1e-6
        q_ff[f] = np.log(self.param_pi[f]) + np.sum(np.log(toLog)) # log domain to avoid underflow
        
      #normalize
      q_ff = q_ff - np.max(q_ff) # raise exponent 
      c = np.log(np.sum(np.exp(q_ff)))
      q_ff = q_ff - c
      q_ff = np.exp(q_ff) # back to probability domain
      self.q_f[t] = q_ff
        
     
      #qm1
      q_mm1 = np.ones((self.num_class, len(self.data[0])))
     
      for f in range(self.num_class):
         toLog = np.power(self.param_alpha[f] * mlab.normpdf(self.data[t], self.param_mu[f], self.param_sigma[f]) , q_ff[f] );
         assert not (toLog < 0).any()
         toLog[toLog < 1e-6] = 1e-6
         q_mm1[f] = np.log(toLog) # log domain to avoid underflow

      q_m1 = np.sum(q_mm1, axis = 0);

      #qm0
      q_mmf0 = np.ones((self.num_class, len(self.data[0])))
      q_mmb0 = np.ones((self.num_class, len(self.data[0])))
     
      for f in range(self.num_class):
         toLog =  np.power((1 - self.param_alpha[f]), q_ff[f]);
         assert not (toLog < 0).any()
         toLog[toLog < 1e-6] = 1e-6
         q_mmf0[f] = np.log(toLog) # log domain to avoid underflow
         
      for b in range(self.num_class):
         toLog =  np.power(mlab.normpdf(self.data[t], self.param_mu[b], self.param_sigma[b]) , self.q_b[t][b] );
         assert not (toLog < 0).any()
         toLog[toLog < 1e-6] = 1e-6
         q_mmb0[b] = np.log(toLog) # log domain to avoid underflow   
         
      q_m0 = np.sum(q_mmf0, axis = 0) + np.sum(q_mmb0, axis = 0);
     
      maxVal = np.max((np.max(q_m1), np.max(q_m0)));
      q_m1 = q_m1 - maxVal;
      q_m0 = q_m0 - maxVal;


      q_m1 = np.exp(q_m1);
      q_m0 = np.exp(q_m0);
     
     #normalize
      # q_m1 = q_m1 / (q_m0 + q_m1);
      # q_m0 = np.ones((1, len(self.data[0]))) - q_m1;
     
     
      self.q_tm1[t] =  q_m1 / (q_m0 + q_m1);
      self.q_tm0[t] =  q_m0 / (q_m0 + q_m1);     
     
      #b
      q_bb = np.ones((self.num_class))
      
      for b in range(self.num_class):
        toLog = np.power(mlab.normpdf(self.data[t],self.param_mu[b],self.param_sigma[b]) , self.q_tm0[t]) ;
        assert not (toLog < 0).any()
        toLog[toLog < 1e-6] = 1e-6
        q_bb[b] = np.log(self.param_pi[b]) + np.sum(np.log(toLog)) # log domain to avoid underflow
        
      #normalize
      q_bb = q_bb - np.max(q_bb) # raise exponent 
      c = np.log(np.sum(np.exp(q_bb)))
      q_bb = q_bb - c
      q_bb = np.exp(q_bb) # back to probability domain
      self.q_b[t] = q_bb
      
      
#       # m
#       q_bfm = np.ones((self.num_class, self.num_class, len(self.data[0])))
#       for b in range(self.num_class):
#         for f in range(self.num_class):
#           q_m1 = self.param_alpha[f] * mlab.normpdf(self.data[t], self.param_mu[f], self.param_sigma[f])
#           q_m0 = (1 - self.param_alpha[f]) * mlab.normpdf(self.data[t], self.param_mu[b], self.param_sigma[b])
#           q_m1[q_m1 < 1e-6] = 1e-6
#           q_m0[q_m0 < 1e-6] = 1e-6
#           q_bfm[b,f] = q_m1 / (q_m1 + q_m0)
#
#       q_bfm[q_bfm < 1e-6] = 1e-6
#
#       self.q_tfm1[t] = np.sum(q_bfm * q_bf[:,:,np.newaxis],axis=0)
#       self.q_tfm1[t][self.q_tfm1[t] < 1e-6] = 1e-6
#
# #      self.q_tbm1[t] = np.sum(q_bfm * q_bf[:,:,np.newaxis],axis=1)
# #      self.q_tbm1[t][self.q_tbm1[t] < 1e-6] = 1e-6
#
# #      self.q_tfm0[t] = np.sum((1 -q_bfm) * q_bf[:,:,np.newaxis],axis=0)
# #      self.q_tfm0[t][self.q_tfm0[t] < 1e-6] = 1e-6
#
#       self.q_tbm0[t] = np.sum((1 - q_bfm) * q_bf[:,:,np.newaxis],axis=1)
#       self.q_tbm0[t][self.q_tbm0[t] < 1e-6] = 1e-6

  def step_M(self):
    # pi
    self.param_pi = 1.0 / (2.0 * len(self.data)) * np.sum(self.q_f + self.q_b, axis=0)

    # alpha
    self.param_alpha = np.sum( self.q_tm1[: ,np.newaxis ,:] * self.q_f[:,:,np.newaxis], axis=0) / np.sum(self.q_f, axis = 0)[:,np.newaxis]

    self.param_alpha[self.param_alpha < 1e-10] = 1e-10
    self.param_alpha[self.param_alpha > 1-1e-10] = 1-1e-10

    # mu
    self.param_mu = np.sum((self.q_tm1[: ,np.newaxis ,:] * self.q_f[:,:,np.newaxis] + self.q_tm0[: ,np.newaxis ,:] * self.q_b[:,:,np.newaxis]) * self.data[:,np.newaxis,:], axis=0) / np.sum((self.q_tm1[: ,np.newaxis ,:] * self.q_f[:,:,np.newaxis] + self.q_tm0[: ,np.newaxis ,:] * self.q_b[:,:,np.newaxis]), axis=0)

    # sigma
    self.param_sigma = np.sqrt(    np.sum(    (    self.q_tm1[: ,np.newaxis ,:] * self.q_f[:,:,np.newaxis] +     self.q_tm0[: ,np.newaxis ,:] * self.q_b[:,:,np.newaxis] )    *     pow(    self.data[:,np.newaxis,:] - self.param_mu, 2    ), axis=0    )     / np.sum((self.q_tm1[: ,np.newaxis ,:] * self.q_f[:,:,np.newaxis] + self.q_tm0[: ,np.newaxis ,:] * self.q_b[:,:,np.newaxis]), axis=0))
    self.param_sigma[self.param_sigma < 1e-6] = 1e-6

    vf = np.sum(self.q_f,axis=0) / len(self.q_f)
    vb = np.sum(self.q_b,axis=0) / len(self.q_b)
    
if __name__ == "__main__":
  sut = modelVariationalEM(limit=None)
  sut.train()
