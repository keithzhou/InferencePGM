import numpy as np
import loadData
import model
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import scipy.misc
import scipy

class modelExactEM(model.model):
  def __init__(self,  num_class = 14, limit = None):
    model.model.__init__(self,num_class, limit)
    
    # hidden variables
    # for each training case, there are:
      # 1. distribution of f and b representing FG/BG class
      # 2. m representing the mask
    self.q_f = np.ones((len(self.data), self.num_class)) * 1.0 / self.num_class
    self.q_b = np.ones((len(self.data), self.num_class)) * 1.0 / self.num_class
    self.q_tbm1 = np.ones((len(self.data), self.num_class, len(self.data[0])))
    self.q_tfm1 = np.ones((len(self.data), self.num_class, len(self.data[0])))
    self.q_tbm0 = np.ones((len(self.data), self.num_class, len(self.data[0])))
    self.q_tfm0 = np.ones((len(self.data), self.num_class, len(self.data[0])))
  
  # def freeEnergy(self):
  #   return 0.0
  
  def step_E(self):
    # print "free energy:%.4f\n", self.freeEnergy()
    if (self.param_alpha > 1).any() :
        print "alpha > 1"
      
    if (self.param_alpha <0).any():
        print "alpha < 0 "
      
    if (self.param_pi > 1).any() :
        print "param_pi > 1"
        
    if (self.param_pi <0).any() :
        print "param_pi <0"
        
    self.param_alpha[self.param_alpha < 1e-100] = 1e-100
                    
    for t in range(len(self.data)):
      # b,f
      q_bf = np.ones((self.num_class, self.num_class))
      
      for b in range(self.num_class):
        normB = mlab.normpdf( self.data[t], self.param_mu[b], self.param_sigma[b]) ;
        normB[normB < 1e-100] = 1e-100
        bPart= np.log( normB)
        
        for f in range(self.num_class):
            # toLog = self.param_alpha[f] *            mlab.normpdf( self.data[t], self.param_mu[f], self.param_sigma[f] ) +            (1 - self.param_alpha[f]) * mlab.normpdf( self.data[t], self.param_mu[b], self.param_sigma[b] );
            innerParts = np.ones(( len(self.data[0]), 2));
            normF = mlab.normpdf( self.data[t], self.param_mu[f], self.param_sigma[f] );
            normF[normF < 1e-100] = 1e-100

            innerParts[:,0] = np.log(self.param_alpha[f]) + np.log( normF );
            innerParts[:,1] = np.log((1 - self.param_alpha[f])) + bPart;
            logged = scipy.misc.logsumexp(innerParts,axis=1);
            # assert not (toLog < 0).any()
            # toLog[toLog < 1e-100] = 1e-100
 #            q_bf[b][f] = np.log(self.param_pi[b]) +  np.log(self.param_pi[f]) + np.sum(np.log(toLog)) # log domain to avoid underflow
            q_bf[b][f] = np.log(self.param_pi[b]) +  np.log(self.param_pi[f]) + np.sum(logged) # log domain to avoid underflow
      
      
      
      # q_bf = q_bf - np.max(q_bf) # raise exponent
      # c = np.log(np.sum(np.exp(q_bf)))
      # q_bf = q_bf - c
      # q_bf = np.exp(q_bf) # back to probability domain
      # self.q_f[t] = np.sum(q_bf, axis=0)
      # self.q_b[t] = np.sum(q_bf, axis=1)

      c2 = scipy.misc.logsumexp(q_bf);
      q_bf = q_bf - c2; #normalized now 
      
      self.q_f[t] = np.exp(scipy.misc.logsumexp(q_bf, axis=0))
      self.q_b[t] = np.exp(scipy.misc.logsumexp(q_bf, axis=1))
      
      # m
      # q_bfm = np.ones((self.num_class, self.num_class, len(self.data[0])))
      q_bfm0 = np.ones((self.num_class, self.num_class, len(self.data[0])))
      q_bfm1 = np.ones((self.num_class, self.num_class, len(self.data[0])))

      for b in range(self.num_class):
        for f in range(self.num_class):
            normB = mlab.normpdf( self.data[t], self.param_mu[b], self.param_sigma[b]) ;
            normB[normB < 1e-100] = 1e-100
            normF = mlab.normpdf( self.data[t], self.param_mu[f], self.param_sigma[f] );
            normF[normF < 1e-100] = 1e-100
            q_m1 = self.param_alpha[f] * normB
            q_m0 = (1 - self.param_alpha[f]) * normF
            q_m1[q_m1 < 1e-100] = 1e-100
            q_m0[q_m0 < 1e-100] = 1e-100
            # q_bfm[b,f] = q_m1 / (q_m1 + q_m0)
          # q_bfm0[b,f] = q_m0
       #    q_bfm1[b,f] = q_m1
          
          # q_m1 = np.log(self.param_alpha[f]) + np.log(mlab.normpdf(self.data[t], self.param_mu[f], self.param_sigma[f]))
 #          q_m0 = np.log((1 - self.param_alpha[f])) + np.log(mlab.normpdf(self.data[t], self.param_mu[b], self.param_sigma[b]))
            q_bfm0[b,f] = q_m0 / (q_m1 + q_m0);
            q_bfm1[b,f] = q_m1 / (q_m1 + q_m0);

      
      # q_bfm[q_bfm < 1e-6] = 1e-6
      #rright now in log domain
      # q_bf3 = q_bf.reshape(self.num_class, self.num_class,1).repeat(len(self.data[0]),2);
      # self.q_tfm1[t] = np.exp(scipy.misc.logsumexp(q_bfm1 + q_bf3,axis=0))
      # self.q_tfm1[t][self.q_tfm1[t] < 1e-6] = 1e-6
      self.q_tfm1[t] = np.sum(q_bfm1 * q_bf[:,:,np.newaxis],axis=0)
      self.q_tfm1[t][self.q_tfm1[t] < 1e-100] = 1e-100

      #self.q_tbm1[t] = np.sum(q_bfm * q_bf[:,:,np.newaxis],axis=1)
      #self.q_tbm1[t][self.q_tbm1[t] < 1e-6] = 1e-6
      
      #self.q_tfm0[t] = np.sum((1 -q_bfm) * q_bf[:,:,np.newaxis],axis=0)
      #self.q_tfm0[t][self.q_tfm0[t] < 1e-6] = 1e-6
      
      self.q_tbm0[t] = np.sum(q_bfm0 * q_bf[:,:,np.newaxis],axis=1)
      self.q_tbm0[t][self.q_tbm0[t] < 1e-100] = 1e-100
      # self.q_tbm0[t] = np.exp(scipy.misc.logsumexp(q_bfm1 + q_bf3,axis=1))
      # self.q_tbm0[t][self.q_tbm0[t] < 1e-6] = 1e-6
  
  def step_M(self):
    # pi
    self.param_pi = 1.0 / (2.0 * len(self.data)) * np.sum(self.q_f + self.q_b, axis=0)
    
    # alpha
    self.param_alpha = np.sum( self.q_tfm1, axis=0) / np.sum(self.q_f, axis = 0)[:,np.newaxis]
    
    # mu
    self.param_mu = np.sum((self.q_tfm1 + self.q_tbm0) * self.data[:,np.newaxis,:], axis=0) / np.sum(self.q_tfm1 + self.q_tbm0, axis=0)
    
    # sigma
    self.param_sigma = np.sqrt(np.sum((self.q_tfm1 + self.q_tbm0)* pow(self.data[:,np.newaxis,:] - self.param_mu, 2), axis=0) / np.sum((self.q_tfm1 + self.q_tbm0), axis=0))
    self.param_sigma[self.param_sigma < 1e-100] = 1e-100
    
    vf = np.sum(self.q_f,axis=0) / len(self.q_f)
    vb = np.sum(self.q_b,axis=0) / len(self.q_b)

if __name__ == "__main__":
  sut = modelExactEM(limit=None)
  sut.train()
