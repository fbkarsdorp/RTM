
import numpy as np
import pymc as pm

K = 2 # number of topics
V = 4 # number of words
D = 3 # number of documents

data = np.array([[0, 1, 2, 0], [1, 2, 1, 1], [1, 1, 1, 1]])

alpha = np.ones(K)
beta = np.ones(V)

theta = pm.Container([pm.Dirichlet("theta_%s" % i, theta=alpha) for i in range(D)])
phi = pm.Container([pm.Dirichlet("phi_%s" % k, theta=beta) for k in range(K)])

z = pm.Container([pm.Categorical('z_%i' % d, 
	                             p = theta[d], 
	                             size=len(data[d]),
	                             value=np.random.randint(K, size=len(data[d])))
                  for d in range(D)])

w = pm.Container([pm.Categorical("w_%i" % d,
	                             value=data[d], 
	                             observed=True, 
	                             p = pm.Lambda("phi_z_%s" % d, lambda phi=phi, z=z[d]: [phi[z[i]] for i in range(len(data[d]))]))
                  for d in range(D)])

model = pm.Model([z, w])
mcmc = pm.MCMC(model)
mcmc.sample(1000, burn=100)

