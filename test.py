import numpy as np
import matplotlib.pyplot as plt

from robot import Robot
from plotmap import plotMap, plotEstimate, plotMeasurement, plotError
from ekf import predict, update

## https://www.cs.utexas.edu/~pstone/Courses/393Rfall11/resources/RC09-Quinlan.pdf

# In[Generate static landmarks]
# Seed as randomisations
np.random.seed(2)
n = 150 # number of static landmarks
mapsize = 60
landmark_xy = mapsize*(np.random.rand(n,2)-0.5)
landmark_id = np.transpose([np.linspace(0,n-1,n,dtype='uint16')])
ls = np.append(landmark_xy,landmark_id,axis=1)

# In[Generate dynamic landmarks]

k = 0 # number of dynamic landmarks
vm = 5 # velocity multiplier
landmark_xy = mapsize*(np.random.rand(k,2)-0.5)
landmark_v = np.random.rand(k,2)-0.5
landmark_id = np.transpose([np.linspace(n,n+k-1,k,dtype='uint16')])
ld = np.append(landmark_xy,landmark_id,axis=1)
ld = np.append(ld,landmark_v,axis=1)


# In[Define and initialize robot parameters]

fov = 80

Rt = 5*np.array([[0.1,0,0],
               [0,0.01,0],
               [0,0,0.01]])
Qt = np.array([[0.01,0],
               [0,0.01]])

x_init = [0,0,0.5*np.pi]

r1 = Robot(x_init, fov, Rt, Qt)

# In[Generate inputs and measurements]

steps = 30
stepsize = 1.5
left_turn = (np.pi) /2
right_turn = -(np.pi)/2

x_true = [x_init]
obs = []

# generate input sequence
u = np.zeros((steps,3))
u[:,0] = stepsize
u[5,1] = left_turn
u[15,1] = right_turn
u[17,1] = right_turn
u[20,1] = left_turn
u[22,1] = left_turn
u[25,1] = right_turn
u[29,1] = left_turn

# Generate random trajectory instead
'''u = np.append(stepsize*np.ones((steps,1),dtype='uint8'),
              curviness*np.random.randn(steps,2),
              axis=1)'''

# generate dynamic landmark trajectories
ldt = ld
for j in range(1,steps):
    # update dynamic landmarks
    F = np.array([[1,0,0,vm,0],
                  [0,1,0,0,vm],
                  [0,0,1,0,0],
                  [0,0,0,1,0],
                  [0,0,0,0,1]])
    for i in range(len(ld)):
        ld[i,:] = F.dot(ld[i,:].T).T
    ldt = np.dstack((ldt,ld))

# generate robot states and observations
for movement, t in zip(u,range(steps)):
    landmarks = np.append(ls,ldt[:,:3,t],axis=0)
    
    # process robot movement
    x_true.append(r1.move(movement))
    obs.append(r1.sense(landmarks))

plotMap(ls,ldt,x_true,r1,mapsize)

# In[Estimation]

# Initialize state matrices
inf = 1e6

mu = np.append(np.array([x_init]).T,np.zeros((2*(n+k),1)),axis=0)
mu_new = mu

cov = inf*np.eye(2*(n+k)+3)
cov[:3,:3] = np.zeros((3,3))

c_prob = 0.5*np.ones((n+k,1))

plotEstimate(mu, cov, r1, mapsize)

for movement, measurement in zip(u, obs):
    mu_new, cov = predict(mu_new, cov, movement, Rt)
    mu = np.append(mu,mu_new,axis=1)
    plotEstimate(mu, cov, r1, mapsize)
    
    print('Measurements: {0:d}'.format(len(measurement)))
    mu_new, cov, c_prob_new = update(mu_new, cov, measurement, c_prob[:,-1].reshape(n+k,1), Qt)
    mu = np.append(mu,mu_new,axis=1)
    c_prob = np.append(c_prob, c_prob_new, axis=1)
    plotEstimate(mu, cov, r1, mapsize)
    plotMeasurement(mu_new, cov, measurement, n)
    
    plotError(mu,x_true[:len(mu[:,0::2])][:])
    print('----------')

plotEstimate(mu, cov, r1, mapsize)
plt.show()