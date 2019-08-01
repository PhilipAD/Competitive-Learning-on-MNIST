import numpy as np
import numpy.matlib
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

#Bounded gaussian distribution
def get_trunc_norm(mean=0.5, sd=0.1, lower=0, upper=1):
    return truncnorm(
        (lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)

train = np.genfromtxt ('letters.csv', delimiter=",")
trainlabels = np.genfromtxt ('letterslabels.csv', delimiter=",")

normTrain = np.sqrt(np.diag(train.T.dot(train)))

#Normalises data
train = train / normTrain
data = train.T
#W = W / normW
# Center data around the mean

[n,m]  = np.shape(train)                    # number of pixels and number of training data
eta    = 0.05                     # learning rate
winit  = 1                                  # parameter controlling magnitude of initial conditions
alpha = 0.999
targetLearning = 0.04 # Target decay learning rate

tmax   = 40000
digits = 10

result = np.zeros((0))

#Sets the decay value relative to the learning rate, target learing and the max number of iterations
decayValue = (eta-targetLearning)/tmax


#random weights with Gaussian distribution
X =  get_trunc_norm()
#for R in range(n):
#    result = np.concatenate((result, X.rvs(digits)))
#W = winit * result.reshape((digits,n))

W = winit * np.random.rand(digits,n)# Weight matrix (rows = output neurons, cols = input neurons)
normW = np.sqrt(np.diag(W.dot(W.T)))
normW = normW.reshape(digits,-1)            # reshape normW into a numpy 2d array

W = W / np.matlib.repmat(normW.T,n,1).T    # normalise using repmat
#W = W / normW                               # normalise using numpy broadcasting -  http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html

counter = np.zeros((1,digits))              # counter for the winner neurons
wCount = np.ones((1,tmax+1)) * 0.25         # running avg of the weight change over time

#for t in range(1,10):
epoch_count = np.zeros((1,digits))
num_zeros = 0
for t in range(1,tmax):
        i = math.ceil(m * np.random.rand())-1   # get a randomly generated index in the input range
        x = train[:,i]                          # pick a training instance using the random index

        h = W.dot(x)/digits                     # get output firing
        h = h.reshape(h.shape[0],-1)            # reshape h into a numpy 2d array

        output = np.max(h)                      # get the max in the output firing vector

        #Noise on output neurons
        #noise = np.random.normal(0,1, output.shape)/100
        #outputn = output + noise

        k = np.argmax(h)                        # get the index of the firing neuron

        counter[0,k] += 1                       # increment counter for winner neuron

        dw = eta * (x.T - W[k,:])               # calculate the change in weights for the k-th output neuron
                                                # get closer to the input (x - W)

        wCount[0,t] = wCount[0,t-1] * (alpha + dw.dot(dw.T)*(1-alpha)) # % weight change over time (running avg)

        #Leaky learning
        #for y in range(digits):
        #        if (y != k):
        #            dwl = 0.000001 * (x.T - W[y,:])               # calculate the change in weights for the k-th output neuron
        #            W[y,:] = W[y,:] + dwl                           # get closer to the input (x - W)



        W[k,:] = W[k,:] + dw

        #neighbour update
        #if k!=digits-1:
        #    W[(k+1),:] = W[k+1,:] + (dw * 0.2)
        #if k!=0:
        #    W[(k-1),:] = W[k-1,:] + (dw * 0.2)

        #neuron decay
        #eta -= decayValue


epoch_count += counter
num_zeros += (epoch_count == 0).sum()
print(num_zeros)


# Plot a prototype
plt.figure(figsize=(20,10))
for x in range(10):
    plt.subplot(4, 15, x+1)
    plt.imshow(W[x,:].reshape((88,88), order = 'F'),interpolation = 'nearest', cmap='inferno')


#correlationelation matrix
correlation = np.zeros((digits,digits))
thresh = 0.025
for f in range(m):
    # Get neuron firings
    x = train[:,(f)]                          # pick a training instance using the random index
    h = W.dot(x)/digits

    outputS = np.zeros_like(h)
    outputS[h <= thresh] = -1

    outputS[h > thresh] = 1
    correlation += np.outer(outputS , outputS )



correlation /= train.shape[0]
print(correlation)
