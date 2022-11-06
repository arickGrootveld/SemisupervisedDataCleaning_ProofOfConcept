### This script houses the simulation code for assessing the performance of various forms of data cleaning, including clustering and svcc classifier

## Importing supporting functions
import numpy as np
from sklearn.cluster import KMeans
from generateSamples import generateSampleset
from utilities import binaryModelLabelChecking


## Simulation parameters

######################################################################
# Sampling process parameters
mixProb = 0.5
rngSeed = 10

# Number of samples from each of the distributions
numSamplesMix = 100
numSamplesF1 = 100

# Number of features in the dataset
numDims = 10

# Parameters of the distributions (diagonal variance value, means)
var1 = 1
var2 = 1

mu1 = 1
mu2 = 0

# Clustering parameters
numClusters = 2
clustRandState = 0

######################################################################

# Generating diagonal covariance matrices
covMat1 = var1 * np.eye(numDims)
covMat2 = var2 * np.eye(numDims)

# Generating constant mean vectors
muVec1 = mu1 * np.ones((numDims,))
muVec2 = mu2 * np.ones((numDims,))


## Generating the data from the mixture and F1 distribution
[F1Samples, F1Labels, mixSamples, mixLabels] = generateSampleset(numSamplesMix=numSamplesMix, numSamplesF1=numSamplesF1, numDims=numDims, 
                                                                    covarF1=covMat1, covarF2=covMat2, muF1=muVec1, 
                                                                    muF2=muVec2, seed=rngSeed, mixtureProb=mixProb)


## Processing the samples to feed into the clustering algorithm

# Concatentating mixture and F1 distribution samples and labels
clusterTrainSamples = np.concatenate((F1Samples, mixSamples))
clusterTrainLabels = np.concatenate((F1Labels, mixLabels)).astype(np.int)

# Setting up clustering
kmeans = KMeans(n_clusters=numClusters, random_state=clustRandState).fit(clusterTrainSamples)

# Correcting the labelling in case the clustering alg got the labels backwards
correctLabelling = binaryModelLabelChecking(kmeans.labels_, clusterTrainLabels)

# Analyzing the results
numCorrect = np.count_nonzero(correctLabelling == clusterTrainLabels)
clustAcc = numCorrect / clusterTrainLabels.shape[0]


print('Finished')

