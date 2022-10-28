import numpy as np

### Function to generate samples from a Gaussian mixture distribution
def generateMixtureSamples(numSamples, dim, muVec1, muVec2, covMat1, covMat2, mixtureProb=0.5, seed=-1):
    if(seed > 0):
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Generate a bunch of bernoulli random variables to tell us which distribution to sample from
    sampleDecisions = rng.binomial(1, mixtureProb, numSamples)

    # Determining the number of samples from each distribution to make life easier later
    numDist1Samples = np.count_nonzero(sampleDecisions)
    numDist2Samples = numSamples - numDist1Samples

    # Generating samples from Distribution 1
    dist1Samples = rng.multivariate_normal(muVec1, covMat1, numDist1Samples)

    # Generating samples from Distribution 2
    dist2Samples = rng.multivariate_normal(muVec2, covMat2, numDist2Samples)

    # Creating a holder variable for the final samples
    mixtureSamples = np.zeros((numSamples, dim))

    # Putting the generated samples in the output matrix
    mixtureSamples[np.where(sampleDecisions)[0],:] = dist1Samples
    mixtureSamples[np.where(sampleDecisions == 0)[0], :] = dist2Samples

    return((sampleDecisions, mixtureSamples))


if(__name__ == '__main__'):
    ## Code for testing purposes
    # Parameters of the simulation
    numSamples = 10
    numDims = 10

    # Parameters of the distributions
    var1 = 1
    var2 = 1

    mu1 = 100
    mu2 = 0
    
    # Generating the diagonal covariance matrices
    covMat1 = var1 * np.eye(numDims)
    covMat2 = var2 * np.eye(numDims)

    # Generating the mean vectors 
    muVec1 = mu1 * np.ones((numDims,))
    muVec2 = mu2 * np.ones((numDims,))

    [labels, samples] = generateMixtureSamples(numSamples, numDims, muVec1, muVec2, covMat1, covMat2, seed=2)
    print(samples)
    print(labels)
    