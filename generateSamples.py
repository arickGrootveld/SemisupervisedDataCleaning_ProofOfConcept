import numpy as np

### Function to generate samples from a Gaussian mixture distribution
def generateMixtureSamples(numSamples, dim, muVec1, muVec2, covMat1, covMat2, mixtureProb=0.5, seed=-1):
    if(seed > 0):
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Generate a bunch of bernoulli random variables to tell us which distribution to sample from
    # a label of 1 means it came from dist 1, and a label of 0 means it came from dist 2
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


def generateSampleset(numSamplesMix, numSamplesF1, numDims, **kwargs):
    # function generateSampleset(numSamplesMix, numSamplesF1, numDims, **kwargs)
    # 
    # numSamplesMix: number of samples to generate from the mixture distribution
    # numSamplesF1: number of samples to generate from F1's distribution
    # numDims: The number of feature dimensions in the generated samples
    # Acceptable **kwargs arguments:
    #   covarF1 [eye(n)]: Covariance matrix of distribution F1
    #   covarF2 [eye(n)]: Covariance matrix of distribution F2
    #   muF1 [zeros(1,n)]: Mean vector of dist. F1
    #   muF2 [zeros(1,n)]: Mean vector of dist. F2 
    #   seed [-1]: The seed of the random number generator
    #   mixtureProb [0.5]: The probability of a sample coming from
    #                      F1 in the mixture distribution

    # Defaulting the kwargs arguments
    defaultArgs = {'covarF2': np.eye(numDims), 'covarF1': np.eye(numDims),
                    'muF1': np.zeros((1,numDims)), 'muF2': np.zeros((1, numDims)),
                    'seed': -1, 'mixtureProb': 0.5}
    
    # Taking default arguments if kwargs are not provided
    functionArgs = {**defaultArgs, **kwargs}

    # Setting up stuff based on the parameters passed
    if(functionArgs['seed'] > 0):
        # If the seed is a valid seed (1 or greater), then we use it for the rng
        rng = np.random.default_rng(functionArgs['seed'])
    else:
        # Otherwise we just use a random seed
        rng = np.random.default_rng()
    

    # Generating the samples and labels from F1's distribution
    F1Samples = rng.multivariate_normal(functionArgs['muF1'], functionArgs['covarF1'], numSamplesF1)
    F1Labels = np.ones((numSamplesF1,))

    # Generating the mixture samples and labels
    [mixtureLabels, mixtureSamples] = generateMixtureSamples(numSamples=numSamplesMix, dim=numDims, muVec1=functionArgs['muF1'], 
                                            muVec2=functionArgs['muF2'], covMat1=functionArgs['covarF1'], covMat2=functionArgs['covarF2'], 
                                            mixtureProb=functionArgs['mixtureProb'], seed=functionArgs['seed'])
    return(F1Samples, F1Labels, mixtureSamples, mixtureLabels)

if(__name__ == '__main__'):
    ## Code for testing purposes
    # Parameters of the simulation
    numSamples = 10
    numDims = 2

    # Parameters of the distributions
    var1 = 1
    var2 = 1

    mu1 = 100
    mu2 = 0

    # Parameters of the simulation
    mixProb = 0.5
    rngSeed=10
    
    # Generating the diagonal covariance matrices
    covMat1 = var1 * np.eye(numDims)
    covMat2 = var2 * np.eye(numDims)

    # Generating the mean vectors 
    muVec1 = mu1 * np.ones((numDims,))
    muVec2 = mu2 * np.ones((numDims,))

    # [labels, samples] = generateMixtureSamples(numSamples, numDims, muVec1, muVec2, covMat1, covMat2, mixtureProb=mixProb, seed=rngSeed)
    # print(samples)
    # print(labels)

    # print()

    [F1Samples, F1Labels, mixSamples, mixLabels] = generateSampleset(numSamples, numSamples, numDims=numDims, covarF1=covMat1, covarF2=covMat2, 
                                                                      muF1=muVec1, muF2=muVec2, seed=rngSeed, mixtureProb=mixProb)
    print(F1Samples)
    print(F1Labels)
    print()
    print(mixSamples)
    print(mixLabels)




