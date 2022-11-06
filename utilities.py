### Utilities functions to make life easier

import numpy as np


# Function to find out if the 0 and 1 labelling scheme of a model matches the 0 and 1 labelling scheme of the true labels,
# and if it doesn't, then we flip the models labels 
def binaryModelLabelChecking(modelLabels, trueLabels):

    # Checking which has more in common, the current binary labels, or their flipped version
    numMatching_noLabelFlip = np.count_nonzero(modelLabels == trueLabels)

    numMatching_labelFlip = np.count_nonzero(np.logical_not(modelLabels) == trueLabels)


    if(numMatching_labelFlip > numMatching_noLabelFlip):
        returnLabels = np.logical_not(modelLabels).astype(np.int)
    else:
        returnLabels = modelLabels

    return(returnLabels)
