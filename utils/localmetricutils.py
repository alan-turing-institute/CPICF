import numpy as np

##############################################################################
#    Helper functions for computing local change in prediction probability   #
##############################################################################



def proba_nearxpoint(xpoint, dist, classifier, numpoints = 10):
    """
    Compute the prediction probabilities near the query point
    on a square grid around the point of interest
    """
    xvals = xpoint[0]+np.linspace(-dist/2, dist/2,num = numpoints)
    yvals = xpoint[1]+np.linspace(-dist/2, dist/2,num = numpoints)
    XX, YY = np.meshgrid(xvals, yvals)
    XXX = np.array([ (x,y) for (x, y) in zip(XX.flatten(), YY.flatten())])
    ZZclassprob = classifier.predict_proba(XXX)[:,0].reshape(XX.shape)
    return ZZclassprob

def probadiff_nearxpoint(xpoint, dist, classifier1, classifier2, numpoints = 10):
    """
    Compute absolute difference in absolute values of probabilities near query point
    """
    xvals = xpoint[0]+np.linspace(-dist/2, dist/2,num = numpoints)
    yvals = xpoint[1]+np.linspace(-dist/2, dist/2,num = numpoints)
    XX, YY = np.meshgrid(xvals, yvals)
    XXX = np.array([ (x,y) for (x, y) in zip(XX.flatten(), YY.flatten())])
    ZZclassprobdiff = classifier2.predict_proba(XXX)[:,0].reshape(XX.shape)-classifier1.predict_proba(XXX)[:,0].reshape(XX.shape)

    return np.abs(ZZclassprobdiff)