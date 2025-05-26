from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.variable import Real, Integer
from pymoo.optimize import minimize

import numpy as np


def get_confidenceinterval(X, lacp, alpha = .2):
    """
    Take an array of shape (N, 2) and return confidence intervals
    Duplicate prediction for smaller arrays to 
    """
    shape = X.shape
    if len(shape)<2:
        print("CF interval input shape incorrect.")
        return None
    if shape[0] < 2:
        X = np.vstack([X, X])
        pred, lower, upper = lacp.predict(X, alpha=alpha)
        return (upper-lower)[0]
    else:
        pred, lower, upper = lacp.predict(X, alpha=alpha)
        return upper - lower

class MixedVariableProblem(ElementwiseProblem):
    """Mixed Variable optimization problem from pymoo
    https://pymoo.org/
    """

    def __init__(self, xinst, ConfInterval, BinClassifier, ll = 1, alpha = 0.2, unconstrained = False, goweronly = False, **kwargs):
        vars = {
            "z1": Real(bounds=(-5, 5)),
            "z2": Real(bounds=(-5, 5)),
        }
        self.xinst = xinst
        self.ll = ll
        self.alpha = alpha
        self.ConfInterval = ConfInterval
        self.BinClassifier = BinClassifier
        self.unconstrained = unconstrained
        self.goweronly = goweronly
        super().__init__(vars=vars, n_obj=1, n_ieq_constr = 0, n_eq_constr = 1, **kwargs)

    def lossfn(self, xcf):
        """
        Loss function for selecting conformal prediction intervals
        """
        if self.unconstrained:
            return 1
        
        D = np.linalg.norm(self.xinst-xcf)

        if self.goweronly:
            return D

        C = self.ConfInterval(xcf, alpha=self.alpha)
        
        try:
            FE = 1.0 / C + self.ll * D
            return FE
        except:
            pass
    
    def _evaluate(self, X, out, *args, **kwargs):
        
        z1, z2 = X["z1"], X["z2"]
        out["F"] = self.lossfn(np.array([[z1, z2]]))
        out["H"] = self.BinClassifier.predict(np.vstack([np.array([z1, z2]), self.xinst])).sum() - 1.0



class CPICF():
    """
    Input: Confidence interval function, binary classifier 
    Output: Counterfactual
    """
    def __init__(self, xinst, ConfInterval, BinClassifier, ll = 1, alpha = 0.2, unconstrained = False, goweronly = False):
        self.problem = MixedVariableProblem(xinst = xinst, ConfInterval = ConfInterval, BinClassifier = BinClassifier, ll = ll, alpha = alpha, unconstrained = unconstrained, goweronly = goweronly)
        self.algorithm = MixedVariableGA(pop_size=30)

    def returnCF(self):
        res = minimize(self.problem,
                self.algorithm,
                termination=('n_evals', 50),
                #seed=1, #more diverse results from random seed
                verbose=False)
        return res