from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from deel.puncc.regression import LocallyAdaptiveCP
from deel.puncc.api.prediction import MeanVarPredictor
import numpy as np
import joblib

###############################
#  XGB conformal predictor    #
###############################


class LACPwrapper():
    """
    Conformal predictor based on https://github.com/deel-ai/puncc
    Sequentially fit \\theta_k regression model, and MAD model
    """

    def __init__(self, X_fit, y_fit_reg, X_calib, y_calib, savepath = None, loadpath = None):  

        if loadpath:
            self.loadpath = loadpath
            self.loadmodel()
        else:

            xgbparams_reg = {
            'n_estimators': [80, 100, 120, 150, 300],
            'max_depth': [8, 9, 10,11,12],
            'learning_rate': [ 0.01, 0.05, 0.1, 0.20],
            'min_child_weight': [1, 2, 3, 4, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [ 0.6, 0.8, 1.0]
            }
                    
            # Define cross-validation strategy
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            model_mu = XGBRegressor()        
            # Perform hyperparameter search

            self.search_cv_mu = RandomizedSearchCV(estimator=model_mu, 
                                        param_distributions=xgbparams_reg,
                                        n_iter=20,
                                        scoring="neg_mean_absolute_error",
                                        n_jobs=-1,
                                        cv=cv,
                                        verbose=1)

            # look at the undersampled version
            self.search_cv_mu.fit(X_fit, y_fit_reg)

            y_fit_disp = np.abs(y_fit_reg - self.search_cv_mu.predict(X_fit) )
            # Define cross-validation strategy
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            model_disp = XGBRegressor()        
            # Perform hyperparameter search

            self.search_cv_disp = RandomizedSearchCV(estimator=model_disp, 
                                        param_distributions=xgbparams_reg,
                                        n_iter=20,
                                        scoring="neg_mean_absolute_error",
                                        n_jobs=-1,
                                        cv=cv,
                                        verbose=1)

            # look at the undersampled version
            self.search_cv_disp.fit(X_fit, y_fit_disp)


                    # Wrap models in a mean/variance predictor
            mean_var_predictor = MeanVarPredictor(models=[self.search_cv_mu, self.search_cv_disp], is_trained = [True, True])

            # CP method initialization
            self.lacp = LocallyAdaptiveCP(mean_var_predictor, train = False)
            self.lacp.fit(X_calib = X_calib, y_calib = y_calib)
            if savepath:
                self.savepath = savepath
                self.savemodel()

    def savemodel(self):
        joblib.dump(self.lacp, '{:s}/lacp.pkl'.format(self.savepath))
        

    def loadmodel(self):
        self.lacp = joblib.load("{:s}/lacp.pkl".format(self.loadpath))