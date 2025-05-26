from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import joblib

class XGBclassifierwrapper():
    """
    Train an XGB classifier 
    https://xgboost.readthedocs.io/en/stable/get_started.html
    Model hyperparameters selected by 5 fold cross validation
    """

    def __init__(self, X_fit, y_fit, savepath= None, loadpath = None):
        self.savepath = savepath
        self.loadpath = loadpath
        # Create XGB classifier
        if loadpath:
            self.loadmodel()
        else:
            xgbparams = {'n_estimators': [ 40, 60, 80, 100, 120],
                                                        'max_depth': [  6, 7, 8, 9, 10],
                                                        'learning_rate': [0.05, 0.1, 0.15, 0.20],
                                                        'min_child_weight': [1, 2, 3, 4, 5],
                                                        'subsample': [0.6, 0.8, 1.0],
                                                        'colsample_bytree': [ 0.6, 0.8, 1.0]
                                                        }

            xgb = XGBClassifier(tree_method = "hist", eval_metric = "auc", verbosity = 2)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            self.search_cv = RandomizedSearchCV(estimator=xgb, 
                                            param_distributions=xgbparams,
                                            n_iter=20,
                                            scoring="roc_auc",
                                            n_jobs=-1,
                                            cv=cv)
            self.search_cv.fit(X_fit, y_fit)
            if savepath:
                self.savemodel()

    def savemodel(self):
        joblib.dump(self.search_cv, '{:s}/search_cv.pkl'.format(self.savepath))

    def loadmodel(self):
        self.search_cv = joblib.load("{:s}/search_cv.pkl".format(self.loadpath))