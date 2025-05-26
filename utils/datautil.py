from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pathlib
import numpy as np


##############
# Data class #
##############

class syntheticdataset():
    """
    Create synthetic dataset based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    """
    

    def __init__(self, random_state = 42, loadpath = None):

        if loadpath:
            self.loaddata(loadpath)
        else:
            # Create synthetic data set
            self.params = {}
            self.params['n_features'] = 2
            self.params['n_samples'] = 30000
            self.params['n_classes'] = 2
            self.params['class_sep'] = 1.5
            self.params['n_redundant'] = 0
            self.params['weights'] = (0.9, 0.1)
            self.params['random_state'] = random_state

            self.X, self.y = make_classification(n_features = self.params['n_features'], n_samples = self.params['n_samples'], 
                                                 n_classes = self.params['n_classes'], class_sep =self.params['class_sep'],
                                                 n_redundant = self.params['n_redundant'], weights=self.params['weights'], random_state=self.params['random_state'])
            self.colordict = {0: '#FE9900', 1: '#BFD641'}

            # Split into training and calibration sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, stratify=self.y, test_size=0.2, random_state=42)
            self.X_fit, self.X_calib, self.y_fit, self.y_calib = train_test_split(self.X_train, self.y_train, stratify=self.y_train, test_size=0.25, random_state=self.params['random_state'])


            

    def savedata(self, savepath):
            dirdatasave = str(savepath)+"/data/"
            pathlib.Path(dirdatasave).mkdir(parents = True)
            np.save(dirdatasave+"/"+"X",self.X)
            np.save(dirdatasave+"/"+"y",self.y)
            np.save(dirdatasave+"/"+"X_train", self.X_train)
            np.save(dirdatasave+"/"+"y_train",self.y_train)
            np.save(dirdatasave+"/"+"X_fit", self.X_fit)
            np.save(dirdatasave+"/"+"y_fit",self.y_fit)
            np.save(dirdatasave+"/"+"X_calib", self.X_calib)
            np.save(dirdatasave+"/"+"y_calib",self.y_calib)
            np.save(dirdatasave+"/"+"X_test", self.X_test)
            np.save(dirdatasave+"/"+"y_test",self.y_test)
        
    def loaddata(self, loadpath):
            dirdataload = str(loadpath)+"/data/"
            self.X = np.load(dirdataload+"/"+"X.npy")
            self.y = np.load(dirdataload+"/"+"y.npy")
            self.X_train = np.load(dirdataload+"/"+"X_train.npy")
            self.y_train = np.load(dirdataload+"/"+"y_train.npy")
            self.X_fit = np.load(dirdataload+"/"+"X_fit.npy")
            self.y_fit = np.load(dirdataload+"/"+"y_fit.npy")
            self.X_calib = np.load(dirdataload+"/"+"X_calib.npy")
            self.y_calib = np.load(dirdataload+"/"+"y_calib.npy")
            self.X_test = np.load(dirdataload+"/"+"X_test.npy")
            self.y_test = np.load(dirdataload+"/"+"y_test.npy")
