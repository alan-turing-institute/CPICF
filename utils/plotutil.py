
import seaborn as sns
import numpy as np
from sklearn.metrics import average_precision_score,precision_recall_curve,PrecisionRecallDisplay
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from utils.CFutils import get_confidenceinterval, CPICF

import matplotlib as mpl

from matplotlib import cm, ticker


def plot_performance(dataset, classifier):
    dftemp = pd.DataFrame({'x1': dataset.X_fit[:,0], 'x2': dataset.X_fit[:,1], 'class': dataset.y_fit})

    fig, axes = plt.subplots(1,3,figsize = (10.8,3.6))

    ax = axes[0]
    g = sns.scatterplot(data = dftemp, x = 'x1', y = 'x2', hue = 'class', alpha = 0.1, ax = ax)
    sns.kdeplot(data = dftemp, x = 'x1', y = 'x2', levels = 5, hue = 'class', ax = ax)
    g.set_xlabel('$x_1$')
    g.set_ylabel('$x_2$')
    g.legend('').set_visible(False)
    g.set_xlim(-5,4)
    g.set_ylim(-5,4)
    plt.tight_layout()
    ax.annotate('(A)', (-4.5, 3.5))

    ax = axes[1]
    y_pred = classifier.predict(dataset.X_test)
    precision, recall, _ = precision_recall_curve(dataset.y_test, y_pred)
    disp = PrecisionRecallDisplay.from_estimator(
        classifier, dataset.X_test, dataset.y_test, name="XGB Classifier",
        plot_chance_level=True, ax = ax)
    ax.legend(loc = 'best')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.tight_layout()
    ax.annotate('(B)', (0.9, 0.95))


    ax = axes[2]
    xvals = np.arange(np.min(dataset.X_fit[:,0]), np.max(dataset.X_fit[:,0]),0.25)
    yvals = np.arange(np.min(dataset.X_fit[:,1]), np.max(dataset.X_fit[:,1]),0.25)
    XX, YY = np.meshgrid(xvals, yvals)
    XXX = np.array([ (x,y) for (x, y) in zip(XX.flatten(), YY.flatten())])
    ZZclass = classifier.predict(XXX).reshape(XX.shape)
    g = sns.scatterplot(data = dftemp, x = 'x1', y = 'x2', hue = 'class', alpha = 0.1)
    ax.contour(XX, YY, ZZclass, levels = [0], colors = 'k', linestyles = 'dashed')
    g.legend('').set_visible(False)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    g.set_xlim(-5,4)
    g.set_ylim(-5,4)
    ax.annotate('(C)', (-4.5, 3.5))



# https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.Formatter
def plot_uncertainty(X_fit_limited_list, y_fit_limited_list, XGBLACP_list):
    # function to make each plot
    def make_plot(X,y,lacp, axis1, axis2, plotcbar = False):

        dftemp_limited= pd.DataFrame({'x1': X[:,0], 'x2': X[:,1], 'class': y})
        xvals = np.linspace(-5, 4,100)
        yvals = np.linspace(-5, 4,100)
        XX, YY = np.meshgrid(xvals, yvals)
        ZZ = np.array(
            [
                get_confidenceinterval( np.array([[x,y]]), lacp.lacp )  
                for x, y in zip(XX.flatten(), YY.flatten()) 
            ]
            )

        XXX = np.array([ (x,y) for (x, y) in zip(XX.flatten(), YY.flatten())])
        
        ZZ[ZZ ==np.inf] = ZZ[ZZ < np.inf].max()
        ZZreshape = ZZ.reshape(XX.shape)

        nlevels = 10
        lmin = 0.0001
        lmax = 1.2
        levels  =[lmin*np.exp((x+1)/(nlevels)*np.log(lmax/lmin)) for x in range(nlevels)]
      
        contour = axis1.contourf(XX, YY, ZZreshape, locator=ticker.LogLocator(),  levels = levels, extend = "both")
        axis1.set_xlim(-5,4)
        axis1.set_ylim(-5,4)
        if plotcbar:
            cbar = plt.colorbar(contour,format = mpl.ticker.LogFormatterSciNotation(base = 10.0, labelOnlyBase = False,minor_thresholds = (4,1)) )

        # Set the colorbar ticks and labels
            cbar.set_ticks([0.001, 0.01, 0.1])
            cbar.set_label('Prediction Interval Width')
        g = sns.scatterplot(data = dftemp_limited, x = 'x1', y = 'x2', hue = 'class', alpha = 0.2, s = 5, ax = axis2)
        sns.kdeplot(data = dftemp_limited, x = 'x1', y = 'x2', levels = 5, hue = 'class', ax = axis2)

        g.legend('').set_visible(False)
        g.set_xlim(-5,4)
        g.set_ylim(-5,4)
        axis1.set_xlabel('$x_1$')
        axis1.set_ylabel('$x_2$')
        
        axis2.set_xlabel('$x_1$')
        axis2.set_ylabel('$x_2$')
        
        #axis1.annotate(label+'{:d})'.format(1), (-4.5, 3.5), color = 'w')
        #axis2.annotate(label+'{:d})'.format(2), (-4.5, 3.5), color = 'k')
        return contour

    fig, ax = plt.subplots(2,3, figsize = (10.8, 7), layout = 'constrained')
    #annotatelist = ['(A','(B','(C']
    for i in range(3):
        if i == 2:
            c = make_plot(X_fit_limited_list[i], y_fit_limited_list[i],XGBLACP_list[i], ax[0,i], ax[1,i], plotcbar = True)
        else:
            c = make_plot(X_fit_limited_list[i], y_fit_limited_list[i],XGBLACP_list[i], ax[0,i], ax[1,i], plotcbar = False)

    

def get_minimised_counterfactual_list(xinstlist, get_confidenceinterval, BinaryClassifier, ll, alpha, ax):
    
    xcflist = []
    classlist = []
    for xinst in tqdm.tqdm(xinstlist):
        # Compute counterfactual    
        CFgen = CPICF(xinst, get_confidenceinterval, BinaryClassifier, ll = ll, alpha = alpha)
        res = CFgen.returnCF()
    
        xcf = res.X['z1'],res.X['z2']
        xcflist.append(xcf)


    
    
    # Start plotting surface
    xvals = np.arange(-5,5,0.25)
    yvals = np.arange(-5,5,0.25)
    XX, YY = np.meshgrid(xvals, yvals)
    
    ZZ = np.array(
        [
            CFgen.problem.lossfn(np.array([[x,y]]) )  
            for x, y in zip(XX.flatten(), YY.flatten()) 
        ]
        )
    
    XXX = np.array([ (x,y) for (x, y) in zip(XX.flatten(), YY.flatten())])
    ZZclass = CFgen.problem.BinClassifier.predict(XXX).reshape(XX.shape)
    ZZreshape = ZZ.reshape(XX.shape)
    
    #fig, ax = plt.subplots()#subplot_kw={"projection": "3d"})

    contour = ax.contourf(XX, YY, ZZreshape, levels = 20, extend = 'both')
    ax.contour(XX, YY, ZZclass, levels = [0.1],  linestyles = 'dashed', alpha = 0.5, colors = 'w')
  
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
  
    cbar = plt.colorbar(contour)
    #ticks = cbar.get_ticks()

    # Set the colorbar ticks and labels
    #cbar.set_ticks([100,1000,1e4])
    #cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])
    cbar.set_label('$L_{total}$')
  
    ax.annotate('(B)', (-4.5, 3.5), color = 'w')
    ax.set_xlim(-5,4)
    ax.set_ylim(-5,4)

    for xinst, xcf in zip(xinstlist, xcflist):
        ax.plot(xinst[0], xinst[1], 'ro')
        ax.plot(xcf[0]+.1, xcf[1]-0.1,'wo')
        ax.arrow(xinst[0], xinst[1], xcf[0]+0.1 - xinst[0], xcf[1]-0.1 - xinst[1])