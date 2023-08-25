'''
Available methods are the followings:
[1] RulebasedWOE
[2] FeatureScore
[3] PlotScore

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-08-2023

'''
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (StandardScaler,
                                   minmax_scale)
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import (confusion_matrix, 
                             precision_score,
                             recall_score, f1_score, 
                             precision_recall_curve)

__all__ = ["RulebasedWOE" , 
           "FeatureScore",
           "PlotScore"]

class ValidateParams:
    
    '''Validate parameters'''
    
    def Interval(self, Param, Value, dtype=int, 
                 left=None, right=None, closed="both"):

        '''
        Validate numerical input.

        Parameters
        ----------
        Param : str
            Parameter's name

        Value : float or int
            Parameter's value

        dtype : {int, float}, default=int
            The type of input.

        left : float or int or None, default=None
            The left bound of the interval. None means left bound is -∞.

        right : float, int or None, default=None
            The right bound of the interval. None means right bound is +∞.

        closed : {"left", "right", "both", "neither"}
            Whether the interval is open or closed. Possible choices are:
            - "left": the interval is closed on the left and open on the 
              right. It is equivalent to the interval [ left, right ).
            - "right": the interval is closed on the right and open on the 
              left. It is equivalent to the interval ( left, right ].
            - "both": the interval is closed.
              It is equivalent to the interval [ left, right ].
            - "neither": the interval is open.
              It is equivalent to the interval ( left, right ).

        Returns
        -------
        Value : float or int
            Parameter's value

        '''
        Options = {"left"    : (np.greater_equal, np.less), # a<=x<b
                   "right"   : (np.greater, np.less_equal), # a<x<=b
                   "both"    : (np.greater_equal, np.less_equal), # a<=x<=b
                   "neither" : (np.greater, np.less)} # a<x<b

        f0, f1 = Options[closed]
        c0 = "[" if f0.__name__.find("eq")>-1 else "(" 
        c1 = "]" if f1.__name__.find("eq")>-1 else ")"
        v0 = "-∞" if left is None else str(dtype(left))
        v1 = "+∞" if right is None else str(dtype(right))
        if left  is None: left  = -np.inf
        if right is None: right = +np.inf
        interval = ", ".join([c0+v0, v1+c1])
        tuples = (Param, dtype.__name__, interval, Value)
        err_msg = "%s must be %s or in %s, got %s " % tuples    

        if isinstance(Value, dtype):
            if not (f0(Value, left) & f1(Value, right)):
                raise ValueError(err_msg)
        else: raise ValueError(err_msg)
        return Value

    def StrOptions(self, Param, Value, options, dtype=str):

        '''
        Validate string or boolean inputs.

        Parameters
        ----------
        Param : str
            Parameter's name
            
        Value : float or int
            Parameter's value

        options : set of str
            The set of valid strings.

        dtype : {str, bool}, default=str
            The type of input.
        
        Returns
        -------
        Value : float or int
            Parameter's value

        '''
        if Value not in options:
            err_msg = f'{Param} ({dtype.__name__}) must be either '
            for n,s in enumerate(options):
                if n<len(options)-1: err_msg += f'"{s}", '
                else: err_msg += f' or "{s}" , got %s'
            raise ValueError(err_msg % Value)
        return Value
    
    def check_range(self, param0, param1):
        
        '''
        Validate number range.
        
        Parameters
        ----------
        param0 : tuple(str, float)
            A lower bound parameter e.g. ("name", -100.)
            
        param1 : tuple(str, float)
            An upper bound parameter e.g. ("name", 100.)
        '''
        if param0[1] >= param1[1]:
            raise ValueError(f"`{param0[0]}` ({param0[1]}) must be less"
                             f" than `{param1[0]}` ({param1[1]}).")

class BinaryData:
    
    ''' Convert to binary inputs'''
    
    def convert(self, X, y=None):
        
        '''Convert `X`, and `y` to binary inputs'''
        # Convert to binary array = {0,1}
        X = pd.DataFrame(self.__array__(X), 
                         columns=self.__columns__(X))
        if y is None: return X
        else: return X, self.__array__(y) 

    def __columns__(self, X):
        
        '''Extract columns'''
        columns = [f"Unnamed: {n}" for n in range(X.shape[1])]
        if isinstance(X, pd.DataFrame): columns = list(X)
        return columns
        
    def __array__(self, a):
        
        '''Convert to binary array'''
        return np.array(a).astype(bool).astype(int).copy()
    
    def check_columns(self, X):
        
        '''Check columns'''
        # Check for missing features
        columns = set(self.columns).difference(list(X))
        if len(columns)>0:
            raise ValueError(f"{len(columns):,d} missing feature(s) "
                             f"i.e. {', '.join(columns)}.")
        
        # Check number of features
        if X.shape[1]!=len(self.columns):
            raise ValueError(f"`X` has {X.shape[1]:,d} features, but "
                             f"`{type(self).__name__}` is expecting "
                             f"{len(self.columns):,d} features as input.")

class RulebasedWOE(ValidateParams, BinaryData):
    
    '''
    Calculate Weight-of-Evidence (WOE) only for binary variable with 
    values of 0 and 1. The formula is LOG(P(y=0|x=1))/P(y=1|x=1)).
    According to rule-based approach, WOE is only calculated when X 
    is 1.
    
    Parameters
    ----------
    decimal : int, default=8
        Number of decimal places to round to. `decimal` must not be
        negative.

    factor : int, default=1
        WOE multiplier.
        
    min_woe : float, defaut=-20
        Minimum value of WOE after applying `factor`.
        
    max_woe : float, default=20
        Maximum value of WOE after applying `factor`
        
    Attributes
    ----------
    woes : dict
        A dict with keys as variable name, and weight-of-evidence as 
        values.
    '''
    def __init__(self, decimal=8, factor=1, 
                 min_woe=-20., max_woe=20.):
        
        # Validate parameters
        args0 = (int, 0, None, "left")
        args1 = (float, None, None, "neither")
        self.decimal = super().Interval("decimal", decimal, *args0)
        self.factor  = super().Interval("factor" , factor , *args0)
        self.min_woe = super().Interval("min_woe", min_woe, *args1)
        self.max_woe = super().Interval("max_woe", max_woe, *args1)
        self.fitted = False
        
        if self.max_woe==self.min_woe:
            raise ValueError(f"`min_woe` must be less than `max_woe`. "
                             f"Got min_woe={self.min_woe}, and "
                             f"max_woe={self.max_woe}")
    
    def fit(self, X, y):
    
        '''
        Convert `X` to binary inputs and then fit model.
        
        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Binary input samples.

        y : array-like of shape (n_samples,) 
            Binary target values.

        Attributes
        ----------
        woes : dictionary
            A dict with keys as variable name, and weight-of-evidence as 
            value.
        '''
        self.woes = dict()
        X, y = super().convert(X, y)
        self.columns = list(X)
        N0 = np.fmax(sum(y==0), 1)
        N1 = np.fmax(sum(y==1), 1)
   
        for c,var in enumerate(self.columns):
            y1 = y[X.values[:,c]==1]
            n1 = np.array([sum(y1==m) for m in [0,1]])
            dist0 = np.fmax(n1[0], 0.5) / N0
            dist1 = np.fmax(n1[1], 0.5) / N1
            woe = np.log(dist0 / dist1) * self.factor
            woe = np.round(woe, self.decimal)
            self.woes[var] = np.fmax(self.min_woe, 
                                     np.fmin(self.max_woe, woe))
        self.fitted = True
        return self
    
    def transform(self, X):
        
        '''
        Transform data.
        
        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Binary input samples.
            
        Returns
        -------
        X_new : pd.DataFrame of shape (n_samples, n_features)
            Transformed array.
        '''
        X_new = super().convert(X)
        self.check_columns(X_new)
        
        if self.fitted==False:
            raise ValueError(f"`{type(self).__name__}` instance is not "
                             f"fitted yet. Call `fit` with appropriate "
                             f"arguments before using this estimator.")
         
        woes = np.array(list(self.woes.values()))
        woes = np.full(X_new.shape, woes.reshape(1,-1))
        return X_new.astype(float) * woes
    
    def fit_transform(self, X, y):
        
        '''
        Fit to data, then transform it.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Binary input samples.
        
        y : array-like of shape (n_samples,) 
            Binary target values.
        
        Returns
        -------
        X_new : pd.DataFrame of shape (n_samples, n_features)
            Transformed array.
        '''
        self.fit(X, y)
        return self.transform(X)

class FeatureScore(ValidateParams, BinaryData):
    
    '''
    Performs a score caculation for binary features.
    
    Parameters
    ----------
    use_woe : bool, default=False
        If True, `X` is converted to Weight-of-Evidence (WOE) to fit
        `estimator`, otherwise `X` remains unchanged and `woe` 
        defaults to 1.
        
    min_woe : float, defaut=-20
        Minimum value of Weight-of-Evidence. This is relevant when 
        `use_woe` is True.
        
    max_woe : float, default=20
        Maximum value of Weight-of-Evidence. This is relevant when 
        `use_woe` is True.
    
    decimal : int, default=4
        Number of decimal places to round to and must not be negative. 
        This is applied to `scores` (attribute).
    
    min_score : float, default=0.
        Minimum feature score.
        
    max_score : float, default=100.
        Maximum feature score.
        
    estimator : estimator object, default=None
        A sklearn LogisticRegression with initial parameters set. If 
        None, it uses default parameters i.e. {"max_iter":1000, 
        "solver":"liblinear", "class_weight":"balanced", and 
        "random_state":0}.
        
    Attributes
    ----------
    raw_scores : ndarray of shape (n_features,)
        An array of initial scores (estimator.coef_ * woes).
        
    woes : dict
        A dict with keys as variable name, and Weight-of-Evidence as 
        value.
    
    scores : dict
        A dict with keys as variable name, and scores as value.
    
    '''
  
    def __init__(self, use_woe=False, min_woe=-20., max_woe=20., 
                 decimal=0, min_score=0., max_score=100., 
                 estimator=None):
        
        # Initial parameters for LogisticRegression.
        if estimator is None:
            params = {"max_iter"     : 1000,
                      "solver"       : "liblinear",
                      "class_weight" : "balanced", 
                      "random_state" : 0}
            self.estimator = LogisticRegression(**params)
        else: self.estimator = clone(estimator)
        self.fitted = False
        
        # Validate `RulebasedWOE`'s parameters
        args = [([True, False], bool), 
                (float, None, None, "both"),
                (int, 0, None, "left")]
        self.use_woe = self.StrOptions('use_woe', use_woe, *args[0])
        self.min_woe = self.Interval("min_woe", min_woe, *args[1])
        self.max_woe = self.Interval("max_woe", max_woe, *args[1])
        self.check_range(("min_woe", self.min_woe),
                         ("max_woe", self.max_woe))
  
        # Validate other parameters
        self.decimal = self.Interval("decimal", decimal, *args[2])
        self.min_score = self.Interval("min_score", min_score, float)
        self.max_score = self.Interval("max_score", max_score, float)
        self.check_range(("min_score", self.min_score),
                         ("max_score", self.max_score))
            
    def fit(self, X, y, sample_weight=None):
        
        '''
        Fit the model according to the given training data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples 
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples. If 
            not provided, then each sample is given unit weight.

        Returns
        -------
        self : estimator
            Fitted estimator.
        '''
        if isinstance(X, pd.DataFrame): 
            self.columns = list(X)
            self.n_features = X.shape[1]
        else: raise ValueError(f"`X` must be pandas.DataFrame. " 
                               f"Got {type(X)} instead.")
            
        # Transform `X` to WOEs   
        if self.use_woe:
            kwds = dict(min_woe=self.min_woe,max_woe=self.max_woe)
            self.woe_converter = RulebasedWOE(**kwds).fit(X, y)
            X = self.woe_converter.transform(X)
            self.woes = self.woe_converter.woes
        else: self.woes = dict(zip(list(X),np.ones(X.shape[1])))
        
        # Calculate raw scores
        self.estimator.fit(X, y, sample_weight)
        coef = self.estimator.coef_.flatten() 
        woes = np.array([self.woes[key] for key in self.columns])
        self.raw_scores = coef * woes
    
        # Initial raw scores (for calibration)
        init_scores = np.full(X.shape, self.raw_scores.reshape(1,-1))
        init_scores = np.array(self.convert(X)) * init_scores
        init_scores = init_scores.sum(1, keepdims=True)
        
        # Calibrate scores
        calibrator = clone(self.estimator)
        calibrator.set_params(**{"class_weight":None})
        calibrator.fit(init_scores, y)
        self.adj_coef = calibrator.coef_.ravel()
        
        # Scale scores (min, max)
        args = (self.raw_scores * self.adj_coef, 
                (self.min_score, self.max_score))
        new_scores  = np.round(minmax_scale(*args), self.decimal)
        self.scores = dict(zip(self.columns, new_scores)) 
        self.fitted = True

        return self
    
    def transform(self, X):
        
        '''
        Transform `X` to scores.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The data to be transformed into scores.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Score array.
        '''
        if not isinstance(X, pd.DataFrame): 
            raise ValueError(f"`X` must be pandas.DataFrame. " 
                             f"Got {type(X)} instead.")
            
        if self.fitted==False:
            raise ValueError(f"`{type(self).__name__}` instance is not "
                             f"fitted yet. Call `fit` with appropriate "
                             f"arguments before using this estimator.")
            
        self.check_columns(X)
        scores = [self.scores[key] for key in self.columns]
        scores = np.full(X.shape, scores)
        scores = self.convert(X)[self.columns].values * scores
        return scores.sum(1)
    
    def fit_transform(self, X, y, sample_weight=None):
        
        '''
        Fits transformer to `X` and `y` and returns a transformed version 
        of `X`.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples 
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples. If 
            not provided, then each sample is given unit weight.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Score array.
        '''
        self.fit(X, y, sample_weight=None)
        return self.transform(X)

class SetProperties:
    
    def __majorlocator__(self):
        
        '''set major locator both x and y'''
        # Set y-axis limit
        y_min, y_max = self.ax.get_ylim()
        self.ax.set_ylim(y_min, y_max/0.9)
        
        # Number of ticks
        t = min(int(self.bins/1.5), 12)
        self.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(t))
        self.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
        
        # Tick label font size
        self.ax.tick_params(axis='x', labelsize=11)
        self.ax.tick_params(axis='y', labelsize=11)
    
    def __prop__(self, ylabel:str=None, xlabel:str=None, title:str=None):
        
        '''set label for both axes as well as title'''
        kwds = dict(fontsize=12)
        self.ax.set_xlabel(xlabel, **kwds)
        self.ax.set_ylabel(ylabel, **kwds)
        self.ax.set_title(title, **kwds)
        self.__majorlocator__()
        self.__spines__()
        
    def __spines__(self):
        
        '''set spines'''
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        
    def __legend__(self):
        
        '''set legend'''
        self.ax.legend(loc="best", edgecolor="grey", ncol=1,
                       borderaxespad=0.2, markerscale=1., 
                       columnspacing=0.5, labelspacing=0.4, 
                       handletextpad=0.2, fontsize=11,
                       handlelength=1)
        
    def __axvline__(self, score, label=None):
        
        '''draw vertical line'''
        kwds  = dict(lw=1, ls="--", color="grey", label=label)
        self.ax.axvline(score, **kwds)
        self.ax.xaxis.set_minor_locator(FixedLocator([score]))
        self.ax.tick_params(axis="x", which="minor", length=3, color="k")
        
    def __axhline__(self, score, label=None):
        
        '''draw horizontal line'''
        kwds  = dict(lw=1, ls="--", color="grey", label=label)
        self.ax.axhline(score, **kwds)
        self.ax.yaxis.set_minor_locator(FixedLocator([score]))
        self.ax.tick_params(axis="y", which="minor", length=3, color="k")
        
    def check_axis(self, ax):
        
        '''check whether the object is matplotlib.axes'''
        if ax is not None:
            if not isinstance(ax, mpl.axes.Axes):
                raise ValueError(f"`ax` must be matplotlib axis. "
                                 f"Got {type(ax)} instead.")
            else: self.ax = ax
        else: self.ax = plt.subplots(figsize=(6,4))[1]

class CalculateParams:
    
    def __bins__(self):
        
        '''
        Create bin-edges, width, and x.
        
        Parameters
        ----------
        self.y_score : ndarray (attribute)
        self.bins : int (attribute)
        
        Attributes
        ----------
        bin_edges : ndarray
        width : float
        x : ndarray
        '''
        start, stop = np.percentile(self.y_score, q=[0,100])
        stop += np.finfo(float).eps
        self.bin_edges = np.linspace(start, stop, self.bins + 1)
        self.width = np.diff(self.bin_edges)[0] * 0.8
        self.x = self.bin_edges[:-1] + np.diff(self.bin_edges)/2
    
    def __scores__(self):
        
        '''
        Calculate precision, recall, and f1 for all thresholds.
        
        Attributes
        ----------
        precision : ndarray of shape (n_thresholds+1,)
        recall : ndarray of shape (n_thresholds+1,)
        f1 : ndarray of shape (n_thresholds+1,)
        threshold : ndarray (n_uniques,)
        '''
        # Caluclate precision and recall
        self.precision, self.recall, self.threshold = \
        precision_recall_curve(self.y_true, self.y_score)
        
        # Calcuate F1-Score
        numer = 2 * self.precision * self.recall
        denom = self.precision + self.recall
        self.f1 = numer/np.where(denom==0,1,denom)
        
    def __cutoff__(self, cutoff:float=None, metric:str="f1", threshold:float=0.5):
        
        '''
        If `cutoff` is None, it defines a new cutoff given metric and 
        its corresponding threshold, otherwise it uses cutoff to 
        determine other components i.e. precision, recall, and f1-score.
        
        Attributes
        ----------
        components : dict, keys={"precision","recall","f1","threshold"}
        pct : dict, keys={"0","1","all"}
        '''
        # Determine index of cutoff
        if cutoff is not None: 
            t = np.clip(cutoff, min(self.y_score), max(self.y_score))
            n = sum(self.threshold<=t) - 1  
        else: 
            t = threshold - getattr(self, metric, 0.)[:-1]
            n = np.argmin(np.where(t < 0, np.inf, t))

        self.components = {"precision" : self.precision[n],
                           "recall"    : self.recall[n],
                           "f1"        : self.f1[n],
                           "threshold" : self.threshold[n]}
        
        self.pct = dict()
        for k,c in dict([("0",0),("1",1),("all",[0,1])]).items():            
            # % that makes the cutoff
            t = np.isin(self.y_true, c)
            p = sum((self.y_score>self.threshold[n]) & t)/sum(t)  
            self.pct[k] = p
            
    def __cumsum__(self):
        
        '''
        Calculate cumulative sum of samples
        
        Attributes
        ----------
        cumsum : dict
        '''
        self.cumsum = dict()
        for k,c in dict([("0",0),("1",1),("all",[0,1])]).items():
            # Last bin includes remaining scores
            t = np.isin(self.y_true, c)
            bins = np.r_[self.threshold, np.inf]
            hist = np.histogram(self.y_score[t], bins=bins)[0]
            cums = (np.cumsum(hist[::-1])/sum(t))[::-1]
            self.cumsum[k] = {"sum":cums}
            
    def __getparams__(self):
        
        '''create all parameters'''
        self.__bins__()
        self.__scores__()
        self.__cumsum__()
        
    def __validate__(self, cutoff, metric, threshold):
        
        '''validate parameters'''
        _ = self.StrOptions('metric', metric, ["f1","precision","recall"], str)
        _ = self.Interval("threshold", threshold, float, 0., 1., "both")
        if cutoff is not None: _ = self.Interval("cutoff", cutoff, float)

class PlotScore(ValidateParams, SetProperties, CalculateParams):
    
    '''
    Plotting class
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary label indicators. 
    
    y_score : array-like of shape (n_samples,)
        Target scores.
    
    bins : int, default=20
        The number of equal-width bins.
        
    colors : list of 3 hex-colors, default=None
        If None, it defaults to ["#008BFB", "#FF0051", "#7E7E7E"].
    
    '''
    def __init__(self, y_true, y_score, bins=20, colors=None):
        
        self.bins = self.Interval("bins", bins, int, 2, None, "left")
        if colors is None: self.colors = ["#008BFB","#FF0051","#7E7E7E"]
        self.y_true  = y_true
        self.y_score = y_score
        self.__getparams__()

    def hist(self, label=2, ax=None):
        
        '''
        Plot histogram
        
        Parameters
        ----------
        label : {0, 1, 2}, default=2
            Label of selected class. If 2, all classes is selected.
            
        ax : axis object, default=None
            If None, it creates an axis with figsize of (6,4).
            
        Returns
        -------
        ax : axis object
        '''
        # Validate all parameters
        label = self.Interval("label", label, int, 0, 2, "both")
        self.check_axis(ax)
        
        # Plot histogram
        groups = dict([(0,[0]),(1,[1]),(2,[0,1])])
        scores = self.y_score[np.isin(self.y_true, groups[label])]
        height = np.histogram(scores, self.bin_edges)[0]
        kwds = dict(width=self.width, align="center", 
                    color=self.colors[label])
        self.ax.bar(self.x, height/sum(height), **kwds)
        
        k = ",".join(np.r_[groups[label]].astype(str))
        title = (f"N({k}) = {(n:=len(scores)):,.0f} "
                 f"({n/len(self.y_true)*100:.3g}%)")
        self.__prop__("Density", "Estimator score", title)
        
        return self.ax
        
    def score(self, cutoff=None, metric="f1", threshold=0.5, ax=None):
        
        '''
        Plot scores i.e. precision, recall, and f1.
        
        Parameters
        ----------
        cutoff : float, default=None
            The score cutoff. Any sample whose score is greater than
            `cutoff` is selected.
            
        metric : {"f1", "precision", "recall"}, default="f1"
            Specify metric to evaluate the performance. This is relevant
            when `cutoff` is not defined.
        
        threshold : float, default=0.5
            A threshold of defined metric. This is relevant when `cutoff`
            is not defined.
      
        ax : axis object, default=None
            If None, it creates an axis with figsize of (6,4).
            
        Returns
        -------
        ax : axis object
        '''        
        # Validate all parameters
        self.check_axis(ax)
        self.__validate__(cutoff, metric, threshold)
        self.__cutoff__(cutoff, metric, threshold)
        
        # Plot scores i.e. precision, recall, and f1
        score = self.components["threshold"]
        for n,key in enumerate(self.components.keys()):
            if key!="threshold":
                args = (key[0].upper()+key[1:], self.components[key])
                self.ax.plot(self.threshold, getattr(self, key)[:-1], 
                             lw=2, color=self.colors[n])
                self.ax.scatter([score], [self.components[key]], s=25,  
                                marker="o", color=self.colors[n], 
                                label= "{} ({:,.0%})".format(*args))
            else:
                label =  r"Threshold > {:,.0f}".format(score)
                self.__axvline__(score, label)
                
        self.__prop__("Score", "Estimator score")
        self.__legend__()
        
        return self.ax
    
    def cumulative(self, cutoff=None, metric="f1", threshold=0.5, 
                   ax=None, show_metric=True):
        
        '''
        Plot scores i.e. precision, recall, and f1.
        
        Parameters
        ----------
        cutoff : float, default=None
            The score cutoff. Any sample whose score is greater than
            `cutoff` is selected.
            
        metric : {"f1", "precision", "recall"}, default="f1"
            Specify metric to evaluate the performance. This is relevant
            when `cutoff` is not defined.
        
        threshold : float, default=0.5
            A threshold of defined metric. This is relevant when `cutoff`
            is not defined.
      
        ax : axis object, default=None
            If None, it creates an axis with figsize of (6,4).
            
        show_metric : bool, default=True
            If True, it displays `metric` in the background.
            
        Returns
        -------
        ax : axis object
        '''
        # Validate all parameters
        self.check_axis(ax)
        self.__validate__(cutoff, metric, threshold)
        self.__cutoff__(cutoff, metric, threshold)
        
        # Plot cumulative number of respective classes
        score = self.components["threshold"]
        groups = dict([("0","N(0)"),("1","N(1)"),("all","N(0,1)")])
        for n,(key,value) in enumerate(self.cumsum.items()):
            label = f"{groups[key]} ({self.pct[key]*100:.3g}%)" 
            self.ax.plot(self.threshold, self.cumsum[key]["sum"], 
                         lw=2, color=self.colors[n])
            self.ax.scatter([score], [self.pct[key]], s=25,  
                            marker="o", label=label, 
                            color=self.colors[n])
        # Show metric
        if show_metric:
            args = (metric[0].upper()+metric[1:], 
                    self.components[metric])
            self.ax.plot(self.threshold, getattr(self, metric)[:-1], 
                         lw=1, ls="-", color="#bdc3c7", zorder=-1)
            self.ax.scatter([score], [self.components[metric]], 
                            s=25, marker="o", color="#bdc3c7", 
                            label= "{} ({:,.0%})".format(*args))

        delta = (self.pct["all"]-1)*100
        title = f"N = {delta+100:.3g}%, $\Delta$ = {delta:-.3g}%"
        label =  r"Estimator score > {:,.0f}".format(score)
        self.__prop__("Cumulative Density", "Estimator score", title)
        self.__axvline__(score, label)
        self.__legend__()
        
        return self.ax