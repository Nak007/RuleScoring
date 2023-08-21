'''
Available methods are the followings:
[1] RulebasedWOE
[2] FeatureScore

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-08-2023

'''
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (StandardScaler,
                                   minmax_scale)
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import (confusion_matrix, 
                             precision_score,
                             recall_score, f1_score)

__all__ = ["RulebasedWOE" , "FeatureScore"]

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
        
    min_woe : float, defaut=-1e5
        Minimum value of WOE after applying `factor`.
        
    max_woe : float, default=1e5
        Maximum value of WOE after applying `factor`
        
    Attributes
    ----------
    woes : dict
        A dict with keys as variable name, and weight-of-evidence as 
        values.
    '''
    def __init__(self, decimal=8, factor=1, 
                 min_woe=-1e5, max_woe=1e5):
        
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
        
    min_woe : float, defaut=-1e5
        Minimum value of Weight-of-Evidence. This is relevant when 
        `use_woe` is True.
        
    max_woe : float, default=1e5
        Maximum value of Weight-of-Evidence. This is relevant when 
        `use_woe` is True.
    
    decimal : int, default=4
        Number of decimal places to round to and must not be negative. 
        This is applied to `scores` (attribute).
    
    min_score : float, default=0.
        Minimum score.
        
    max_score : float, default=100.
        Maximum score.
        
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
  
    def __init__(self, use_woe=False, min_woe=-1e5, max_woe=1e5, 
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
        self.use_woe = super().StrOptions('use_woe', use_woe, *args[0])
        self.min_woe = super().Interval("min_woe", min_woe, *args[1])
        self.max_woe = super().Interval("max_woe", max_woe, *args[1])
        super().check_range(("min_woe", self.min_woe),
                            ("max_woe", self.max_woe))
  
        # Validate other parameters
        self.decimal = super().Interval("decimal", decimal, *args[2])
        self.min_score = super().Interval("min_score", min_score, float)
        self.max_score = super().Interval("max_score", max_score, float)
        super().check_range(("min_score", self.min_score),
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
        init_scores = np.array(super().convert(X)) * init_scores
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
            
        super().check_columns(X)
        scores = [self.scores[key] for key in self.columns]
        scores = np.full(X.shape, scores)
        scores = super().convert(X)[self.columns].values * scores
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