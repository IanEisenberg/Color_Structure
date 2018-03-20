import numpy as np
from past.utils import old_div
from psychopy.data.fit import _baseFunctionFit
from psychopy.data import FitCumNormal, FitWeibull
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

def fit_choice_fun(df, stim_col='speed_change'):
    """ Fits a logistic regression to predict choice """
    df = df.query('exp_stage != "pause" and rt==rt')
    clf = LogisticRegression(C=1e20)
    X=(df.loc[:,stim_col]).values.reshape(-1, 1)
    clf.fit(X, df.binarized_response)
    loss = log_loss(df.binarized_response, clf.predict_proba(X))
    accuracy = accuracy_score(df.binarized_response, clf.predict(X))
    return clf, {'log-loss': loss,
                 'accuracy': accuracy}
    

def response_probs(responseFun, X):
   probs = list(responseFun.eval(X))
   probs = [[1-i, i] for i in probs]
   return probs

def fit_response_fun(df, kind='lapseWeibull', fit_kwargs=None):
    """ Fits a response function to accuracy data """ 
    df = df.query('exp_stage != "pause" and rt==rt')
    sigma = 1
    fit_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = fit_kwargs
    if kind == 'CumNorm':
        fun = FitCumNormal
    elif kind == 'Weibull':
        fun = FitWeibull
        #param guess
        fit_kwargs['guess'] = [df.quest_estimate.iloc[-1], 3.5]
    elif kind == 'lapseWeibull':
        fun = FitLapseWeibull
        #param guess
        fit_kwargs['guess'] = [df.quest_estimate.iloc[-1], 3.5, .05]
        fit_kwargs['optimize_kws'] = {'bounds': ([-np.inf, -np.inf, 0], [np.inf, np.inf, 1])}
    out = fun(df.decision_var, df.FB, sigma, **fit_kwargs)
    probs = response_probs(out, df.decision_var)
    loss = log_loss(df.FB, probs)
    accuracy = accuracy_score(df.FB, [i>.5 for i in list(out.eval(df.decision_var))])
    return out, {'log-loss': loss,
                 'accuracy': accuracy}


global _chance
_chance = .5
class FitLapseWeibull(_baseFunctionFit):
    """Fit a Weibull function (either 2AFC or YN)
    of the form::

        y = chance + (1.0-chance)*(1-exp( -(xx/alpha)**(beta) ))

    and with inverse::

        x = alpha * (-log((1.0-y)/(1-chance)))**(1.0/beta)

    After fitting the function you can evaluate an array of x-values
    with ``fit.eval(x)``, retrieve the inverse of the function with
    ``fit.inverse(y)`` or retrieve the parameters from ``fit.params``
    (a list with ``[alpha, beta]``)
    """
    # static methods have no `self` and this is important for
    # optimise.curve_fit
    @staticmethod
    def _eval(xx, alpha, beta, lapse):
        global _chance
        xx = np.asarray(xx)
        yy = _chance + (1.0 - _chance - lapse) * (1 - np.exp(-(old_div(xx, alpha))**beta))
        return yy

    @staticmethod
    def _inverse(yy, alpha, beta, lapse):
        global _chance
        xx = alpha * (-np.log(old_div((1.0 - yy), (1 - _chance - lapse)))) ** (old_div(1.0, beta))
        return xx