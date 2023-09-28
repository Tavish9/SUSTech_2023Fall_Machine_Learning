import numpy as np
import itertools
import functools

class PolynomialFeature(object):
    """
    polynomial features

    transforms input array with polynomial features

    Example
    =======
    x =
    [[a, b],
    [c, d]]

    y = PolynomialFeatures(degree=2).transform(x)
    y =
    [[1, a, b, a^2, a * b, b^2],
    [1, c, d, c^2, c * d, d^2]]
    """

    def __init__(self, degree=2):
        """
        construct polynomial features

        Parameters
        ----------
        degree : int
            degree of polynomial
        """
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        """
        transforms input array with polynomial features

        Parameters
        ----------
        x : (sample_size, n) ndarray
            input array

        Returns
        -------
        output : (sample_size, 1 + nC1 + ... + nCd) ndarray
            polynomial features
        """
        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()
    
class Regression(object):
    """
    Base class for regressors
    """
    pass
    
class LinearRegression(Regression):
    """
    Linear regression model
    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def rmse(self, a, b):
        return np.sqrt(np.mean(np.square(a - b)))

    def fit(self, X:np.ndarray, t:np.ndarray):
        """
        perform least squares fitting

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) np.ndarray
            training dependent variable
        """
        self.w = np.linalg.pinv(X) @ t
        self.var = np.mean(np.square(X @ self.w - t))

    def predict(self, X:np.ndarray, return_std:bool=False):
        """
        make prediction given input

        Parameters
        ----------
        X : (N, D) np.ndarray
            samples to predict their output
        return_std : bool, optional
            returns standard deviation of each predition if True

        Returns
        -------
        y : (N,) np.ndarray
            prediction of each sample
        y_std : (N,) np.ndarray
            standard deviation of each predition
        """
        y = X @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y
    
def main():
    n, m = input().split()
    n, m = int(n), int(m)
    train = np.empty(shape=(2, n), dtype=float)
    test = np.empty(shape=(2, m), dtype=float)
    for i in range(n):
        x, y = input().split()
        train[0, i] = float(x)
        train[1, i] = float(y)
    for i in range(m):
        x, y = input().split()
        test[0, i] = float(x)
        test[1, i] = float(y)
    x_train, y_train = train[0], train[1]
    x_test, y_test = test[0], test[1]

    errors = []
    for i in range(11):
        p = PolynomialFeature(i)
        model = LinearRegression()
        model.fit(p.transform(x_train), y_train)
        y_pred, std = model.predict(p.transform(x_test), return_std=True)
        rmse = model.rmse(y_pred, y_test)
        errors.append((rmse, i, std))
    min_err = min(errors)
    print(min_err[1])
    print(min_err[2][0].round(6))

if __name__ == '__main__':
    main()
    