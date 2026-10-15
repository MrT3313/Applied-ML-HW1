import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class SVDRegression(RegressorMixin, BaseEstimator):
	"""Least squares solved by SVD with numpy's lstsq: the closed form for columns that repeat each other.

	Singular values below numpy's standard cutoff (machine precision times the larger of rows and columns, relative to
	the largest) count as 0, so when some columns are exact combinations of others the solution is the one with the
	smallest coefficients. LinearRegression (scipy's lstsq) only drops singular values below machine precision, so
	rounding noise in a repeated column can give coefficients near 1e14 and predictions just as far off.
	"""

	def fit(self, X, y):
		X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
		self.x_mean_, self.y_mean_ = X.mean(axis=0), y.mean()  # centered, as LinearRegression does, so the intercept is the mean
		self.coef_ = np.linalg.lstsq(X - self.x_mean_, y - self.y_mean_, rcond=None)[0]
		return self

	def predict(self, X):
		return (np.asarray(X, dtype=float) - self.x_mean_) @ self.coef_ + self.y_mean_
