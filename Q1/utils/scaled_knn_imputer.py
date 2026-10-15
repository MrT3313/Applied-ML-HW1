from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class ScaledKNNImputer(TransformerMixin, BaseEstimator):
	"""KNNImputer with the distances measured on standardized attributes, returning every attribute in its own units.

	make_pipeline(StandardScaler(), KNNImputer()) fills the same values but returns them standardized, which a shape
	with a threshold in the attribute's units (ex a floor at 56.8) cannot use.
	"""

	def fit(self, X, y=None):
		self.scaler_ = StandardScaler().fit(X)
		self.knn_ = KNNImputer().fit(self.scaler_.transform(X))
		return self

	def transform(self, X):
		return self.scaler_.inverse_transform(self.knn_.transform(self.scaler_.transform(X)))
