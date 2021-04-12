from sklearn.cluster import KMeans, SpectralClustering
from ValidationMethods import ClusteringValidationMethod
from ValidationMethods import EstimatorValidationMethod
from EmbeddingsLoader import W2VEmbeddingsLoader, SafeEmbeddingsLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

clusteringValidationMethod = ClusteringValidationMethod()
kmeans = KMeans(n_clusters = -1)
spectralClustering = SpectralClustering(n_clusters= -1)

clusteringValidationMethod.validateK(SafeEmbeddingsLoader(), [kmeans, spectralClustering])
clusteringValidationMethod.validateK(W2VEmbeddingsLoader(), [kmeans, spectralClustering])

estimatorValidation = EstimatorValidationMethod()
estimatorValidation.validate(SafeEmbeddingsLoader(), [RandomForestClassifier(), SVC(), XGBClassifier(verbosity = 0)], 0.3)
estimatorValidation.validate(W2VEmbeddingsLoader(), [RandomForestClassifier(), SVC(), XGBClassifier(verbosity = 0)], 0.3)


