
from DetermineKMethods import SilhouetteMethod
from EmbeddingsLoader import Code2VecEmbeddingsLoader, InfercodeEmbeddingsLoader, W2VEmbeddingsLoader, SafeEmbeddingsLoader, TfidfEmbeddingsLoader, EmbeddingsLoader
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from IValidationPipelines import IValidationPipelines

class DetermineKPipelines(IValidationPipelines):
    def k_clustering_pipeline(self):
        print("Running KClusteringPipeline...")

        clustering_validation_method = SilhouetteMethod()

        cluster_algos = [KMeans(), SpectralClustering(), AgglomerativeClustering(affinity="cosine",linkage="average")]
        embeddings_list = [self._split_embeddings_data_per_problem(Code2VecEmbeddingsLoader()), 
                             self._split_embeddings_data_per_problem(SafeEmbeddingsLoader()), 
                             self._split_embeddings_data_per_problem(TfidfEmbeddingsLoader()),
                             self._split_embeddings_data_per_problem(InfercodeEmbeddingsLoader()),
                             self._split_embeddings_data_per_problem(W2VEmbeddingsLoader())]

        csv = self._create_csv_validation("NumKCluster")

        self._append_to_csv(csv, ['Problem', 'Embedding', 'Clustering', "PredNumClusters", "TrueNumClusters", 'Score'])

        for embeddings in embeddings_list:
            for clusterAlgo in cluster_algos:
                problems = clustering_validation_method.determineK(embeddings, clusterAlgo)

                for name in problems.keys():
                   k_scores = problems[name]
                   
                   for pred_k, true_k, score in k_scores:
                        self._append_to_csv(csv, [name, embeddings['name'], type(clusterAlgo).__name__, str(pred_k), str(true_k), str(score)])

        self._close_csv(csv)