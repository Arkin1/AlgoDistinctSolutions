from EmbeddingsLoader import EmbeddingsLoader
import numpy as np
import itertools
class ClusteringValidationMethod:
    
    def validateK(self, embeddingsLoader:EmbeddingsLoader, clusterAlgo):
        vector_size = embeddingsLoader.GetSize()

        X = []
        Y = []

        for problem in embeddingsLoader:
            problemEmbedding = np.mean(np.array(problem["embeddings"]), 1)
            X.append(problemEmbedding)
            Y.append(problem["label"])

        allLabels = list(set(Y))
        k = len(allLabels)

        X = np.array(X)
        Y = np.array(Y)

        labels = clusterAlgo.fit_transform(X)
        
        bestPermutation = range(0, k)
        for permutation in itertools.permutations(range(0, k)):
            permutedLabels = []

            for index in range(labels):
                permutedY.append()

        

        
