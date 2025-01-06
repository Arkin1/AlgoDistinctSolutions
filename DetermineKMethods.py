import numpy as np
from sklearn.metrics import silhouette_score
from Constants import SEED
import pickle
import os
class SilhouetteMethod:
    def determineK(self, embeddings, cluster_algo, max_k = 10, cache = 'Data/Cached_Simple_Clusterings'):
        print(f"Determining the number k using Silhouette score without multiview {embeddings['name']}:")
        problem_dict = embeddings['problemDict']
        
        problem_validation_cluster_k = {}
        for problem , data in problem_dict.items():
            X = np.array(data['X'])
            Y = np.array(data['Y'])
            k_scores = []
            true_k = list(set(Y))
            for k in range(2, max_k + 1):
                score = self.determine_silhouette_score(X, cluster_algo, problem, k, embeddings['name'], cache)

                k_scores.append((k, true_k, score))

            k_scores = sorted(k_scores, key = lambda l : l[2], reverse=True)
            problem_validation_cluster_k[problem] = k_scores
        
        return problem_validation_cluster_k
    
    def determine_silhouette_score(self, X, cluster_algo, problem, k, embeddings_name, cache_path = 'Data/Cached_Simple_Clusterings'):
        if not os.path.exists(f"{cache_path}/{type(cluster_algo).__name__}_{embeddings_name}_{problem}_{k}_labels.bin") or not os.path.exists(f"{cache_path}/{type(cluster_algo).__name__}_{embeddings_name}_{problem}_{k}.bin"):
            try:
                cluster_algo.set_params(n_clusters = k, random_state = SEED)
            except:
                cluster_algo.set_params(n_clusters = k)

            print(f'Determining Silhouette score for {k} on {problem} using cluster algorithm {type(cluster_algo).__name__} where ')

            try:
                cluster_labels = cluster_algo.fit_predict(X, n_jobs = -1)
            except:
                cluster_labels = cluster_algo.fit_predict(X)
            
            #with open(f"{cache_path}/{type(cluster_algo).__name__}_{embeddings_name}_{problem}_{k}_labels.bin", 'wb') as fp:
            #    pickle.dump(cluster_labels, fp)
            #with open(f"{cache_path}/{type(cluster_algo).__name__}_{embeddings_name}_{problem}_{k}.bin", 'wb') as fp:
            #    pickle.dump(cluster_algo, fp)
        else:
            #raise Exception("NOT LEGAL")
            print("Using Cache")
            with open(f"{cache_path}/{type(cluster_algo).__name__}_{embeddings_name}_{problem}_{k}_labels.bin", 'rb') as fp:
                cluster_labels = pickle.load(fp)
            # with open(f"{cache_path}/{type(cluster_algo).__name__}_{embeddings_name}_{problem}_{k}.bin", 'rb') as fp:
            #     cluster_algo = pickle.load(fp)

        score = -1
        try:
            score = silhouette_score(X, cluster_labels)
        except Exception as e:
            print(e)

        return score