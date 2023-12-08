import numpy as np
from sklearn.metrics import silhouette_score
from Constants import SEED
import pickle
class SilhouetteMethod:
    def determineK(self, embeddings, cluster_algo, max_k = 10):
        print(f"Determining the number k using Silhouette score without multiview {embeddings['name']}:")
        problem_dict = embeddings['problemDict']
        
        problem_validation_cluster_k = {}
        for problem , data in problem_dict.items():
            k_scores = []
            for k in range(2, max_k + 1):
                X = np.array(data['X'])
                Y = np.array(data['Y'])

                all_labels = list(set(Y))
                true_k = len(all_labels)
            
                try:
                    cluster_algo.set_params(n_clusters = k, random_state = SEED)
                except:
                    cluster_algo.set_params(n_clusters = k)

                print(f'Determining Silhouette score for {k} on {problem} using cluster algorithm {type(cluster_algo).__name__} where ')

                try:
                    cluster_labels = cluster_algo.fit_predict(X, n_jobs = -1)
                except:
                    cluster_labels = cluster_algo.fit_predict(X)
                
                with open(f"Data/Clusterings/{type(cluster_algo).__name__}_{embeddings['name']}_{problem}_{k}_labels.bin", 'wb') as fp:
                    pickle.dump(cluster_labels, fp)
                with open(f"Data/Clusterings/{type(cluster_algo).__name__}_{embeddings['name']}_{problem}_{k}.bin", 'wb') as fp:
                    pickle.dump(cluster_algo, fp)

                k_scores.append((k, true_k, silhouette_score(X, cluster_labels)))

            k_scores = sorted(k_scores, key = lambda l : l[2], reverse=True)
            problem_validation_cluster_k[problem] = k_scores

                
        return problem_validation_cluster_k