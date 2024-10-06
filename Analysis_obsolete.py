# import pandas as pd
# from sklearn.metrics import accuracy_score

# df = pd.read_csv('Data/Validation/NumKClusterUnsupervised_new.csv')

# def voting_at_k(df:pd.DataFrame, k):
#     sorted_df = df.sort_values('Score', ascending=False)

#     true = 0
#     for idx in range(k):
#         if(sorted_df.iloc[idx]['PredNumClusters'] == sorted_df.iloc[idx]['TrueNumClusters']):
#             true+=1
    
#     return pd.DataFrame({f"Voting@{k}": [true / k]})



# df_at_x = df.groupby(['Problem']).apply(lambda x: voting_at_k(x, 1))

# for i in range(2,11):
#     df_at_x = df_at_x.merge(df.groupby(['Problem']).apply(lambda x: voting_at_k(x, i)), on = 'Problem')

# df_at_x.to_csv('Data/Validation/UnsupervisedVotingKIntersection.csv')

# import copy


# class Node:
#     def __init__(self, cluster):
#         self.children = []
#         self.cluster = cluster
#         self.count = 0

# class MultiAssigmentClustering():
#     def __init__(self, grouped_paths, num_views, num_k):
#         self.num_views = num_views
#         self.num_k = num_k

#         self.found_k = 0
#         self.current_score = 0 
#         self.current_matching = [[]]
#         self.best_score = -1
#         self.best_matchings = None
#         self.is_used = [[0]*num_k for _ in range(num_views)]

#         self.root = Node(-1)

#         for path, v in grouped_paths.items():
#             self._add_to_trie(self.root, [int(x) for x in path.split('$')], num_views, len(v))


#     def _add_to_trie(self, root, v, num_views, value):
#         next_node = root
#         for level in range(num_views):
#             view_cluster = v[level]

#             found_node = None
#             for child in next_node.children:
#                 if child.cluster == view_cluster:
#                     found_node = child
#                     break

#             if found_node is None:
#                 found_node = Node(view_cluster)
#                 next_node.children.append(found_node)
#             next_node = found_node

#         next_node.count = value

#     def _fit(self, initial_root, root, level):
#         if len(root.children) == 0:
#             self.found_k += 1
#             self.current_score += root.count
#             self.current_matching.append([])

#             if self.found_k == self.num_k:
#                 if self.best_score < self.current_score:
#                     self.best_score = self.current_score
#                     self.best_matchings = copy.deepcopy(self.current_matching)

#                 self.current_score -= root.count
#                 self.found_k -= 1
#                 self.current_matching.pop()
#                 return
#             else:
#                 next_root = None
#                 i = 0
#                 for child in initial_root.children:
#                     if i == self.found_k:
#                         next_root = child
#                         break
#                     i += 1
#                 if next_root is not None:
#                     self.current_matching[-1].append(next_root.cluster)
#                     self._fit(initial_root, next_root, 1)
#                 self.current_score -= root.count
#                 self.found_k -= 1
#                 self.current_matching.pop()

#         for child in root.children:
#             if not self.is_used[level][child.cluster]:
#                 self.current_matching[-1].append(child.cluster)
#                 self.is_used[level][child.cluster] = True

#                 self._fit(initial_root, child, level + 1)
#                 self.is_used[level][child.cluster] = False
#                 self.current_matching[-1].pop()

#     def fit(self):
#         self.current_matching[-1].append(self.root.children[0].cluster)
#         self._fit(self.root, self.root.children[0], 1)




# from ValidationMethods import ClusteringValidationMethod, EstimatorValidationMethod
# from EmbeddingsLoader import Code2VecEmbeddingsLoader, InfercodeEmbeddingsLoader, W2VEmbeddingsLoader, SafeEmbeddingsLoader, TfidfEmbeddingsLoader, EmbeddingsLoader
# from IValidationPipelines import IValidationPipelines
# import numpy as np
# from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
# import itertools
# import pickle
# import json
# def _split_embeddings_data_per_problem(embeddings_loader:EmbeddingsLoader):
#         print(f"Extracting the {embeddings_loader.get_name()} embeddings from disk")

#         problem_dict = {}

#         for solution in embeddings_loader.get_embeddings():
#             function_embeddings = np.array(solution["embeddings"])
#             solution_embedding = np.mean(function_embeddings, 0)

#             problem = solution['label'].split("$")[0]
#             if(problem not in problem_dict):
#                 problem_dict[problem] = {'indexes':[], 'X' : [], 'Y': []}

#             problem_dict[problem]['indexes'].append(solution['index'])
#             problem_dict[problem]['X'].append(solution_embedding)
#             problem_dict[problem]['Y'].append(solution['label'])
        
#         embeddings = {}
#         embeddings['name'] = embeddings_loader.get_name()
#         embeddings['problemDict'] = problem_dict

#         return embeddings



# class DetermineKUnsupervised(IValidationPipelines):
#     def determine(self):
#         clustering_algos = [KMeans(), SpectralClustering(), AgglomerativeClustering(affinity="cosine",linkage="average")]

#         embeddings_list = [_split_embeddings_data_per_problem(Code2VecEmbeddingsLoader()), 
#                         _split_embeddings_data_per_problem(SafeEmbeddingsLoader()), 
#                         _split_embeddings_data_per_problem(TfidfEmbeddingsLoader()),
#                         _split_embeddings_data_per_problem(InfercodeEmbeddingsLoader()),
#                         _split_embeddings_data_per_problem(W2VEmbeddingsLoader())]


#         embeddings_list2_combinations = itertools.combinations(embeddings_list, 2)
#         clustering_algos2_cartesian_product = [[x,y] for x in clustering_algos for y in clustering_algos]
#         all_hyper_parametrization2 = itertools.product(embeddings_list2_combinations,  clustering_algos2_cartesian_product)

#         embeddings_list3_combinations = itertools.combinations(embeddings_list, 3)
#         clustering_algos3_cartesian_product = [[x,y,z] for x in clustering_algos for y in clustering_algos  for z in clustering_algos]
#         all_hyper_parametrization3 = itertools.product(embeddings_list3_combinations,  clustering_algos3_cartesian_product)

#         all_hyper_parametrization = [all_hyper_parametrization2, all_hyper_parametrization3]
#         csv = self._create_csv_validation("NumKClusterUnsupervised_new")
#         self._append_to_csv(csv, ['Problem', 'Embedding', 'Clustering', "Pred_Num_Clusters", "True_Num_Clusters", 'Score', 'Samples_size'])
#         for hyperparametrization in all_hyper_parametrization:
#             for cartesian_product in hyperparametrization:
#                 embeddings_used = str.join('/', [emb['name'] for emb in cartesian_product[0]])
#                 clustering_used = str.join('/', [type(clus).__name__ for clus in cartesian_product[1]])

#                 embeddings = list(cartesian_product[0])
#                 clustering_algos = cartesian_product[1]
#                 problem_dicts = []

#                 for i in range(0, len(embeddings)):
#                     if(embeddings[i]['name'] == 'w2v' or embeddings[i]['name'] == 'tfidf'):
#                         embeddings[0], embeddings[i] = embeddings[i], embeddings[0]
#                         break

#                 for embedding in embeddings:
#                     problem_dicts.append(embedding['problemDict'])


#                 for problem , data in problem_dicts[0].items():
#                     for num_clusters in range(2,11):
#                         common_clusters_per_view = []
#                         clusters_n = []

                        
#                         for i in range(len(embeddings)):
#                             common_clusters_per_view.append([])
#                             with open(f"Data/Clusterings/{type(clustering_algos[i]).__name__}_{embeddings[i]['name']}_{problem}_{num_clusters}_labels.bin", 'rb') as fp:
#                                 clusters_n.append(pickle.load(fp))

#                         Y = []
#                         for index in data['indexes']:
#                                 index_in_all_embeddings = True

#                                 for problem_dict in problem_dicts:
#                                     if index not in problem_dict[problem]['indexes']:
#                                         index_in_all_embeddings = False
#                                         break

#                                 if(index_in_all_embeddings):
#                                     i = 0
#                                     for problem_dict in problem_dicts:
#                                         problem_data = problem_dict[problem]
#                                         common_clusters_per_view[i].append(clusters_n[i][problem_data['indexes'].index(index)])
#                                         i+=1
#                                     Y.append(data['Y'][data['indexes'].index(index)])

#                         all_labels = list(set(Y))

                                    
#                         label_n = np.array(common_clusters_per_view).T

#                         grouped_paths = {}
                        
#                         i = 0
#                         for path in label_n:
#                             key = ""

#                             for x in path:
#                                 key+=str(x) + '$'
#                             key = key[:-1]

#                             if(key in grouped_paths):
#                                 grouped_paths[key].append(i)
#                             else:
#                                 grouped_paths[key] = [i]
#                             i+=1
                
#                         mac = MultiAssigmentClustering(grouped_paths, len(cartesian_product[0]), num_clusters)
#                         mac.fit()
#                         best_score = mac.best_score
#                         best_matchings = mac.best_matchings
                        
#                         if(best_score != -1):
#                             best_matchings.pop()
     
#                         with open(f"Data/UnsupervisedResults/{embeddings_used.replace('/', '_')}-{clustering_used.replace('/', '_')}-{num_clusters}-{problem}.json", 'w') as fp:
#                             json.dump({
#                                 "best_score":best_score,
#                                 "best_matchings":best_matchings
#                             }, fp)

#                         print(f"Executing on problem {problem} with {num_clusters} clusters, embeddings {embeddings_used} and clustering methods {clustering_used}: Best score {best_score}")

#                         self._append_to_csv(csv, [problem, embeddings_used, clustering_used, str(num_clusters), str(len(all_labels)), str(best_score), str(label_n.shape[0])])


# d = DetermineKUnsupervised()

# d.determine()


from ValidationMethods import ClusteringValidationMethod, EstimatorValidationMethod
from EmbeddingsLoader import Code2VecEmbeddingsLoader, InfercodeEmbeddingsLoader, W2VEmbeddingsLoader, SafeEmbeddingsLoader, TfidfEmbeddingsLoader, EmbeddingsLoader
from IValidationPipelines import IValidationPipelines
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import itertools
import pickle
import json

def _split_embeddings_data_per_problem(embeddings_loader:EmbeddingsLoader):
        print(f"Extracting the {embeddings_loader.get_name()} embeddings from disk")

        problem_dict = {}

        for solution in embeddings_loader.get_embeddings():
            function_embeddings = np.array(solution["embeddings"])
            solution_embedding = np.mean(function_embeddings, 0)

            problem = solution['label'].split("$")[0]
            if(problem not in problem_dict):
                problem_dict[problem] = {'indexes':[], 'X' : [], 'Y': []}

            problem_dict[problem]['indexes'].append(solution['index'])
            problem_dict[problem]['X'].append(solution_embedding)
            problem_dict[problem]['Y'].append(solution['label'])
        
        embeddings = {}
        embeddings['name'] = embeddings_loader.get_name()
        embeddings['problemDict'] = problem_dict

        return embeddings

embeddings_list = [_split_embeddings_data_per_problem(Code2VecEmbeddingsLoader()), 
                _split_embeddings_data_per_problem(SafeEmbeddingsLoader()), 
                _split_embeddings_data_per_problem(TfidfEmbeddingsLoader()),
                _split_embeddings_data_per_problem(InfercodeEmbeddingsLoader()),
                _split_embeddings_data_per_problem(W2VEmbeddingsLoader())]

import os
import json

for ur_file_name in os.listdir('Data/UnsupervisedResults'):
    with open(f'Data/UnsupervisedResults/{ur_file_name}', 'r') as fp:
        result = json.load(ur_file_name)

# import pandas as pd
# from sklearn.metrics import accuracy_score

# df = pd.read_csv('Data/Validation/NumKClusterUnsupervised.csv')


# df = df.groupby(['Problem', 'Embedding', 'PredNumClusters']).apply(lambda f: f.sort_values('Score', ascending=False).iloc[0])
# df = df.drop(['Problem', 'Embedding', 'PredNumClusters'], axis = 1)
# df = df.reset_index()
# df = df.groupby(['Problem', 'Embedding']).apply(lambda f : f.sort_values('Score', ascending=False).iloc[0])[['PredNumClusters', 'TrueNumClusters', 'Score']]

# df['Target'] = df['PredNumClusters'] == df['TrueNumClusters']
# df = df.groupby(['Embedding']).apply(lambda f: len(f[f['Target']]) / len(f['Target']))

# df.to_csv('Data/Validation/SiluetteScoreUnsupervisedClusteringK.csv')