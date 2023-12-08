import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv('Data/Validation/NumKClusterUnsupervised.csv')

df = df[df.PredNumClusters <= 4]

def precision_at_k(df:pd.DataFrame, k):
    sorted_df = df.sort_values('Score', ascending=False)

    true = 0
    for idx in range(k):
        if(sorted_df.iloc[idx]['PredNumClusters'] == sorted_df.iloc[idx]['TrueNumClusters']):
            true+=1
    
    return pd.DataFrame({f"Precision@{k}": [true / k]})



df_at_1 = df.groupby(['Problem']).apply(lambda x: precision_at_k(x, 1))
df_at_3 = df.groupby(['Problem']).apply(lambda x: precision_at_k(x, 3))
df_at_5 = df.groupby(['Problem']).apply(lambda x: precision_at_k(x, 5))
df_at_10 = df.groupby(['Problem']).apply(lambda x: precision_at_k(x, 10))

df = df_at_1.merge(df_at_3, on='Problem').merge(df_at_5, on='Problem').merge(df_at_10, on='Problem')
df.to_csv('Data/Validation/SiluetteScoreUnsupervisedClusteringK.csv')




# from ValidationMethods import ClusteringValidationMethod, EstimatorValidationMethod
# from EmbeddingsLoader import Code2VecEmbeddingsLoader, InfercodeEmbeddingsLoader, W2VEmbeddingsLoader, SafeEmbeddingsLoader, TfidfEmbeddingsLoader, EmbeddingsLoader
# from IValidationPipelines import IValidationPipelines
# import numpy as np
# from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
# import itertools
# import pickle

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
#         csv = self._create_csv_validation("NumKClusterUnsupervised")
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
#                     for num_clusters in range(2,5):
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

#                         def check_distinct_combination(combination):
#                             for x in combination:
#                                 for y in combination:
#                                     if x != y:
#                                         clustersX = x.split("$")
#                                         clustersY = y.split("$")

#                                         for i in range(len(clustersX)):
#                                             if(clustersX[i] == clustersY[i]):
#                                                 return False
#                             return True
                                    
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
                
#                         best_score = - 1
#                         best_combination = []
#                         grouped_paths_keys = grouped_paths.keys()

#                         for combination in itertools.combinations(grouped_paths_keys, num_clusters):
#                             if(check_distinct_combination(combination)):
#                                 score = 0

#                                 for key in combination:
#                                     score += len(grouped_paths[key])
                                
#                                 if(score > best_score):
#                                     best_score = score
#                                     best_combination = combination

#                         best_score = best_score
#                         print(f"Executing on problem {problem} with {num_clusters} clusters, embeddings {embeddings_used} and clustering methods {clustering_used}: Best score {best_score}")

#                         self._append_to_csv(csv, [problem, embeddings_used, clustering_used, str(num_clusters), str(len(all_labels)), str(best_score), str(label_n.shape[0])])


# d = DetermineKUnsupervised()

# d.determine()

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