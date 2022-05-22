
from ValidationMethods import ClusteringValidationMethod, EstimatorValidationMethod
from EmbeddingsLoader import Code2VecEmbeddingsLoader, InfercodeEmbeddingsLoader, W2VEmbeddingsLoader, SafeEmbeddingsLoader, TfidfEmbeddingsLoader, EmbeddingsLoader
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import f1_score
import os
import itertools


class ValidationPipelines:
    def k_clustering_pipeline(self):
        print("Running KClusteringPipeline...")

        clustering_validation_method = ClusteringValidationMethod()

        cluster_algos = [KMeans(), SpectralClustering(), AgglomerativeClustering(affinity="cosine",linkage="average")]
        embeddings_list = [self._split_embeddings_data_per_problem(Code2VecEmbeddingsLoader()), 
                             self._split_embeddings_data_per_problem(SafeEmbeddingsLoader()), 
                             self._split_embeddings_data_per_problem(TfidfEmbeddingsLoader()),
                             self._split_embeddings_data_per_problem(InfercodeEmbeddingsLoader()),
                             self._split_embeddings_data_per_problem(W2VEmbeddingsLoader())]

        csv = self._create_csv_validation("KCluster")

        self._append_to_csv(csv, ['Problem', 'Embedding', 'Clustering', 'F1_micro', 'F1_macro', 'F1_weight', 'Samples_size'])

        for embeddings in embeddings_list:
            for clusterAlgo in cluster_algos:
                problems = clustering_validation_method.validateK(embeddings, clusterAlgo)

                for name in problems.keys():
                    Y_true, Y_predicted = problems[name]

                    microF1 = str(f1_score(Y_true, Y_predicted, average='micro'))
                    macroF1 = str(f1_score(Y_true, Y_predicted, average='macro'))
                    averageF1 = str(f1_score(Y_true, Y_predicted, average='weighted'))
                    sampleSize = str(len(Y_true))
                    
                    self._append_to_csv(csv, [name, embeddings['name'], type(clusterAlgo).__name__, microF1, macroF1, averageF1, sampleSize])

        self._close_csv(csv)
        
    def estimator_pipeline(self):
        print("Running EstimatorPipeline...")
        estimatorValidationMethod = EstimatorValidationMethod()

        estimators = [RandomForestClassifier(), SVC(kernel='rbf'), XGBClassifier(verbosity = 0)]

        embeddingsList = [self._split_embeddings_data_per_problem(Code2VecEmbeddingsLoader()), 
                             self._split_embeddings_data_per_problem(SafeEmbeddingsLoader()), 
                             self._split_embeddings_data_per_problem(TfidfEmbeddingsLoader()),
                             self._split_embeddings_data_per_problem(InfercodeEmbeddingsLoader()),
                             self._split_embeddings_data_per_problem(W2VEmbeddingsLoader())]

        csv = self._create_csv_validation("Estimator")

        self._append_to_csv(csv, ['Problem', 'Embedding', 'Estimator', 'F1_micro', 'F1_macro', 'F1_weight', 'Samples_size'])

        for embeddings in embeddingsList:
            for estimator in estimators:
                problems = estimatorValidationMethod.validate(embeddings, estimator, 0.2)

                for name in problems.keys():
                    Y_true, Y_predicted = problems[name]

                    microF1 = str(f1_score(Y_true, Y_predicted, average='micro'))
                    macroF1 = str(f1_score(Y_true, Y_predicted, average='macro'))
                    averageF1 = str(f1_score(Y_true, Y_predicted, average='weighted'))
                    sampleSize = str(len(Y_true))
                    
                    self._append_to_csv(csv, [name, embeddings['name'], type(estimator).__name__, microF1, macroF1, averageF1, sampleSize])

        self._close_csv(csv)   

    def unsupervised_voting_pipeline(self):
        print("Running SemisupervisedVoting pipeline...")
        cluster_validation_method = ClusteringValidationMethod()

        estimators = [RandomForestClassifier(), SVC(kernel='rbf'), XGBClassifier(verbosity = 0)]
        clustering_algos = [KMeans(), SpectralClustering(), AgglomerativeClustering(affinity="cosine",linkage="average")]

        embeddings_list = [self._split_embeddings_data_per_problem(Code2VecEmbeddingsLoader()), 
                             self._split_embeddings_data_per_problem(SafeEmbeddingsLoader()), 
                             self._split_embeddings_data_per_problem(TfidfEmbeddingsLoader()),
                             self._split_embeddings_data_per_problem(InfercodeEmbeddingsLoader()),
                             self._split_embeddings_data_per_problem(W2VEmbeddingsLoader())]

        csv = self._create_csv_validation("SemisupervisedVoting")

        self._append_to_csv(csv, ['Problem', 'Embedding', 'Clusterings', 'Estimators', 'F1_before_micro', 'F1_before_macro', 'F1_before_weight', 'Samples_before_size', 'F1_after_micro', 'F1_after_macro', 'F1_after_weight', 'Samples_after_size'])

        embeddings_list2_combinations = itertools.combinations(embeddings_list, 2)
        clustering_algos2_cartesian_product = [[x,y] for x in clustering_algos for y in clustering_algos]
        estimator2_cartesian_product = [[x,y] for x in estimators for y in estimators]
        all_hyper_parametrization2 = itertools.product(embeddings_list2_combinations,  clustering_algos2_cartesian_product, estimator2_cartesian_product)

        embeddings_list3_combinations = itertools.combinations(embeddings_list, 3)
        clustering_algos3_cartesian_product = [[x,y,z] for x in clustering_algos for y in clustering_algos for z in clustering_algos]
        estimator3_cartesian_product = [[x,y,z] for x in estimators for y in estimators for z in estimators]
        all_hyper_parametrization3 = itertools.product(embeddings_list3_combinations,  clustering_algos3_cartesian_product, estimator3_cartesian_product)

        all_hyper_parametrization = [all_hyper_parametrization2, all_hyper_parametrization3]

        for hyper_parametrization in all_hyper_parametrization:
            for cartesian_product in hyper_parametrization:
                embeddings_used = str.join('/', [emb['name'] for emb in cartesian_product[0]])
                clustering_used = str.join('/', [type(clus).__name__ for clus in cartesian_product[1]])
                estimator_used = str.join('/', [type(esti).__name__ for esti in cartesian_product[2]])
                print(len(cartesian_product))
                try:
                    problems = cluster_validation_method.validate_unsupervised(list(cartesian_product[0]), cartesian_product[1], cartesian_product[2])

                    for name in problems.keys():
                        before, after = problems[name]

                        Y_before_true, Y_before_predicted = before
                        Y_after_true, Y_after_predicted = after

                        micro_before_F1 = str(f1_score(Y_before_true, Y_before_predicted, average='micro'))
                        macro_before_F1 = str(f1_score(Y_before_true, Y_before_predicted, average='macro'))
                        average_before_F1 = str(f1_score(Y_before_true, Y_before_predicted, average='weighted'))
                        sample_before_size = str(len(Y_before_true))

                        micro_after_F1 = str(f1_score(Y_after_true, Y_after_predicted, average='micro'))
                        macro_after_F1 = str(f1_score(Y_after_true, Y_after_predicted, average='macro'))
                        average_after_F1 = str(f1_score(Y_after_true, Y_after_predicted, average='weighted'))
                        sample_after_size = str(len(Y_after_true))

                        self._append_to_csv(csv, [name, embeddings_used, clustering_used, estimator_used, micro_before_F1, macro_before_F1, average_before_F1, sample_before_size, micro_after_F1, macro_after_F1, average_after_F1, sample_after_size])
                except Exception as e:
                    print(e)
                    self._append_to_csv(csv, ["Error", embeddings_used, clustering_used, estimator_used])
        self._close_csv(csv)    

    def semi_supervised_multiview_spectral_clustering(self):
        print("Running MultiviewSpectralClustering Pipeline...")

        clustering_validation_method = ClusteringValidationMethod()

        embeddings_list = [self._split_embeddings_data_per_problem(Code2VecEmbeddingsLoader()), 
                             self._split_embeddings_data_per_problem(SafeEmbeddingsLoader()), 
                             self._split_embeddings_data_per_problem(TfidfEmbeddingsLoader()),
                             self._split_embeddings_data_per_problem(InfercodeEmbeddingsLoader()),
                             self._split_embeddings_data_per_problem(W2VEmbeddingsLoader())]

        csv = self._create_csv_validation("MultiViewSpectralClustering")

        self._append_to_csv(csv, ['Problem', 'Embeddings', 'F1_micro', 'F1_macro', 'F1_weight', 'Samples_size'])

        embeddings_list2_combinations = itertools.combinations(embeddings_list, 2)
        embeddings_list3_combinations = itertools.combinations(embeddings_list, 3)
        embeddings_list4_combinations = itertools.combinations(embeddings_list, 4)
        embeddings_list5 = [embeddings_list]

        all_hyper_parametrization = [ embeddings_list2_combinations, embeddings_list3_combinations, embeddings_list4_combinations, embeddings_list5]

        for hyper_parametrization in all_hyper_parametrization:
            for embeddings in hyper_parametrization:
                embeddings_used = str.join('/', [emb['name'] for emb in embeddings])
                problems = clustering_validation_method.validateClusteringMultiView(embeddings)

                for name in problems.keys():
                    Y_true, Y_predicted = problems[name]

                    micro_F1 = str(f1_score(Y_true, Y_predicted, average='micro'))
                    macro_F1 = str(f1_score(Y_true, Y_predicted, average='macro'))
                    average_F1 = str(f1_score(Y_true, Y_predicted, average='weighted'))
                    sample_size = str(len(Y_true))
                    
                    self._append_to_csv(csv, [name, embeddings_used, micro_F1, macro_F1, average_F1, sample_size])

    def _create_csv_validation(self, name):
        if(not os.path.exists('Data/Validation')):
            os.mkdir('Data/Validation')

        csv = open(f'Data/Validation/{name}.csv', "w")

        return csv

    def _append_to_csv(self, csv, row):
        csv.write(f"{str.join(',', row)}\n")
    
    def _close_csv(self, csv):
        csv.close()

    def _split_embeddings_data_per_problem(self, embeddings_loader:EmbeddingsLoader):

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
