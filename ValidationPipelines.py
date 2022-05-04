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

    def KClusteringPipeline(self):
        print("Running KClusteringPipeline...")

        clusteringValidationMethod = ClusteringValidationMethod()

        clusterAlgos = [KMeans(), SpectralClustering(), AgglomerativeClustering(affinity="cosine",linkage="average")]
        embeddingsList = [self.__splitEmbeddingsDataPerProblem(Code2VecEmbeddingsLoader()), 
                             self.__splitEmbeddingsDataPerProblem(SafeEmbeddingsLoader()), 
                             self.__splitEmbeddingsDataPerProblem(TfidfEmbeddingsLoader()),
                             self.__splitEmbeddingsDataPerProblem(InfercodeEmbeddingsLoader()),
                             self.__splitEmbeddingsDataPerProblem(W2VEmbeddingsLoader())]

        csv = self.__createCsvValidation("KCluster")

        self.__appendToCsv(csv, ['Problem', 'Embedding', 'Clustering', 'F1_micro', 'F1_macro', 'F1_weight', 'Samples_size'])

        for embeddings in embeddingsList:
            for clusterAlgo in clusterAlgos:
                problems = clusteringValidationMethod.validateK(embeddings, clusterAlgo)

                for name in problems.keys():
                    Y_true, Y_predicted = problems[name]

                    microF1 = str(f1_score(Y_true, Y_predicted, average='micro'))
                    macroF1 = str(f1_score(Y_true, Y_predicted, average='macro'))
                    averageF1 = str(f1_score(Y_true, Y_predicted, average='weighted'))
                    sampleSize = str(len(Y_true))
                    
                    self.__appendToCsv(csv, [name, embeddings['name'], type(clusterAlgo).__name__, microF1, macroF1, averageF1, sampleSize])

        self.__closeCsv(csv)
        
    def EstimatorPipeline(self):
        print("Running EstimatorPipeline...")
        estimatorValidationMethod = EstimatorValidationMethod()

        estimators = [RandomForestClassifier(), SVC(kernel='rbf'), XGBClassifier(verbosity = 0)]

        embeddingsList = [self.__splitEmbeddingsDataPerProblem(Code2VecEmbeddingsLoader()), 
                             self.__splitEmbeddingsDataPerProblem(SafeEmbeddingsLoader()), 
                             self.__splitEmbeddingsDataPerProblem(TfidfEmbeddingsLoader()),
                             self.__splitEmbeddingsDataPerProblem(InfercodeEmbeddingsLoader()),
                             self.__splitEmbeddingsDataPerProblem(W2VEmbeddingsLoader())]

        csv = self.__createCsvValidation("Estimator")

        self.__appendToCsv(csv, ['Problem', 'Embedding', 'Estimator', 'F1_micro', 'F1_macro', 'F1_weight', 'Samples_size'])

        for embeddings in embeddingsList:
            for estimator in estimators:
                problems = estimatorValidationMethod.validate(embeddings, estimator, 0.2)

                for name in problems.keys():
                    Y_true, Y_predicted = problems[name]

                    microF1 = str(f1_score(Y_true, Y_predicted, average='micro'))
                    macroF1 = str(f1_score(Y_true, Y_predicted, average='macro'))
                    averageF1 = str(f1_score(Y_true, Y_predicted, average='weighted'))
                    sampleSize = str(len(Y_true))
                    
                    self.__appendToCsv(csv, [name, embeddings['name'], type(estimator).__name__, microF1, macroF1, averageF1, sampleSize])

        self.__closeCsv(csv)   

    def SemisupervisedVotingPipeline(self):
        print("Running SemisupervisedVoting pipeline...")
        clusterValidationMethod = ClusteringValidationMethod()

        estimators = [RandomForestClassifier(), SVC(kernel='rbf'), XGBClassifier(verbosity = 0)]
        clusteringAlgos = [KMeans(), SpectralClustering(), AgglomerativeClustering(affinity="cosine",linkage="average")]

        embeddingsList = [self.__splitEmbeddingsDataPerProblem(Code2VecEmbeddingsLoader()), 
                             self.__splitEmbeddingsDataPerProblem(SafeEmbeddingsLoader()), 
                             self.__splitEmbeddingsDataPerProblem(TfidfEmbeddingsLoader()),
                             self.__splitEmbeddingsDataPerProblem(InfercodeEmbeddingsLoader()),
                             self.__splitEmbeddingsDataPerProblem(W2VEmbeddingsLoader())]

        csv = self.__createCsvValidation("SemisupervisedVoting")

        self.__appendToCsv(csv, ['Problem', 'Embedding', 'Clusterings', 'Estimators', 'F1_before_micro', 'F1_before_macro', 'F1_before_weight', 'Samples_before_size', 'F1_after_micro', 'F1_after_macro', 'F1_after_weight', 'Samples_after_size'])

        embeddingsList2Combinations = itertools.combinations(embeddingsList, 2)
        clusteringAlgos2CartesianProduct = [[x,y] for x in clusteringAlgos for y in clusteringAlgos]
        estimator2CartesianProduct = [[x,y] for x in estimators for y in estimators]

        allHyperParametrization2 = itertools.product(embeddingsList2Combinations,  clusteringAlgos2CartesianProduct, estimator2CartesianProduct)

        clusteringAlgos3CartesianProduct = [[x,y,z] for x in clusteringAlgos for y in clusteringAlgos for z in clusteringAlgos]
        estimator3CartesianProduct = [[x,y,z] for x in estimators for y in estimators for z in estimators]
        allHyperParametrization3 = itertools.product([embeddingsList],  clusteringAlgos3CartesianProduct, estimator3CartesianProduct)

        allHyperParametrization = [allHyperParametrization3, allHyperParametrization2]

        for hyperParametrization in allHyperParametrization:
            for cartesianProduct in hyperParametrization:
                embeddingsUsed = str.join('/', [emb['name'] for emb in cartesianProduct[0]])
                clusteringUsed = str.join('/', [type(clus).__name__ for clus in cartesianProduct[1]])
                estimatorUsed = str.join('/', [type(esti).__name__ for esti in cartesianProduct[2]])

                try:
                    problems = clusterValidationMethod.validateSemiSupervised(list(cartesianProduct[0]), cartesianProduct[1], cartesianProduct[2])

                    for name in problems.keys():
                        before, after = problems[name]

                        Y_before_true, Y_before_predicted = before
                        Y_after_true, Y_after_predicted = after

                        microBeforeF1 = str(f1_score(Y_before_true, Y_before_predicted, average='micro'))
                        macroBeforeF1 = str(f1_score(Y_before_true, Y_before_predicted, average='macro'))
                        averageBeforeF1 = str(f1_score(Y_before_true, Y_before_predicted, average='weighted'))
                        sampleBeforeSize = str(len(Y_before_true))

                        microAfterF1 = str(f1_score(Y_after_true, Y_after_predicted, average='micro'))
                        macroAfterF1 = str(f1_score(Y_after_true, Y_after_predicted, average='macro'))
                        averageAfterF1 = str(f1_score(Y_after_true, Y_after_predicted, average='weighted'))
                        sampleAfterSize = str(len(Y_after_true))

                        self.__appendToCsv(csv, [name, embeddingsUsed, clusteringUsed, estimatorUsed, microBeforeF1, macroBeforeF1, averageBeforeF1, sampleBeforeSize, microAfterF1, macroAfterF1, averageAfterF1, sampleAfterSize])
                except Exception as e:
                    print(e)
                    self.__appendToCsv(csv, ["Error", embeddingsUsed, clusteringUsed, estimatorUsed])
        self.__closeCsv(csv)    

    def SemiSupervisedMultiviewSpectralClustering(self):
        print("Running MultiviewSpectralClustering Pipeline...")

        clusteringValidationMethod = ClusteringValidationMethod()

        embeddingsList = [self.__splitEmbeddingsDataPerProblem(Code2VecEmbeddingsLoader()), 
                             self.__splitEmbeddingsDataPerProblem(SafeEmbeddingsLoader()), 
                             self.__splitEmbeddingsDataPerProblem(TfidfEmbeddingsLoader()),
                             self.__splitEmbeddingsDataPerProblem(InfercodeEmbeddingsLoader()),
                             self.__splitEmbeddingsDataPerProblem(W2VEmbeddingsLoader())]

        csv = self.__createCsvValidation("MultiViewSpectralClustering")

        self.__appendToCsv(csv, ['Problem', 'Embeddings', 'F1_micro', 'F1_macro', 'F1_weight', 'Samples_size'])

        embeddingsList2Combinations = itertools.combinations(embeddingsList, 2)
        embeddingsList3 = [embeddingsList]

        allHyperParametrization = [ embeddingsList2Combinations,embeddingsList3]

        for hyperParametrization in allHyperParametrization:
            for embeddings in hyperParametrization:
                    embeddingsUsed = str.join('/', [emb['name'] for emb in embeddings])
                    problems = clusteringValidationMethod.validateClusteringMultiView(embeddings)

                    for name in problems.keys():
                        Y_true, Y_predicted = problems[name]

                        microF1 = str(f1_score(Y_true, Y_predicted, average='micro'))
                        macroF1 = str(f1_score(Y_true, Y_predicted, average='macro'))
                        averageF1 = str(f1_score(Y_true, Y_predicted, average='weighted'))
                        sampleSize = str(len(Y_true))
                        
                        self.__appendToCsv(csv, [name, embeddingsUsed, microF1, macroF1, averageF1, sampleSize])

    def __createCsvValidation(self, name):
        if(not os.path.exists('Data/Validation')):
            os.mkdir('Data/Validation')

        csv = open(f'Data/Validation/{name}.csv', "w")

        return csv

    def __appendToCsv(self, csv, row):
        csv.write(f"{str.join(',', row)}\n")
    
    def __closeCsv(self, csv):
        csv.close()

    def __splitEmbeddingsDataPerProblem(self, embeddingsLoader:EmbeddingsLoader):

        print(f"Extracting the {embeddingsLoader.GetName()} embeddings from disk")

        problemDict = {}

        for solution in embeddingsLoader.GetEmbeddings():
            functionEmbeddings = np.array(solution["embeddings"])
            solutionEmbedding = np.mean(functionEmbeddings, 0)

            problem = solution['label'].split("$")[0]
            if(problem not in problemDict):
                problemDict[problem] = {'indexes':[], 'X' : [], 'Y': []}

            problemDict[problem]['indexes'].append(solution['index'])
            problemDict[problem]['X'].append(solutionEmbedding)
            problemDict[problem]['Y'].append(solution['label'])
        
        embeddings = {}
        embeddings['name'] = embeddingsLoader.GetName()
        embeddings['problemDict'] = problemDict

        return embeddings
