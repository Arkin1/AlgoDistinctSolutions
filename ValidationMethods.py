from EmbeddingsLoader import EmbeddingsLoader, W2VEmbeddingsLoader, TfidfEmbeddingsLoader
import numpy as np
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import math
from multiview import MVSC

def SplitEmbeddingsDataPerProblem(embeddingsLoader:EmbeddingsLoader):
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
    
    return problemDict

class ClusteringValidationMethod:
    
    def validateK(self, embeddingsLoader:EmbeddingsLoader, clusterAlgos):
        print(f'Validating using the known k clustering validation method using embeddings {embeddingsLoader}:')
        problemDict = SplitEmbeddingsDataPerProblem(embeddingsLoader)
        
        for problem , data in problemDict.items():
            X = np.array(data['X'])
            Y = np.array(data['Y'])

            for clusterAlgo in clusterAlgos:
                allLabels = list(set(Y))
                k = len(allLabels)
            
                clusterAlgo.set_params(n_clusters = k)

                print(f'Validating problem {problem} using cluster algorithm {clusterAlgo}')

                clusterAlgo.fit(X)
                labels = clusterAlgo.labels_
                
                bestScore = -1

                for permutation in itertools.permutations(range(0, k)):
                    permutedLabels = []

                    for index in range(len(labels)):
                        permutedLabels.append(allLabels[permutation[labels[index]]])
                    
                    permutedLabels = np.array(permutedLabels)
                    score = f1_score(permutedLabels, Y, average='weighted')

                    if(score > bestScore):
                        bestScore = score
                        bestPermutation = permutedLabels

                print(classification_report(Y, bestPermutation))

    def listCommonValue(self, list1, list2):
        if(len(list1) != len(list2)):
            raise Exception()
        
        for i in range(len(list1)):
            if(list1[i] == list2[i]):
                return True
        
        return False
        


    def validateSemiSupervised(self, embeddingsLoaders, clusterAlgo, classifiers):
        print(f'Validating using the semiSupervised validation method using embeddings {embeddingsLoaders}:')

        if(len(embeddingsLoaders) <=1):
            print("There must be at least two embeddings")
            raise Exception()

        problemDicts = []

        for i in range(0, len(embeddingsLoaders)):
            if(type(embeddingsLoaders[i]) is W2VEmbeddingsLoader or type(embeddingsLoaders[i]) is TfidfEmbeddingsLoader):
                embeddingsLoaders[0], embeddingsLoaders[i] = embeddingsLoaders[i], embeddingsLoaders[0]
                break

        for embeddingsLoader in embeddingsLoaders:
            problemDicts.append(SplitEmbeddingsDataPerProblem(embeddingsLoader))
    
           
        for problem , data in problemDicts[0].items():
            print(f'Validating problem {problem} using cluster algorithm {clusterAlgo}')

            X_all_indices = data['indexes']
            Y_all_indices = data['Y']

            X_train_indices, X_test_indices, Y_train, Y_test = train_test_split(X_all_indices, Y_all_indices, test_size = 0.3, random_state=42)
            
            Xn = []
            Xn_test = []
            
            for i in range(len(embeddingsLoaders)):
                Xn.append([])
                Xn_test.append([])

            Y = []
            Y_test=[]
            for index in data['indexes']:
                    indexInAllEmbeddings = True

                    for problemDict in problemDicts:
                        if index not in problemDict[problem]['indexes']:
                            indexInAllEmbeddings = False
                            break

                    if(indexInAllEmbeddings):
                        i = 0
                        for problemDict in problemDicts:
                            problemData = problemDict[problem]
                            if(index in X_train_indices):
                                 Xn[i].append(problemData['X'][problemData['indexes'].index(index)])
                            else:
                                 Xn_test[i].append(problemData['X'][problemData['indexes'].index(index)])
                            i+=1

                        if(index in X_train_indices):
                             Y.append(data['Y'][data['indexes'].index(index)])
                        else:
                             Y_test.append(data['Y'][data['indexes'].index(index)])

            
            allLabels = list(set(Y))
            k = len(allLabels)

            clusterAlgo.set_params(n_clusters = k)

            label_n = []

            for X in Xn:
                label_n.append(clusterAlgo.fit(X).labels_)

            label_n = np.array(label_n).T

            groupedPaths = {}
            
            i = 0
            for path in label_n:
                key = ""

                for x in path:
                    key+=str(x) + '$'
                key = key[:-1]

                if(key in groupedPaths):
                    groupedPaths[key].append(i)
                else:
                    groupedPaths[key] = [i]
                i+=1
     
            
            bestScore = -1
            best_key1 = -1
            best_key2 = -1
            for key1 in groupedPaths:
                for key2 in groupedPaths:
                    if(key1 != key2 and self.listCommonValue(key1.split("$"), key2.split("$")) == False):
                        if(len(groupedPaths[key1]) + len(groupedPaths[key2]) > bestScore):
                            bestScore = len(groupedPaths[key1]) + len(groupedPaths[key2])
                            best_key1 = key1
                            best_key2 = key2

            labels = []
            X_result = []
            Y_result = []
            for i in groupedPaths[best_key1]:
                labels.append(label_n[i][0])
                X_result.append(Xn[0][i])
                Y_result.append(Y[i])

            for i in groupedPaths[best_key2]:
                labels.append(label_n[i][0])
                X_result.append(Xn[0][i])
                Y_result.append(Y[i])

            bestScore = -1
            for permutation in itertools.permutations(range(0, k)):
                permutedLabels = []

                for index in range(len(labels)):
                    permutedLabels.append(allLabels[permutation[labels[index]]])
                
                permutedLabels = np.array(permutedLabels)
                score = f1_score(permutedLabels, Y_result, average='weighted')

                if(score > bestScore):
                    bestScore = score
                    bestPermutation = permutedLabels



            print(classification_report(Y_result, bestPermutation))

            for estimator in classifiers:
                print(f"Validating the semi supervised clustering method with classifier {estimator}")
                estimator.fit(np.array(X_result), np.array(bestPermutation))

                X_test = Xn_test[0]

                labels = estimator.predict(np.array(X_test))

                print("Validating the semi supervised clustering method using estimator")

                print(classification_report(Y_test, labels))

    def validateClusteringMultiView(self, embeddingsLoaders):

        print(f'Validating using the multiview clustering validation method using embeddings {embeddingsLoaders}:')

        problemDicts = []

        for embeddingsLoader in embeddingsLoaders:
            problemDicts.append(SplitEmbeddingsDataPerProblem(embeddingsLoader))
           
        for problem , data in problemDicts[0].items():
            print(f'Validating problem {problem}')
            Xn = []
            
            for i in range(len(embeddingsLoaders)):
                Xn.append([])

            Y = []
            for index in data['indexes']:
                indexInAllEmbeddings = True

                for problemDict in problemDicts:
                    if index not in problemDict[problem]['indexes']:
                        indexInAllEmbeddings = False
                        break

                if(indexInAllEmbeddings):
                    i = 0
                    for problemDict in problemDicts:
                        problemData = problemDict[problem]
                        Xn[i].append(problemData['X'][problemData['indexes'].index(index)])
                        i+=1
                    Y.append(data['Y'][data['indexes'].index(index)])
            
            allLabels = list(set(Y))
            k = len(allLabels)

            multiView = MVSC(k)

            labels = multiView.fit([np.array(Xn[0]),np.array(Xn[1])], np.array([False, False])).embedding_

            bestScore = -1
            for permutation in itertools.permutations(range(0, k)):
                permutedLabels = []

                for index in range(len(labels)):
                    permutedLabels.append(allLabels[permutation[labels[index]]])
                
                permutedLabels = np.array(permutedLabels)
                score = f1_score(permutedLabels, Y, average='weighted')

                if(score > bestScore):
                    bestScore = score
                    bestPermutation = permutedLabels

            print(classification_report(Y, bestPermutation))



        
class EstimatorValidationMethod:

    def validate(self, embeddingsLoader:EmbeddingsLoader, estimators, testSize):
        print(f'Validating using the estimator validation method using embeddings {embeddingsLoader}:')
        problemDict = SplitEmbeddingsDataPerProblem(embeddingsLoader)
        
        for problem , data in problemDict.items():
            X = np.array(data['X'])
            Y = np.array(data['Y'])

            for estimator in estimators:
                print(f'Validating problem {problem} using estimator {estimator}')
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = testSize, random_state=42)

                print(f"Cross validation score : {cross_val_score(estimator, X_train, Y_train)}")

                estimator.fit(X_train, Y_train)

                labels = estimator.predict(X_test)

                print(classification_report(Y_test, labels))

                


        

        
