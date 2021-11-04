from EmbeddingsLoader import EmbeddingsLoader, W2VEmbeddingsLoader, TfidfEmbeddingsLoader
import numpy as np
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import math
from multiview import MVSC


def checkCombinationAllDistinct(combination):

    for x in combination:
        for y in combination:
            if x != y:
                clustersX = x.split("$")
                clustersY = y.split("$")

                for i in range(len(clustersX)):
                    if(clustersX[i] == clustersY[i]):
                        return False
    return True

class ClusteringValidationMethod:
    
    def validateK(self, embeddings, clusterAlgo):
        print(f"Validating using the known k clustering validation method using embeddings {embeddings['name']}:")
        problemDict = embeddings['problemDict']
        
        problemValidationData = {}
        for problem , data in problemDict.items():
            X = np.array(data['X'])
            Y = np.array(data['Y'])

            allLabels = list(set(Y))
            k = len(allLabels)
        
            clusterAlgo.set_params(n_clusters = k)

            print(f'Validating problem {problem} using cluster algorithm {type(clusterAlgo).__name__}')

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

            problemValidationData[problem] = (Y, bestPermutation)

        return problemValidationData
        

    def validateSemiSupervised(self, embeddings, clusterAlgos, classifiersUnfit):
        embeddingsUsed = str.join('/', [emb['name'] for emb in embeddings])
        clusterAlgosUsed = str.join('/', [type(clusterAlgo).__name__ for clusterAlgo in clusterAlgos])
        classifiersUsed = str.join('/', [type(classifier).__name__ for classifier in classifiersUnfit])
        print(f'Validating using the semiSupervised validation method using embeddings {embeddingsUsed}:')

        if(len(embeddings) <=1):
            print("There must be at least two embeddings")
            raise Exception()

        problemDicts = []

        for i in range(0, len(embeddings)):
            if(embeddings[i]['name'] == 'w2v' or embeddings[i]['name'] == 'tfidf'):
                embeddings[0], embeddings[i] = embeddings[i], embeddings[0]
                break

        for embedding in embeddings:
            problemDicts.append(embedding['problemDict'])
    
        problemValidationData ={}
        for problem , data in problemDicts[0].items():
            print(f'Validating problem {problem} using cluster algorithms {clusterAlgosUsed} and estimators {classifiersUsed}')

            classifiers = [clone(classifier) for classifier in classifiersUnfit]

            X_all_indices = data['indexes']
            Y_all_indices = data['Y']
            
            Xn = []
            Xn_test = []
            
            for i in range(len(embeddings)):
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
                            Xn[i].append(problemData['X'][problemData['indexes'].index(index)])
                            i+=1

                        Y.append(data['Y'][data['indexes'].index(index)])

            
            allLabels = list(set(Y))
            k = len(allLabels)

            for clusterAlgo in clusterAlgos:
                clusterAlgo.set_params(n_clusters = k)

            label_n = []

            index = 0
            for X in Xn:
                label_n.append(clusterAlgos[index].fit(X).labels_)
                index+=1

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
     
            bestScore = - 1
            best_combination = []
            groupedPathsKeys = groupedPaths.keys()

            for combination in itertools.combinations(groupedPathsKeys, k):
                if(checkCombinationAllDistinct(combination)):
                    score = 0

                    for key in combination:
                        score+= len(groupedPaths[key])
                    
                    if(score > bestScore):
                        bestScore = score
                        best_combination = combination

            if(bestScore == -1):
                raise Exception(f"Couldn't find {k} vectors, make k smaller")

            labels = []
            X_result = []
            for _ in range(len(embeddings)):
                X_result.append([])

            Y_result = []
            for comb in best_combination:
                for i in groupedPaths[comb]:
                    labels.append(label_n[i][0])
                    Y_result.append(Y[i])

                    for j in range(len(embeddings)):
                        X_result[j].append(Xn[j][i])

            Y_test = []
            for key, v in groupedPaths.items():
                if(key not in best_combination):
                    for i in v:
                        for j in range(len(embeddings)):
                            Xn_test[j].append(Xn[j][i])
                        Y_test.append(Y[i])


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

            beforeClassifiers = (Y_result.copy(), bestPermutation)

            disagreedSolutions_X = Xn_test
            disagreedSolutions_Y = Y_test
            agreedSolutions_X = X_result
            agreedSolutions_Y = bestPermutation.tolist()

            lastIterationSize = 0
            while(len(disagreedSolutions_Y) > 0 and len(agreedSolutions_Y) - lastIterationSize > 20):

                lastIterationSize = len(agreedSolutions_Y)

                predictedSolutions =[]
                for i in range(len(classifiers)):
                    classifiers[i].fit(np.array(agreedSolutions_X[i]), np.array(agreedSolutions_Y))
                    predictedSolutions.append(classifiers[i].predict(np.array(disagreedSolutions_X[i])))

                predictedSolutions = np.array(predictedSolutions).T

                nextDisagreedSolutions_X=[[] for _ in range(len(embeddings))]
                nextDisagreedSolutions_Y = []
                for solutionIndex in range(len(predictedSolutions)):
                    allAgree = True

                    for i in range(1, len(classifiers)):
                        if(not np.char.equal(predictedSolutions[solutionIndex][i-1], predictedSolutions[solutionIndex][i])):
                            allAgree = False
                            break

                    if(allAgree== True):
                        for i in range(len(classifiers)):
                            agreedSolutions_X[i].append(disagreedSolutions_X[i][solutionIndex])
                        agreedSolutions_Y.append(str(predictedSolutions[solutionIndex][0]))
                        Y_result.append(disagreedSolutions_Y[solutionIndex])
                    else:
                        for i in range(len(classifiers)):
                            nextDisagreedSolutions_X[i].append(disagreedSolutions_X[i][solutionIndex])
                        nextDisagreedSolutions_Y.append(disagreedSolutions_Y[solutionIndex])

                disagreedSolutions_X = nextDisagreedSolutions_X
                disagreedSolutions_Y = nextDisagreedSolutions_Y

                afterClassifiers = (agreedSolutions_Y,  Y_result)

                problemValidationData[problem] = (beforeClassifiers, afterClassifiers)

        return problemValidationData   

    def validateClusteringMultiView(self, embeddings):
        embeddingsUsed = str.join('/', [emb['name'] for emb in embeddings])
        print(f'Validating using the multiview clustering validation method using embeddings {embeddingsUsed}:')

        problemDicts = []

        for embedding in embeddings:
            problemDicts.append(embedding['problemDict'])

        problemValidationData = {}
           
        for problem , data in problemDicts[0].items():
            print(f'Validating problem {problem}')
            Xn = []
            
            for i in range(len(embeddings)):
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

            views = [np.array(Xi) for Xi in Xn]
            is_distance = np.array([False for Xi in Xn])
            
            labels = multiView.fit(views, is_distance).embedding_

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

            problemValidationData[problem] = (Y, bestPermutation)
    
        return problemValidationData


class EstimatorValidationMethod:

    def validate(self, embeddings, estimatorModel, testSize):
        print(f"Validating using the estimator validation method using embeddings {embeddings['name']}:")
        problemDict = embeddings['problemDict']
        
        problemValidationData = {}
        for problem , data in problemDict.items():

            estimator = clone(estimatorModel)

            X = np.array(data['X'])
            Y = np.array(data['Y'])

            print(f'Validating problem {problem} using estimator {type(estimator).__name__}')
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = testSize, random_state=42)
            
            estimator = clone(estimator)

            print(f"Cross validation score : {cross_val_score(estimator, X_train, Y_train)}")

            estimator.fit(X_train, Y_train)

            labels = estimator.predict(X_test)

            problemValidationData[problem] = (Y_test, labels)

        return problemValidationData

                


        

        
