import itertools
from EmbeddingsLoader import EmbeddingsLoader, W2VEmbeddingsLoader, TfidfEmbeddingsLoader
import numpy as np
from multiview import MVSC

def SplitEmbeddingsDataPerProblem(embeddingsLoader:EmbeddingsLoader):
    problemDict = {}

    for solution in embeddingsLoader.GetEmbeddings():
        functionEmbeddings = np.array(solution["embeddings"])
        solutionEmbedding = np.mean(functionEmbeddings, 0)

        problem = solution['label'].split("$")[0]
        if(problem not in problemDict):
            problemDict[problem] = {'indexes':[], 'X' : []}

        problemDict[problem]['indexes'].append(solution['index'])
        problemDict[problem]['X'].append(solutionEmbedding)
    
    return problemDict

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

class PredictionMethods:
    def predictK(self, problem, embeddings, clusterAlgo, k):
        print(f"Validating using the known k clustering validation method using embeddings {embeddings['name']}:")
        problemDict = embeddings['problemDict']

        predicted_Y = []
        for data in problemDict[problem]:
            X = np.array(data['X'])
        
            clusterAlgo.set_params(n_clusters = k)

            print(f'Predicting problem {problem} using cluster algorithm {type(clusterAlgo).__name__}')

            clusterAlgo.fit(X)
            predicted_Y = clusterAlgo.labels_
        
        return [data['indexes'], predicted_Y]
            


    def predictUnsupervisedVoting(self, problem, embeddingsLoaders, clusterAlgos, classifiers, k):
        print(f'Predicting using the unsupervised method using embeddings {embeddingsLoaders}:')

        if(len(embeddingsLoaders) <=1):
            print("There must be at least two embeddings")
            raise Exception()
        
        if(len(embeddingsLoaders)!= len(clusterAlgos)):
            print("Embeddings and clusters list must have the same size")

        problemDicts = []

        for i in range(0, len(embeddingsLoaders)):
            if(type(embeddingsLoaders[i]) is W2VEmbeddingsLoader or type(embeddingsLoaders[i]) is TfidfEmbeddingsLoader):
                embeddingsLoaders[0], embeddingsLoaders[i] = embeddingsLoaders[i], embeddingsLoaders[0]
                break

        for embeddingsLoader in embeddingsLoaders:
            problemDicts.append(SplitEmbeddingsDataPerProblem(embeddingsLoader))

        data = problemDicts[0][problem]
        
        print(f'Predicting problem {problem} using cluster algorithms {clusterAlgos}')
        
        Xn = []
        Xn_indices = []
        
        for i in range(len(embeddingsLoaders)):
            Xn.append([])

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
                    Xn_indices.append(index)

    
            
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
    
        
        bestScore = -1

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

        X_train = []
        X_train_indices =[]
        for _ in range(len(embeddingsLoaders)):
            X_train.append([])
        
        labels = []
        for c in best_combination:
            for i in groupedPaths[c]:
                for emb in range(len(embeddingsLoaders)):
                    X_train[emb].append(Xn[emb][i])
                X_train_indices.append(i)
                labels.append(c.split("$")[0])

        X_test = []
        X_test_indices = []
        for _ in range(len(embeddingsLoaders)):
            X_test.append([])
        for key, v in groupedPaths.items():
            if(key not in best_combination):
                for i in v:
                    for emb in range(len(embeddingsLoaders)):
                         X_test[emb].append(Xn[emb][i])
                    X_test_indices.append(i)

        disagreedSolutions_X = X_test
        disagreedSolution_indices = X_test_indices
        agreedSolutions_X = X_train
        agreedSolutions_indices = X_train_indices
        agreedSolutions_Y = labels

        lastIterationSize = 0
        while(len(disagreedSolutions_X[0]) > 0 and len(agreedSolutions_Y) - lastIterationSize > 20):
            lastIterationSize = len(agreedSolutions_Y)

            predictedSolutions =[]
            for i in range(len(classifiers)):
                classifiers[i].fit(np.array(agreedSolutions_X[i]), np.array(agreedSolutions_Y))
                predictedSolutions.append(classifiers[i].predict(np.array(disagreedSolutions_X[i])))

            predictedSolutions = np.array(predictedSolutions).T

            nextDisagreedSolutions_X=[]
            nextDisagreedSolutions_indices=[]
            for _ in range(len(embeddingsLoaders)):
                 nextDisagreedSolutions_X.append([])
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
                    agreedSolutions_indices.append(disagreedSolution_indices[solutionIndex])
                    agreedSolutions_Y.append(str(predictedSolutions[solutionIndex][0]))
                else:
                    for i in range(len(classifiers)):
                        nextDisagreedSolutions_X[i].append(disagreedSolutions_X[i][solutionIndex])
                    nextDisagreedSolutions_indices.append(disagreedSolution_indices[solutionIndex])

            disagreedSolutions_X = nextDisagreedSolutions_X
            disagreedSolution_indices = nextDisagreedSolutions_indices


        if(len(disagreedSolutions_X[0]) > 0):
            predictedSolutions =[]
            for i in range(len(classifiers)):
                classifiers[i].fit(np.array(agreedSolutions_X[i]), np.array(agreedSolutions_Y))
                predictedSolutions.append(classifiers[i].predict(np.array(disagreedSolutions_X[i])))

            predictedSolutions = np.array(predictedSolutions).T

            for solutionIndex in range(len(predictedSolutions)):
                for i in range(len(classifiers)):
                    agreedSolutions_X[i].append(disagreedSolutions_X[i][solutionIndex])
                agreedSolutions_indices.append(disagreedSolution_indices[solutionIndex])
                agreedSolutions_Y.append(str(predictedSolutions[solutionIndex][0]))


        solutions_indices = []

        for index in agreedSolutions_indices:
            solutions_indices.append(Xn_indices[index])

        return (solutions_indices, agreedSolutions_Y)

    def predictClusteringMultiView(self, problem, embeddings, k):
        embeddingsUsed = str.join('/', [emb['name'] for emb in embeddings])
        print(f'Predicting using the multiview clustering validation method using embeddings {embeddingsUsed}:')

        problemDicts = []

        for embedding in embeddings:
            problemDicts.append(embedding['problemDict'])
        
        prediction_Y = []
        Xn_indices = []
        for data in problemDicts[0].items():
            print(f'Predicting problem {problem}')
            Xn = []
            
            for i in range(len(embeddings)):
                Xn.append([])

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
                    Xn_indices.append(index)

            multiView = MVSC(k)
            
            views = [np.array(Xi) for Xi in Xn]
            is_distance = np.array([False for Xi in Xn])

            prediction_Y = multiView.fit(views, is_distance).embedding_
    
        return [Xn_indices, prediction_Y]


                

