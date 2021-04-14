from EmbeddingsLoader import EmbeddingsLoader
import numpy as np
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def SplitEmbeddingsDataPerProblem(embeddingsLoader:EmbeddingsLoader):
    problemDict = {}

    for solution in embeddingsLoader.GetEmbeddings():
        functionEmbeddings = np.array(solution["embeddings"])
        solutionEmbedding = np.mean(functionEmbeddings, 0)

        problem = solution['label'].split("$")[0]
        if(problem not in problemDict):
            problemDict[problem] = {'X' : [], 'Y': []}
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



        

        
