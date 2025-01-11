import itertools
from EmbeddingsLoader import EmbeddingsLoader, W2VEmbeddingsLoader, TfidfEmbeddingsLoader
import numpy as np


def split_problem_embeddings(embeddingsLoader:EmbeddingsLoader):
    problemDict = {}

    for solution in embeddingsLoader.get_embeddings():
        functionEmbeddings = np.array(solution["embeddings"])
        solutionEmbedding = np.mean(functionEmbeddings, 0)

        problem = solution['label'].split("$")[0]
        if(problem not in problemDict):
            problemDict[problem] = {'indexes':[], 'X' : []}

        problemDict[problem]['indexes'].append(solution['index'])
        problemDict[problem]['X'].append(solutionEmbedding)
    
    return problemDict

def check_distinct_combination(combination):
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
    def _validateK(self, problem, embeddings, cluster_algo):
        print(f"Validating using the known k clustering validation method using embeddings {embeddings['name']}:")
        problem_dict = embeddings['problemDict']
        
        predicted_Y = []
        for data in problem_dict[problem]:
            X = np.array(data['X'])
            Y = np.array(data['Y'])

            all_labels = list(set(Y))
            k = len(all_labels)
        
            cluster_algo.set_params(n_clusters = k)

            print(f'Predicting problem {problem} using cluster algorithm {type(cluster_algo).__name__}')

            cluster_algo.fit(X)
        
        return [data['indexes'], predicted_Y]

    def _predict_unsupervised(self, problem, embeddings_loaders, cluster_algos, classifiers, k):
        print(f'Predicting using the semi-supervised method using embeddings {embeddings_loaders}:')

        if(len(embeddings_loaders) <=1):
            print("There must be at least two embeddings")
            raise Exception()
        
        if(len(embeddings_loaders)!= len(cluster_algos)):
            print("Embeddings and clusters list must have the same size")

        problem_dicts = []

        for i in range(0, len(embeddings_loaders)):
            if(type(embeddings_loaders[i]) is W2VEmbeddingsLoader or type(embeddings_loaders[i]) is TfidfEmbeddingsLoader):
                embeddings_loaders[0], embeddings_loaders[i] = embeddings_loaders[i], embeddings_loaders[0]
                break

        for embeddings_loader in embeddings_loaders:
            problem_dicts.append(split_problem_embeddings(embeddings_loader))

        data = problem_dicts[0][problem]
        
        print(f'Predicting problem {problem} using cluster algorithms {cluster_algos}')
        
        Xn = []
        Xn_indices = []
        
        for i in range(len(embeddings_loaders)):
            Xn.append([])

        for index in data['indexes']:
                index_in_all_embeddings = True

                for problem_dict in problem_dicts:
                    if index not in problem_dict[problem]['indexes']:
                        index_in_all_embeddings = False
                        break

                if(index_in_all_embeddings):
                    i = 0
                    for problem_dict in problem_dicts:
                        problem_data = problem_dict[problem]
                        Xn[i].append(problem_data['X'][problem_data['indexes'].index(index)])
                        i+=1
                    Xn_indices.append(index)
            
        for cluster_algo in cluster_algos:
            cluster_algo.set_params(n_clusters = k)

        label_n = []
        index = 0
        for X in Xn:
            label_n.append(cluster_algos[index].fit(X).labels_)
            index+=1

        label_n = np.array(label_n).T

        grouped_paths = {}
        
        i = 0
        for path in label_n:
            key = ""

            for x in path:
                key+=str(x) + '$'
            key = key[:-1]

            if(key in grouped_paths):
                grouped_paths[key].append(i)
            else:
                grouped_paths[key] = [i]
            i+=1
        
        best_score = -1
        best_combination = []
        grouped_paths_keys = grouped_paths.keys()

        for combination in itertools.combinations(grouped_paths_keys, k):
            if(check_distinct_combination(combination)):
                score = 0

                for key in combination:
                    score+= len(grouped_paths[key])
                
                if(score > best_score):
                    best_score = score
                    best_combination = combination

        X_train = []
        X_train_indices =[[] * len(embeddings_loaders)]

        labels = []
        for c in best_combination:
            for i in grouped_paths[c]:
                for emb in range(len(embeddings_loaders)):
                    X_train[emb].append(Xn[emb][i])
                X_train_indices.append(i)
                labels.append(c.split("$")[0])

        X_test = [[] * len(embeddings_loaders)]
        X_test_indices = []

        for key, v in grouped_paths.items():
            if(key not in best_combination):
                for i in v:
                    for emb in range(len(embeddings_loaders)):
                         X_test[emb].append(Xn[emb][i])
                    X_test_indices.append(i)

        disagreed_solutions_X = X_test
        disagreed_solution_indices = X_test_indices
        agreed_solutions_X = X_train
        agreed_solutions_indices = X_train_indices
        agreed_solutions_Y = labels

        last_iteration_size = 0
        while(len(disagreed_solutions_X[0]) > 0 and len(agreed_solutions_Y) - last_iteration_size > 20):
            last_iteration_size = len(agreed_solutions_Y)

            predicted_solutions =[]
            for i in range(len(classifiers)):
                classifiers[i].fit(np.array(agreed_solutions_X[i]), np.array(agreed_solutions_Y))
                predicted_solutions.append(classifiers[i].predict(np.array(disagreed_solutions_X[i])))

            predicted_solutions = np.array(predicted_solutions).T

            next_disagreed_solutions_X=[]
            next_disagreed_solutions_indices=[[] * len(embeddings_loaders)]

            for solutionIndex in range(len(predicted_solutions)):
                all_agree = True

                for i in range(1, len(classifiers)):
                    if(not np.char.equal(predicted_solutions[solutionIndex][i-1], predicted_solutions[solutionIndex][i])):
                        all_agree = False
                        break

                if(all_agree== True):
                    for i in range(len(classifiers)):
                        agreed_solutions_X[i].append(disagreed_solutions_X[i][solutionIndex])
                    agreed_solutions_indices.append(disagreed_solution_indices[solutionIndex])
                    agreed_solutions_Y.append(str(predicted_solutions[solutionIndex][0]))
                else:
                    for i in range(len(classifiers)):
                        next_disagreed_solutions_X[i].append(disagreed_solutions_X[i][solutionIndex])
                    next_disagreed_solutions_indices.append(disagreed_solution_indices[solutionIndex])

            disagreed_solutions_X = next_disagreed_solutions_X
            disagreed_solution_indices = next_disagreed_solutions_indices

        if(len(disagreed_solutions_X[0]) > 0):
            predicted_solutions =[]
            for i in range(len(classifiers)):
                classifiers[i].fit(np.array(agreed_solutions_X[i]), np.array(agreed_solutions_Y))
                predicted_solutions.append(classifiers[i].predict(np.array(disagreed_solutions_X[i])))

            predicted_solutions = np.array(predicted_solutions).T

            for solutionIndex in range(len(predicted_solutions)):
                for i in range(len(classifiers)):
                    agreed_solutions_X[i].append(disagreed_solutions_X[i][solutionIndex])
                agreed_solutions_indices.append(disagreed_solution_indices[solutionIndex])
                agreed_solutions_Y.append(str(predicted_solutions[solutionIndex][0]))

        solutions_indices = []

        for index in agreed_solutions_indices:
            solutions_indices.append(Xn_indices[index])

        return (solutions_indices, agreed_solutions_Y)

    def _validate_clustering_multi_view(self, problem, embeddings):
        embeddings_used = str.join('/', [emb['name'] for emb in embeddings])
        print(f'Predicting using the multiview clustering validation method using embeddings {embeddings_used}:')

        problem_dicts = []

        for embedding in embeddings:
            problem_dicts.append(embedding['problemDict'])
        
        prediction_Y = []
        Xn_indices = []
        for data in problem_dicts[0].items():
            print(f'Predicting problem {problem}')
            Xn = []
            
            for i in range(len(embeddings)):
                Xn.append([])

            for index in data['indexes']:
                index_in_all_embeddings = True

                for problem_dict in problem_dicts:
                    if index not in problem_dict[problem]['indexes']:
                        index_in_all_embeddings = False
                        break

                if(index_in_all_embeddings):
                    i = 0
                    for problem_dict in problem_dicts:
                        problem_data = problem_dict[problem]
                        Xn[i].append(problem_data['X'][problem_data['indexes'].index(index)])
                        i+=1
                    Xn_indices.append(index)
            
            all_labels = list(set(Y))
            k = len(all_labels)

            multi_view = MVSC(k)
            
            views = [np.array(Xi) for Xi in Xn]
            is_distance = np.array([False for Xi in Xn])

            prediction_Y = multi_view.fit(views, is_distance).embedding_
    
        return [Xn_indices, prediction_Y]
