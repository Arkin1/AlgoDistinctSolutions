import numpy as np
import itertools
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import math
from multiview import MVSC


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


class ClusteringValidationMethod:
    def validateK(self, embeddings, cluster_algo):
        print(f"Validating using the known k clustering validation method using embeddings {embeddings['name']}:")
        problem_dict = embeddings['problemDict']
        
        problem_validation_data = {}
        for problem , data in problem_dict.items():
            X = np.array(data['X'])
            Y = np.array(data['Y'])

            all_labels = list(set(Y))
            k = len(all_labels)
        
            cluster_algo.set_params(n_clusters = k)

            print(f'Validating problem {problem} using cluster algorithm {type(cluster_algo).__name__}')

            try:
                cluster_algo.fit(X, n_jobs = -1)
                labels = cluster_algo.labels_
            except:
                cluster_algo.fit(X)
                labels = cluster_algo.labels_
            
            best_score = -1

            for permutation in itertools.permutations(range(0, k)):
                permuted_labels = []

                for index in range(len(labels)):
                    permuted_labels.append(all_labels[permutation[labels[index]]])
                
                permuted_labels = np.array(permuted_labels)
                score = f1_score(permuted_labels, Y, average='weighted')

                if(score > best_score):
                    best_score = score
                    bestPermutation = permuted_labels

            problem_validation_data[problem] = (Y, bestPermutation)

        return problem_validation_data
        

    def validate_unsupervised(self, embeddings, clusterAlgos, classifiersUnfit):
        embeddings_used = str.join('/', [emb['name'] for emb in embeddings])
        cluster_algos_used = str.join('/', [type(clusterAlgo).__name__ for clusterAlgo in clusterAlgos])
        classifiers_used = str.join('/', [type(classifier).__name__ for classifier in classifiersUnfit])
        print(f'Validating using the semiSupervised validation method using embeddings {embeddings_used}:')

        if(len(embeddings) <=1):
            print("There must be at least two embeddings")
            raise Exception()

        problem_dicts = []

        for i in range(0, len(embeddings)):
            if(embeddings[i]['name'] == 'w2v' or embeddings[i]['name'] == 'tfidf'):
                embeddings[0], embeddings[i] = embeddings[i], embeddings[0]
                break

        for embedding in embeddings:
            problem_dicts.append(embedding['problemDict'])
    
        problem_validation_data ={}
        for problem , data in problem_dicts[0].items():
            print(f'Validating problem {problem} using cluster algorithms {cluster_algos_used} and estimators {classifiers_used}')

            classifiers = [clone(classifier) for classifier in classifiersUnfit]
            
            Xn = []
            Xn_test = []
            for i in range(len(embeddings)):
                Xn.append([])
                Xn_test.append([])

            Y = []
            Y_test=[]
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

                        Y.append(data['Y'][data['indexes'].index(index)])

            
            all_labels = list(set(Y))
            k = len(all_labels)

            for cluster_algo in clusterAlgos:
                cluster_algo.set_params(n_clusters = k)

            label_n = []
            index = 0
            for X in Xn:
                label_n.append(clusterAlgos[index].fit(X).labels_)
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
     
            best_score = - 1
            best_combination = []
            grouped_paths_keys = grouped_paths.keys()

            for combination in itertools.combinations(grouped_paths_keys, k):
                if(check_distinct_combination(combination)):
                    score = 0

                    for key in combination:
                        score += len(grouped_paths[key])
                    
                    if(score > best_score):
                        best_score = score
                        best_combination = combination

            if(best_score == -1):
                raise Exception(f"Couldn't find {k} vectors, make k smaller")

            labels = []
            X_result = []
            for _ in range(len(embeddings)):
                X_result.append([])

            Y_result = []
            for comb in best_combination:
                for i in grouped_paths[comb]:
                    labels.append(label_n[i][0])
                    Y_result.append(Y[i])

                    for j in range(len(embeddings)):
                        X_result[j].append(Xn[j][i])

            Y_test = []
            for key, v in grouped_paths.items():
                if(key not in best_combination):
                    for i in v:
                        for j in range(len(embeddings)):
                            Xn_test[j].append(Xn[j][i])
                        Y_test.append(Y[i])


            best_score = -1
            for permutation in itertools.permutations(range(0, k)):
                permuted_labels = []

                for index in range(len(labels)):
                    permuted_labels.append(all_labels[permutation[labels[index]]])
                
                permuted_labels = np.array(permuted_labels)
                score = f1_score(permuted_labels, Y_result, average='weighted')

                if(score > best_score):
                    best_score = score
                    bestPermutation = permuted_labels

            before_classifiers = (Y_result.copy(), bestPermutation)

            disagreed_solutions_X = Xn_test
            disagreed_solutions_Y = Y_test
            agreed_solutions_X = X_result
            agreed_solutions_Y = bestPermutation.tolist()

            last_iteration_size = 0
            while(len(disagreed_solutions_Y) > 0 and len(agreed_solutions_Y) - last_iteration_size > 20):

                last_iteration_size = len(agreed_solutions_Y)

                predicted_solutions =[]
                for i in range(len(classifiers)):
                    classifiers[i].fit(np.array(agreed_solutions_X[i]), np.array(agreed_solutions_Y))
                    predicted_solutions.append(classifiers[i].predict(np.array(disagreed_solutions_X[i])))

                predicted_solutions = np.array(predicted_solutions).T

                next_disagreed_solutions_X=[[] for _ in range(len(embeddings))]
                next_disagreed_solutions_Y = []
                for solution_index in range(len(predicted_solutions)):
                    all_agree = True

                    for i in range(1, len(classifiers)):
                        if(not np.char.equal(predicted_solutions[solution_index][i-1], predicted_solutions[solution_index][i])):
                            all_agree = False
                            break

                    if(all_agree== True):
                        for i in range(len(classifiers)):
                            agreed_solutions_X[i].append(disagreed_solutions_X[i][solution_index])
                        agreed_solutions_Y.append(str(predicted_solutions[solution_index][0]))
                        Y_result.append(disagreed_solutions_Y[solution_index])
                    else:
                        for i in range(len(classifiers)):
                            next_disagreed_solutions_X[i].append(disagreed_solutions_X[i][solution_index])
                        next_disagreed_solutions_Y.append(disagreed_solutions_Y[solution_index])

                disagreed_solutions_X = next_disagreed_solutions_X
                disagreed_solutions_Y = next_disagreed_solutions_Y

                afterClassifiers = (agreed_solutions_Y,  Y_result)
                problem_validation_data[problem] = (before_classifiers, afterClassifiers)

        return problem_validation_data   

    def validateClusteringMultiView(self, embeddings):
        embeddings_used = str.join('/', [emb['name'] for emb in embeddings])
        print(f'Validating using the multiview clustering validation method using embeddings {embeddings_used}:')

        problem_dicts = []

        for embedding in embeddings:
            problem_dicts.append(embedding['problemDict'])

        problem_validation_data = {}
           
        for problem , data in problem_dicts[0].items():
            print(f'Validating problem {problem}')
            Xn = []
            
            for i in range(len(embeddings)):
                Xn.append([])

            Y = []
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
                    Y.append(data['Y'][data['indexes'].index(index)])
            
            all_labels = list(set(Y))
            k = len(all_labels)

            multiview = MVSC(k)

            views = [np.array(Xi) for Xi in Xn]
            is_distance = np.array([False for Xi in Xn])
            
            labels = multiview.fit(views, is_distance).embedding_

            best_score = -1
            for permutation in itertools.permutations(range(0, k)):
                permuted_labels = []

                for index in range(len(labels)):
                    permuted_labels.append(all_labels[permutation[labels[index]]])
                
                permuted_labels = np.array(permuted_labels)
                score = f1_score(permuted_labels, Y, average='weighted')

                if(score > best_score):
                    best_score = score
                    best_permutation = permuted_labels

            problem_validation_data[problem] = (Y, best_permutation)
    
        return problem_validation_data


class EstimatorValidationMethod:

    def validate(self, embeddings, estimator_model, test_size):
        print(f"Validating using the estimator validation method using embeddings {embeddings['name']}:")
        problem_dict = embeddings['problemDict']
        
        problem_validation_data = {}
        for problem , data in problem_dict.items():

            estimator = clone(estimator_model)

            X = np.array(data['X'])
            Y = np.array(data['Y'])

            print(f'Validating problem {problem} using estimator {type(estimator).__name__}')
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state=42)
            
            estimator = clone(estimator)

            print(f"Cross validation score : {cross_val_score(estimator, X_train, Y_train)}")

            estimator.fit(X_train, Y_train)

            labels = estimator.predict(X_test)

            problem_validation_data[problem] = (Y_test, labels)

        return problem_validation_data
