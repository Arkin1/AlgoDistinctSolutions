from PredictionMethods import PredictionMethods
class PredictionPipeline:

    def predict(problem, method, k, clusteringMethods, embeddingsMethods):
        predictionMethods = PredictionMethods()
        
        prediction = []

        if(method == 'kclustering'):
            prediction = predictionMethods.predictK(problem, embeddingsMethods, k)
        elif(method == 'mv_spectral_clustering'):
            prediction = predictionMethods.predictClusteringMultiView(problem, embeddingsMethods, k)
        elif(method == 'unsupervised_voting'):
            prediction = predictionMethods.predictUnsupervisedVoting(problem, embeddingsMethods, clusteringMethods, k)
        else:
            raise Exception(f'{method} is an invalid method')

        


            

