from PredictionMethods import PredictionMethods
from ValidationPipelines import ValidationPipelines
from AlgoLabelHelper import AlgoLabelHelper
from OriginalDataDownloader import OriginalDataDownloader
import argparse
import warnings
import os
import shutil
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Code embeddings in the context of different solutions in a competitive programming problem')

parser.add_argument('--download', action='store_true', help='Download the original files (raw dataset, preprocessed dataset and embeddings)')
parser.add_argument('--transform', action='store_true', help='Transform the raw dataset in a format which can be used by AlgoLabel')
parser.add_argument('--embeddings', dest='embeddingsTypes',action='append', choices=['w2v', 'safe', 'tfidf'], help='Computes the embeddings for transformed dataset')
parser.add_argument('--evaluate', action='store_true', help='Evaluates how well the embeddings contribute in the distinct solutions problem')


args = parser.parse_args()

algoLabelHelper = AlgoLabelHelper()

if(args.download is True):
    OriginalDataDownloader().download()

if(args.transform is True):
    algoLabelHelper.transformFolderStructureValidFormat("Data/RawDataset")

if(args.embeddingsTypes is not None and len(args.embeddingsTypes) > 0):
    algoLabelHelper.compute_embeddings(args.embeddingsTypes)

if(args.evaluate is True):
    validationPipelines = ValidationPipelines()

    validationPipelines.KClusteringPipeline()
    #validationPipelines.EstimatorPipeline()
    #validationPipelines.SemisupervisedVotingPipeline()
    #validationPipelines.SemiSupervisedMultiviewSpectralClustering()
    '''
    clusteringValidationMethod = ClusteringValidationMethod()
    kmeans = KMeans(n_clusters = -1)
    spectralClustering = SpectralClustering(n_clusters= -1)

    clusteringValidationMethod.validateK(SafeEmbeddingsLoader(), [kmeans, spectralClustering])
    clusteringValidationMethod.validateK(W2VEmbeddingsLoader(), [kmeans, spectralClustering])
    clusteringValidationMethod.validateK(TfidfEmbeddingsLoader(), [kmeans, spectralClustering])

    estimatorValidation = EstimatorValidationMethod()

    estimatorValidation.validate(SafeEmbeddingsLoader(), [RandomForestClassifier(), SVC(), XGBClassifier(verbosity = 0)], 0.2)
    estimatorValidation.validate(W2VEmbeddingsLoader(), [RandomForestClassifier(), SVC(), XGBClassifier(verbosity = 0)], 0.2)
    estimatorValidation.validate(TfidfEmbeddingsLoader(), [RandomForestClassifier(), SVC(), XGBClassifier(verbosity = 0)], 0.2)


    clusteringValidationMethod = ClusteringValidationMethod()
    clusteringValidationMethod.validateSemiSupervised([W2VEmbeddingsLoader(), TfidfEmbeddingsLoader(), SafeEmbeddingsLoader()], [kmeans, kmeans, kmeans], [XGBClassifier(verbosity = 0), SVC(), RandomForestClassifier()])


    clusteringValidationMethod = ClusteringValidationMethod()
    clusteringValidationMethod.validateClusteringMultiView([W2VEmbeddingsLoader(), TfidfEmbeddingsLoader(),  SafeEmbeddingsLoader()])
    '''
'''
predictionMethods = PredictionMethods()
k = 2
problem ="cmmdc"
indices, Y = predictionMethods.validateSemiSupervised(problem, [W2VEmbeddingsLoader(), TfidfEmbeddingsLoader()], [KMeans(), SpectralClustering()], [XGBClassifier(verbosity = 0), SVC(kernel='rbf')], k)

problemPath = f"Data/RawDataset/{problem}"

if(os.path.exists(problemPath + "/clusters")):
    shutil.rmtree(problemPath + "/clusters")

os.mkdir(problemPath + "/clusters")

for i in range(k):
    os.mkdir(problemPath + "/clusters/" + str(i))

for i in range(len(indices)):
    shutil.copyfile(problemPath + '/100/' + f'{indices[i]}.cpp', problemPath + "/clusters/" + f'{Y[i]}/{indices[i]}.cpp')
'''