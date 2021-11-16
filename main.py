from PredictionMethods import PredictionMethods
from ValidationPipelines import ValidationPipelines
from AlgoLabelHelper import AlgoLabelHelper
from DatasetStatistics import storeDatasetStatistics
from OriginalDataDownloader import OriginalDataDownloader
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Code embeddings in the context of different solutions in a competitive programming problem')

parser.add_argument('--download', action='store_true', help='Download the original files (raw dataset and embeddings)')
parser.add_argument('--statistics', action = 'store_true', help='Get info about number of source codes per solution as well as compilable source codes')
parser.add_argument('--transform', action='store_true', help='Transform the raw dataset in a format which can be used by AlgoLabel')
parser.add_argument('--embeddings', dest='embeddingsTypes',action='append', choices=['w2v', 'safe', 'tfidf'], help='Computes the embeddings for transformed dataset')
parser.add_argument('--evaluate', action='store_true', help='Evaluates how well the embeddings contribute in the distinct solutions problem')
parser.add_argument('--predict', action='store_true', help='Predict the distinct solutions for a specific problem in the problems folder')


args = parser.parse_args()

algoLabelHelper = AlgoLabelHelper()

if(args.download is True):
    OriginalDataDownloader().download()

if(args.statistics is True):
    storeDatasetStatistics("Data/RawDataset", "Data/statistics.csv")

if(args.transform is True):
    algoLabelHelper.transformFolderStructureValidFormat("Data/RawDataset")

if(args.embeddingsTypes is not None and len(args.embeddingsTypes) > 0):
    algoLabelHelper.compute_embeddings(args.embeddingsTypes)

if(args.evaluate is True):
    validationPipelines = ValidationPipelines()

    validationPipelines.KClusteringPipeline()
    validationPipelines.EstimatorPipeline()
    validationPipelines.UnsupervisedVotingPipeline()
    validationPipelines.UnsupervisedMultiviewSpectralClustering()

if(args.predict is True):
    predictionMethods = PredictionMethods()

