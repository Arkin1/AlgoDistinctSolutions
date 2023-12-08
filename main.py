# from PredictionMethods import PredictionMethods
from ValidationPipelines import ValidationPipelines
from DetermineKValidationPipelines import DetermineKPipelines

from AlgoLabelHelper import AlgoLabelHelper
from StatisticsHelper import StatisticsHelper
from OriginalDataDownloader import OriginalDataDownloader
import argparse
import warnings
import os
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Code embeddings in the context of different solutions in a competitive programming problem')

parser.add_argument('--download', action='store_true', help='Download the original files (raw dataset and embeddings)')
parser.add_argument('--statistics', dest='statisticsType', action = 'append', choices=['dataset', 'incremental_tfidf'], help='Get statistics based on the choice (dataset, source code, results)')
parser.add_argument('--transform', action='store_true', help='Transform the raw dataset in a format which can be used by AlgoLabel')
parser.add_argument('--prepare' , action='store_true', help='Prepare the dataset for the training/validation pipeline')
parser.add_argument('--embeddings', dest='embeddingsTypes',action='append', choices=['w2v', 'c2v' , 'safe', 'tfidf', 'infercode', 'incremental_tfidf'], help='Computes the embeddings for transformed dataset')
parser.add_argument('--evaluate', action='store_true', help='Evaluates how well the embeddings contribute in the distinct solutions problem')
parser.add_argument('--evaluate-k-selection', action='store_true', help='Evaluates how well the embeddings contribute to determining the optimal number k')

warnings.filterwarnings("ignore")

if(not os.path.exists('tmp')):
    os.mkdir('tmp')

args = parser.parse_args()

algoLabelHelper = AlgoLabelHelper()
statisticsHelper = StatisticsHelper()

if(args.download is True):
    OriginalDataDownloader().download()

if(args.statisticsType is not None and len(args.statisticsType) > 0):
    statisticsHelper.handleStatistics(args.statisticsType)

if(args.transform is True):
    algoLabelHelper.transformFolderStructureValidFormat("Data/RawDataset")

if(args.prepare is True):
    algoLabelHelper.prepare_embeddings()

if(args.embeddingsTypes is not None and len(args.embeddingsTypes) > 0):
    algoLabelHelper.compute_embeddings(args.embeddingsTypes)

if(args.evaluate is True):
    validationPipelines = ValidationPipelines()

    validationPipelines.k_clustering_pipeline()
    validationPipelines.estimator_pipeline()
    validationPipelines.unsupervised_voting_pipeline()
    validationPipelines.semi_supervised_multiview_spectral_clustering()

if(args.evaluate_k_selection is True):
    kvalidationPipelines = DetermineKPipelines()

    kvalidationPipelines.k_clustering_pipeline()

