import sys
from gensim.models import Word2Vec
from CodeEmbedder import Code2VecEmbedder, InferCodeEmbedder
from Constants import *
from Embeddings.SafeEmbedder import SafeEmbedder
from Preprocesser import Preprocesser
from Preprocessing.RemoveCommentsPreprocessingOp import RemoveCommentsPreprocessingOp
from Preprocessing.RemoveIncludesUsingPreprocessingOp import RemoveIncludesUsingPreprocessingOp
from Preprocessing.RemoveNonAsciiPreprocessingOp import RemoveNonAsciiPreprocessingOp
from Preprocessing.RemoveUnusedFunctionsPreprocessingOp import RemoveUnusedFunctionsPreprocessingOp

sys.path.insert(0, 'AlgoLabel')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import json
import itertools
import pickle
import numpy as np


class AlgoLabelHelper():
    def __init__(self):
        self.config                        = json.load(open(self.__getRelativePath('config.json')))
        self.w2vDictionaryEmbeddings       = "Data/Embeddings/w2v/w2v_128_dict.emb"
        self.w2vFileName                   = "Data/Embeddings/w2v/w2vEmbeddings.json"
        self.c2vFileName                   = "Data/Embeddings/c2v/c2vEmbeddings.json"
        self.infercodeFileName             = "Data/Embeddings/infercode/infercodeEmbeddings.json"
        self.safeFileName                  = "Data/Embeddings/safe/safeEmbeddings.json"
        self.tfidfFileName                 = "Data/Embeddings/tfidf/tfidfEmbeddings.pkl"
        self.incrementalTfidfFileName      = "Data/Embeddings/incremental_tfidf/incrementalTfidfEmbeddings"
        self.codebertFileName              = os.getcwd() + '/' + 'Models/codebert'
        self.preprocessedDatasetFolderPath = os.getcwd() + '/' + "AlgoLabel/datasets"
        self.rawDatasetFolderPath          = os.getcwd() + '/' + "Data/RawDataset"
        self.c2vModelFileName              = os.getcwd() + '/' + "Models/c2v/saved_model_iter2"
        self.config['raw_dataset']         = 'datasets/raw_dataset.json'
        self.params                        = self.config["prepare"]["source"]
        self.__persistConfig()

        self.PREPROCESSING_OPTIONS = [RemoveUnusedFunctionsPreprocessingOp(), RemoveCommentsPreprocessingOp(),
                                        RemoveIncludesUsingPreprocessingOp(), RemoveNonAsciiPreprocessingOp()]

    def transformFolderStructureValidFormat(self, pathFolder:str):
        print("Transforming the raw dataset...")

        json_data = []
        tags = []
        for root, _, files in os.walk(f"{pathFolder}"):
            if(len(files) > 0):
                index= root.replace(f"{pathFolder}/", "").replace("/","$")
                for file_name in files:
                    if(file_name.endswith(".cpp")):
                        cpp_file = open(root + "/" + file_name,"r", encoding='utf-8')
                        tags.append(f'{index}')
                        problem_data = {
                            "index": f'{file_name[:-4]}',
                            "tags":[ f'{index}'],
                            "solutions":[
                                {
                                    "index":f'{file_name[:-4]}',
                                    "code": cpp_file.read()
                                }
                            ]
                        }
                        json_data.append(problem_data)
        self.config['split']['labels'] = list(set(tags))
        self.__persistConfig()
        truth_clusters_file = open(self.__getRelativePath(f'{self.config["raw_dataset"]}'), "w")

        return json.dump(json_data, truth_clusters_file, indent = 4)

    def prepare_embeddings(self):
        print("Preprocessing dataset...")
        preprocesser = Preprocesser(self.rawDatasetFolderPath, self.preprocessedDatasetFolderPath, self.params["compiler"])
        preprocesser.preprocess(self.PREPROCESSING_OPTIONS, 3)
        dataset = self.__getRawPreparedJson()
        self.__split_dataset(dataset)
        
    def compute_embeddings(self, embeddings:list):
        print("Computing embeddings...")

        if('tfidf' in embeddings):
            self.__compute_tfidf_embeddings()

        if('incremental_tfidf' in embeddings):
            self.__compute_incremental_tfidf_embeddings()

        if('infercode' in embeddings):
            self.__compute_infercode_embeddings()

        if('c2v' in embeddings):
            self.__compute_c2v_embeddings();

        if('w2v' in embeddings):
            self.__compute_w2v_embeddings()
        
        if('safe' in embeddings):
            self.__compute_safe_embeddings()

    def __compute_tfidf_embeddings(self):

        print("\tComputing TFIDF embeddings...")
        dataset = self.__getRawPreparedJson()
        
        tfidf_tokens = []

        for problem in dataset:
            tfidf_tokens.append(" ".join(problem['tokens']))
  
        vectorizer = TfidfVectorizer(token_pattern=TFIDF_TOKEN_PATTERN, stop_words=TFIDF_STOP_WORDS)

        matrix = vectorizer.fit_transform(tfidf_tokens)

        tfidf_embeddings = []
        
        for problem_idx in range(len(dataset)):
            tfidf_embeddings.append({
                    "index":problem["id"],
                    "label":problem["algorithmic_solution"],
                    "tfidf": matrix[problem_idx]
                })

        os.makedirs(os.path.dirname(self.tfidfFileName), exist_ok=True)
        pickle.dump(tfidf_embeddings, open(self.tfidfFileName, "wb"))

    def __compute_incremental_tfidf_embeddings(self):
        train_sizes = [5, 10, 20, 30, 40, 50, 80]
        
        dataset = self.__getRawPreparedJson()

        train_data = []
        for train_size in train_sizes:
            train_per_sizes = []
            for _, value in itertools.groupby(dataset, lambda x: x['algorithmic_solution']):
                train, _ = train_test_split(list(value), train_size=train_size)
                train_per_sizes.extend(train)
            train_data.append(train_per_sizes)
        
        for train_size_idx in range(len(train_sizes)):
            print(f"\tComputing incremental TFIDF embeddings for train size {train_sizes[train_size_idx]}")
            vectorizer = TfidfVectorizer(token_pattern=TFIDF_TOKEN_PATTERN, stop_words=TFIDF_STOP_WORDS)
            tfidf_tokens = []

            for problem in train_data[train_size_idx]:
                tfidf_tokens.append(" ".join(problem['tokens']))
            
            vectorizer.fit(tfidf_tokens)
            
            tfidf_embeddings = []
            for problem in dataset:
                tokens = " ".join(problem['tokens'])
                tfidf_embeddings.append({
                    "index":problem["id"],
                    "label":problem["algorithmic_solution"],
                    "tfidf": vectorizer.transform([tokens])
                })

            os.makedirs(os.path.dirname(f"{self.incrementalTfidfFileName}_{train_sizes[train_size_idx]}.json"), exist_ok=True)
            pickle.dump(tfidf_embeddings, open(f"{self.incrementalTfidfFileName}_{train_sizes[train_size_idx]}.json", "wb"))

    def __compute_infercode_embeddings(self):
        print("\tComputing InferCode embeddings...")

        dataset = self.__getRawPreparedJson()
        infercode_embeddings = []
        infercode = InferCodeEmbedder()
        infercode_embeddings = infercode.GetEmbeddingsFromDataset(dataset)

        os.makedirs(os.path.dirname(self.infercodeFileName), exist_ok=True)
        json.dump(infercode_embeddings, open(self.infercodeFileName, 'w'), indent=4)

    def __compute_c2v_embeddings(self):
        print("\tComputing c2v embeddings...")
        
        dataset = self.__getRawPreparedJson()

        c2vEmbedder = Code2VecEmbedder(self.c2vModelFileName)

        c2v_embeddings = c2vEmbedder.GetEmbeddingsFromFiles(self.rawDatasetFolderPath)
        
        for embedding in c2v_embeddings:
            try:
                value = next(item for item in dataset if item["id"] == embedding["index"])
                embedding['label'] = value['algorithmic_solution']
            except StopIteration:
                print(f"Problem with index {embedding['id']} wasn't found")
        
        os.makedirs(os.path.dirname(self.c2vFileName), exist_ok=True)
        json.dump(c2v_embeddings, open(self.c2vFileName, 'w'), indent=4)

    def __compute_w2v_embeddings(self):
        print("\tComputing w2v embeddings...")
        dataset = self.__getRawPreparedJson()
        X = self.__getTrainJson()
        inputs = [problem['tokens'] for problem in X]
        np.random.shuffle(inputs)

        w2v_settings = self.config['features']['types']['word2vec']
        w2v_model = Word2Vec(inputs,  window=w2v_settings["window"],
                            min_count=w2v_settings["min_count"],
                            workers=w2v_settings["workers"])
        
        w2v_embeddings = []
        for problem in dataset:
            tokens = problem['tokens']

            w2v_embeddings.append({
                    "index":problem["id"],
                    "label":problem["algorithmic_solution"],
                    "tokens": tokens
                })

        os.makedirs(os.path.dirname(self.w2vDictionaryEmbeddings), exist_ok=True)
        w2v_model.save(self.w2vDictionaryEmbeddings)
        json.dump(w2v_embeddings, open(self.w2vFileName, 'w'), indent=4)

    def __compute_safe_embeddings(self):
        print("\tComputing safe embeddings...")

        dataset = self.__getRawPreparedJson()
        os.chdir("AlgoLabel")
        safeEmbedder = SafeEmbedder(dataset)

        safe_embeddings = safeEmbedder.compute_embeddings()
        os.chdir("..")
        os.makedirs(os.path.dirname(self.safeFileName), exist_ok=True)
        json.dump(safe_embeddings, open(self.safeFileName, 'w'), indent=4)

    def __split_dataset(self, dataset):
        y = np.array([problem['algorithmic_solution'] for problem in dataset])
        dataset = np.array(dataset)
        train, test, _, y_test = train_test_split(dataset, y, test_size=self.config['split']['percentage'], stratify=y)
        dev, test = train_test_split(test, test_size=0.5, stratify=y_test)
        
        json.dump(list(train), open(self.__getRelativePath(self.config['raw_dataset'][:-5] + "_train.json"), "w"), indent=4)
        json.dump(list(dev), open(self.__getRelativePath(self.config['raw_dataset'][:-5] + "_dev.json"), "w"), indent=4)
        json.dump(list(test), open(self.__getRelativePath(self.config['raw_dataset'][:-5] + "_test.json"), "w"), indent=4)

    def __persistConfig(self):
        json.dump(self.config, open(self.__getRelativePath('config.json'), 'w'), indent= 4)

    def __getRelativePath(self, relativePath:str):
        return f'AlgoLabel/{relativePath}'

    def __getTrainJson(self):
        return json.load(open(self.__getRelativePath(self.config["raw_dataset"][:-5] + "_train.json"), "r"))
    
    def __getRawPreparedJson(self):
        return json.load(open(self.__getRelativePath(self.config["raw_dataset"][:-5] + "_prepared.json"), "r"))
