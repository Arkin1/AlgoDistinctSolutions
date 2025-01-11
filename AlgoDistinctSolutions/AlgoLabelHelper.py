import sys
from gensim.models import Word2Vec
from CodeEmbedder import Code2VecEmbedder, InferCodeEmbedder, SafeEmbedder
from Constants import *
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
        self._config = json.load(open(self._get_relative_path('config.json')))
        self._w2v_dict_embeddings_path = "Data/Embeddings/w2v/w2v_128_dict.emb"
        self._w2v_file_path = "Data/Embeddings/w2v/w2vEmbeddings.json"
        self._c2v_file_path = "Data/Embeddings/c2v/c2vEmbeddings.json"
        self._infercode_file_path = "Data/Embeddings/infercode/infercodeEmbeddings.json"
        self._safe_file_path = "Data/Embeddings/safe/safeEmbeddings.json"
        self._tfidf_file_path = "Data/Embeddings/tfidf/tfidfEmbeddings.pkl"
        self._inc_tfidf_file_path = "Data/Embeddings/incremental_tfidf/incrementalTfidfEmbeddings"
        self._preprocessed_dataset_folder_path = os.getcwd() + '/' + "AlgoLabel/datasets"
        self._raw_dataset_folder_path = os.getcwd() + '/' + "Data/RawDataset"
        self._c2v_model_file_path = os.getcwd() + '/' + "Models/c2v/saved_model_iter2"
        self._config['raw_dataset'] = 'datasets/raw_dataset.json'
        self._params = self._config["prepare"]["source"]
        self._persist_config()

        self._PREPROCESSING_OPTIONS = [RemoveUnusedFunctionsPreprocessingOp(), RemoveCommentsPreprocessingOp(),
                                        RemoveIncludesUsingPreprocessingOp(), RemoveNonAsciiPreprocessingOp()]

    def transform_folder_structure_valid_format(self, folder_path:str):
        print("Transforming the raw dataset...")

        json_data = []
        tags = []
        for root, _, files in os.walk(f"{folder_path}"):
            if(len(files) > 0):
                index= root.replace(f"{folder_path}/", "").replace("/","$")
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
        self._config['split']['labels'] = list(set(tags))
        self._persist_config()
        truth_clusters_file = open(self._get_relative_path(f'{self._config["raw_dataset"]}'), "w")

        return json.dump(json_data, truth_clusters_file, indent = 4)

    def prepare_embeddings(self):
        print("Preprocessing dataset...")
        preprocesser = Preprocesser(self._raw_dataset_folder_path, self._preprocessed_dataset_folder_path, self._params["compiler"])
        preprocesser.preprocess(self._PREPROCESSING_OPTIONS, 3)
        dataset = self._get_raw_prepared_json()
        self._split_dataset(dataset)
        
    def compute_embeddings(self, embeddings:list):
        print("Computing embeddings...")

        if('tfidf' in embeddings):
            self._compute_tfidf_embeddings()

        if('incremental_tfidf' in embeddings):
            self._compute_incremental_tfidf_embeddings()

        if('infercode' in embeddings):
            self._compute_infercode_embeddings()

        if('c2v' in embeddings):
            self._compute_c2v_embeddings();

        if('w2v' in embeddings):
            self._compute_w2v_embeddings()
        
        if('safe' in embeddings):
            self._compute_safe_embeddings()

    def _compute_tfidf_embeddings(self):

        print("\tComputing TFIDF embeddings...")
        dataset = self._get_raw_prepared_json()
        
        tfidf_tokens = []

        for problem in dataset:
            tfidf_tokens.append(" ".join(problem['tokens']))
  
        vectorizer = TfidfVectorizer(token_pattern=TFIDF_TOKEN_PATTERN, stop_words=TFIDF_STOP_WORDS)

        matrix = vectorizer.fit_transform(np.array(tfidf_tokens))

        tfidf_embeddings = []
        
        for problem_idx in range(len(dataset)):
            tfidf_embeddings.append({
                    "index":dataset[problem_idx]["id"],
                    "label":dataset[problem_idx]["algorithmic_solution"],
                    "tfidf": matrix[problem_idx]
                })

        os.makedirs(os.path.dirname(self._tfidf_file_path), exist_ok=True)
        pickle.dump(tfidf_embeddings, open(self._tfidf_file_path, "wb"))

    def _compute_incremental_tfidf_embeddings(self):
        train_sizes = [5, 10, 20, 30, 40, 50, 80]
        
        dataset = self._get_raw_prepared_json()

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

            os.makedirs(os.path.dirname(f"{self._inc_tfidf_file_path}_{train_sizes[train_size_idx]}.json"), exist_ok=True)
            pickle.dump(tfidf_embeddings, open(f"{self._inc_tfidf_file_path}_{train_sizes[train_size_idx]}.json", "wb"))

    def _compute_infercode_embeddings(self):
        print("\tComputing InferCode embeddings...")

        dataset = self._get_raw_prepared_json()
        infercode_embeddings = []
        infercode = InferCodeEmbedder()
        infercode_embeddings = infercode.get_embeddings_from_dataset(dataset)

        os.makedirs(os.path.dirname(self._infercode_file_path), exist_ok=True)
        json.dump(infercode_embeddings, open(self._infercode_file_path, 'w'), indent=4)

    def _compute_c2v_embeddings(self):
        print("\tComputing c2v embeddings...")
        
        dataset = self._get_raw_prepared_json()

        c2vEmbedder = Code2VecEmbedder(self._c2v_model_file_path)

        c2v_embeddings = c2vEmbedder.get_embeddings_from_files(self._raw_dataset_folder_path)
        
        for embedding in c2v_embeddings:
            try:
                value = next(item for item in dataset if item["id"] == embedding["index"])
                embedding['label'] = value['algorithmic_solution']
            except StopIteration:
                print(f"Problem with index {embedding['id']} wasn't found")
        
        os.makedirs(os.path.dirname(self._c2v_file_path), exist_ok=True)
        json.dump(c2v_embeddings, open(self._c2v_file_path, 'w'), indent=4)

    def _compute_w2v_embeddings(self):
        print("\tComputing w2v embeddings...")
        dataset = self._get_raw_prepared_json()
        X = self._get_train_json()
        inputs = [problem['tokens'] for problem in X]
        np.random.shuffle(inputs)

        w2v_settings = self._config['features']['types']['word2vec']
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

        os.makedirs(os.path.dirname(self._w2v_dict_embeddings_path), exist_ok=True)
        w2v_model.save(self._w2v_dict_embeddings_path)
        json.dump(w2v_embeddings, open(self._w2v_file_path, 'w'), indent=4)

    def _compute_safe_embeddings(self):
        print("\tComputing safe embeddings...")

        dataset = self._get_raw_prepared_json()
        os.chdir("AlgoLabel")
        safeEmbedder = SafeEmbedder(dataset)

        safe_embeddings = safeEmbedder.compute_embeddings()
        os.chdir("..")
        os.makedirs(os.path.dirname(self._safe_file_path), exist_ok=True)
        json.dump(safe_embeddings, open(self._safe_file_path, 'w'), indent=4)

    def _split_dataset(self, dataset):
        y = np.array([problem['algorithmic_solution'] for problem in dataset])
        dataset = np.array(dataset)
        train, test, _, y_test = train_test_split(dataset, y, test_size=self._config['split']['percentage'], stratify=y)
        dev, test = train_test_split(test, test_size=0.5, stratify=y_test)
        
        json.dump(list(train), open(self._get_relative_path(self._config['raw_dataset'][:-5] + "_train.json"), "w"), indent=4)
        json.dump(list(dev), open(self._get_relative_path(self._config['raw_dataset'][:-5] + "_dev.json"), "w"), indent=4)
        json.dump(list(test), open(self._get_relative_path(self._config['raw_dataset'][:-5] + "_test.json"), "w"), indent=4)

    def _persist_config(self):
        json.dump(self._config, open(self._get_relative_path('config.json'), 'w'), indent= 4)

    def _get_relative_path(self, relativePath:str):
        return f'AlgoLabel/{relativePath}'


    def _get_train_json(self):
        return json.load(open(self._get_relative_path(self._config["raw_dataset"][:-5] + "_train.json"), "r"))
    def _get_raw_prepared_json(self):
        return json.load(open(self._get_relative_path(self._config["raw_dataset"][:-5] + "_prepared.json"), "r"))
