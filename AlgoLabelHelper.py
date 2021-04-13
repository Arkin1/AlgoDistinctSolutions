
import sys
from shutil import copyfile

sys.path.insert(0, 'AlgoLabel')

import os

from AlgoLabel.models.utils import preprocess_dataset, split_dataset, prepare_embeddings
from AlgoLabel.models.utils import prepare_input, train, test

from gensim.models import Word2Vec

import json
import numpy as np


class AlgoLabelHelper():
    def __init__(self):
        self.config = json.load(open(self.__getRelativePath('config.json')))
        self.w2vDictionaryEmbeddings="Data/Embeddings/w2v/w2v_128_dict.emb"
        self.w2vFileName ="Data/Embeddings/w2v/w2vEmbeddings.json"
        self.safeFileName ="Data/Embeddings/safe/safeEmbeddings.json"

        self.config['raw_dataset'] = 'datasets/raw_dataset.json'
        self.__persistConfig()

    def transformFolderStructureValidFormat(self, pathFolder:str):
        print("Transforming the raw dataset...")

        json_data = []
        tags = []
        for root, dirs, files in os.walk(f"{pathFolder}"):
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

    def compute_embeddings(self, embeddings:list):
        os.chdir("AlgoLabel")
        print("Preprocessing dataset...")
        preprocess_dataset(self.config)
        print("Split dataset...")
        split_dataset(self.config)
        os.chdir("..")

        print("Computing embeddings...")

        if('w2v' in embeddings):
            self.__compute_w2v_embeddings()
        
        if('safe' in embeddings):
            self.__compute_safe_embeddings()

    
    def __compute_w2v_embeddings(self):
        print("\tComputing w2v embeddings...")

        self.config['pretrain'] = 'word2vec_code'
        self.__persistConfig()

        os.chdir("AlgoLabel")
        prepare_embeddings(self.config)
        os.chdir("..")

        
        os.makedirs(os.path.dirname(self.w2vDictionaryEmbeddings), exist_ok=True)
        copyfile(self.__getRelativePath("data/embeddings/w2v_code_128.emb"), self.w2vDictionaryEmbeddings)

        w2v_embeddings = []

        splits = [self.__getTrainJson(), self.__getTestJson(), self.__getDevJson()]
        
        for split in splits:
            for problem in split:
                tokens = problem['tokens']

                w2v_embeddings.append({
                        "index":problem["index"],
                        "label":problem["tags"][0],
                        "tokens": tokens
                    })

        json.dump(w2v_embeddings, open(self.w2vFileName, 'w'), indent=4)

    def __compute_safe_embeddings(self):
        print("\tComputing safe embeddings...")

        self.config['pretrain'] = 'safe'
        self.__persistConfig()
        
        os.chdir("AlgoLabel")
        prepare_embeddings(self.config)
        os.chdir("..") 

        safe_embeddings = []

        splits = [self.__getTrainJson(), self.__getTestJson(), self.__getDevJson()]
        
        for split in splits:
            for problem in split:
                if("safe" in problem):
                    embeddings = problem['safe']

                    safe_embeddings.append({
                            "index":problem["index"],
                            "label":problem["tags"][0],
                            "safe": embeddings
                        })

        os.makedirs(os.path.dirname(self.safeFileName), exist_ok=True)
        json.dump(safe_embeddings, open(self.safeFileName, 'w'), indent=4)


    def __persistConfig(self):
        json.dump(self.config, open(self.__getRelativePath('config.json'), 'w'), indent= 4)

    def __getRelativePath(self, relativePath:str):
        return f'AlgoLabel/{relativePath}'

    def __getTrainJson(self):
        return json.load(open(self.__getRelativePath(self.config["raw_dataset"][:-5] + "_train.json"), "r"))
    def __getTestJson(self):
        return json.load(open(self.__getRelativePath(self.config["raw_dataset"][:-5] + "_test.json"), "r"))
    def __getDevJson(self):
        return json.load(open(self.__getRelativePath(self.config["raw_dataset"][:-5] + "_dev.json"), "r"))



