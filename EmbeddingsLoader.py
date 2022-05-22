from abc import ABC, abstractmethod
import json
import numpy as np
from gensim.models import Word2Vec
import pickle

class EmbeddingsLoader(ABC):

    @abstractmethod
    def get_embeddings(self):
        pass

    @abstractmethod
    def get_name(self):
        pass


class W2VEmbeddingsLoader(EmbeddingsLoader):

    def get_embeddings(self):
        model = Word2Vec.load(self._get_relative_path("w2v_128_dict.emb"))
        matrix = np.array(model.wv.vectors)
        vocab = model.wv.vocab

        problems = json.load(open(self._get_relative_path("w2vEmbeddings.json"), "r"))

        for problem in problems:
            tokens = problem['tokens']

            yield {
                    "index":problem["index"],
                    "label":problem["label"],
                    "embeddings": [matrix[vocab[token].index].tolist() for token in tokens if(token in vocab)],
                    "tokens": tokens
                }
            
    def get_name(self):
        return 'w2v'
    
    def _get_relative_path(self, path):
        return f'Data/Embeddings/w2v/{path}'


class SafeEmbeddingsLoader(EmbeddingsLoader):

    def get_embeddings(self):
        problems = json.load(open(self._get_relative_path("safeEmbeddings.json"), "r"))

        for problem in problems:
            yield {
                    "index":problem["index"],
                    "label":problem["label"],
                    "embeddings": problem["embeddings"]
                }
        
    def get_name(self):
        return 'safe'

    def _get_relative_path(self, path):
        return f'Data/Embeddings/safe/{path}'


class Code2VecEmbeddingsLoader(EmbeddingsLoader):
    
    def get_embeddings(self):
        problems = json.load(open(self._get_relative_path("c2vEmbeddings.json"), "r"))

        for problem in problems:
            yield {
                    "index":problem["index"],
                    "label":problem["label"],
                    "embeddings": problem["embeddings"]
                }
        
    def get_name(self):
        return 'c2v'

    def _get_relative_path(self, path):
        return f'Data/Embeddings/c2v/{path}'    

class InfercodeEmbeddingsLoader(EmbeddingsLoader):
    
    def get_embeddings(self):
        problems = json.load(open(self._get_relative_path("infercodeEmbeddings.json"), "r"))

        for problem in problems:
            yield {
                "index":problem["index"],
                "label":problem["label"],
                "embeddings": problem["embeddings"]
                }

    def get_name(self):
        return 'infercode'
    
    def _get_relative_path(self, path):
        return f'Data/Embeddings/infercode/{path}'

class TfidfEmbeddingsLoader(EmbeddingsLoader):
    
    def get_embeddings(self):
        problems = pickle.load(open(self._get_relative_path("tfidfEmbeddings.pkl"), "rb"))

        for problem in problems:
            yield {
                    "index":problem["index"],
                    "label":problem["label"],
                    "embeddings": problem["tfidf"].toarray().tolist()
                }
            
    def get_name(self):
        return 'tfidf'
    
    def _get_relative_path(self, path):
        return f'Data/Embeddings/tfidf/{path}'