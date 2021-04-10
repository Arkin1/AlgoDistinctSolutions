from abc import ABC, abstractmethod
import json
class EmbeddingsLoader(ABC):

    @abstractmethod
    def GetEmbeddings(self):
        pass

    @abstractmethod
    def GetSize(self):
        pass


class W2VEmbeddingsLoader(EmbeddingsLoader):

    def GetEmbeddings(self):
        model = Word2Vec.load(self.__getRelativePath("embeddings/w2v/w2v_128_dict.emb"))
        matrix = np.array(model.wv.vectors)
        vocab = model.wv.vocab

        problems = json.load(open("w2vEmbeddings.json", "r"))

        for problem in problems:
            tokens = problem['tokens']

            yield {
                    "index":problem["index"],
                    "label":problem["tags"][0],
                    "embeddings": [matrix[vocab.get(token).index].tolist() for token in tokens if(vocab.get(token) is not None)],
                    "tokens": tokens
                }
            
    def GetSize(self):
        return 128

class SafeEmbeddingsLoader(EmbeddingsLoader):

    def GetEmbeddings(self):
        problems = json.load(open("safeEmbeddings.json", "r"))

        for problem in problems:
            yield {
                    "index":problem["index"],
                    "label":problem["tags"][0],
                    "embeddings": problem["safe"]
                }
        
    def GetSize(self):
        return 100
        

