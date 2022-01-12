import json
from tqdm.contrib.concurrent import process_map
from Constants import TMP_PATH

from safe.safe import SAFE
import os

class SafeEmbedder:
    def __init__(self, preprocesssed_dataset_path):
        self.preprocessed_dataset_path = preprocesssed_dataset_path
        self.safe = SAFE(model_path='safe/data/safe.pb', instr_conv='safe/data/i2v/word2id.json')

    def compute_embeddings(self, max_workers = 1):
        f = open(f'{self.preprocessed_dataset_path}/preprocessed_dataset.json', 'r', encoding='utf8')
        preprocesssed_dataset = json.load(f)
        
        chunks_result = process_map(self.compute_embeddings2, preprocesssed_dataset, max_workers = max_workers, chunksize = 1)

        nr = 0
        for chunk in chunks_result:
            for emb in chunk:
                if(emb != None):
                    preprocesssed_dataset[nr]['safe_embeddings'] = emb
                    nr+=1
        
        return preprocesssed_dataset

    
    def compute_embeddings2(self, source_code):

        f = open(f'{TMP_PATH}/tmp_safe.cpp', encoding='utf8')
        f.write(source_code)
        f.close()

        result = os.system('g++ {TMP_PATH}/tmp_safe.cpp -O3 -o {TMP_PATH}/object_code.o')
        
        if(result):
            embeddings = safe.embedd_functions(str(f'{TMP_PATH}/object_code.o'))
            embeddings = [x.tolist()[0] for x in embeddings if x is not None]
        else:
            return None
    
    
        
