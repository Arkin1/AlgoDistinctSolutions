from preprocessing_operations.ReplaceMacrosPreprocessingOp import ReplaceMacrosPreprocessingOp
from preprocessing_operations.RemoveIncludesUsingPreprocessingOp import RemoveIncludesUsingPreprocessingOp 
from preprocessing_operations.RemoveCommentsPreprocessingOp import RemoveCommentsPreprocessingOp 
from preprocessing_operations.RemoveNonAsciiPreprocessingOp import RemoveNonAsciiPreprocessingOp 
from preprocessing_operations.RemoveUnusedFunctionsPreprocessingOp import RemoveUnusedFunctionsPreprocessingOp
from preprocessing_operations.TokenizePreprocessingOp import TokenizePreprocessingOp

import gensim
import logging
from typing import Any, Tuple
from tqdm import tqdm
import json
import os
from utils import tqdm_multiprocess_map
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle

logger = logging.getLogger()

class TfidfFacade():

    def pretrain(self, dataset_info_path:str, 
                 destination_dir:str, 
                 preprocessing_workers:int,
                 **kwargs):
        logger.info(f"Reading the entire dataset {dataset_info_path} for tfidf pretraining")
        with open(dataset_info_path, 'r', encoding='utf-8') as fp:
            dataset_info = json.load(fp)
        
        source_codes = self._read_all_source_code(dataset_info, os.path.dirname(dataset_info_path))

        logger.info(f"Preprocessing the entire dataset {dataset_info_path} for tfidf pre-training")
        
        preprocessed_source_codes = tqdm_multiprocess_map(self._preprocessing_fn, source_codes, max_workers= preprocessing_workers, chunksize = 64)

        logger.info("Pretraining tfidf model")
        tfidf = TfidfVectorizer(**kwargs)
        tfidf.fit(preprocessed_source_codes)

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        with open(os.path.join(destination_dir, 'tfidf.model'), 'wb') as fp:
            pickle.dump(tfidf, fp)

        
    def generate_embeddings(self, dataset_info_path:str,
                                  destination_dir:str, 
                                  preprocessing_workers:int,
                                  model_path:str):
        logger.info(f"Reading the entire dataset {dataset_info_path} for generating tfidf embeddings")

        with open(dataset_info_path, 'r', encoding='utf-8') as fp:
            dataset_info = json.load(fp)
       
        source_codes = self._read_all_source_code(dataset_info, os.path.dirname(dataset_info_path))

        logger.info(f"Preprocessing the entire dataset {dataset_info_path} for tfidf embedding generation")
        preprocessed_source_codes = tqdm_multiprocess_map(self._preprocessing_fn, source_codes, max_workers= preprocessing_workers, chunksize = 64)

        with open(model_path, 'rb') as fp:
            tfidf = pickle.load(fp)
        
        logger.info(f"Generating embeddings")
        source_codes_embeddings = tfidf.transform(preprocessed_source_codes).toarray()
        source_codes_embeddings = np.vsplit(source_codes_embeddings, source_codes_embeddings.shape[0])
        source_codes_embeddings = [np.squeeze(sce) for sce in source_codes_embeddings]


        df = pd.DataFrame({"id":[sample['id'] for sample in dataset_info],  
                           "embeddings": source_codes_embeddings})

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        df.to_parquet(os.path.join(destination_dir, 'embeddings.parquet'))

        

        
    @staticmethod
    def _preprocessing_fn(source_code:Tuple[int, str]) -> str:
        replace_macros_pre_op = ReplaceMacrosPreprocessingOp()
        remove_includes_using_pre_op = RemoveIncludesUsingPreprocessingOp()
        remove_comments_pre_op = RemoveCommentsPreprocessingOp()
        remove_non_ascii_pre_op = RemoveNonAsciiPreprocessingOp()
        remove_unused_functions_pre_op = RemoveUnusedFunctionsPreprocessingOp()
        tokenizer_pre_op = TokenizePreprocessingOp()
        
        preprocessed_source_code = remove_comments_pre_op.preprocess(source_code)
        preprocessed_source_code = remove_unused_functions_pre_op.preprocess(preprocessed_source_code)
        preprocessed_source_code = remove_includes_using_pre_op.preprocess(preprocessed_source_code)
        preprocessed_source_code = replace_macros_pre_op.preprocess(preprocessed_source_code)
        preprocessed_source_code = remove_non_ascii_pre_op.preprocess(preprocessed_source_code)

        tokens = tokenizer_pre_op.preprocess(preprocessed_source_code)
        return " ".join(tokens)
        

    def _read_all_source_code(self, dataset_info, dataset_root_path:str):
        source_codes = []
        for submission_info in tqdm(dataset_info):
            submission_path = submission_info['path']

            submission_path = os.path.join(dataset_root_path, submission_path)

            with open(submission_path, 'r', encoding='utf-8') as fp:
                source_code = fp.read()

            source_codes.append(source_code)
        return source_codes
