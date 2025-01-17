from preprocessing_operations.RemoveUnusedFunctionsPreprocessingOp import RemoveUnusedFunctionsPreprocessingOp
from preprocessing_operations.RemoveIncludesUsingPreprocessingOp import RemoveIncludesUsingPreprocessingOp
from preprocessing_operations.RemoveCommentsPreprocessingOp import RemoveCommentsPreprocessingOp
from preprocessing_operations.ReplaceMacrosPreprocessingOp import ReplaceMacrosPreprocessingOp
from preprocessing_operations.ExtractFunctionsAndMethodsProcessingOp import ExtractFunctionsAndMethodsPreprocessingOp

import logging
from typing import Any, Tuple
from tqdm import tqdm
import json
import os
from utils import tqdm_multiprocess_map

import pandas as pd
import numpy as np
import torch
from utils import to_batches
from unixcoder.unixcoder import UniXcoder

logger = logging.getLogger()

class UniXcoderFacade():

    def pretrain(self, dataset_info_path:str, 
                 destination_dir:str, 
                 preprocessing_workers:int,
                 **kwargs):
        raise NotImplementedError()

        
    def generate_embeddings(self, dataset_info_path:str,
                                  destination_dir:str, 
                                  preprocessing_workers:int,
                                  batch_size:int = 4,
                                  max_length:int = 512,
                                  padding:bool = True):
        logger.info(f"Reading the entire dataset {dataset_info_path} for generating UniXcoder embeddings")

        with open(dataset_info_path, 'r', encoding='utf-8') as fp:
            dataset_info = json.load(fp)
       
        source_codes = self._read_all_source_code(dataset_info, os.path.dirname(dataset_info_path))[:10]

        logger.info(f"Preprocessing the entire dataset {dataset_info_path} for UniXcoder embedding generation")
        functions_per_source_code = tqdm_multiprocess_map(self._preprocessing_fn, source_codes, max_workers= preprocessing_workers, chunksize = 64)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UniXcoder("microsoft/unixcoder-base")
        model.to(device)

        logger.info(f"Generating embeddings")
        source_codes_embeddings = []

        for functions in tqdm(functions_per_source_code):
            func_embeddings = []
            for batch_func in to_batches(functions, batch_size):
                tokens_ids = model.tokenize(batch_func,
                                            max_length=max_length,
                                            padding = padding,
                                            mode="<encoder-only>")
                source_ids = torch.tensor(tokens_ids).to(device)
                _,max_func_embedding = model(source_ids)

                func_embeddings.append(max_func_embedding)
            func_embeddings = torch.concat(func_embeddings)
            func_embeddings = func_embeddings.mean(0).detach().cpu().numpy()

            source_codes_embeddings.append(func_embeddings)

        df = pd.DataFrame({"id":[sample['id'] for sample in dataset_info[:10]],  
                           "embeddings": source_codes_embeddings})

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        df.to_parquet(os.path.join(destination_dir, 'embeddings.parquet'))

    
    @staticmethod
    def _preprocessing_fn(source_code:Tuple[int, str]) -> str:
        remove_comments_pre_op = RemoveCommentsPreprocessingOp()
        remove_includes_pre_op = RemoveIncludesUsingPreprocessingOp()
        replace_macros_pre_op = ReplaceMacrosPreprocessingOp()
        remove_unused_functions_pre_op = RemoveUnusedFunctionsPreprocessingOp()
        extract_functions_pre_op = ExtractFunctionsAndMethodsPreprocessingOp('cpp')
        
        preprocessed_source_code = remove_comments_pre_op.preprocess(source_code)
        preprocessed_source_code = remove_includes_pre_op.preprocess(preprocessed_source_code)
        preprocessed_source_code = replace_macros_pre_op.preprocess(preprocessed_source_code)
        preprocessed_source_code = remove_unused_functions_pre_op.preprocess(preprocessed_source_code)
        functions = extract_functions_pre_op.preprocess(preprocessed_source_code)
 
        return functions
        

    def _read_all_source_code(self, dataset_info, dataset_root_path:str):
        source_codes = []
        for submission_info in tqdm(dataset_info):
            submission_path = submission_info['path']

            submission_path = os.path.join(dataset_root_path, submission_path)

            with open(submission_path, 'r', encoding='utf-8') as fp:
                source_code = fp.read()

            source_codes.append(source_code)
        return source_codes