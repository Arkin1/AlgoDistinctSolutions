from preprocessing_operations.RemoveUnusedFunctionsPreprocessingOp import RemoveUnusedFunctionsPreprocessingOp
from preprocessing_operations.CompilePreprocessingOp import CompilePreprocessingOp

import logging
from typing import Any, Tuple
from tqdm import tqdm
import json
import os
from utils import tqdm_multiprocess_map
from safe.safe import SAFE

import tensorflow
import numpy as np

logger = logging.getLogger()

class SafeFacade():

    def pretrain(self, dataset_info_path:str, 
                 destination_dir:str, 
                 preprocessing_workers:int,
                 **kwargs):
        raise NotImplementedError()

        
    def generate_embeddings(self, dataset_info_path:str,
                                  destination_dir:str, 
                                  preprocessing_workers:int,
                                  model_path:str,
                                  preprocessing_cache_dir:str = None):
        logger.info(f"Reading the entire dataset {dataset_info_path} for generating safe embeddings")

        with open(dataset_info_path, 'r', encoding='utf-8') as fp:
            dataset_info = json.load(fp)
       
        source_codes = self._read_all_source_code(dataset_info, os.path.dirname(dataset_info_path))[:10]

        logger.info(f"Preprocessing the entire dataset {dataset_info_path} for safe embedding generation")
        compiled_source_code_paths = tqdm_multiprocess_map(self._preprocessing_fn, source_codes, max_workers= preprocessing_workers, chunksize = 64)

        embedder = SAFE(model_path = 'Data/Models/safe/safe_trained_X86.pb',
                        instr_conv= 'Data/Models/safe/i2v/word2id.json',
                        max_instr = 150)

        logger.info(f"Generating embeddings")
        source_codes_embeddings = []

        for compiled_path_source_code in tqdm(compiled_source_code_paths):
            if compiled_path_source_code is not None:
                function_embeddings = embedder.embedd_functions(compiled_path_source_code)
                function_embeddings = np.concatenate(function_embeddings, 0)
                source_embeddings = np.mean(function_embeddings, 0)
                source_codes_embeddings.append(source_embeddings)
            else:
                source_codes_embeddings.append(None)

        # for x in tqdm(source_codes):
        #     self._preprocessing_fn(x)

        # df = pd.DataFrame({"id":[sample['id'] for sample in dataset_info],  
        #                    "embeddings": source_codes_embeddings})

        # if not os.path.exists(destination_dir):
        #     os.makedirs(destination_dir)
        # df.to_parquet(os.path.join(destination_dir, 'embeddings.parquet'))

        

        
    @staticmethod
    def _preprocessing_fn(source_code:Tuple[int, str]) -> str:
        remove_unused_functions_pre_op = RemoveUnusedFunctionsPreprocessingOp()
        compile_pre_op = CompilePreprocessingOp()
        
        preprocessed_source_code = remove_unused_functions_pre_op.preprocess(source_code)
        compiled_source_code_path = compile_pre_op.preprocess(preprocessed_source_code)
 
        return compiled_source_code_path
        

    def _read_all_source_code(self, dataset_info, dataset_root_path:str):
        source_codes = []
        for submission_info in tqdm(dataset_info):
            submission_path = submission_info['path']

            submission_path = os.path.join(dataset_root_path, submission_path)

            with open(submission_path, 'r', encoding='utf-8') as fp:
                source_code = fp.read()

            source_codes.append(source_code)
        return source_codes
