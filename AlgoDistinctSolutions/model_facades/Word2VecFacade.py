from preprocessing_operations.ReplaceMacrosPreprocessingOp import ReplaceMacrosPreprocessingOp
from preprocessing_operations.RemoveIncludesUsingPreprocessingOp import RemoveIncludesUsingPreprocessingOp 
from preprocessing_operations.RemoveCommentsPreprocessingOp import RemoveCommentsPreprocessingOp 
from preprocessing_operations.RemoveNonAsciiPreprocessingOp import RemoveNonAsciiPreprocessingOp 
from preprocessing_operations.RemoveUnusedFunctionsPreprocessingOp import RemoveUnusedFunctionsPreprocessingOp
from preprocessing_operations.TokenizePreprocessingOp import TokenizePreprocessingOp

import gensim
import logging
from typing import Any
from tqdm import tqdm
import json
import os

logger = logging.getLogger()

class Word2VecFacade():
    def __init__(self):
        def preprocessing_fn(source_code:str) -> str:
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
            return tokens
        
        self.preprocessing_fn = preprocessing_fn

    def pretrain(self, dataset_info_path:str, destination_path:str, **kwargs):
        logger.debug("Reading the entire dataset for pretraining")
        source_codes = self._read_all_source_code(dataset_info_path)

        logger.debug("Preprocessing the entire dataset for pre-training")
        preprocessed_source_codes = []
        for sc in tqdm(source_codes):
            preprocessed_source_codes.append(self.preprocessing_fn(sc))

        w2v = gensim.models.Word2Vec(preprocessed_source_codes, **kwargs)

        w2v.save(destination_path)

        
    def generate_embeddings(self, source_codes: list[str]):
        pass

    def _read_all_source_code(self, dataset_info_path:str):
        with open(dataset_info_path, 'r', encoding='utf-8') as fp:
            dataset_info = json.load(fp)
        
        source_codes = []
        for submission_info in tqdm(dataset_info):
            submission_path = submission_info['path']
            dataset_path = os.path.dirname(dataset_info_path)

            submission_path = os.path.join(dataset_path, submission_path)

            with open(submission_path, 'r', encoding='utf-8') as fp:
                source_code = fp.read()

            source_codes.append(source_code)
        return source_codes
