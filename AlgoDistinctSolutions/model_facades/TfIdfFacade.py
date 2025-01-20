from preprocessing_operations.ReplaceMacrosPreprocessingOp import ReplaceMacrosPreprocessingOp
from preprocessing_operations.RemoveIncludesUsingPreprocessingOp import RemoveIncludesUsingPreprocessingOp 
from preprocessing_operations.RemoveCommentsPreprocessingOp import RemoveCommentsPreprocessingOp 
from preprocessing_operations.RemoveNonAsciiPreprocessingOp import RemoveNonAsciiPreprocessingOp 
from preprocessing_operations.RemoveUnusedFunctionsPreprocessingOp import RemoveUnusedFunctionsPreprocessingOp
from preprocessing_operations.TokenizePreprocessingOp import TokenizePreprocessingOp

import logging
from typing import Any
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

from model_facades.BaseFacade import BaseFacade

logger = logging.getLogger()

class TfidfFacade(BaseFacade):
    def _preprocessing_fn(self, source_code:str) -> str:
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

    def _pretrain_fn(self, preprocessed_source_codes: list[Any], destination_dir:str, **kwargs):
        tfidf = TfidfVectorizer(**kwargs)
        tfidf.fit(preprocessed_source_codes)

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        with open(os.path.join(destination_dir, 'tfidf.model'), 'wb') as fp:
            pickle.dump(tfidf, fp)
    
    def _generate_embeddings_fn(self, preprocessed_source_codes: list[Any], **kwargs):
        model_path = kwargs.pop('model_path')

        with open(model_path, 'rb') as fp:
            tfidf = pickle.load(fp)
        
        source_codes_embeddings = tfidf.transform(preprocessed_source_codes).toarray()
        source_codes_embeddings = np.vsplit(source_codes_embeddings, source_codes_embeddings.shape[0])
        source_codes_embeddings = [np.squeeze(sce) for sce in source_codes_embeddings]

        return source_codes_embeddings
