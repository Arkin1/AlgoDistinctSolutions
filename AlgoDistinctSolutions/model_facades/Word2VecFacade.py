from preprocessing_operations.ReplaceMacrosPreprocessingOp import ReplaceMacrosPreprocessingOp
from preprocessing_operations.RemoveIncludesUsingPreprocessingOp import RemoveIncludesUsingPreprocessingOp 
from preprocessing_operations.RemoveCommentsPreprocessingOp import RemoveCommentsPreprocessingOp 
from preprocessing_operations.RemoveNonAsciiPreprocessingOp import RemoveNonAsciiPreprocessingOp 
from preprocessing_operations.RemoveUnusedFunctionsPreprocessingOp import RemoveUnusedFunctionsPreprocessingOp
from preprocessing_operations.TokenizePreprocessingOp import TokenizePreprocessingOp

import gensim
import logging
from tqdm import tqdm
import os
import numpy as np
from typing import Any

from model_facades.BaseFacade import BaseFacade

logger = logging.getLogger()

class Word2VecFacade(BaseFacade):
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
        return tokens

    def _pretrain_fn(self, preprocessed_source_codes: list[Any], destination_dir:str, **kwargs):
        w2v = gensim.models.Word2Vec(preprocessed_source_codes, **kwargs)

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        w2v.save(os.path.join(destination_dir, 'w2v.model'))


    def _generate_embeddings_fn(self, preprocessed_source_codes: list[Any], **kwargs):
        model_path:str = kwargs.pop('model_path')

        w2v = gensim.models.Word2Vec.load(model_path)
        wv = w2v.wv
        
        logger.info(f"Generating embeddings")
        source_codes_embeddings = []
        for sc in tqdm(preprocessed_source_codes):
            embedding_per_words = []

            for w in sc:
                if w in wv:
                    embedding_per_words.append(wv[w])
            embedding_per_words = np.stack(embedding_per_words)

            source_codes_embeddings.append(embedding_per_words.mean(axis = 0))

        return source_codes_embeddings