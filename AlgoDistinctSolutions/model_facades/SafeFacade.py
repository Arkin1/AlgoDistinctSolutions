from preprocessing_operations.RemoveUnusedFunctionsPreprocessingOp import RemoveUnusedFunctionsPreprocessingOp
from preprocessing_operations.CompilePreprocessingOp import CompilePreprocessingOp

import logging
from typing import Any
from tqdm import tqdm
from safe.safe import SAFE
import numpy as np

from model_facades.BaseFacade import BaseFacade

logger = logging.getLogger()

class SafeFacade(BaseFacade):
    def _preprocessing_fn(self, source_code:str) -> str:
        remove_unused_functions_pre_op = RemoveUnusedFunctionsPreprocessingOp()
        compile_pre_op = CompilePreprocessingOp()
        
        preprocessed_source_code = remove_unused_functions_pre_op.preprocess(source_code)
        compiled_source_code_path = compile_pre_op.preprocess(preprocessed_source_code)
 
        return compiled_source_code_path
    
    def _generate_embeddings_fn(self, preprocessed_source_codes: list[Any], **kwargs):
        model_path = kwargs.pop('model_path')
        instr_conv_path = kwargs.pop('instr_conv_path')
        max_instr = kwargs.pop('max_instr')

        embedder = SAFE(model_path = model_path,
                        instr_conv= instr_conv_path,
                        max_instr = max_instr)

        source_codes_embeddings = []

        for compiled_path_source_code in tqdm(preprocessed_source_codes):
            if compiled_path_source_code is not None:
                function_embeddings = embedder.embedd_functions(compiled_path_source_code)
                function_embeddings = np.concatenate(function_embeddings, 0)
                source_embeddings = np.mean(function_embeddings, 0)
                source_codes_embeddings.append(source_embeddings)
            else:
                source_codes_embeddings.append(None)
        return source_codes_embeddings
    