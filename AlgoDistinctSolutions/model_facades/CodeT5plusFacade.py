from preprocessing_operations.RemoveUnusedFunctionsPreprocessingOp import RemoveUnusedFunctionsPreprocessingOp
from preprocessing_operations.RemoveIncludesUsingPreprocessingOp import RemoveIncludesUsingPreprocessingOp
from preprocessing_operations.RemoveCommentsPreprocessingOp import RemoveCommentsPreprocessingOp
from preprocessing_operations.ReplaceMacrosPreprocessingOp import ReplaceMacrosPreprocessingOp
from preprocessing_operations.ExtractFunctionsAndMethodsProcessingOp import ExtractFunctionsAndMethodsPreprocessingOp

import logging
from tqdm import tqdm
from typing import Any
import torch
from utils import to_batches
from transformers import AutoTokenizer, AutoModel

from model_facades.BaseFacade import BaseFacade

logger = logging.getLogger()

class CodeT5plusFacade(BaseFacade):
    def _preprocessing_fn(self, source_code:str) -> str:
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

    def _generate_embeddings_fn(self, preprocessed_source_codes:list[Any], **kwargs):
        batch_size:int = kwargs.pop('batch_size')
        padding:bool = kwargs.pop('padding')
        model_path:str = kwargs.pop('model_path')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code = True).to(device)

        logger.info(f"Generating embeddings")
        source_codes_embeddings = []

        for functions in tqdm(preprocessed_source_codes):
            if len(functions) == 0:
                source_codes_embeddings.append(None)
                continue
            
            func_embeddings = []
            for batch_func in to_batches(functions, batch_size):
                with torch.no_grad():
                    tokens_ids = tokenizer(batch_func,
                                        padding = padding,
                                        truncation = True,
                                        return_tensors = 'pt').to(device)
                    max_func_embedding = model(**tokens_ids)

                func_embeddings.append(max_func_embedding)
            func_embeddings = torch.concat(func_embeddings)
            func_embeddings = func_embeddings.mean(0).detach().cpu().numpy()

            source_codes_embeddings.append(func_embeddings)
        return source_codes_embeddings