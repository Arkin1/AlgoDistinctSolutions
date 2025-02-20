from tqdm import tqdm
from typing import Any
import json
import os
import pandas as pd
from utils import tqdm_multiprocess_map
import logging

logger = logging.getLogger()

class BaseFacade():
    def pretrain(self, dataset_info_path:str, 
                 destination_dir:str, 
                 preprocessing_workers:int,
                 **kwargs):
        _, source_codes = self._read_dataset(dataset_info_path)
        preprocessed_source_codes = self._preprocessing(source_codes, preprocessing_workers)
        logger.info(f"Pretraining {type(self)} model")
        self._pretrain_fn(preprocessed_source_codes, destination_dir, **kwargs)

        
    def generate_embeddings(self, dataset_info_path:str,
                                  destination_dir:str, 
                                  preprocessing_workers:int,
                                  **kwargs):
        dataset_info, source_codes = self._read_dataset(dataset_info_path)
        preprocessed_source_codes = self._preprocessing(source_codes, preprocessing_workers)

        logger.info(f"Generating embeddings using {type(self)}")

        source_codes_embeddings = self._generate_embeddings_fn(preprocessed_source_codes, **kwargs)

        sample_ids = [sample['id'] for sample in dataset_info]

        sample_ids_filtered = []
        source_codes_embeddings_filtered = []

        for sample_id, source_code_embedding in zip(sample_ids, source_codes_embeddings):
            if source_code_embedding is not None:
                sample_ids_filtered.append(sample_id)
                source_codes_embeddings_filtered.append(source_code_embedding)

        df = pd.DataFrame({"id":sample_ids_filtered,  
                           "embeddings": source_codes_embeddings_filtered})

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        df.to_parquet(os.path.join(destination_dir, 'embeddings.parquet'))


    def _preprocessing_fn(self, source_code:str) -> str:
        raise NotImplementedError()

    def _pretrain_fn(self, preprocessed_source_codes: list[Any], destination_dir:str, **kwargs):
        raise NotImplementedError()
    
    def _generate_embeddings_fn(self, preprocessed_source_codes: list[Any], **kwargs):
        raise NotImplementedError()

    def _preprocessing(self, source_codes: list[str], preprocessing_workers:int):
        logger.info(f"Preprocessing the source codes for {type(self)}")
        preprocessed = tqdm_multiprocess_map(self._preprocessing_fn, source_codes, max_workers= preprocessing_workers, chunksize = 64)
        return preprocessed

    def _read_dataset(self, dataset_info_path:str):
        logger.info(f"Reading the dataset_info from path {dataset_info_path}")
        with open(dataset_info_path, 'r', encoding='utf-8') as fp:
            dataset_info = json.load(fp)
        
        dataset_root_path = os.path.dirname(dataset_info_path)
        
        logger.info(f"Reading the source codes based on dataset info")
        source_codes = []
        for submission_info in tqdm(dataset_info):
            submission_path = submission_info['path']

            submission_path = os.path.join(dataset_root_path, submission_path)

            with open(submission_path, 'r', encoding='utf-8') as fp:
                source_code = fp.read()

            source_codes.append(source_code)
        return dataset_info, source_codes
        