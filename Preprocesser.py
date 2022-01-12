import os
from collections import defaultdict
import json

from Preprocessing.RemoveCommentsPreprocessingOp import RemoveCommentsPreprocessingOp
from Preprocessing.RemoveIncludesUsingPreprocessingOp import RemoveIncludesUsingPreprocessingOp
from Preprocessing.RemoveUnusedFunctionsPreprocessingOp import RemoveUnusedFunctionsPreprocessingOp
from Preprocessing.RemoveNonAsciiPreprocessingOp import RemoveNonAsciiPreprocessingOp
from Preprocessing.ReplaceMacrosPreprocessingOp import ReplaceMacrosPreprocessingOp

from tqdm.contrib.concurrent import process_map

class Preprocesser:
    def __init__(self, raw_dataset_path, preprocess_dataset_path):
        self.raw_dataset_path = raw_dataset_path
        self.preprocess_dataset_path = preprocess_dataset_path

        if(not os.path.exists(self.preprocess_dataset_path)):
            os.mkdir(self.preprocess_dataset_path)
    
    def preprocess(self, preprocessing_operations, max_workers = 1):
        self.preprocessing_operations = preprocessing_operations
        cpp_files_preprocess = []
        for root, _, files in os.walk(self.raw_dataset_path):
            for f in files:
                if(f.endswith(".cpp")):
                    cpp_files_preprocess.append(f'{root}/{f}'.replace(self.raw_dataset_path, "")[1:])


        chunks_result = process_map(self.preprocess2, cpp_files_preprocess, max_workers = max_workers, chunksize = 1)

        preprocessed_source_codes = defaultdict(list)

        for result in chunks_result:
            for key, value in result.items():
                for v in value:
                    preprocessed_source_codes[key].append(v)

        f_result = open(f"{self.preprocess_dataset_path}/preprocessed_dataset.json", 'w', encoding='utf8')
        json.dump(preprocessed_source_codes, f_result)
        f_result.close()

            

    def preprocess2(self, file_name):
        preprocessed_source_codes = defaultdict(list)

        problem_name, algorithmic_solution, source_code_id = file_name.split('/')

        source_code_id = source_code_id.replace('.cpp', '')

        f = open(f'{self.raw_dataset_path}/{file_name}', 'r', encoding='utf8')
        source_code = f.read()
        f.close()

        source_code_preprocessed_data = {}

        source_code_preprocessed_data['id'] = source_code_id
        source_code_preprocessed_data['algorithmic_solution'] = algorithmic_solution
        source_code_preprocessed_data['raw'] = source_code
        source_code_preprocessed_data['preprocessed'] = source_code

        for operation in self.preprocessing_operations:
            source_code_preprocessed_data = operation.preprocess(source_code_preprocessed_data)
        
        preprocessed_source_codes[problem_name].append(source_code_preprocessed_data)

        return preprocessed_source_codes