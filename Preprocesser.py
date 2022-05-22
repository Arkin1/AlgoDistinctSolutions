import os
import json
from sys import platform
from tqdm.contrib.concurrent import process_map
from subprocess import run

class Preprocesser:
    def __init__(self, raw_dataset_path, preprocess_dataset_path, compiler):
        self._raw_dataset_path = raw_dataset_path
        self._preprocess_dataset_path = preprocess_dataset_path
        self._compiler = compiler

        self._tokens_path = f"{os.getcwd()}/AlgoLabel/data/tmp/solution.tokens"
        self._solution_path = f"{os.getcwd()}/AlgoLabel/data/tmp/solution.cpp"

        if(not os.path.exists(self._preprocess_dataset_path)):
            os.mkdir(self._preprocess_dataset_path)
    
    def preprocess(self, preprocessing_operations, max_workers = 1):
        self.preprocessing_operations = preprocessing_operations
        cpp_files_preprocess = []
        for root, _, files in os.walk(self._raw_dataset_path):
            for f in files:
                if(f.endswith(".cpp")):
                    cpp_files_preprocess.append(f'{root}/{f}'.replace(self._raw_dataset_path, "")[1:])

        chunks_result = process_map(self._preprocess_helper, cpp_files_preprocess, max_workers = max_workers, chunksize = 1)

        preprocessed_source_codes = [result[0] for result in chunks_result]

        for solution in preprocessed_source_codes:
            # Code taken from AlgoLabel repository
            solution['tokens'] = self._splitSourceCodeTokens(solution['preprocessed'])
        
        f_result = open(f"{self._preprocess_dataset_path}/raw_dataset_prepared.json", 'w', encoding='utf8')
        json.dump(preprocessed_source_codes, f_result, indent=4)
        f_result.close()


    def _preprocess_helper(self, file_name):
        preprocessed_source_codes = []

        problem_name, algorithmic_solution, source_code_id = file_name.split('/')

        source_code_id = source_code_id.replace('.cpp', '')

        f = open(f'{self._raw_dataset_path}/{file_name}', 'r', encoding='utf8')
        source_code = f.read()
        f.close()

        source_code_preprocessed_data = {}

        source_code_preprocessed_data['id'] = source_code_id
        source_code_preprocessed_data['algorithmic_solution'] = f"{problem_name}${algorithmic_solution}"
        source_code_preprocessed_data['raw'] = source_code
        source_code_preprocessed_data['preprocessed'] = source_code

        for operation in self.preprocessing_operations:
            source_code_preprocessed_data = operation.preprocess(source_code_preprocessed_data)

        preprocessed_source_codes.append(source_code_preprocessed_data)
        return preprocessed_source_codes

    def _splitSourceCodeTokens(self, code):
        f = open(self._solution_path, 'w')
        f.write(code)
        f.close()

        tokenizer = self._ensure_tokenizer_exists()
        tokenizer_cmd = "{} {}".format(tokenizer, self._solution_path)
        with open(self._tokens_path, "w") as f:
            rc = run(tokenizer_cmd, shell=True, stdout=f, stderr=f)
            if not rc:
                raise Exception("Error while tokenizing source code")

        with open(self._tokens_path, "r") as f:
            tokens = []
            for line in f:
                if "//" not in line and "/*" not in line:
                    tokens.append(line.strip())
                elif "EOF encountered" in line:
                    raise Exception("Failure occured during tokenizer!")

        return tokens


    def _ensure_tokenizer_exists(self):
        tokenizer_dir  = f"{os.getcwd()}/AlgoLabel/tokenizer/src"

        if platform == "win32":
            tokenizer_exe = "tokenizer.exe"
        else:
            tokenizer_exe = "tokenizer"

        tokenizer_path = f"{tokenizer_dir}/{tokenizer_exe}"

        if not os.path.exists(tokenizer_path):
            current_path = os.getcwd()
            os.chdir(tokenizer_dir)
            run("{} *.cpp *.h -o {}".format(self._compiler, tokenizer_path))
            os.chdir(current_path)

        return tokenizer_path