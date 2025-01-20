import os
from Constants import TMP_FOLDER_PATH
from uuid import uuid4
from preprocessing_operations.PreprocessingOp import PreprocessingOp

class ReplaceMacrosPreprocessingOp(PreprocessingOp):
    def preprocess(self, source_code:str) -> str:
        if not isinstance(source_code, str):
            raise Exception(f'Source code should be a string. Instead it is {type(source_code)}')
        
        id_ = f'tmp_{uuid4()}'

        source_code_tmp_path = os.path.join(TMP_FOLDER_PATH, f'{id_}.cpp')
        
        with open(source_code_tmp_path, 'w', encoding='utf8') as fp:
            fp.write(source_code)

        preprocessed_source_code_tmp_path = os.path.join(TMP_FOLDER_PATH, f'preprocessed_{id_}.cpp')

        os.system(f"g++ -E -P {source_code_tmp_path} -o {preprocessed_source_code_tmp_path}")

        with open(preprocessed_source_code_tmp_path, encoding='utf8') as fp:
            preprocessed_source_code= fp.read()

        os.remove(source_code_tmp_path)
        os.remove(preprocessed_source_code_tmp_path)
        
        return preprocessed_source_code

        