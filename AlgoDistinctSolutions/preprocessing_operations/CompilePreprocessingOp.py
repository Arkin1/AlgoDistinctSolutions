import os
from Constants import TMP_FOLDER_PATH
from uuid import uuid4
import logging
from preprocessing_operations.PreprocessingOp import PreprocessingOp

logger = logging.getLogger()
class CompilePreprocessingOp(PreprocessingOp):
    def __init__(self):
        self.arguments_to_try = ['-std=gnu++17 -O2 -Wno-error -w',
                                 '-std=gnu++14 -O2 -Wno-error -w',
                                 '-std=gnu++03 -O2 -Wno-error -w',
                                 '-std=gnu++98 -O2 -Wno-error -w']
    def preprocess(self, source_code:str) -> str:
        if not isinstance(source_code, str):
            raise Exception(f'Source code should be a string. Instead it is {type(source_code)}')
        
        id_ = f'tmp_{uuid4()}'

        source_code_tmp_path = os.path.join(TMP_FOLDER_PATH, f'{id_}.cpp')
        
        with open(source_code_tmp_path, 'w', encoding='utf8') as fp:
            fp.write(source_code)

        compiled_source_code_tmp_path = os.path.join(TMP_FOLDER_PATH, f'compiled_{id_}.o')

        can_compile = False

        for arguments_to_try in self.arguments_to_try:
            
            if os.system(f"g++ {arguments_to_try} {source_code_tmp_path} -o {compiled_source_code_tmp_path}") != 0:
                logger.info(f"Compilation failed with args {arguments_to_try}.")
                continue
            can_compile = True
            break
        
        if can_compile == False:
            logger.info(f"No g++ version worked to compile {source_code_tmp_path}")
            return None
            
        os.remove(source_code_tmp_path)
        
        return compiled_source_code_tmp_path

        