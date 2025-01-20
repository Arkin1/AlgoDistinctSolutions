import re
from preprocessing_operations.PreprocessingOp import PreprocessingOp

class RemoveCommentsPreprocessingOp(PreprocessingOp):
    def __init__(self):
        self.regex = '\/\/.*|\/\*(\S|\s)*\*\/'
    
    def preprocess(self, source_code: str) -> str:
        if not isinstance(source_code, str):
            raise Exception(f'Source code should be a string. Instead it is {type(source_code)}')
        
        return re.sub(self.regex, '', source_code)