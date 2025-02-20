import re
from preprocessing_operations.PreprocessingOp import PreprocessingOp

class RemoveIncludesUsingPreprocessingOp(PreprocessingOp):
    def __init__(self):
        self.regex = 'using namespace (\s)*.*?;|#(\s)*(include|import)(\s)*<.*?>|#(\s)*(include|import)(\s)*\".*?\"'
    
    def preprocess(self, source_code: str) -> str:
        if not isinstance(source_code, str):
            raise Exception(f'Source code should be a string. Instead it is {type(source_code)}')
        
        return re.sub(self.regex, '', source_code)