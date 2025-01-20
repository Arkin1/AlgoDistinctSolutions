from preprocessing_operations.PreprocessingOp import PreprocessingOp

class RemoveNonAsciiPreprocessingOp(PreprocessingOp):    
    def preprocess(self, source_code: str) -> str:
        if not isinstance(source_code, str):
            raise Exception(f'Source code should be a string. Instead it is {type(source_code)}')
        
        return source_code.encode("ascii", "ignore").decode()