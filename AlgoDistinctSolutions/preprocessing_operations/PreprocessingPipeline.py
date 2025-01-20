from preprocessing_operations.PreprocessingOp import PreprocessingOp

class PreprocessingPipeline():
    def __init__(self, operations: list[PreprocessingOp]):
        self.operations = operations

    def preprocess(self, source_code:str):
        for op in self.operations:
            source_code = op.preprocess(source_code)
        return source_code