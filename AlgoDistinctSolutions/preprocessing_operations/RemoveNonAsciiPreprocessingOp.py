class RemoveNonAsciiPreprocessingOp:    
    def preprocess(self, source_code: str) -> str:
        return source_code.encode("ascii", "ignore").decode()