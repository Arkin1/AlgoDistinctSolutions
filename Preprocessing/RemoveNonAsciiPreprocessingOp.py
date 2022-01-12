class RemoveNonAsciiPreprocessingOp:
    def preprocess(self, source_code_data):
        source_code_data['preprocessed'] = source_code_data['preprocessed'].encode("ascii", "ignore").decode()
        return source_code_data