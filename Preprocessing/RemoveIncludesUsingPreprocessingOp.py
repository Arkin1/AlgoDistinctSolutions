import re
class RemoveIncludesUsingPreprocessingOp:
    def __init__(self):
        self.regex = 'using namespace (\s|\S)*?;|#include(\s|\S)*?>'
    
    def preprocess(self, source_code_data):
        source_code_data['preprocessed'] = re.sub(self.regex, '', source_code_data['preprocessed'])
        return source_code_data