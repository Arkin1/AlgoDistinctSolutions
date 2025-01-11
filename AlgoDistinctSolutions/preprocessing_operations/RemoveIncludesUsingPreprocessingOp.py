import re

class RemoveIncludesUsingPreprocessingOp:
    def __init__(self):
        self.regex = 'using namespace (\s|\S)*?;|#include(\s|\S)*?>'
    
    def preprocess(self, source_code: str) -> str:
        return re.sub(self.regex, '', source_code)