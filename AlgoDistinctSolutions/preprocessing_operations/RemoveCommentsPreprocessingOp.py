import re
from copy import deepcopy
class RemoveCommentsPreprocessingOp:
    def __init__(self):
        self.regex = '\/\/.*|\/\*(\S|\s)*\*\/'
    
    def preprocess(self, source_code: str) -> str:
        return re.sub(self.regex, '', source_code)