import re
class RemoveCommentsPreprocessingOp:
    def __init__(self):
        self.regex = '\/\/.*|\/\*(\S|\s)*\*\/'
    
    def preprocess(self, source_code_data):
        source_code_data['preprocessed'] = re.sub(self.regex, '', source_code_data['preprocessed'])
        return source_code_data