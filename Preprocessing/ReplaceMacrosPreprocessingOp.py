import os
from Constants import TMP_PATH
import subprocess


class ReplaceMacrosPreprocessingOp:
    def preprocess(self, source_code_data):
        code = source_code_data['preprocessed']
        id = source_code_data['id']

        f = open(f'{TMP_PATH}/{id}.cpp', 'w', encoding='utf8')
        f.write(code)
        f.close()

        os.system(f"g++ -E -P {TMP_PATH}/{id}.cpp -o {TMP_PATH}/{id}.preprocessed.cpp")

        f = open(f'{TMP_PATH}/{id}.preprocessed.cpp' )
        source_code_data['preprocessed'] = f.read()
        f.close()

        os.remove(f'{TMP_PATH}/{id}.preprocessed.cpp')
        os.remove(f'{TMP_PATH}/{id}.cpp')
        
        return source_code_data

        