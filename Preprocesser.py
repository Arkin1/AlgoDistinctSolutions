import os
import re
from multiprocessing import Pool

class Preprocesser:
    def __init__(self, rawDatasetPath = 'Data/RawDataset', preprocessedDatasetPath = 'Data/PreprocessedDataset'):
        self.rawDatasetPath = rawDatasetPath
        self.preprocessedDatasetPath = preprocessedDatasetPath

        self.comments_regex = re.compile('//.*|/\*.*\*/')
        self.include_using_regex = re.compile('#include<.+>|using namespace .+;')

    def preprocess(self, **kwargs):
        
        if(kwargs['number_proc'] <= 0):
            raise Exception('number_threads parameter should be greater than 0')

        files = []

        for root, folder, files in os.walk(self.rawDatasetPath):
            for f in files:
                if(f.endswith('.cpp')):
                    files.append(f'{root}/{f}')
    
        chunkSize = len(files) // kwargs['number_proc']

        chunks = [files[i:min(i+chunkSize, len(files))] for i in range(0, len(data), chunkSize)]

        with Pool(kwargs['number_proc']) as pool:
            p.map(__preprocess, chunks, **kwargs)

    def __preprocess(self, chunk, **kwargs):

        for f in chunk:
            source_code = open(f, 'r', encoding='utf8')

            if(kwargs['remove_comments']):
                source_code = self.comments_regex.sub('', source_code)

            if(kwargs['remove_include_using_directives']):
                source_code = self.include_regex.sub('', source_code)
                




