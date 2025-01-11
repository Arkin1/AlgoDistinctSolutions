import sys

sys.path.insert(0,"code2vec")

import os
import shutil
import pandas
import re
import hashlib
import code2vec as c2v
import logging
from Constants import TMP_PATH

sys.path.insert(0, "AlgoLabel")

from config import Config
from infercode.client.infercode_client import InferCodeClient
from safe.safe import SAFE

class Code2VecEmbedder():
    def __init__(self, path):
        self.path = path
        self.config = Config(set_defaults=True)
        self.config.PREDICT = True
        self.config.MODEL_LOAD_PATH = path
        self.config.DL_FRAMEWORK = 'tensorflow'
        self.config.SEPARATE_OOV_AND_PAD = False
        self.config.EXPORT_CODE_VECTORS = True
        self.model = c2v.load_model_dynamically(self.config)
        self.astminer = "Algolabel/astminer/lib-0.5.jar"

    def _split_tokens(self, str):
        matches = re.findall(r'[a-zA-Z]+', str)

        tokens = []
        for match in matches:
            tokens_matches = re.findall(r'[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[A-Z]', match)

            for token in tokens_matches:
                tokens.append(token)
        result = '|'.join(tokens)

        if(result ==''):
            return 'unknown'

        return  result

    def _extract_c2v_format_project(self, preprocessFolder,  maxL, maxW, maxContexts, maxTokens, maxPaths):
        self.currentFolder = os.getcwd()
        tmpFolder = os.getcwd() + '/' + 'tmp'

        for root, _, files in os.walk(f"{preprocessFolder}"):
            if(len(files) > 0):
                cppFiles = [f for f in files if f.endswith(".cpp")]
                if(len(cppFiles) > 0):
                    print(f"Copying files to tmp folder")

                    for f in files:
                        if(os.path.isdir(tmpFolder)):
                            shutil.rmtree(tmpFolder)
                        os.mkdir(tmpFolder)

                        shutil.copy(f"{root}/{f}", f"{tmpFolder}/{f}")

                        os.chdir("astminer")
                        os.system(f"./cli.sh code2vec --lang cpp --project {tmpFolder} --output {tmpFolder} --maxL {maxL} --maxW {maxW} --maxContexts {maxContexts} --maxTokens {maxTokens} --maxPaths {maxPaths}  --split-tokens --granularity method")
                        os.chdir("..")

                        tokens = pandas.read_csv(f'{tmpFolder}/cpp/tokens.csv')
                        paths = pandas.read_csv(f'{tmpFolder}/cpp/paths.csv')
                        node_types =  pandas.read_csv(f'{tmpFolder}/cpp/node_types.csv')

                        tokens_dict = dict(zip(tokens.iloc[:,0], tokens.iloc[:, 1]))
                        paths_dict = dict(zip(paths.iloc[:,0], paths.iloc[:,1]))
                        node_types_dict = dict(zip(node_types.iloc[:, 0], node_types.iloc[:, 1]))
                        functions = []
                        
                        with open(f"{tmpFolder}/cpp/path_contexts.csv","r") as path_contexts_file:
                            for line in path_contexts_file:
                                line_splitted = line.split(" ")

                                dataset_entry = ""

                                func_name = line_splitted[0]
                                i = 1
                                ok = 1
                                while(ok):
                                    try:
                                        token1_id, path_id, token2_id = line_splitted[i].split(",")
                                        ok = 0
                                    except:
                                        func_name+=line_splitted[i]
                                        i+=1
                                
                                dataset_entry+= f"{self._split_tokens(func_name).lower()} "

                                for path_context in line_splitted[i:]:
                                    token1_id, path_id, token2_id = path_context.split(",")

                                    dataset_entry += f"{tokens_dict[int(token1_id)]},"
                                    path = ""
                                    for node_type_id in paths_dict[int(path_id)].split(" "):
                                        path += f"{node_types_dict[int(node_type_id)].split(' ')[0]}"

                                    path = hashlib.sha1(path.encode('utf-8')).hexdigest()

                                    dataset_entry+= f"{path},"
                                    dataset_entry += f"{tokens_dict[int(token2_id)]} "
                                functions.append(dataset_entry)
                        yield  (f, functions)

    def get_embeddings_from_files(self, folder):
        files = self._extract_c2v_format_project(folder, 10, 10, self.config.MAX_CONTEXTS, 100000, 100000)
        
        fileEmbeddings = []
        
        for file_name, file_functions in files:
            functionsPathContexts = []

            c2vEmbeddings = []
            
            for func in file_functions:
                split =  func.split(" ")
                f_name = split[0]
                path_contexts = split[1:]

                if(len(path_contexts) > self.config.MAX_CONTEXTS):
                    path_contexts = path_contexts[:self.config.MAX_CONTEXTS]
                else:
                    path_contexts.extend(['']*(self.config.MAX_CONTEXTS - len(path_contexts)))

                functionsPathContexts.append(f_name + " " + " ".join(path_contexts))

            functions_embeddings = self.model.predict(functionsPathContexts)

            for function in functions_embeddings:
                c2vEmbeddings.append(function.code_vector.tolist())

            fileEmbeddings.append({
                    "index":file_name.split('.')[0],
                    "embeddings": c2vEmbeddings
                })
        return fileEmbeddings


class InferCodeEmbedder():
    def __init__(self) :
        logging.basicConfig(level=logging.INFO)
        self._infercode = InferCodeClient(language='cpp')
        self._infercode.init_from_config()

    def get_embeddings_from_dataset(self, dataset):
        embeddings = []
        problematic_files = []

        for problem in dataset:
            index = problem['id']
            label = problem['algorithmic_solution']
            code = problem['preprocessed']
            
            try:
                file_embeddings = self._infercode.encode([code])
        
                embeddings.append({
                            "index": index,
                            "label": label,
                            "embeddings": [file_embeddings[0].tolist()]
                        })
            except:
                embeddings.append({
                            "index": index,
                            "label": label,
                            "embeddings": [[]]
                        })
                problematic_files.append(f"{index}.cpp")
            
        return embeddings


class SafeEmbedder:
    def __init__(self, dataset):
        self._dataset = dataset
        self._safe = SAFE(model_path='safe/data/safe.pb', instr_conv='safe/data/i2v/word2id.json')
        os.makedirs(TMP_PATH, exist_ok=True)
        sys.path.insert(0, 'AlgoLabel')

    def compute_embeddings(self):
        safe_embeddings = []

        for entry in self._dataset:
            res = self._compute_embeddings_helper(entry)
            safe_embeddings.append({
                "index": entry['id'],
                "label": entry['algorithmic_solution'],
                "embeddings": res
            })
        
        return safe_embeddings

    
    def _compute_embeddings_helper(self, entry):
        
        f = open(f'{TMP_PATH}/tmp_safe.cpp', 'w', encoding='utf8')
        f.write(entry['raw'])
        f.close()

        _ = os.system(f'g++ {TMP_PATH}/tmp_safe.cpp -O3 -o {TMP_PATH}/object_code.o')

        try:
            embeddings = self._safe.embedd_functions(str(f'{TMP_PATH}/object_code.o'))
            embeddings = [x.tolist()[0] for x in embeddings if x is not None]

            return embeddings
        except:
            return None
