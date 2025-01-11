import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

class DatasetMetadataExtractor():
    def extract_metadata(self, datasetPath: str, outputPath:str, determine_compilable = True, determine_size_characters = True):
        all_source_code_paths = self._determine_source_files_paths(datasetPath)

        data = {"Path":[], "Id":[], "Problem":[], "Solution":[], "IsCompilable":[], "NumCharacters":[]}

        for source_code_path in tqdm(all_source_code_paths):
            ss_path = Path(source_code_path)

            id = ss_path.name
            solution = ss_path.parent.name
            problem = ss_path.parent.parent.name
            is_compilable = self._determine_is_compilable_source_code(source_code_path)
            num_characters = self._determine_num_characters_solution(source_code_path)

            data['Path'].append(source_code_path)
            data['Id'].append(id)
            data['Problem'].append(problem)
            data['Solution'].append(solution)
            data['IsCompilable'].append(is_compilable)
            data['NumCharacters'].append(num_characters)

        df = pd.DataFrame(data)
        df.to_csv(outputPath)
        
    def _determine_source_codes_paths(self, datasetPath:str)->list[str]:
        source_files_paths = []
        for root, _, files in os.walk(datasetPath):
                for f in files:
                    source_files_paths.extend(os.path.join(root, f))
        return source_files_paths


    def _determine_is_compilable_source_code(self, source_code_path:str) -> bool:
        if(os.system(f"g++ {source_code_path} -fsyntax-only") == 0):
            return True
        else:
            return False
        
    def _determine_num_characters_solution(self, source_code_path:str) -> str:
        with open(source_code_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()

        return source_code

        


