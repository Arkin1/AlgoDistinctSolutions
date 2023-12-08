import os
import numpy as np
from EmbeddingsLoader import EmbeddingsLoader

class IValidationPipelines():
    def _create_csv_validation(self, name):
        if(not os.path.exists('Data/Validation')):
            os.mkdir('Data/Validation')

        csv = open(f'Data/Validation/{name}.csv', "w")

        return csv

    def _append_to_csv(self, csv, row):
        csv.write(f"{str.join(',', row)}\n")
    
    def _close_csv(self, csv):
        csv.close()

    def _split_embeddings_data_per_problem(self, embeddings_loader:EmbeddingsLoader):

        print(f"Extracting the {embeddings_loader.get_name()} embeddings from disk")

        problem_dict = {}

        for solution in embeddings_loader.get_embeddings():
            function_embeddings = np.array(solution["embeddings"])
            solution_embedding = np.mean(function_embeddings, 0)

            problem = solution['label'].split("$")[0]
            if(problem not in problem_dict):
                problem_dict[problem] = {'indexes':[], 'X' : [], 'Y': []}

            problem_dict[problem]['indexes'].append(solution['index'])
            problem_dict[problem]['X'].append(solution_embedding)
            problem_dict[problem]['Y'].append(solution['label'])
        
        embeddings = {}
        embeddings['name'] = embeddings_loader.get_name()
        embeddings['problemDict'] = problem_dict

        return embeddings