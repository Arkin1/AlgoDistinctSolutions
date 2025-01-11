import Constants
import os
import shutil
import pandas as pd
class DatasetStatistics:
    def compute_all_dataset_statistics(datasetPath: str, output_folder: str):
        not_compilable_path = ""

        for root, folders, files in os.walk(datasetPath):
            if(len(folders)==0):
                problem = root.split("/")[-2]
                solution = root.split("/")[-1]
                
                number_files = len(files)
                number_compilable_files = 0
                nr = 0
                for f in files:
                    print(f'{problem}/{solution} {nr}/{number_files - 1}')
                    
                    if(os.system(f"g++ {root}/{f} -fsyntax-only") == 0):
                        number_compilable_files+=1
                    else:
                        print('-----------------------------')
                        problem_path = root.split(solution)[0]
                        not_compilable_path = f"{problem_path}{solution}_{Constants.NOT_COMPILABLE_PATH}"   
                        
                        if(not os.path.exists(not_compilable_path)):
                            os.mkdir(not_compilable_path)
                        
                        shutil.move(f"{root}/{f}", f"{not_compilable_path}/{f}")

                    nr+=1
                
                dataset_statistics.write(f'{problem}, {solution}, {number_files}, {number_compilable_files}\n')
        dataset_statistics.close()
    
    def determine_metadata_dataset():
        pass
    def _determine_if_solution_is_compilable():
        pass

    def _store_dataset_statistics(self, datasetPath, statisticsCsvPath):
        