import os
import shutil

from Constants import NOT_COMPILABLE

def storeDatasetStatistics(datasetPath, statisticsCsvPath):
    datasetStatistics = open(statisticsCsvPath, "w")

    notCompilablePath = ""

    for root, folders, files in os.walk(datasetPath):
        if(len(folders)==0):
            problem = root.split("/")[-2]
            solution = root.split("/")[-1]
            
            numberFiles = len(files)
            numberCompilableFiles = 0
            nr = 0
            for f in files:
                print(f'{problem}/{solution} {nr}/{numberFiles - 1}')
                
                if(os.system(f"g++ {root}/{f} -fsyntax-only") == 0):
                    numberCompilableFiles+=1
                else:
                    print('-----------------------------')
                    problemPath = root.split(solution)[0]
                    notCompilablePath = f"{problemPath}{solution}_{NOT_COMPILABLE_PATH}"   
                    
                    if(not os.path.exists(notCompilablePath)):
                        os.mkdir(notCompilablePath)
                    
                    shutil.move(f"{root}/{f}", f"{notCompilablePath}/{f}")

                nr+=1
            
            datasetStatistics.write(f'{problem}, {solution}, {numberFiles}, {numberCompilableFiles}\n')
