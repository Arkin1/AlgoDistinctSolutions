import os

def storeDatasetStatistics(datasetPath, statisticsCsvPath):
    datasetStatistics = open(statisticsCsvPath, "w")

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
                nr+=1
            
            datasetStatistics.write(f'{problem}, {solution}, {numberFiles}, {numberCompilableFiles}\n')