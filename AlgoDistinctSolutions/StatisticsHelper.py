import itertools
import os
import shutil
import pickle
from matplotlib import pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from Constants import NOT_COMPILABLE_PATH

import Constants

class StatisticsHelper():
    def __init__(self):
        self._incrementalTfidfFileName = "Data/Embeddings/incremental_tfidf"
        self._statisticsFolderPath = "Data/Statistics"

        if not os.path.exists(self._statisticsFolderPath):
            os.mkdir(self._statisticsFolderPath)
        
    def handleStatistics(self, statisticsType:str):
        if "dataset" in statisticsType:
            self._store_dataset_statistics(Constants.RAW_DATASET_PATH, f"{self._statisticsFolderPath}/dataset_statistics.csv")
        if "tfidf_incremental" in statisticsType:
            self._store_tfidf_incremental_statistics(f"{self._statisticsFolderPath}/tfidf_incremental_statistics.csv")


    def _load_tfidf_embeddings(self, inputPath):
        problems = pickle.load(open(inputPath, "rb"))
        embeddings = []

        for problem in problems:
            embeddings.append({
                'index': problem['index'],
                'label': problem['label'],
                'embeddings': problem['tfidf'].toarray()[0]
            })
        
        return embeddings

    def _store_tfidf_incremental_statistics(self, outputPath):
        tfidf_incremental_statistics = open(outputPath, "w")
        embeddings_file_names = [f for f in os.listdir(f"{self._incrementalTfidfFileName}/") if f.endswith(".json")]
        classification_reports = {}
        f1_scores = {}
        train_size_list = []
        
        for embeddings_file_name in embeddings_file_names:
            embeddings = self._load_tfidf_embeddings(f"{self._incrementalTfidfFileName}/{embeddings_file_name}")
            train_size = int(embeddings_file_name.split('_')[-1].split('.')[0])
            train_size_list.append(train_size)
            labels = list(set([x['label'] for x in embeddings]))

            train_data = []
            test_data = []
            for _, value in itertools.groupby(embeddings, lambda x: x['label']):
                train, test = train_test_split(list(value), train_size=train_size)                
                train_data.extend(train)
                test_data.extend(test)

            lr_clf = LogisticRegression()

            X_train = np.array([x['embeddings'] for x in train_data])
            y_train = [x['label'] for x in train_data]

            X_test = np.array([x['embeddings'] for x in test_data])
            y_test = [x['label'] for x in test_data]

            lr_clf.fit(X_train, y_train)
            y_pred = lr_clf.predict(X_test)
            
            f1_scores[train_size] = f1_score(y_test, y_pred, average='weighted') 
            classification_reports[train_size] = classification_report(y_true=y_test, y_pred=y_pred, target_names=labels, output_dict=True)
        
        sorted_f1_scores = sorted(f1_scores.items(), key=lambda x: x[0])
        plt_f1_scores = [x[1] for x in sorted_f1_scores]

        plt.plot(sorted(train_size_list), plt_f1_scores, label = "line 1")
        plt.show()
        plt.savefig('Data/f1_scores_tfidf.png')

        sorted_classification_reports = dict(sorted(classification_reports.items(), key=lambda x: x[0]))
        
        tfidf_incremental_statistics.write("Problem, Size_train, F1_score, Precision, Recall\n")

        for train_size, generated_classification_report in sorted_classification_reports.items():
            for problem, score in generated_classification_report.items():
                try:
                    tfidf_incremental_statistics.write(f"{problem.replace('$','/')}, {train_size}, {score['f1-score']}, {score['precision']}, {score['recall']}\n")
                except:
                    pass
        
        tfidf_incremental_statistics.close()
