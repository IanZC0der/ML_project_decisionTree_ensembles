import zipfile
import sys
import os
import io
import glob
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

class dataExtractor:
    def __init__(self, zipPath):
        self.allData = self._initializeAllData()
        self.zipPath = os.path.join("./", zipPath)
        self.folderName = None
        self._extractFile()

    
    def _initializeAllData(self):
        result = {}
        for clause in [300, 500, 1000, 1500, 1800]:
            clauseDict = {}
            for example in [100, 1000, 5000]:
                clauseDict[example] = clauseDict.get(example, {"train": None, "valid": None, "test": None})
            result[clause] = result.get(clause, clauseDict)
        return result
    def _extractFile(self):
        with zipfile.ZipFile(self.zipPath, "r") as folder:
            csvList = [_ for _ in folder.namelist() if _.endswith(".csv")]
            for oneCsv in csvList:
                csvName = oneCsv.split("/")[-1]
                dataType = csvName.split("_")[0]
                clauseNumber = int(csvName.split("_")[1][1:])
                exampleNumber = int(csvName.split("_")[2][1:].split(".")[0])
                with folder.open(oneCsv) as f:
                    dataSet = np.genfromtxt(f, delimiter=",", dtype=int)
                    self.allData[clauseNumber][exampleNumber][dataType] = dataSet


class decisionTreeExperiment(dataExtractor):
    def __init__(self, zipPath):
        super.__init__(self, zipPath)
        self.bestParams = self._initializeAllData()
        self._initializeBestParams()
    
    def _initializeBestParams(self):
        for key1, val1 in self.bestParams.items():
            for key2, val2 in val1.items():
                val2 = {"bestParams": None, "accuracy": None, "F1Score": None}
        



# ext = dataExtractor("project2_data.zip")