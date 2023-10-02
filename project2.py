import zipfile
import sys
import os
import io
import glob
import numpy as np

class dataExtractor:
    def __init__(self, zipPath):
        self.allData = None
        self._initializeAllData()
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
        self.allData = result
    def _extractFile(self):
        counter = 0
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
                    counter += 1
        print(counter)

ext = dataExtractor("project2_data.zip")