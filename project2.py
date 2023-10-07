import zipfile
import sys
import os
import io
import glob
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

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
        super().__init__(zipPath)
        self.bestParams = self._initialization()
        self.testResults = self._initialization()
        self.scoring = {
            'f1_score': make_scorer(f1_score, zero_division=0.0, average="binary"),
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0.0, average="binary"),
            'recall': make_scorer(recall_score, average="binary")
        }
        self.paramGrid = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": list(range(2, 53, 2)),
            "max_features": [None, "sqrt", "log2"],
            "min_samples_split": list(range(2, 101, 2))
        }
    
    def _initialization(self):
        result = {}
        for clause in [300, 500, 1000, 1500, 1800]:
            clauseDict = {}
            for example in [100, 1000, 5000]:
                clauseDict[example] = clauseDict.get(example, {"bestParams": None, "accuracy": None, "F1Score": None})
            result[clause] = result.get(clause, clauseDict)
        return result
    
    def _randomizedGridSearch(self, X, Y):
        model = DecisionTreeClassifier()
        search = RandomizedSearchCV(estimator=model, param_distributions=self.paramGrid, n_iter=50, scoring=self.scoring, n_jobs=-1, refit="f1_score")
        search.fit(X, Y)
        return search.best_params_, search.best_estimator_
    def searchForEachDataSet(self):
        for clauseNumber, val1 in self.allData.items():
            for exampleNumber, val2 in val1.items():
                X, Y = val2["valid"][:, :-1], val2["valid"][:, -1]
                self.bestParams[clauseNumber][exampleNumber]["bestParams"], model = self._randomizedGridSearch(X, Y)
                self.testResults[clauseNumber][exampleNumber]["bestParams"] = self.bestParams[clauseNumber][exampleNumber]["bestParams"]
                [self.bestParams[clauseNumber][exampleNumber]["accuracy"], self.bestParams[clauseNumber][exampleNumber]["F1Score"]] = self._calc(Y, model.predict(X))
    
    def test(self):
        for clauseNumber, val1 in self.allData.items():
            for exampleNumber, val2 in val1.items():
                X, Y = np.vstack((self.allData[clauseNumber][exampleNumber]["train"][:,:-1], self.allData[clauseNumber][exampleNumber]["valid"][:,:-1])), np.concatenate((self.allData[clauseNumber][exampleNumber]["train"][:,-1], self.allData[clauseNumber][exampleNumber]["valid"][:,-1]))
                tree = DecisionTreeClassifier(**(self.testResults[clauseNumber][exampleNumber]["bestParams"]))
                tree.fit(X, Y)
                YPred = tree.predict(self.allData[clauseNumber][exampleNumber]["test"][:, :-1])
                [self.testResults[clauseNumber][exampleNumber]["accuracy"], self.testResults[clauseNumber][exampleNumber]["F1Score"]] = self._calc(self.allData[clauseNumber][exampleNumber]["test"][:, -1], YPred)
    
    def _calc(self, Y, YPred):
        accuracy = accuracy_score(Y, YPred)
        fscore = f1_score(Y, YPred, zero_division=0.0, average="binary")
        temp = [accuracy, fscore]
        return [round(_, 3) for _ in temp]
    
    def outputToTextFile(self):
        filePath = "./results_decisionTree.txt"
        with open(filePath, "w") as file:
            sys.stdout = file
            for clauseNumber, val1 in self.allData.items():
                for exampleNumber, val2 in val1.items():
                    print(f"\nRandomized Grid Search Result, hyperparametes tuned on validation data, for the dataSet generated by {clauseNumber} clauses with {exampleNumber} examples: \n")
                    print("\nBest Params:\n")
                    print(self.bestParams[clauseNumber][exampleNumber]["bestParams"])
                    print("\nAccuracy\n")
                    print(self.bestParams[clauseNumber][exampleNumber]["accuracy"])
                    print("\nf1_score:\n")
                    print(self.bestParams[clauseNumber][exampleNumber]["F1Score"])
                    print("\n--------test accuracy and f1score\n")
                    print("\nAccuracy\n")
                    print(self.testResults[clauseNumber][exampleNumber]["accuracy"])
                    print("\nf1_score:\n")
                    print(self.testResults[clauseNumber][exampleNumber]["F1Score"])
                    print("\n--------test accuracy and f1score\n")

            sys.stdout = sys.__stdout__


def main():
    experiment = decisionTreeExperiment("project2_data.zip")
    experiment.searchForEachDataSet()
    experiment.test()
    experiment.outputToTextFile()

if __name__ == "__main__":
    main()


# ext = dataExtractor("project2_data.zip")