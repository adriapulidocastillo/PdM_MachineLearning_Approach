import pickle
import string
from os import path
import pandas as pd
import numpy as np
import time

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

filePath = "models/"

class IClassifier():
    def fit(X, y):
        raise NotImplemented

    def predict(X):
        raise NotImplemented

    def set_params(params):
        raise NotImplemented

class FitRecord():
    def __init__(self, model_name, model: IClassifier, accuracy: float, fit_time: float, cross_validation: float, confusion_matrix, classification_report):
        self.model_name = model_name
        self.model = model
        self.accuracy = accuracy
        self.fit_time = fit_time
        self.cross_validation = cross_validation
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report

    def print(self):
        print("%s accuaracy: %.3f" % (self.model_name, self.accuracy))

class Classifier():
    def __init__(self, name: string, classifier: IClassifier, parameters: dict = {}, needsGridSearch: bool = False, reFit: bool = False):
        self.name = name
        self.fileName = filePath + name.replace(" ", "_") + ".sav"
        self.classifier = classifier
        self.reFit = reFit
        self.fitRecord = None
        self.parameters = parameters
        self.needsGridSearch = needsGridSearch

    def getFitRecord(self) -> FitRecord:
        return self.fitRecord

    def getTrainedModel(self) -> IClassifier:
        return self.fitRecord.model 

    def process(self, X_train, y_train, X_test, y_test):
        if(not self.reFit):
            if(not path.exists(self.fileName)):
                raise Exception("Path [%s] does not exists" % self.fileName)

            self.fitRecord = pickle.load(open(self.fileName, 'rb'))
            return
        
        self._fitModel(X_train, y_train, X_test, y_test)

    def _fitModel(self, X_train, y_train, X_test, y_test):
        print("\nFitting '%s' model" % self.name)

        cross_validation = 0
        best_parameters = {}
        if self.needsGridSearch:
            gs = GridSearchCV(self.classifier, self.parameters, cv=3, verbose=3, n_jobs=-1, error_score="raise")
            gs_results = gs.fit(X_train, y_train)
            best_parameters =  gs_results.best_params_
            print("Best parameters: " + str(best_parameters))
            self.classifier.set_params(**best_parameters)
            cross_validation = np.max(gs_results.cv_results_["mean_test_score"])

        else:
            cross_validation = cross_val_score(self.classifier, X_train, y_train, cv=3)
            cross_validation = cross_validation.mean()

        print("Cross validation: %.3f" % cross_validation)

        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        fit_time = time.time() - start_time
        print("Fit time: %.2fs" % fit_time)

        # Calculating the model accuracy
        predict = self.classifier.predict(X_test)
        accuracity = accuracy_score(y_test, predict)
        print("Accuaracy: %.3f" % accuracity)

        # Confusion matrix
        cm = pd.DataFrame(confusion_matrix(y_test, predict))
        print("Confusion matrix:")
        print(cm)

        # classification report
        cr = classification_report(y_test, predict)
        print("Classification report:")
        print(cr)

        self.fitRecord = FitRecord(self.name, self.classifier, accuracity, fit_time, cross_validation, cm, cr)

        pickle.dump(self.fitRecord, open(self.fileName, 'wb'))

    def printFitRecord(self):
        if self.fitRecord:
            self.fitRecord.print()

    def getAccuracy(self):
        tn, fp, fn, tp = np.array(self.fitRecord.confusion_matrix).ravel()
        return (tp + tn) / (fp + fn + tp + tn)

    def getSensitivity(self):
        tn, fp, fn, tp = np.array(self.fitRecord.confusion_matrix).ravel()
        return tp / (fn + tp)

    def getSpecificity(self):
        tn, fp, fn, tp = np.array(self.fitRecord.confusion_matrix).ravel()
        return tn / (tn + fp)

    def getPrecision(self):
        tn, fp, fn, tp = np.array(self.fitRecord.confusion_matrix).ravel()
        return tp / (tp + fp)

    def getF1Score(self):
        return 2 * (self.getPrecision() * self.getSensitivity()) / (self.getPrecision() + self.getSensitivity())

    def getCrossValidation(self):
        return self.fitRecord.cross_validation
