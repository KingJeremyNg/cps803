import csv
import numpy as np


def readDataset(path):
    features = []
    result = []
    with open(path, 'r') as csvFile:
        rows = csv.DictReader(csvFile)
        for row in rows:
            features.append([
                float(row['fixed acidity']),
                float(row['volatile acidity']),
                float(row['citric acid']),
                float(row['residual sugar']),
                float(row['chlorides']),
                float(row['free sulfur dioxide']),
                float(row['total sulfur dioxide']),
                float(row['density']),
                float(row['pH']),
                float(row['sulphates']),
                float(row['alcohol'])
            ])
            result.append(int(row['quality']))
    csvFile.close()
    return (np.array(features), np.array(result))
