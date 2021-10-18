import sys
import readDataset as reader
import numpy as np


def main(redDataset, whiteDataset):
    redFeatures, redResult = reader.readDataset(redDataset)
    whiteFeatures, whiteResult = reader.readDataset(whiteDataset)


if __name__ == '__main__':
    main(redDataset='../data/winequality-red.csv',
         whiteDataset='../data/winequality-white.csv')
