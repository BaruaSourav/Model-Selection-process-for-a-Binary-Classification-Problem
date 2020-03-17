import pandas as pd 
import csv
import sys
import numpy as np
import decimal 
import scipy.stats as stats
import sklearn.datasets as datasets

#pd.set_option('display.max_colwidth', -1)

def main():
    for args in sys.argv:
        print(str(args))
    # reading test and train dataset from the names supplied on command line
    test_dataset = pd.read_csv(sys.argv[1],sep='\t')
    train_dataset = pd.read_csv(sys.argv[2],sep='\t')
    # print("Test")
    # print(test_dataset)
    # print(train_dataset)

if __name__== '__main__':
    main()
