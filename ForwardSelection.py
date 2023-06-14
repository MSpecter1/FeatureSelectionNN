import KFoldCrossValidation as kf
import numpy as np
import pandas as pd
import math
import time
import random
import sys

class ForwardSelectionSearch():
    
    def search(self, data): # data as a numpy array
        feature_set = set()
        col_size = data.shape[1]
        best_acc = 0

        print("Starting Search:")
        print("Default Rate = ", kf.defaultRate(data, 2, 0))
        for i in range(1, col_size):
            print("Level: ", i)
            feature_to_add = -1
            this_accuracy = 0

            for k in range(1, col_size):
                if k not in feature_set:
                    accuracy = 0
                    test_set = set(feature_set)
                    test_set.add(k)
                    print(" -  Adding ", k, " to feature set: Accuracy of set ", test_set, ": ", end='')
                    accuracy = kf.findAccuracy(data, test_set)
                    print(accuracy)
                    if accuracy>best_acc:
                        best_acc = accuracy
                        this_accuracy = accuracy
                        feature_to_add = k
            
            if feature_to_add != -1: 
                feature_set.add(feature_to_add)
                print("Adding ", feature_to_add,", ", feature_set," accuracy is ",this_accuracy, "\n")
            else: 
                print("Adding more features decreases accuracy, ending early\n")
                break

        print("SOLUTION: ", best_acc)
        print(feature_set)
        return 0
    
    def searchIris(self, data): # data as a numpy array
        feature_set = set()
        col_size = data.shape[1]
        best_acc = 0

        print("Starting Search:")
        print("Default Rate = ", kf.defaultRateIris(data))
        for i in range(0, col_size-1):
            print("Level: ", i)
            feature_to_add = -1
            this_accuracy = 0

            for k in range(0, col_size-1):
                if k not in feature_set:
                    accuracy = 0
                    test_set = set(feature_set)
                    test_set.add(k)
                    print(" -  Adding ", k, " to feature set: Accuracy of set ", test_set, ": ", end='')
                    accuracy = kf.findAccuracyIris(data, test_set)
                    print(accuracy)
                    if accuracy>best_acc:
                        best_acc = accuracy
                        this_accuracy = accuracy
                        feature_to_add = k
            
            if feature_to_add != -1: 
                feature_set.add(feature_to_add)
                print("Adding ", feature_to_add,", ", feature_set," accuracy is ",this_accuracy, "\n")
            else: 
                print("Adding more features decreases accuracy, ending early\n")
                break

        print("SOLUTION: ", best_acc)
        print(feature_set)
        return 0

# path = 'output_Small17.txt'
# sys.stdout = open(path, 'w')
# df = pd.read_csv(r"CS170_small_Data__17.txt",header=None, delim_whitespace=True)
# df = df.to_numpy()
# start = time.time()
# fs = ForwardSelectionSearch()
# result = fs.search(df)
# print("Execution Time: ", time.time()-start)

# path = 'output_Large15.txt'
# sys.stdout = open(path, 'w')
# df = pd.read_csv(r"CS170_large_Data__15.txt",header=None, delim_whitespace=True)
# df = df.to_numpy()
# start = time.time()
# fs = ForwardSelectionSearch()
# result = fs.search(df)
# print("Execution Time: ", time.time()-start)

# path = 'output_XLarge11.txt'
# sys.stdout = open(path, 'w')
# df = pd.read_csv(r"CS170_XXXlarge_Data__11.txt",header=None, delim_whitespace=True)
# df = df.to_numpy()
# start = time.time()
# fs = ForwardSelectionSearch()
# result = fs.search(df)
# print("Execution Time: ", time.time()-start)
