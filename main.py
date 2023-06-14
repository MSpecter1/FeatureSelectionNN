import ForwardSelection as forward_selection
import numpy as np
import pandas as pd
import math
import time
import random
import sys

print("Feature Selection with NN")

print("\t1: Forward Selection")
print("\t2: Backward Selection")

alg = input("Enter algorithm selection: ")

print("\t1: Small 17")
print("\t2: Large 15")
print("\t3: XXXLarge 11")
print("\t4: Iris Data (Real World Dataset)")

file_input = input("Select a file ")

if int(alg)==1:
    match int(file_input):
        case 1:
            path = 'output_Forward_Small17.txt'
            sys.stdout = open(path, 'w')
            df = pd.read_csv(r"CS170_small_Data__17.txt",header=None, delim_whitespace=True)
            df = df.to_numpy()
            start = time.time()
            fs = forward_selection.ForwardSelectionSearch()
            result = fs.search(df)
            print("Execution Time: ", time.time()-start)
        case 2:
            path = 'output_Forward_Large15.txt'
            sys.stdout = open(path, 'w')
            df = pd.read_csv(r"CS170_large_Data__15.txt",header=None, delim_whitespace=True)
            df = df.to_numpy()
            start = time.time()
            fs = forward_selection.ForwardSelectionSearch()
            result = fs.search(df)
            print("Execution Time: ", time.time()-start)
        case 3:
            path = 'output_Forward_XXXLarge11.txt'
            sys.stdout = open(path, 'w')
            df = pd.read_csv(r"CS170_XXXlarge_Data__11.txt",header=None, delim_whitespace=True)
            df = df.to_numpy()
            start = time.time()
            fs = forward_selection.ForwardSelectionSearch()
            result = fs.search(df)
            print("Execution Time: ", time.time()-start)
        case 4:
            path = 'output_Forward_Iris.txt'
            sys.stdout = open(path, 'w')
            df = pd.read_csv(r"iris\iris.data",header=None)
            df = df.to_numpy()

            start = time.time()
            fs = forward_selection.ForwardSelectionSearch()
            result = fs.searchIris(df)
            print(result)
            print("Execution Time: ", time.time()-start)