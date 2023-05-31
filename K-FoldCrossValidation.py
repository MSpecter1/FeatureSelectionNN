import numpy as np
import pandas as pd
import math
import time


# Euclidean distance using linear algebra : https://stackoverflow.com/questions/60574862/calculating-pairwise-euclidean-distance-between-all-the-rows-of-a-dataframe
# Basic structure from Professor Keogh's project 2 briefing : https://www.dropbox.com/sh/ftzvcnntl2j5eiu/AADbGDjXQFeXnqmfAsOsppTIa/Project_2_Briefing.pptx?dl=0
def findAccuracy(input_df, feature_subset):
    row_size = input_df.shape[0]
    col_size = input_df.shape[1]
    correct_cnt = 0

    # Remove features not in subset
    np_df = np.copy(input_df)
    for col in range(1, col_size):
        if (col not in feature_subset):
            np_df[:,col]= 0

    for i in range(0,row_size):
        # cur_class = df.iat[i,0]
        cur_class = np_df[i][0]
        nn_dist = None
        nn_index = None
        nn_class = None

        cur_instance = np_df[i, 1:]
        
        # test one data instance, i, against all others
        # find distance from current instance to all other rows
        for j in range(0,row_size):
            if(j!=i):
                # dist = np.linalg.norm(cur_instance - df.iloc[j, 1:col_size])
                dist = np.linalg.norm(cur_instance - np_df[j, 1:])
                if not nn_dist or dist < nn_dist:
                    nn_dist = dist
                    nn_index = j
                    # nn_class = df.iat[j,0]
                    nn_class = np_df[j][0]

        if cur_class == nn_class:
            correct_cnt+=1
    return correct_cnt/row_size

class Test:
    def testS32(self):
        df = pd.read_csv(r"CS170_small_Data__32.txt",header=None, delim_whitespace=True)
        df = df.to_numpy()
        feature_subset = {3,1,5} # EXPECTED VAL: 0.954

        start = time.time()
        result = findAccuracy(df, feature_subset)
        print("Execution Time: ", time.time()-start)
        if(result==0.954): return True
        return False
    
    def testS33(self):
        df = pd.read_csv(r"CS170_small_Data__33.txt",header=None, delim_whitespace=True)
        df = df.to_numpy()
        feature_subset = {8,7,3} # EXPECTED VAL: 0.949

        start = time.time()
        result = findAccuracy(df, feature_subset)
        print("Execution Time: ", time.time()-start)
        if(result==0.949): return True
        return False
    
    def testL32(self):
        df = pd.read_csv(r"CS170_large_Data__32.txt",header=None, delim_whitespace=True)
        df = df.to_numpy()
        feature_subset = {3,7,6} # EXPECTED VAL: 0.963

        start = time.time()
        result = findAccuracy(df, feature_subset)
        print("Execution Time: ", time.time()-start)
        if(result==0.963): return True
        return False
    
    def testL33(self):
        df = pd.read_csv(r"CS170_large_Data__33.txt",header=None, delim_whitespace=True)
        df = df.to_numpy()
        feature_subset = {4,5,10} # EXPECTED VAL: 0.9655

        start = time.time()
        result = findAccuracy(df, feature_subset)
        print("Execution Time: ", time.time()-start)
        if(result==0.9655): return True
        return False
    
    def testXL1(self):
        df = pd.read_csv(r"CS170_XXXlarge_Data__1.txt",header=None, delim_whitespace=True)
        df = df.to_numpy()
        feature_subset = {4,5,10} # EXPECTED VAL: 0.9655

        start = time.time()
        result = findAccuracy(df, feature_subset)
        print("Execution Time: ", time.time()-start)
        print("Result: ", result)
        # if(result==0.9655): return True
        # return False
    
    def test(self):
        fail = 0
        if not self.testS32(): 
            print("S32 failed")
            fail+=1
        if not self.testS33(): 
            print("S33 failed")
            fail+=1
        if not self.testL32(): 
            print("L32 failed")
            fail+=1
        if not self.testL33(): 
            print("L33 failed")
            fail+=1
        print("Tests failed: ", fail)

# test = Test()
# test.test()
# test.testXL1()
