import numpy as np
import pandas as pd
import math
import time


# euclidean distance using linear algebra : https://stackoverflow.com/questions/60574862/calculating-pairwise-euclidean-distance-between-all-the-rows-of-a-dataframe
def findAccuracy(input_df, feature_subset):
    row_size = input_df.shape[0]
    col_size = input_df.shape[1]
    correct_cnt = 0

    # Remove features not in subset
    # df = input_df.copy()
    # for col in range(1, col_size):
    #     if (col not in feature_subset):
    #         df[col].values[:] = 0
    np_df = np.copy(input_df)
    for col in range(1, col_size):
        if (col not in feature_subset):
            np_df[:,col]= 0

    # MANUAL TEST
    # test1 = df.iloc[0, 1:col_size]
    # test2 = df.iloc[1, 1:col_size]
    # dist = np.linalg.norm(test1-test2)
    # print(dist)
    # dist2 = math.sqrt(pow((df.iat[0,1]-df.iat[1,1]),2)+pow((df.iat[0,2]-df.iat[1,2]),2)+pow((df.iat[0,3]-df.iat[1,3]),2))
    # print(dist2)

    # np_df = df.to_numpy()
    # cur_instance = np_df[0, :]
    # cur_instance2 = np_df[1, :]
    # dist = np.linalg.norm(cur_instance - cur_instance2)
    # print(dist)

    for i in range(0,row_size):
        # cur_class = df.iat[i,0]
        cur_class = np_df[i][0]
        nn_dist = None
        nn_index = None
        nn_class = None

        # cur_instance = df.iloc[i, 1:col_size]
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

df = pd.read_csv(r"CS170_small_Data__32.txt",header=None, delim_whitespace=True)
df = df.to_numpy()
feature_subset = {3,1,5} # EXPECTED VAL: 0.954

start = time.time()
result = findAccuracy(df, feature_subset)
print("Execution Time: ", time.time()-start)
print(result)


