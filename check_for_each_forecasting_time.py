import numpy as np

import pandas as pd 

df = pd.read_csv("hres.csv")

test_df = pd.read_csv("test_index.csv")

list_prd = [[],[],[],[],[],[],[]] 
list_grt =  [[],[],[],[],[],[],[]] 
for i in range(len(test_df)):
    idx = test_df["nwp_id"][i]
    list_prd[idx].append(df['Prediction'][i])
    list_grt[idx].append(df['Ground truth'][i])
ab = []
for i in range(7):
    print(np.mean(list_prd[i]))
print("\n")
for i in range(7):
    print(np.std(list_prd[i]))
print("\n")
for i in range(7):
    print(np.mean(list_grt[i]))

print("\n")
for i in range(7):
    print(np.std(list_grt[i]))
