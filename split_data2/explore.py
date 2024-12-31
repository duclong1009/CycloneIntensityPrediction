import numpy as np
import pandas as pd
data = np.load("data/unzipdata3/1402_KAJIKI/arr/2014013100_4HD.npz")
df = pd.read_csv("data/single-tc-bessttrack/1402.csv")
t = data['time_strings']
arr = np.load("cutted_data/train/train_data.npz")
x,y = arr['x_arr'], arr['groundtruth']
