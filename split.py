import pandas as pd 


df = pd.read_csv("cropped_data/file_index2.csv")

train_df=  df.values[:908]
valid_df = df.values[908:1045]
test_df = df.values[1045:]

pd.DataFrame(train_df, columns=df.columns).to_csv("cropped_data/train_index.csv")
pd.DataFrame(valid_df, columns=df.columns).to_csv("cropped_data/valid_index.csv")
pd.DataFrame(test_df, columns=df.columns).to_csv("cropped_data/test_index.csv")
