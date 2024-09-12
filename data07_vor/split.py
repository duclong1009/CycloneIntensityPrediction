import pandas as pd 


data_dir = "data07"
df = pd.read_csv("data07/file_index0_7.csv")

train_df=  df.values[:1497]
valid_df = df.values[1497:1765]
test_df = df.values[1765:]

pd.DataFrame(train_df, columns=df.columns).to_csv(f"{data_dir}/train_index.csv")
pd.DataFrame(valid_df, columns=df.columns).to_csv(f"{data_dir}/valid_index.csv")
pd.DataFrame(test_df, columns=df.columns).to_csv(f"{data_dir}/test_index.csv")
breakpoint()