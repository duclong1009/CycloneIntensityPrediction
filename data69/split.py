import pandas as pd 


data_dir = "data69"
df = pd.read_csv("data69/file_index69.csv")

train_df=  df.values[:594]
valid_df = df.values[594:766]
test_df = df.values[766:]

pd.DataFrame(train_df, columns=df.columns).to_csv(f"{data_dir}/train_index.csv")
pd.DataFrame(valid_df, columns=df.columns).to_csv(f"{data_dir}/valid_index.csv")
pd.DataFrame(test_df, columns=df.columns).to_csv(f"{data_dir}/test_index.csv")
breakpoint()