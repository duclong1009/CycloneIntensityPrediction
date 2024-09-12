import pandas as pd 

# Đọc dữ liệu từ file CSV
path = "data07_vor/valid_index.csv"
df = pd.read_csv(path)

# Thay thế "data/unzipdata3" bằng "haha" trong cột 'nwp_path'
df['nwp_path'] = df['nwp_path'].apply(lambda x: x.replace("data/unzipdata3", "data/added_features_data"))


df.to_csv(path,index=False)
breakpoint()