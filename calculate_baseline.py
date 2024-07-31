import pandas as pd 

test_df = pd.read_csv("cropped_data/test_index.csv")
list_tracker = []
list_besttrack = []
current_file_name = ""

for i in range(len(test_df)):
    
    _ , _, bt_path, bt_idx, nwp_path, nwp_id = test_df.values[i]
    try:
        if bt_path != current_file_name:
            cyclone_id_name = nwp_path.split("/")[2]
            file_name = nwp_path.split("/")[-1].split(".")[0].split("_")[0]
            year = file_name[:4]
            df_path = f"data/{year}/{cyclone_id_name}/tracker/{file_name}/tracker.csv"
            current_file_name = bt_path
            df = pd.read_csv(df_path)
        list_tracker.append(df['NWP Maximum sustained wind speed'].values[nwp_id])
        list_besttrack.append(df['Best-track Maximum sustained wind speed'].values[nwp_id])
    except:
        breakpoint()

from repo import model_utils

mae, mse, mape, rmse, r2, corr = model_utils.cal_acc(list_tracker, list_besttrack)
breakpoint()