import pandas as pd
import os
import numpy as np

list_folder = os.listdir("data/unzipdata3")
list_folder.sort()

def find_first_idx(cutted_time_df, besttrack_df,idx):
    value_to_find = besttrack_df['Time of analysis'][idx]
    return cutted_time_df.index[cutted_time_df['Time of analysis'] == value_to_find]

def mapping_idx(current_time, time_list):
    return np.where(np.array(time_list) == current_time)

list_results = []
for folder_name in list_folder:
    storm_id = folder_name.split("_")[0]
    cyclone_id = int(folder_name[:4])
    
    try:
        list_file_npz = os.listdir(f"data/unzipdata3/{folder_name}/arr")
        besttrack_df = pd.read_csv(f"data/single-tc-bessttrack/{storm_id}.csv")
    except:
        print(f"data/single-tc-bessttrack/{storm_id}.csv")
        continue
    for file_npz in list_file_npz:
        # print(file_npz)
        try:
            arrz = np.load(f"data/unzipdata3/{folder_name}/arr/{file_npz}")
        except:
            print(f"data/unzipdata3/{folder_name}/arr/{file_npz}")
            continue 
        time_arrz= arrz['time_strings'][6:10]
        
        # breakpoint()
        for bt_idx in range(len(besttrack_df)):
            
            kkk = mapping_idx(besttrack_df["Time of analysis"][bt_idx], time_arrz)[0]
            if kkk.shape[0] == 1:
                list_results.append([f"data/single-tc-bessttrack/{storm_id}.csv", bt_idx, f"data/unzipdata3/{folder_name}/arr/{file_npz}", kkk[0]])
            else:
                pass

pd.DataFrame(list_results, columns=['besttrack_path', "besttrack_id", "nwp_path","nwp_id"]).to_csv("file_index6_9.csv")


# breakpoint()