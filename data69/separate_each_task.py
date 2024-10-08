import pandas as pd
import os
import numpy as np



def find_first_idx(cutted_time_df, besttrack_df,idx):
    value_to_find = besttrack_df['Time of analysis'][idx]
    return cutted_time_df.index[cutted_time_df['Time of analysis'] == value_to_find]

def mapping_idx(current_time, time_list):
    return np.where(np.array(time_list) == current_time)


n_tasks = 7
list_seperated_task = [[]] * n_tasks
lsit_seperated_output = [[]] * n_tasks

list_all_task = [] 
list_all_output = []


npz_folder_path = "data/unzipdata3"

list_folder = os.listdir(npz_folder_path)
list_folder.sort()


for folder_name in list_folder:
    storm_id = folder_name.split("_")[0]
    cyclone_id = int(folder_name[:4])
    
    try:
        list_file_npz = os.listdir(f"{npz_folder_path}/{folder_name}/arr")
        besttrack_df = pd.read_csv(f"data/single-tc-bessttrack/{storm_id}.csv")
    except:
        print(f"data/single-tc-bessttrack/{storm_id}.csv")
        continue
    
    for file_npz in list_file_npz:
        try:
            arrz = np.load(f"{npz_folder_path}/{folder_name}/arr/{file_npz}")
        except:
            print(f"{npz_folder_path}/{folder_name}/arr/{file_npz}")
            continue 
        time_arrz= arrz['time_strings'][:n_tasks]
        
        
        
        
        first_time_of_npz_file = time_arrz[0]
        
        for task_id in range(n_tasks):
            task_time = time_arrz[task_id]
            besttrack_index = np.where(besttrack_df["Time of analysis"].values == task_time)[0]
            if besttrack_index.shape[0] == 1:
                
                breakpoint()
            else:
                print("Not exist")
        #     breakpoint()
        # # breakpoint()
        # for bt_idx in range(len(besttrack_df)):
            
        #     kkk = mapping_idx(besttrack_df["Time of analysis"][bt_idx], time_arrz[0])[0]
        #     breakpoint()
        #     if kkk.shape[0] == 1:
        #         list_results.append([f"data/single-tc-bessttrack/{storm_id}.csv", bt_idx, f"{npz_folder_path}/{folder_name}/arr/{file_npz}", kkk[0]])
        #     else:
        #         pass