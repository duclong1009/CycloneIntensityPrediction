## Our modethod
python repo/main.py --model_type prompt_vit6_2  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --group_name BasedYear_Data5 --seed 79 --body_model_name vit --image_size 50
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --group_name BasedYear_Data5 --seed 79 --body_model_name vit --image_size 50

python repo/main.py --model_type prompt_vit6_2  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name BasedYear_Data5 --seed 79 --body_model_name vit --image_size 50
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name BasedYear_Data5 --seed 79 --body_model_name vit --image_size 50

## Without historical data
python repo/main.py --model_type prompt_vit6  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --group_name BasedYear_Data5 --seed 79 --body_model_name vit --image_size 50
python repo/main.py --model_type prompt_vit6  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name BasedYear_Data5 --seed 79 --body_model_name vit --image_size 50

## Cropped size
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name AB_IS_BasedYear_Data5 --seed 79 --body_model_name vit --image_size 100
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name AB_IS_BasedYear_Data5 --seed 79 --body_model_name vit --image_size 90
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name AB_IS_BasedYear_Data5 --seed 79 --body_model_name vit --image_size 80
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name AB_IS_BasedYear_Data5 --seed 79 --body_model_name vit --image_size 70
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name AB_IS_BasedYear_Data5 --seed 79 --body_model_name vit --image_size 60
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name AB_IS_BasedYear_Data5 --seed 79 --body_model_name vit --image_size 40
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name AB_IS_BasedYear_Data5 --seed 79 --body_model_name vit --image_size 30
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name AB_IS_BasedYear_Data5 --seed 79 --body_model_name vit --image_size 20
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --freeze --group_name AB_IS_BasedYear_Data5 --seed 79 --body_model_name vit --image_size 10

## Baseline 
python repo/main.py --model_type cnn  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --group_name BasedYear_Data5 --seed 79
python repo/main.py --model_type convlstm1  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data/data5 --_use_wandb --group_name BasedYear_Data5 --seed 79
