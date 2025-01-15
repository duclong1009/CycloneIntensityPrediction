## Our modethod
python repo/main.py --model_type prompt_vit6_2  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data_24h/data4 --_use_wandb --group_name basedyear_data_24h4 --seed 79 --body_model_name vit --image_size 50
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data_24h/data4 --_use_wandb --group_name basedyear_data_24h4 --seed 79 --body_model_name vit --image_size 50

python repo/main.py --model_type prompt_vit6_2  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data_24h/data4 --_use_wandb --freeze --group_name basedyear_data_24h4 --seed 79 --body_model_name vit --image_size 50
python repo/main.py --model_type prompt_vit6_3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data_24h/data4 --_use_wandb --freeze --group_name basedyear_data_24h4 --seed 79 --body_model_name vit --image_size 50

## Without historical data
python repo/main.py --model_type prompt_vit3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data_24h/data4 --_use_wandb --group_name basedyear_data_24h4 --seed 79 --body_model_name vit --image_size 50
python repo/main.py --model_type prompt_vit3  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data_24h/data4 --_use_wandb --freeze --group_name basedyear_data_24h4 --seed 79 --body_model_name vit --image_size 50

## Baseline 
python repo/main.py --model_type cnn  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data_24h/data4 --_use_wandb --group_name basedyear_data_24h4 --seed 79
python repo/main.py --model_type convlstm1  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir /home/user01/aiotlab/longnd/cyclone_prediction/raw_data/basedyear_data_24h/data4 --_use_wandb --group_name basedyear_data_24h4 --seed 79
