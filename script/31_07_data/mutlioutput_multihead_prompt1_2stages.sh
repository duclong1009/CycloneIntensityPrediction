# python repo/main2.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 1e-7 --epochs 1000 --patience 10 --data_dir data07/cropped_data  --transform_groundtruth --loss_func mse  --_use_wandb



### Tuning patch_size
python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages
python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages

python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 1e-7 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages
python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 1e-7 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages

python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages
python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 1e-5 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages

python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 1e-4 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages
python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 1e-4 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages

python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 5e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages
python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 5e-6 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages

python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 5e-7 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages
python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 5e-7 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages

python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 5e-5 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages
python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 5e-5 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages

python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 5e-4 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages
python repo/train_cluster.py --model_type mutlioutput_multihead_prompt1_2stages  --batch_size 32 --lr 5e-4 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name mutlioutput_multihead_prompt1_2stages