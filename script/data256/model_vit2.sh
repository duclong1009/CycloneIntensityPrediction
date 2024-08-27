# python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-7 --epochs 1000 --patience 10 --data_dir data07/cropped_data  --transform_groundtruth --loss_func mse  --_use_wandb



### Tuning patch_size
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name tuning_vit2
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name tuning_vit2 --patch_size 5
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name tuning_vit2 --patch_size 20

python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name tuning_vit2
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name tuning_vit2 --patch_size 5
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name tuning_vit2 --patch_size 20


python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name tuning_vit2 --heads 16
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name tuning_vit2 --heads 16 --patch_size 5
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name tuning_vit2 --heads 16 --patch_size 20

python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name tuning_vit2 --heads 16
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name tuning_vit2 --heads 16 --patch_size 5
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name tuning_vit2 --heads 16 --patch_size 20

python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name tuning_vit2 --heads 32
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name tuning_vit2 --heads 32 --patch_size 5
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --group_name tuning_vit2 --heads 32 --patch_size 20

python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name tuning_vit2 --heads 32
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name tuning_vit2 --heads 32 --patch_size 5
python repo/main2.py --model_type simple_vit2  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --_use_wandb --group_name tuning_vit2 --heads 32 --patch_size 20