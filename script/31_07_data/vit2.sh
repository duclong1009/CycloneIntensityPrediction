
python repo/main2.py --model_type simple_vit  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir 31_07_data --transform_groundtruth --_use_wandb --heads 8
python repo/main2.py --model_type simple_vit  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir 31_07_data --transform_groundtruth --_use_wandb --patch_size 5 --heads 8
python repo/main2.py --model_type simple_vit  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir 31_07_data --transform_groundtruth --_use_wandb --patch_size 20 --heads 8
