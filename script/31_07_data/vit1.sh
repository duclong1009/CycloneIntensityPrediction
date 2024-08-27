

python repo/main2.py --model_type simple_vit  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb
python repo/main2.py --model_type simple_vit  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --patch_size 5
python repo/main2.py --model_type simple_vit  --batch_size 32 --lr 1e-6 --epochs 1000 --patience 10 --data_dir cropped_data --transform_groundtruth --_use_wandb --patch_size 20