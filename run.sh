### CNN
python repo/main2.py --batch_size 32 --lr 1e-5 --epochs 50 --model_type simple_cnn --_use_wandb

python repo/main2.py --model_type cnn --backbone_name resnet18 --batch_size 32 --lr 1e-6 --epochs 100 --patience 10 --_use_wandb
python repo/main2.py --model_type cnn --backbone_name resnet18 --batch_size 32 --lr 5e-6 --epochs 100 --patience 10 --_use_wandb
python repo/main2.py --model_type cnn --backbone_name resnet18 --batch_size 32 --lr 1e-7 --epochs 100 --patience 10 --_use_wandb
python repo/main2.py --model_type cnn --backbone_name resnet18 --batch_size 32 --lr 5e-7 --epochs 100 --patience 10 --_use_wandb

python repo/main2.py --model_type cnn --backbone_name vgg16 --batch_size 32 --lr 1e-6 --epochs 100 --patience 10

CUDA_VISIBLE_DEVICES=1 python repo/main2.py --model_type g_attention_cnn --backbone_name resnet18 --batch_size 32 --lr 1e-6 --epochs 100 --patience 10 --_use_wandb

CUDA_VISIBLE_DEVICES=1 python repo/main2.py --model_type c_attention_cnn --backbone_name resnet18 --batch_size 32 --lr 1e-6 --epochs 100 --patience 10 --_use_wandb


python repo/main2.py --model_type cnn --backbone_name vgg16 --batch_size 32 --lr 1e-6 --epochs 100 --patience 10 --_use_wandb

python repo/regression.py --model_type regression --batch_size 32 --lr 1e-3 --epochs 1000 --patience 10 --_use_wandb
python repo/regression.py --model_type regression2 --batch_size 32 --lr 1e-3 --epochs 1000 --patience 10 --_use_wandb

python repo/main2.py --model_type simple_cnn  --batch_size 32 --lr 1e-6 --epochs 100 --patience 10 --data_dir cropped/data --_use_wandb
python repo/main2.py --model_type simple_cnn  --batch_size 32 --lr 1e-6 --epochs 100 --patience 10 --data_dir cropped/data --transform_groundtruth --_use_wandb



python repo/main2.py --model_type simple_vit  --batch_size 32 --lr 1e-6 --epochs 100 --patience 10 --data_dir cropped/data --transform_groundtruth