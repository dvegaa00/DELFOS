CUDA_VISIBLE_DEVICES=4 python main_img_kfold.py --img_pretrain True --img_model medvit --loss_factor 2 --num_epochs 30 
CUDA_VISIBLE_DEVICES=6 python main_img_kfold.py --img_pretrain True --img_model vit_small --loss_factor 2 --num_epochs 30 
CUDA_VISIBLE_DEVICES=7 python main_img_kfold.py --img_pretrain True --img_model vit_tiny --loss_factor 2 --num_epochs 30 