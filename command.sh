####### train command:
# about 12GB memory
CUDA_VISIBLE_DEVICES=0 python train.py --is_train True --n_shot 1 --norm_shift True --n_train_class 15 --n_query 8 --tau1 0.25 --iter 0 --n_folder 0

####### test command:
# inductive
CUDA_VISIBLE_DEVICES=0 python train.py --is_train False --n_shot 1 --norm_shift True --transfer True --tau1 0.35 --tau2 0.45 --beta 0.8 --iter 0 --n_folder 0
# transuctive
CUDA_VISIBLE_DEVICES=0 python train.py --is_train False --n_shot 1 --norm_shift True --transfer False --tau1 0.25 --tau2 0.45 --beta 0.8 --iter 10 --n_folder 0
# semi-supervice
CUDA_VISIBLE_DEVICES=0 python train.py --is_train False --n_shot 1 --norm_shift True --transfer False --tau1 0.25 --tau2 0.45 --beta 0.8 --iter 7 --semi_super True --n_extra 30 --n_folder 0
# distractor semi-supervice
CUDA_VISIBLE_DEVICES=0 python train.py --is_train False --n_shot 1 --norm_shift True --transfer False --tau1 0.25 --tau2 0.45 --beta 0.8 --iter 3 --semi_super True --n_extra 30 --n_distractor 3 --n_folder 0



CUDA_VISIBLE_DEVICES=0 python train.py --is_train True --n_shot 1 --norm_shift True --n_train_class 13 --n_query 8 --tau1 0.25 --iter 0 --n_folder B_FM_reg_813



CUDA_VISIBLE_DEVICES=0 python train.py --is_train True --n_shot 1 --norm_shift False --n_train_class 15 --n_query 8 --tau1 0.25 --iter 0 --n_folder B_dense_reg_21_-norm
CUDA_VISIBLE_DEVICES=0 python train.py --is_train True --n_shot 1 --norm_shift False --n_train_class 18 --n_query 8 --transfer True --tau1 0.35 --tau2 0.45 --beta 0.8 --iter 0 --n_folder B_FM_dense_reg_201_-norm_evalflip

