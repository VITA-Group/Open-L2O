# Lasso 25 50 - LISTA
python train.py --model_name lista --task lasso --lasso_lam 0.005 \
	--base_dir $BASE_DIR --data_dir $BASE_DIR/data/25_50 \
	--num_train_images 12800 --val_batch_size 1280 --num_val_images 1280 \
	--exp_name Lista_lasso-0.005_m25_n50

python train.py --model_name lista --task lasso --lasso_lam 0.005 \
	--base_dir $BASE_DIR --data_dir $BASE_DIR/data/25_50 \
	--num_train_images 12800 --val_batch_size 1280 --num_val_images 1280 \
	--exp_name Lista_lasso-0.005_m25_n50 \
	--test --test_files test_data.npy

# Lasso 25 50 - Step LISTA
python train.py --model_name step_lista --task lasso --model_lam 0.005 --lasso_lam 0.005 \
	--base_dir $BASE_DIR --data_dir $BASE_DIR/data/25_50 \
	--num_train_images 12800 --val_batch_size 1280 --num_val_images 1280 \
	--exp_name StepLista_lasso-0.005_m25_n50

python train.py --model_name step_lista --task lasso --model_lam 0.005 --lasso_lam 0.005 \
	--base_dir $BASE_DIR --data_dir $BASE_DIR/data/25_50 \
	--num_train_images 12800 --val_batch_size 1280 --num_val_images 1280 \
	--exp_name StepLista_lasso-0.005_m25_n50 \
	--test --test_files test_data.npy

# Lasso 5 10 - LISTA
python train.py --model_name lista --task lasso --lasso_lam 0.005 \
	--base_dir $BASE_DIR --data_dir $BASE_DIR/data/5_10 \
	--num_train_images 12800 --val_batch_size 1280 --num_val_images 1280 \
	--exp_name Lista_lasso-0.005_m5_n10

python train.py --model_name lista --task lasso --lasso_lam 0.005 \
	--base_dir $BASE_DIR --data_dir $BASE_DIR/data/5_10 \
	--num_train_images 12800 --val_batch_size 1280 --num_val_images 1280 \
	--exp_name Lista_lasso-0.005_m5_n10 \
	--test --test_files test_data.npy

# Lasso 5 10 - Step LISTA
python train.py --model_name step_lista --task lasso --model_lam 0.005 --lasso_lam 0.005 \
	--base_dir $BASE_DIR --data_dir $BASE_DIR/data/5_10 \
	--num_train_images 12800 --val_batch_size 1280 --num_val_images 1280 \
	--exp_name StepLista_lasso-0.005_m5_n10

python train.py --model_name step_lista --task lasso --model_lam 0.005 --lasso_lam 0.005 \
	--base_dir $BASE_DIR --data_dir $BASE_DIR/data/5_10 \
	--num_train_images 12800 --val_batch_size 1280 --num_val_images 1280 \
	--exp_name StepLista_lasso-0.005_m5_n10 \
	--test --test_files test_data.npy

# sc 256 512 noiseless
## LISTA
python train.py --model_name lista --task sc --base_dir $BASE_DIR --data_dir $BASE_DIR/data/256_512_noiseless --exp_name Lista_sc_256_512_noiseless

## LISTA-CP
python train.py --model_name lista_cp --task sc --base_dir $BASE_DIR --data_dir $BASE_DIR/data/256_512_noiseless --exp_name ListaCp_sc_256_512_noiseless

## LISTA-CPSS
python train.py --model_name lista_cpss --task sc --base_dir $BASE_DIR --data_dir $BASE_DIR/data/256_512_noiseless --exp_name ListaCpss_sc_256_512_noiseless

## ALISTA
python train.py --model_name alista --task sc --base_dir $BASE_DIR --data_dir $BASE_DIR/data/256_512_noiseless --exp_name Alista_sc_256_512_noiseless

