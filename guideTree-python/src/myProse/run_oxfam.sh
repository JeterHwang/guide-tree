export MAX_N_PID_4_TCOFFEE=4194304

python train.py --eval_dataset oxfam-small --align_prog mafft --gpu 1
python train.py --eval_dataset oxfam-small --align_prog famsa --gpu 1
python train.py --eval_dataset oxfam-small --align_prog clustalo --gpu 1
python train.py --eval_dataset oxfam-small --align_prog t_coffee --gpu 1

python train.py --eval_dataset oxfam-small --align_prog mafft --no_tree
python train.py --eval_dataset oxfam-small --align_prog famsa --no_tree
python train.py --eval_dataset oxfam-small --align_prog clustalo --no_tree
python train.py --eval_dataset oxfam-small --align_prog t_coffee --no_tree

python train.py --eval_dataset oxfam-medium --align_prog mafft --gpu 1
python train.py --eval_dataset oxfam-medium --align_prog famsa --gpu 1
python train.py --eval_dataset oxfam-medium --align_prog clustalo --gpu 1
python train.py --eval_dataset oxfam-medium --align_prog t_coffee --gpu 1

python train.py --eval_dataset oxfam-medium --align_prog mafft --no_tree
python train.py --eval_dataset oxfam-medium --align_prog famsa --no_tree
python train.py --eval_dataset oxfam-medium --align_prog clustalo --no_tree
python train.py --eval_dataset oxfam-medium --align_prog t_coffee --no_tree

python train.py --eval_dataset oxfam-large --align_prog mafft --gpu 1
python train.py --eval_dataset oxfam-large --align_prog famsa --gpu 1
python train.py --eval_dataset oxfam-large --align_prog clustalo --gpu 1
python train.py --eval_dataset oxfam-large --align_prog t_coffee --gpu 1

python train.py --eval_dataset oxfam-large --align_prog mafft --no_tree
python train.py --eval_dataset oxfam-large --align_prog famsa --no_tree
python train.py --eval_dataset oxfam-large --align_prog clustalo --no_tree
python train.py --eval_dataset oxfam-large --align_prog t_coffee --no_tree
