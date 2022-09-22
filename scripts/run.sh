export MAX_N_PID_4_TCOFFEE=4194304

python train.py --eval_dataset exthomfam-small --align_prog mafft --gpu 1
python train.py --eval_dataset exthomfam-small --align_prog famsa --gpu 1
python train.py --eval_dataset exthomfam-small --align_prog clustalo --gpu 1
python train.py --eval_dataset exthomfam-small --align_prog t_coffee --gpu 1

python train.py --eval_dataset exthomfam-small --align_prog mafft --no_tree
python train.py --eval_dataset exthomfam-small --align_prog famsa --no_tree
python train.py --eval_dataset exthomfam-small --align_prog clustalo --no_tree
python train.py --eval_dataset exthomfam-small --align_prog t_coffee --no_tree

python train.py --eval_dataset exthomfam-medium --align_prog mafft --gpu 1
python train.py --eval_dataset exthomfam-medium --align_prog famsa --gpu 1
python train.py --eval_dataset exthomfam-medium --align_prog clustalo --gpu 1
python train.py --eval_dataset exthomfam-medium --align_prog t_coffee --gpu 1

python train.py --eval_dataset exthomfam-medium --align_prog mafft --no_tree
python train.py --eval_dataset exthomfam-medium --align_prog famsa --no_tree
python train.py --eval_dataset exthomfam-medium --align_prog clustalo --no_tree
python train.py --eval_dataset exthomfam-medium --align_prog t_coffee --no_tree

python train.py --eval_dataset exthomfam-huge --align_prog mafft --gpu 1
python train.py --eval_dataset exthomfam-huge --align_prog famsa --gpu 1
python train.py --eval_dataset exthomfam-huge --align_prog clustalo --gpu 1
python train.py --eval_dataset exthomfam-huge --align_prog t_coffee --gpu 1

python train.py --eval_dataset exthomfam-huge --align_prog mafft --no_tree
python train.py --eval_dataset exthomfam-huge --align_prog famsa --no_tree
python train.py --eval_dataset exthomfam-huge --align_prog clustalo --no_tree
python train.py --eval_dataset exthomfam-huge --align_prog t_coffee --no_tree

python train.py --eval_dataset exthomfam-large --align_prog mafft --gpu 1
python train.py --eval_dataset exthomfam-large --align_prog famsa --gpu 1
python train.py --eval_dataset exthomfam-large --align_prog clustalo --gpu 1
python train.py --eval_dataset exthomfam-large --align_prog t_coffee --gpu 1

python train.py --eval_dataset exthomfam-large --align_prog mafft --no_tree
python train.py --eval_dataset exthomfam-large --align_prog famsa --no_tree
python train.py --eval_dataset exthomfam-large --align_prog clustalo --no_tree
python train.py --eval_dataset exthomfam-large --align_prog t_coffee --no_tree

python train.py --eval_dataset exthomfam-xlarge --align_prog mafft --gpu 1
python train.py --eval_dataset exthomfam-xlarge --align_prog famsa --gpu 1
python train.py --eval_dataset exthomfam-xlarge --align_prog clustalo --gpu 1
python train.py --eval_dataset exthomfam-xlarge --align_prog t_coffee --gpu 1

python train.py --eval_dataset exthomfam-xlarge --align_prog mafft --no_tree
python train.py --eval_dataset exthomfam-xlarge --align_prog famsa --no_tree
python train.py --eval_dataset exthomfam-xlarge --align_prog clustalo --no_tree
python train.py --eval_dataset exthomfam-xlarge --align_prog t_coffee --no_tree