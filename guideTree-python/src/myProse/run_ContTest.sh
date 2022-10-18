python train.py --eval_dataset ContTest-small --align_prog mafft --dist_type NW --gpu 1
python train.py --eval_dataset ContTest-small --align_prog famsa --dist_type NW --gpu 1
python train.py --eval_dataset ContTest-small --align_prog clustalo --dist_type NW --gpu 1
python train.py --eval_dataset ContTest-small --align_prog tcoffee --dist_type NW --gpu 1

python train.py --eval_dataset ContTest-small --align_prog mafft --dist_type NW --no_tree
python train.py --eval_dataset ContTest-small --align_prog famsa --dist_type NW --no_tree
python train.py --eval_dataset ContTest-small --align_prog clustalo --dist_type NW --no_tree
python train.py --eval_dataset ContTest-small --align_prog tcoffee --dist_type NW --no_tree

python train.py --eval_dataset ContTest-medium --align_prog mafft --dist_type NW --gpu 1
python train.py --eval_dataset ContTest-medium --align_prog famsa --dist_type NW --gpu 1
python train.py --eval_dataset ContTest-medium --align_prog clustalo --dist_type NW --gpu 1
python train.py --eval_dataset ContTest-medium --align_prog tcoffee --dist_type NW --gpu 1

python train.py --eval_dataset ContTest-medium --align_prog mafft --dist_type NW --no_tree
python train.py --eval_dataset ContTest-medium --align_prog famsa --dist_type NW --no_tree
python train.py --eval_dataset ContTest-medium --align_prog clustalo --dist_type NW --no_tree
python train.py --eval_dataset ContTest-medium --align_prog tcoffee --dist_type NW --no_tree

python train.py --eval_dataset ContTest-large --align_prog mafft --dist_type NW --gpu 1
python train.py --eval_dataset ContTest-large --align_prog famsa --dist_type NW --gpu 1
python train.py --eval_dataset ContTest-large --align_prog clustalo --dist_type NW --gpu 1
python train.py --eval_dataset ContTest-large --align_prog tcoffee --dist_type NW --gpu 1

python train.py --eval_dataset ContTest-large --align_prog mafft --dist_type NW --no_tree
python train.py --eval_dataset ContTest-large --align_prog famsa --dist_type NW --no_tree
python train.py --eval_dataset ContTest-large --align_prog clustalo --dist_type NW --no_tree
python train.py --eval_dataset ContTest-large --align_prog tcoffee --dist_type NW --no_tree
