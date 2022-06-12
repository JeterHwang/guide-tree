---
tags: MSA
---
# guide-tree
## TODO
- ~~Output guide tree file~~
- Short Term
    - ~~Mass-produce guide tree on bb3_release dataset and compare SP, TC with original clustalO~~
    - ~~Load big dataset by dataloader of esm~~
    - ~~convert guide tree format to that compatible with MAFFT and use MAFFT as reference program~~
    - ~~normalize the distance Matrix to [0,1]~~
    - ~~**Go through SSA/prose_mt's citations to see how others use their embeddings and SSA algorithm**~~
    - ~~Replace fast UPGMA with quadtree UPGMA~~
    - ~~Know how TC, SP works~~
    - ~~***Write script get score on Homfam/small***~~
    - Add K-means to prose-mt
    - Do test on homfam/medium,large
- Long Term
    - Replace K-means with kd-tree K-means
    - Transfer learning on NW-score either by adding CNN after Bi-LSTM embedding layer or by adding a decoder layer after Bi-LSTM and use UniRef dataset to finetune the model

## Useful Commands
1. mBed
```
python main.py
```
2. esm
:::info
python main.py [--embedding] [--toks_per_batch] [--esm_ckpt] [--outputFolder]
:::
```
python main.py --embedding esm --toks_per_batch 4096 --esm_ckpt ./ckpt/esm/esm1b_t33_650M_UR50S.pt --outputFolder ./output/bb3_release/esm-650M
```
3. clustal omega output guide tree
```
clustalo -i data/bb3_release/RV30/BB30003.tfa -o output/BB30003.msf --guidetree-out=output/BB30003.dnd --clustering-out=output/BB30003.aux --cluster-size=10 --force
```
4. clustal omega input guide tree
```
clustalo -i data/bb3_release/RV50/BB50003.tfa -o output/BB50003-python.msf --guidetree-in=output/BB50003-python.dnd --force
```
5. mafft output guide tree ***(The generated tree will be in the same directory with the input sequences)***
```
mafft --localpair (--parttree / --dpparttree / --fastaparttree) --treeout [input] > [output]
```
6. mafft input guide tree
```
mafft --localpair --treein [guidTreeFile] [input] > [output]
```

## Balibase Benchmark Workflow
1. ==Get the output guide trees from all sources(e.g. clustalO, muscle, mafft, kalign, python) and put it in **"/b07068/MSA/guide-tree/Benchmarks/trees/bb3_release"**==

![](https://i.imgur.com/h0WXKhZ.png)

2. ==Run ***align_generate.py*** in ***"/home/b07068/MSA/guide-tree/Benchmarks/programs/"***==
:::info
python align_generate.py [--tree_dir] [--output_dir] [--tree_type]
:::
Example
```
python align_generate.py --tree_type esm-43M
```
3. ==Execute run_alignment.sh generated from previous step, and the msf files will be store in ***[--output_dir]*** you just specify in previous step==
```
[Under /home/b07068/MSA/guide-tree/Benchmarks/programs]
./run_alignment.sh
```
4. ==Run score_generate.py under the same directory==
:::info
python score_generate.py [--msf_dir] [--bash_path]
:::
Example:
```
python score_generate.py --msf_dir ../msf/bb3_release/esm-43M --bash_path score_bb3_esm-43M.sh
```

5. ==Execute bash file generated from previous step==

you can execute either by command line 
```
chmod 755 score_bb3_esm-43M.sh
./score_bb3_esm-43M.sh > scores_balibase_esm-43M.csv
```
or edit ***score.sh*** and execute multiple scoring bash files
```
[after edit score.sh]
./score.sh
```
6.==Download the csv files generated from previous step to ***D:\\IC-design\\Project\\Yi-Chang Lu\\plot\\data\\balibase\\compare*** and run ***plot_balibase.ipynb*** under ***D:\\IC-design\\Project\\Yi-Chang Lu\\plot***, the output figure path is specified in the notebook==

## Progress
- 2022/4/18
    - finish k-means++ bugs fixing
    
    ![](https://i.imgur.com/4jt5Efn.png)

    - finish esm one batch validation
    - generate guide tree from MUSCLE and CLUSTALO(in Benchmarks/trees) on workstation
- 2022/5/2
    - finish prose_mt embedding and SSA algorithm
    - 


## Meeting Notes
- 2022/3/22
    - Transformer weights must store in on-chip DRAM, even when choosing the smallest model(41MB), so choose the bigger one may be better
    - Can use docker on workstation

## Debug
1. libc.so.6 Remove
```
LD_PRELOAD=/lib/x86_64-linux-gnu/libc-2.27.so ln -s /lib/x86_64-linux-gnu/libc-2.27.so /lib/x86_64-linux-gnu/libc.so.6
```
https://blog.csdn.net/hongguo87/article/details/118378891

2. sequence length above 1024
![](https://i.imgur.com/LbueIdz.png)

https://github.com/facebookresearch/esm/issues/21
![](https://i.imgur.com/rBhS7nC.png)
