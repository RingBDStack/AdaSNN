

 # AdaSNN

 Code for "Adaptive Subgraph Neural Network with Reinforced Critical Structure Mining"

   # Environment

   * Python(verified: v3.6)
   * CUDA(verified: v11.7)
   * Other dependencies can be installed using the following command:
   ```
   pip install -r requirements.txt
   ```
   # File

   ```bash
   ├── model
   │   ├── agent_chain.py  | the RL pipeline
   │   ├── depth_selector.py | depth selection agent
   │   ├── neighbor_selector.py | neighbor selection agent
   │   ├── QLearning.py 
   │   ├── QNetwork.py 
   │   └── Sugar.py | backbone GNN
   ├── one_fold_main.py | training on one fold 
   ├── load_best_main.py | test the reproducibility
   └──  main.py
   ```

# Dataset

All the datasets (i.e. IMDB-B, MUTAG, REDDIT-B) are provided by [PYG](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)

   # Usage

   ## **1. Train from scratch**

   Train and evaluate the model:

   ```bash
   python main.py --dataset <dataset_name>
   ```

   More parameters can be find in the main.py-init-args or type

   ```bash
   python main.py --help
   ```
For instance:

* python main.py --dataset MUTAG --lr 0.01

```bash
============================0/10==================================
Training Meta-policy on Validation Set
  0%|          | 0/5 [00:00<?, ?it/s]
 20%|██        | 1/5 [00:06<00:26,  6.62s/it]
 40%|████      | 2/5 [00:13<00:19,  6.63s/it]
 60%|██████    | 3/5 [00:19<00:13,  6.65s/it]
 80%|████████  | 4/5 [00:26<00:06,  6.69s/it]
100%|██████████| 5/5 [00:33<00:00,  6.68s/it]
100%|██████████| 5/5 [00:33<00:00,  6.69s/it]
Training Meta-policy: 1 Train_Acc: 0.38461538461538464 Train_Loss 1.3143666 Val_Acc: 0.39506172839506176 Val_Loss: 1.2668587

  0%|          | 0/5 [00:00<?, ?it/s]
 20%|██        | 1/5 [00:06<00:26,  6.65s/it]
 40%|████      | 2/5 [00:13<00:20,  6.67s/it]
 60%|██████    | 3/5 [00:20<00:13,  6.76s/it]
 80%|████████  | 4/5 [00:27<00:06,  6.73s/it]
100%|██████████| 5/5 [00:33<00:00,  6.73s/it]
100%|██████████| 5/5 [00:33<00:00,  6.75s/it]
Training Meta-policy: 2 Train_Acc: 0.380952380952381 Train_Loss 1.2847203 Val_Acc: 0.3974358974358974 Val_Loss: 1.2082058
....
Training Meta-policy: 5 Train_Acc: 0.37037037037037035 Train_Loss 1.5145593 Val_Acc: 0.4105263157894737 Val_Loss: 1.3875005


Training GNNs with learned meta-policy
 0%|          | 1/500 [00:11<1:38:57, 11.90s/it, train_acc:0.3000, eva_acc:0.4107, best_acc:0.4107]

  0%|          | 2/500 [00:28<1:32:54, 11.19s/it, train_acc:0.3774, eva_acc:0.3704, best_acc:0.4107]
  ... 

100%|██████████| 500/500 [1:19:38<00:00,  9.47s/it, train_acc:0.9417, eva_acc:0.9011, best_acc:0.9176]
============================1/10==================================
...
```

   **Note:**

   - `main.py` uses the 10-fold validation setting from [PyG](https://github.com/pyg-team/pytorch_geometric/tree/master/benchmark/kernel).
   - `one_fold_main.py` is training on one fold.
   - The datasets we used in this paper can be loaded from [PyG](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) automatically.

   - We follow the two-step training fashion： 1) train the RL agents and GNN simultaneously 2) train GNN with fixed RL agent parameters. 

   ## **2. Reproducibility**

   We provide the snapshot of the RL model. 
   To easily reproduce the results you can follow the next steps:

   1. download the best_save folder, and extract it to the ./

      [google driver](https://drive.google.com/file/d/1C1wuoSOWfqSPUWMtnDjaGCJH5OtRczJU/view?usp=sharing)

      [Lanzou](https://wwt.lanzouj.com/if9Tq05khjeb)

2. load the saved models:

```bash
python load_best_main.py --dataset <dataset_name>
```

you can change the dataset settings in the `load_best_main.py`

For instance: python load_best_main.py --dataset PROTEINS

```bash
./best_save/PROTEINS/fold_0/0.72321
PROTEINS: max_size:620 min_size:4 avg_size:39.05750224618149 node_label:3
test_acc:0.7232142857142857 test_loss:2.729332685470581
./best_save/PROTEINS/fold_1/0.73214
PROTEINS: max_size:620 min_size:4 avg_size:39.05750224618149 node_label:3
test_acc:0.7321428571428571 test_loss:7.343456540788923 
./best_save/PROTEINS/fold_2/0.75000
PROTEINS: max_size:620 min_size:4 avg_size:39.05750224618149 node_label:3
test_acc:0.75 test_loss:6.009946346282959
./best_save/PROTEINS/fold_3/0.80180
PROTEINS: max_size:620 min_size:4 avg_size:39.05750224618149 node_label:3
test_acc:0.8018018018018018 test_loss:1.3272351026535034
./best_save/PROTEINS/fold_4/0.78378
PROTEINS: max_size:620 min_size:4 avg_size:39.05750224618149 node_label:3
test_acc:0.783782451121254 test_loss:7.612959384918213
./best_save/PROTEINS/fold_5/0.77477
PROTEINS: max_size:620 min_size:4 avg_size:39.05750224618149 node_label:3
test_acc:0.7747747747747747 test_loss:1.141286015510559
./best_save/PROTEINS/fold_6/0.78378
PROTEINS: max_size:620 min_size:4 avg_size:39.05750224618149 node_label:3
test_acc:0.783782451121254 test_loss:1.3817527294158936
```

   **Note:** Due to the change of machine and software version, there may contains slight difference between the logged resutls and your reproduced one.





