# augmented KPGNN 

This project builds on the code for the Web Conference 2021 paper [Knowledge-Preserving Incremental Social Event Detection via Heterogeneous GNNs](https://arxiv.org/pdf/2101.08747.pdf).

# To run KPGNN

Step 1) cd /KPGNN

Step 2) run *generate_initial_features.py* to generate the initial features for the messages (please see Figure 1(b) and Section 3.2 of the paper for more details).

Step 3) run *custom_message_graph.py* to construct incremental message graphs. To construct small message graphs for test purpose, set *test=True* when calling *construct_incremental_dataset_0922()*. To use all the messages (see Table. 4 of the paper for a statistic of the number of messages in the graphs), set *test=False*.

Step 4) run *main.py*

# To run KPGNN augmented with GODE

```
python3 main.py --encoder 1
```

# To run KGPNN augmented with KL-divergence loss

```
python3 main.py --use_kl 1
```