# Bloom QDE

This repository stores the code, data and experiments for Bloom QDE. This is the accompanying repository to the paper *[BloomQDE: Leveraging Bloom's Taxonomy for Question Difficulty Estimation](https://www.springer.com/de)*

```bash
pip install -r requirements.txt
```

## Reproduce Experiment Result on Binary Data with LSTM

On reproducibility:

Original results were optimized by applying a genetic algorithm. Additionally randomness and floating point precision introduces possible issues as described [here](https://pytorch.org/docs/stable/notes/randomness.html) and [here](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)

On binary SQuAD 2.0

```bash
python3 train_lstm.py --pretrained ./saves/model --test ./data/annotation_results/pos_tagged/sq_fc_th2_bin.csv
```

## Train Customized Model for QDE

1) Binary BloomLSTM

```bash
python3 train_lstm.py --train './data/shuffled_data/binary/train.csv' --gpu True/False --epochs 5
```

2) Multi-class BloomLSTM

```bash
python3 train_lstm.py --train './data/shuffled_data/multiclass/train.csv' --test './data/annotation_results/pos_tagged/arc_fc_th2_mc.csv'  --gpu True/False --epochs 5
```

For customizable parameters please refer to the script 'train_lstm.py' for name and description