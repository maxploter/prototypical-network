## Train

### Train Autoencoder

```shell
python main.py --model autoencoder --epochs 2 --batch_size 64 --lr 0.001 --embedding_dim 64 --output_dir ./checkpoints
```

### Train Prototypical Network

```shell
python main.py --model prototypical_autoencoder --epochs 2 --n_way 5 --k_shot 5 --q_query 15 --num_episodes 100 --lr 0.001 --embedding_dim 64 --output_dir ./checkpoints
```
