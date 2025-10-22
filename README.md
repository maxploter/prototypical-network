## Train

### Standard Python Script

```shell
python main.py --n_way 5 --k_shot 5 --q_query 15 --num_episodes 100 --lr 0.001 --embedding_dim 64 --output_dir ./checkpoints
```

### Arguments

- `--n_way`: Number of classes per episode (default: 5)
- `--k_shot`: Number of support samples per class (default: 5)
- `--q_query`: Number of query samples per class (default: 15)
- `--num_episodes`: Number of training episodes (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--embedding_dim`: Embedding dimension (default: 64)
- `--output_dir`: Directory to save model checkpoints (default: ./checkpoints)
