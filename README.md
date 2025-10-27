## Dataset Preparation

### TMNIST

Download and prepare the TMNIST dataset with train/val/test splits:

```shell
python data/tmnist/prepare_dataset.py \
  --dataset_name nimishmagre/tmnist-glyphs-1812-characters \
  --output_dir ./data/tmnist \
  --seed 42
```

This will:
- Download the dataset from Kaggle to `./data/tmnist/tmnist-glyphs-1812-characters/`
- Create split files: `train_labels.txt`, `val_labels.txt`, `test_labels.txt`
- Default split ratios: 64% train, 16% val, 20% test

## Training

### Train Autoencoder on TMNIST

```shell
python main.py \
  --model autoencoder \
  --dataset_name tmnist \
  --dataset_path ./data/tmnist/tmnist-glyphs-1812-characters/tmnist-glyphs-1812-characters.csv \
  --dataset_split train \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.001 \
  --embedding_dim 64 \
  --output_dir ./checkpoints
```

### Train Prototypical Network on TMNIST

```shell
python main.py \
  --model prototypical_autoencoder \
  --dataset_name tmnist \
  --dataset_path ./data/tmnist/tmnist-glyphs-1812-characters/tmnist-glyphs-1812-characters.csv \
  --dataset_split train \
  --autoencoder_path ./checkpoints/autoencoder.pth \
  --epochs 2 \
  --n_way 5 \
  --k_shot 5 \
  --q_query 15 \
  --num_episodes 100 \
  --lr 0.001 \
  --embedding_dim 64 \
  --output_dir ./checkpoints
```
