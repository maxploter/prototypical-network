from torch.utils.data import RandomSampler, BatchSampler, Subset
from torchvision import datasets, transforms

from dataset.autoencoder_dataset import AutoencoderDataset
from dataset.dataset_reduction import apply_dataset_reduction
from dataset.episode_sampler import EpisodeSampler
from dataset.tmnist import TMNISTDataset
from utils import is_thresholded_dataset


def build_dataset(args, split='train'):
  """Build dataset for training or validation

  Args:
    args: Arguments containing dataset configuration
    split: Dataset split to use ('train', 'val', or 'test')
  """
  if args.dataset_name == 'mnist':
    # MNIST uses standard grayscale normalization (0-255 -> 0-1 -> normalized)
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(
      root='./',
      train=(split == 'train'),
      download=True,
      transform=transform
    )
  elif args.dataset_name == 'tmnist':
    # Check if dataset is thresholded (binary) or original (grayscale)
    is_thresholded = is_thresholded_dataset(args.dataset_name, args.dataset_path)

    if is_thresholded:
      transform = None
    else:
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
      ])

    dataset = TMNISTDataset(
      dataset_path=args.dataset_path,
      split=split,
      transform=transform
    )
  else:
    raise ValueError(f"Unknown dataset: {args.dataset_name}. Supported: 'mnist', 'tmnist'")

  # Apply dataset reduction if specified (only for training split)
  if split == 'train' and hasattr(args, 'dataset_reduction') and args.dataset_reduction < 1.0:
    reduction_strategy = getattr(args, 'dataset_reduction_strategy', 'percentage')
    dataset = apply_dataset_reduction(
      dataset,
      reduction_factor=args.dataset_reduction,
      strategy=reduction_strategy
    )

  # Wrap with autoencoder dataset if using autoencoder model
  if args.model == 'autoencoder':
    dataset = AutoencoderDataset(dataset)

  return dataset


def build_sampler(args, dataset):
  if args.model == 'autoencoder':
    # For autoencoder, use the base dataset for sampler
    base_dataset = dataset.base_dataset if isinstance(dataset, AutoencoderDataset) else dataset
    sampler = RandomSampler(base_dataset)
    sampler = BatchSampler(
      sampler,
      batch_size=args.batch_size,
      drop_last=False
    )
  else:
    # Extract targets, handling Subset objects from dataset reduction
    if isinstance(dataset, Subset):
      base_dataset = dataset.dataset
      all_targets = base_dataset.targets
      targets = all_targets[dataset.indices]
    else:
      targets = dataset.targets

    samples_per_class = args.k_shot + args.q_query

    sampler = EpisodeSampler(
      labels=targets,
      classes_per_iteration=args.n_way,
      samples_per_class=samples_per_class,
      number_of_iterations=args.num_episodes
    )

  return sampler
