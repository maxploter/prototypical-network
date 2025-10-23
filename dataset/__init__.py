from torchvision import datasets, transforms
from dataset.episode_sampler import EpisodeSampler
from dataset.autoencoder_dataset import AutoencoderDataset
from torch.utils.data import RandomSampler, BatchSampler


def build_dataset(args):
  """Build MNIST dataset for training"""
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])

  dataset = datasets.MNIST(
    root='./',
    train=(args.dataset_split == 'train'),
    download=True,
    transform=transform
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
    samples_per_class = args.k_shot + args.q_query

    sampler = EpisodeSampler(
      labels=dataset.targets,
      classes_per_iteration=args.n_way,
      samples_per_class=samples_per_class,
      number_of_iterations=args.num_episodes
    )

  return sampler
