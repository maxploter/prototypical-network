from torchvision import datasets, transforms
from dataset.episode_sampler import EpisodeSampler


def build_dataset(args):
    """Build MNIST dataset for training"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root='./',
        train=True,
        download=True,
        transform=transform
    )

    return dataset


def build_sampler(args, dataset):
    """Build episode sampler for few-shot learning"""
    samples_per_class = args.k_shot + args.q_query

    sampler = EpisodeSampler(
        labels=dataset.targets,
        classes_per_iteration=args.n_way,
        samples_per_class=samples_per_class,
        number_of_iterations=args.num_episodes
    )

    return sampler
