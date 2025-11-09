from torch.utils.data import Dataset


class AutoencoderDataset(Dataset):
  """
  Wrapper dataset for autoencoder training.
  Returns flattened images as both input and target for reconstruction.
  """

  def __init__(self, base_dataset):
    """
    Args:
        base_dataset: The base MNIST dataset
    """
    self.base_dataset = base_dataset

  def __len__(self):
    return len(self.base_dataset)

  def __getitem__(self, idx):
    """
    Returns:
        tuple: (flattened_image, flattened_image) for reconstruction task
    """
    image, _ = self.base_dataset[idx]

    # Flatten the image
    flattened = image.view(-1)

    # Return flattened image as both input and target
    return flattened, flattened
