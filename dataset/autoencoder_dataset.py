from torch.utils.data import Dataset


class AutoencoderDataset(Dataset):
  """
  Wrapper dataset for autoencoder training.
  Returns flattened data as both input and target for reconstruction.

  For different loss types:
  - MSE/BCE (MNIST, TMNIST): input=float, target=float
  - CrossEntropy (Chess): input=float, target=int64
  """

  def __init__(self, base_dataset):
    """
    Args:
        base_dataset: The base dataset (MNIST, TMNIST, Chess, etc.)
    """
    self.base_dataset = base_dataset

  def __len__(self):
    return len(self.base_dataset)

  def __getitem__(self, idx):
    data, _ = self.base_dataset[idx]
    flattened = data.view(-1)
    target_tensor = flattened.long()
    return flattened, target_tensor
