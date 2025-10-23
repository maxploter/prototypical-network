import torch
from tqdm import tqdm

def train_one_epoch(model, criterion, dataloader, optimizer, args, epoch=None):
    """
    Train the model for one epoch.

    Args:
        model: The prototypical network model
        dataloader: DataLoader with episode sampler
        optimizer: Optimizer for training
        args: Training arguments
        epoch: Current epoch number (optional)
    """
    model.train()

    total_loss = 0.0

    # Progress bar
    desc = f'Training Epoch {epoch}' if epoch is not None else 'Training'
    pbar = tqdm(dataloader, desc=desc)

    for episode_idx, (images, labels) in enumerate(pbar):
        # Move data to device
        images = images.to(args.device)
        labels = labels.to(args.device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass: get embeddings
        embeddings = model(images)

        # Compute prototypical loss
        loss = criterion(
            embeddings,
            labels,
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()

        # Update progress bar
        avg_loss = total_loss / (episode_idx + 1)
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
        })

    # Print final statistics
    print(f'\nTraining complete:')
    print(f'Average Loss: {avg_loss:.4f}')

    return avg_loss
