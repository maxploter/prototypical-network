import torch
from tqdm import tqdm

def train_one_epoch(model, criterion, dataloader, optimizer, args, epoch=None):
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        criterion: Loss function
        dataloader: DataLoader with training data
        optimizer: Optimizer for training
        args: Training arguments
        epoch: Current epoch number (optional)
    """
    model.train()

    total_loss = 0.0

    # Progress bar
    desc = f'Training Epoch {epoch}' if epoch is not None else 'Training'
    pbar = tqdm(dataloader, desc=desc)

    for episode_idx, (inputs, targets) in enumerate(pbar):
        # Move data to device
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

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

def evaluate(model, criterion, dataloader, args, epoch=None):
    """
    Evaluate the model on validation/test data.

    Args:
        model: The model to evaluate
        criterion: Loss function
        dataloader: DataLoader with validation/test data
        args: Training arguments
        epoch: Current epoch number (optional)

    Returns:
        avg_loss: Average loss over all episodes
        avg_accuracy: Average accuracy over all episodes
    """
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0

    # Progress bar
    desc = f'Validation Epoch {epoch}' if epoch is not None else 'Validation'
    pbar = tqdm(dataloader, desc=desc)

    with torch.no_grad():
        for episode_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Compute accuracy (for prototypical networks)
            if hasattr(criterion, 'compute_accuracy'):
                accuracy = criterion.compute_accuracy(outputs, targets)
            else:
                # Simple accuracy for autoencoder or other models
                # For prototypical loss, outputs are distances/logits
                if outputs.dim() == 2:  # Classification output
                    predictions = torch.argmax(outputs, dim=1)
                    accuracy = (predictions == targets).float().mean().item()
                else:
                    accuracy = 0.0  # For autoencoder, accuracy doesn't apply

            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy

            # Update progress bar
            avg_loss = total_loss / (episode_idx + 1)
            avg_accuracy = total_accuracy / (episode_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_accuracy:.4f}',
            })

    # Print final statistics
    print(f'\nValidation complete:')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Average Accuracy: {avg_accuracy:.4f}')

    return avg_loss, avg_accuracy
