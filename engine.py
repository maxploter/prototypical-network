import torch
from tqdm import tqdm


def train_one_epoch(model, criterion, dataloader, optimizer, args, epoch=None):
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        criterion: Loss function that returns a dict of losses
        dataloader: DataLoader with training data
        optimizer: Optimizer for training
        args: Training arguments
        epoch: Current epoch number (optional)
    """
    model.train()

    total_loss = 0.0
    total_loss_dict = {}

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

        # Compute loss dict
        loss_dict = criterion(outputs, targets)

        # Get weight dict from criterion
        weight_dict = criterion.weight_dict if hasattr(criterion, 'weight_dict') else {}

        # Compute weighted loss for backprop
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Backward pass
        losses.backward()
        optimizer.step()

        # Accumulate losses for logging
        loss_value = losses.item()
        total_loss += loss_value

        for k, v in loss_dict.items():
            if k not in total_loss_dict:
                total_loss_dict[k] = 0.0
            total_loss_dict[k] += v.item()

        # Update progress bar
        avg_loss = total_loss / (episode_idx + 1)
        postfix = {'loss': f'{avg_loss:.4f}'}

        # Add individual losses to progress bar (only weighted ones)
        for k in loss_dict.keys():
            if k in weight_dict:
                avg_k_loss = total_loss_dict[k] / (episode_idx + 1)
                postfix[k] = f'{avg_k_loss:.4f}'

        pbar.set_postfix(postfix)

    # Print final statistics
    print(f'\nTraining complete:')
    print(f'Average Loss (weighted): {avg_loss:.4f}')
    for k in total_loss_dict.keys():
        avg_k_loss = total_loss_dict[k] / len(dataloader)
        print(f'Average {k}: {avg_k_loss:.4f}')

    return avg_loss


def evaluate(model, criterion, dataloader, args, epoch=None):
    """
    Evaluate the model on validation/test data.

    Args:
        model: The model to evaluate
        criterion: Loss function that returns a dict of losses
        dataloader: DataLoader with validation/test data
        args: Training arguments
        epoch: Current epoch number (optional)

    Returns:
        tuple: (avg_loss, avg_loss_dict) where avg_loss_dict contains all individual losses
    """
    model.eval()

    total_loss = 0.0
    total_loss_dict = {}

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

            # Compute loss dict
            loss_dict = criterion(outputs, targets)

            # Get weight dict from criterion
            weight_dict = criterion.weight_dict if hasattr(criterion, 'weight_dict') else {}

            # Compute weighted loss
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Accumulate losses
            loss_value = losses.item()
            total_loss += loss_value

            for k, v in loss_dict.items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = 0.0
                total_loss_dict[k] += v.item()

            # Update progress bar
            avg_loss = total_loss / (episode_idx + 1)
            postfix = {'loss': f'{avg_loss:.4f}'}

            # Add individual losses to progress bar
            for k in loss_dict.keys():
                avg_k_loss = total_loss_dict[k] / (episode_idx + 1)
                # Mark weighted losses with asterisk
                label = f'{k}*' if k in weight_dict else k
                postfix[label] = f'{avg_k_loss:.4f}'

            pbar.set_postfix(postfix)

    # Print final statistics
    print(f'\nValidation complete:')
    print(f'Average Loss (weighted): {avg_loss:.4f}')

    # Compute and print average losses
    avg_loss_dict = {}
    for k, v in total_loss_dict.items():
        avg_k_loss = v / len(dataloader)
        avg_loss_dict[k] = avg_k_loss
        is_weighted = '(weighted)' if k in weight_dict else '(unweighted)'
        print(f'Average {k} {is_weighted}: {avg_k_loss:.4f}')

    return avg_loss, avg_loss_dict
