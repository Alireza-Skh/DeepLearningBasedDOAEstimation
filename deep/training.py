from tqdm import tqdm
import torch
from torch.nn import functional as F
import torchmetrics
import copy


def _run_one_epoch(model, loader, loss_fns, loss_weights, metric_objs, device, optimizer=None):
    """Helper function to run one epoch of training or validation."""
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0

    # Reset all metrics
    for m in metric_objs.values():
        m.reset()

    # Use tqdm for a progress bar
    for X, Y_tuple in tqdm(loader, desc="Training" if is_train else "Validation", leave=False):
        X = X.to(device)

        if not isinstance(Y_tuple, (list, tuple)):
            Y_tuple = (Y_tuple,)
        Y_tuple = tuple(y.to(device) for y in Y_tuple)

        # Forward pass
        preds_tuple = model(X)
        if not isinstance(preds_tuple, (list, tuple)):
            preds_tuple = (preds_tuple,)

        # --- Loss Calculation ---
        total_loss = 0
        for preds, target, loss_fn, weight in zip(preds_tuple, Y_tuple, loss_fns, loss_weights):
            total_loss += weight * loss_fn(preds, target)

        # --- Backward pass and optimization (only in training) ---
        if is_train:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        epoch_loss += total_loss.item()

        # --- Update metrics ---
        for i, m in enumerate(metric_objs.values()):
            preds = preds_tuple[i] if i < len(preds_tuple) else preds_tuple[-1]
            target = Y_tuple[i] if i < len(Y_tuple) else Y_tuple[-1]
            m.update(preds, target)

    avg_epoch_loss = epoch_loss / len(loader)

    # Compute final metric values
    computed_metrics = {name: m.compute().detach().cpu().item() for name, m in metric_objs.items()}

    return avg_epoch_loss, computed_metrics


def trainer_multioutputs(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fns: list,
    loss_weights: list,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    scheduler,
    patience: int = 5,
    metric_objs: dict[str, torchmetrics.Metric] | None = None,
    scheduler_metric: str = "val_loss"
):
    """
    Flexible trainer for models with one or more outputs.

    Args:
        model (torch.nn.Module): The model to train. Must return a tuple of output tensors.
        train_loader (torch.utils.data.DataLoader): DataLoader for training. 
            Must yield (X, Y_tuple), where Y_tuple is a tuple of target tensors.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation.
        loss_fns (list): A list of loss function objects, one for each output.
        loss_weights (list): A list of floats to weight each output's loss.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
        epochs (int): Number of epochs to train.
        scheduler: Learning rate scheduler.
        patience (int, optional): Patience for early stopping. Defaults to 5.
        metric_objs (dict[str, torchmetrics.Metric] | None, optional): 
            A dictionary of torchmetrics objects. The order must match the model outputs.
            Example: {'angle_mae': MAE(), 'distance_rmse': RMSE()}
        scheduler_metric (str, optional): The metric to monitor for the LR scheduler 
            and early stopping. Must be 'val_loss' or a key from val_metrics. Defaults to "val_loss".

    Returns:
        tuple: A tuple containing:
            - dict: The state dictionary of the best model based on validation loss.
            - dict: A history of training and validation metrics and losses.
    """
    if metric_objs is None:
        metric_objs = {}

    # Ensure all metric objects are on the correct device
    for name, metric in metric_objs.items():
        metric_objs[name] = metric.to(device)

    # history = {
    #     "train_loss": [], "val_loss": [],
    #     **{f"train_{name}": [] for name in metric_objs},
    #     **{f"val_{name}": [] for name in metric_objs}
    # }

    best_monitor_value = float("inf")
    patience_counter = 0
    best_state_dict = None

    for epoch in range(epochs):
        # ---------- Training ----------
        train_loss, train_metrics = _run_one_epoch(
            model, train_loader, loss_fns, loss_weights, metric_objs, device, optimizer
        )
        # history["train_loss"].append(train_loss)
        # for name, val in train_metrics.items():
        #     history[f"train_{name}"].append(val)

        # ---------- Validation ----------
        with torch.inference_mode():
            val_loss, val_metrics = _run_one_epoch(
                model, val_loader, loss_fns, loss_weights, metric_objs, device, optimizer=None
            )
        # history["val_loss"].append(val_loss)
        # for name, val in val_metrics.items():
        #     history[f"val_{name}"].append(val)

        # ---------- Scheduler and Early Stopping ----------
        # Get the value to monitor (e.g., validation loss or a specific metric)
        if scheduler_metric == "val_loss":
            current_monitor_value = val_loss
        elif scheduler_metric in val_metrics:
            current_monitor_value = val_metrics[scheduler_metric]
        else:
            raise ValueError(f"scheduler_metric '{scheduler_metric}' not found. "
                             f"Available options: 'val_loss' or keys in val_metrics: {list(val_metrics.keys())}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_monitor_value)
        else:
            scheduler.step()

        if current_monitor_value < best_monitor_value:
            best_monitor_value = current_monitor_value
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        # ---------- Logging ----------
        log_message = (
            f"Epoch {epoch+1:02d} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"train_loss: {train_loss:.6f} | "
            f"val_loss: {val_loss:.6f}"
        )

        metrics_log_parts = []
        for name in metric_objs.keys():
            metrics_log_parts.append(f"train_{name}: {train_metrics[name]:.4f}")
            metrics_log_parts.append(f"val_{name}: {val_metrics[name]:.4f}")

        if metrics_log_parts:
            log_message += " | " + " | ".join(metrics_log_parts)

        print(log_message)

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1} after {patience} epochs with no improvement.")
            break

    if best_state_dict is None:
        print("Warning: No best model state was saved. Returning the last model state.")
        # return copy.deepcopy(model.state_dict()), history
        return copy.deepcopy(model.state_dict())

    # return best_state_dict, history
    return best_state_dict
