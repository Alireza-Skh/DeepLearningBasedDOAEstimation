import torch
from torch.utils.data import DataLoader


def evaluate(
    model,
    dataloader: DataLoader,
    loss_fn,
    device: torch.device,
    metrics: dict = None,
    return_outputs: bool = False
):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for m in metrics.values():
        m.reset()

    all_outputs = [] if return_outputs else None

    with torch.inference_mode():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)

            outputs = model(X)
            loss = loss_fn(outputs, Y)

            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for name, metric in metrics.items():
                metric.update(outputs, Y)

            if return_outputs:
                all_outputs.append((outputs.cpu(), Y.cpu()))

    avg_loss = total_loss / total_samples
    results = {"loss": avg_loss}
    for name, metric in metrics.items():
        results[name] = metric.compute().item()

    if return_outputs:
        results["outputs"] = all_outputs

    return results
