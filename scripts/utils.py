import torch
def cal_token_acc(batch, logits):
    """
    Calculates token accuracy for tokenization models.

    Args:
        batch (dict): A batch of data containing 'labels' and 'label_ids'.
        logits (torch.Tensor): The model's output logits.

    Returns:
        torch.Tensor: The calculated token accuracy.
    """
    labels = batch['labels']
    shift_labels = labels[:, 1:].to(logits.device)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_preds = shift_logits.argmax(dim=-1)
    mask = batch['label_ids'][:, 1:].to(logits.device)
    correct_preds = (shift_labels == shift_preds) & mask.bool()
    action_acc = correct_preds.sum().float() / mask.sum().float()
    return action_acc