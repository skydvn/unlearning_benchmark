import torch

def forget_acc(model, forget_loader, device="cuda"):
    """
    calculates the accuracy of the model on the forget_set (or any other dataloader).
    
    args:
        model: architecture inherited from BaseArchitecture.
        forget_loader: DataLoader containing the data to evaluate.
        device: "cuda" or "cpu".
        
    returns:
        float: the accuracy as a ratio (between 0.0 and 1.0).
    """
    correct = 0
    total = 0
    
    for batch in forget_loader:
        # safely extracts images and true labels, ignoring domains if present
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        # use the inference method which automatically sets eval() and no_grad()
        # we only need the logits for accuracy, so we ignore the features
        logits, _ = model.inference(images)
        
        # get the predicted class indices
        predictions = torch.argmax(logits, dim=1)
        
        # count correct predictions
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
    accuracy = correct / total
    return accuracy