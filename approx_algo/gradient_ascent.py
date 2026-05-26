import torch

def gradient_ascent(model, criteria, optimizer, forget_loader, device="cuda"):
    """
    performs gradient ascent on the forget_set to make the model "unlearn" data.
    
    args:
        model: architecture inherited from BaseArchitecture.
        criteria: loss function (e.g., nn.CrossEntropyLoss()).
        optimizer: PyTorch optimizer (e.g., Adam, SGD).
        forget_loader: DataLoader containing the data to forget.
        device: "cuda" or "cpu".
        
    returns:
        float: the average original loss (positive value) on the forget_set.
    """
    # ensure the model is in train mode
    model.train()
    
    total_loss = 0.0
    
    for batch in forget_loader:
        # safely extracts images and labels, ignoring domains if they are present in the batch
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        # clears the gradients from the previous step
        optimizer.zero_grad()
        
        # performs the forward pass to get logits and features
        logits, features = model.forward_with_grad(images)
        
        # calculates the original loss
        loss = criteria(logits, labels)
        
        # reverses the loss sign to perform gradient ascent.
        # the optimizer will try to minimize (-loss), which effectively maximizes the actual (loss).
        ascent_loss = -loss
        
        # performs backpropagation and updates the weights
        ascent_loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(forget_loader)
    return avg_loss