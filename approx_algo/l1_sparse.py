import torch

def l1_sparse(model, criteria, optimizer, forget_loader, alpha=0.25, device="cuda"):
    """
    performs unlearning by applying an l1 penalty to the model's weights 
    while training on the forget_set. this forces the weights towards zero, 
    degrading the learned representations for this specific data.
    
    args:
        model: architecture inherited from BaseArchitecture.
        criteria: loss function (e.g., nn.CrossEntropyLoss()).
        optimizer: PyTorch optimizer (e.g., Adam, SGD).
        forget_loader: DataLoader containing the data to forget.
        alpha: the scaling factor for the l1 penalty (higher = more aggressive unlearning).
        device: "cuda" or "cpu".
        
    returns:
        float: the average total loss (cross-entropy + l1 penalty).
    """
    # ensure the model is in train mode
    model.train()
    
    total_batch_loss = 0.0
    
    for batch in forget_loader:
        # safely extract images and true labels
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        # clear gradients
        optimizer.zero_grad()
        
        # forward pass (we only need logits to calculate the classification error)
        logits, _ = model.forward_with_grad(images)
        
        # 1. calculate standard cross-entropy loss against true labels
        ce_loss = criteria(logits, labels)
        
        # 2. calculate the l1 penalty
        # we iterate through the model's parameters and sum the absolute values of the weights.
        # we skip batchnorm layers ('bn') and biases, focusing only on connection weights.
        l1_penalty = 0.0
        for name, param in model.named_parameters():
            # timm usually names batchnorm layers with 'bn' or 'norm'
            if 'weight' in name and 'bn' not in name and 'norm' not in name:
                l1_penalty += torch.sum(torch.abs(param))
        
        # 3. combine the losses
        total_loss = ce_loss + (alpha * l1_penalty)
        
        # backpropagate and update weights
        total_loss.backward()
        optimizer.step()
        
        total_batch_loss += total_loss.item()
        
    avg_loss = total_batch_loss