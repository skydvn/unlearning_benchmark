import torch

def random_labeling(model, criteria, optimizer, forget_loader, device="cuda"):
    """
    performs unlearning by training the model on the forget_set using incorrect, randomized labels.
    dynamically infers the number of classes from the model's output shape.
    
    args:
        model: architecture inherited from BaseArchitecture.
        criteria: loss function (e.g., nn.CrossEntropyLoss()).
        optimizer: PyTorch optimizer (e.g., Adam, SGD).
        forget_loader: DataLoader containing the data to forget.
        device: "cuda" or "cpu".
        
    returns:
        float: the average loss against the randomized labels.
    """
    # ensure the model is in train mode
    model.train()
    
    total_loss = 0.0
    
    for batch in forget_loader:
        # safely extracts images and true labels, ignoring domains if they are present in the batch
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        # clear gradients
        optimizer.zero_grad()
        
        # 1. forward pass first to get the logits
        logits, _ = model.forward_with_grad(images)
        
        # 2. dynamically infer the number of classes from the logit shape (batch_size, num_classes)
        num_classes = logits.shape[1]
        
        # 3. generate random shifts between 1 and (num_classes - 1)
        # this guarantees the new label is always different from the true label
        shifts = torch.randint(
            low=1, 
            high=num_classes, 
            size=labels.shape, 
            dtype=labels.dtype, 
            device=device
        )
        
        # 4. apply shift and wrap around using modulo to create the fake labels
        random_labels = (labels + shifts) % num_classes
        
        # 5. calculate standard loss against the fake random labels
        loss = criteria(logits, random_labels)
        
        # 6. backpropagate and update weights to learn the fake labels
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(forget_loader)
    return avg_loss