import torch

def boundary_shrink(model, criteria, optimizer, forget_loader, epsilon=0.1, device="cuda"):
    """
    performs unlearning by shifting the forget data towards the nearest decision boundary.
    it does this by generating adversarial labels using fgsm, then training the model
    to predict those adversarial labels for the original images.
    
    args:
        model: architecture inherited from BaseArchitecture.
        criteria: loss function (e.g., nn.CrossEntropyLoss()).
        optimizer: PyTorch optimizer (e.g., Adam, SGD).
        forget_loader: DataLoader containing the data to forget.
        epsilon: step size for the adversarial perturbation.
        device: "cuda" or "cpu".
        
    returns:
        float: the average loss against the adversarial labels.
    """
    total_loss = 0.0
    
    for batch in forget_loader:
        # safely extracts images and true labels
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        # ==========================================
        # phase 1: generate adversarial labels
        # ==========================================
        # set to eval mode to freeze batchnorm/dropout during attack generation
        model.eval()
        
        # clone images and enable gradient tracking for the input tensor
        images_adv = images.detach().clone().requires_grad_(True)
        
        # forward pass (we call the model directly to keep the computation graph alive)
        adv_logits, _ = model(images_adv)
        
        # calculate loss against true labels to find the gradient direction
        loss_adv = criteria(adv_logits, labels)
        
        # clear any existing gradients, then backpropagate to the input images
        optimizer.zero_grad() 
        loss_adv.backward()
        
        # extract the sign of the input gradients and create perturbed images (fgsm)
        grad_sign = images_adv.grad.detach().sign()
        images_perturbed = images_adv.detach() + epsilon * grad_sign
        
        # pass the perturbed images through the network to get the adversarial predictions
        # we can safely use torch.no_grad() here to save memory
        with torch.no_grad():
            perturbed_logits, _ = model(images_perturbed)
            # get the predicted class indices for the adversarial images
            adv_labels = torch.argmax(perturbed_logits, dim=1)
            
        # ==========================================
        # phase 2: boundary shrink update
        # ==========================================
        # clear gradients accumulated during the adversarial backward pass
        optimizer.zero_grad()
        
        # forward pass original images (uses your forward_with_grad to set train mode)
        ori_logits, _ = model.forward_with_grad(images)
        
        # calculate the loss forcing the original images to map to the adversarial labels
        loss = criteria(ori_logits, adv_labels.detach())
        
        # update the model weights
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(forget_loader)
    return avg_loss