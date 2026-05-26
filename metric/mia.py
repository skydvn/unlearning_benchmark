import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

def mia(model, forget_loader, unseen_loader, device="cuda"):
    """
    performs a membership inference attack using logistic regression on the loss values.
    an ideal unlearned model will yield an mia score close to 0.5 (random chance), 
    meaning the forget set is indistinguishable from the unseen set.
    
    args:
        model: architecture inherited from BaseArchitecture.
        forget_loader: DataLoader containing the data we tried to forget.
        unseen_loader: DataLoader containing completely unseen/test data.
        device: "cuda" or "cpu".
        
    returns:
        float: the mean accuracy of the logistic regression attacker.
    """
    # use reduction="none" to get the individual loss for EVERY single image in the batch
    criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_losses(loader):
        """helper function to extract all loss values for a given dataloader."""
        all_losses = []
        
        # no need to manually set model.eval() or torch.no_grad() 
        # because your model.inference() method already handles it safely.
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            
            # extract logits using the custom inference method
            logits, _ = model.inference(images)
            
            # calculate losses for the batch, detach, move to cpu, and convert to numpy
            losses = criterion(logits, labels).cpu().detach().numpy()
            all_losses.extend(losses)
            
        return np.array(all_losses)

    # 1. gather all individual loss values
    forget_losses = compute_losses(forget_loader)
    unseen_losses = compute_losses(unseen_loader)

    # 2. balance the datasets to prevent the attacker from guessing the majority class
    min_len = min(len(forget_losses), len(unseen_losses))
    if min_len == 0:
        raise ValueError("Length of forget set or unseeen set = 0")
        
    forget_losses = forget_losses[:min_len]
    unseen_losses = unseen_losses[:min_len]

    # 3. prepare the dataset for the logistic regression attacker
    # reshape to (-1, 1) because sklearn expects 2d arrays for features
    samples_mia = np.concatenate((unseen_losses, forget_losses)).reshape((-1, 1))
    
    # assign label 0 to unseen data, and label 1 to forget data
    labels_mia = [0] * min_len + [1] * min_len

    # 4. train and evaluate the attacker using cross-validation
    attack_model = LogisticRegression()
    cv = StratifiedShuffleSplit(n_splits=10, random_state=42)
    
    mia_scores = cross_val_score(
        attack_model, samples_mia, labels_mia, cv=cv, scoring="accuracy"
    )
    
    mia_mean = mia_scores.mean()
    
    return mia_mean