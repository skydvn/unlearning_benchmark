from torchvision import transforms

def get_unseen_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])