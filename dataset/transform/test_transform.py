from torchvision import transforms

def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])