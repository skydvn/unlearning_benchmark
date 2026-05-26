from torchvision import transforms

def get_retain_test_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])