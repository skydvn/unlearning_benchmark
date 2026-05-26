from torchvision import transforms

def get_forget_test_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])