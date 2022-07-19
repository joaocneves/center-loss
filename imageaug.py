from torchvision import transforms


def transform_for_training(image_shape):
    return transforms.Compose(
       [transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
    )


def transform_for_infer(image_shape):
    return transforms.Compose(
       [transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
    )