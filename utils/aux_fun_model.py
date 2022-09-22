import torch

from models import Resnet50FaceModel, Resnet18FaceModel

def load_attributes(atributes_file):
    soft_feat = np.load(atributes_file)
    soft_feat = soft_feat.astype('float32')

    return soft_feat

def load_model(arch, weights_file, device):
    if arch == 'resnet18':
        model_class = Resnet18FaceModel
    if arch == 'resnet50':
        model_class = Resnet50FaceModel

    checkpoint = torch.load(weights_file)
    num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]

    model = model_class(num_classes, feat_normalization='none').to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    return model