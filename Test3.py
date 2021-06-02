import torch

from Model import C3D

corresp_name1 = {       # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }


pretrained_model_path = '/Users/robertbrowning/Desktop/UCF/CRCV/Data/Pretrained/c3d-pretrained.pth'
pretrained_model_path1 = '/Users/robertbrowning/Desktop/UCF/CRCV/Data/Pretrained/c3d-pretrained_keys.pth'

c3d = C3D(num_classes=101, pretrained_path=None, pretrained=None)
y = c3d.state_dict()
y.pop('fc8.weight', None)
y.pop('fc8.bias', None)

x = torch.load(pretrained_model_path)
x.pop('classifier.6.weight', None)
x.pop('classifier.6.bias', None)
for old_key in x:
    new_key = corresp_name1[old_key]
    y[new_key] = x[old_key]


torch.save(y, pretrained_model_path1)