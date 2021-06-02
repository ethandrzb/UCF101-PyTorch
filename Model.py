import os

import torch
import torch.nn as nn
# from torchsummary import summary

class C3D(nn.Module):

    def __init__(self, num_classes, pretrained_path, res_ckpt_path='./checkpoint/ckpt.t7', pretrained=False, resume=False):

        # assert pretrained != resume, print("You have 'pretrain' and 'resume' set to True. "
        #                                    "Please set at least one to False.")

        self.pretrained_path = pretrained_path
        self.res_ckpt_path = res_ckpt_path

        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            print('Loading Pretrained Weights From: %s' % self.pretrained_path)
            self.__load_pretrained_weights()
        elif resume:
            print('Loading Previous Checkpoint From:', self.res_ckpt_path)
            self.best_acc, self.start_epoch = self.__resume()
        else:
            print('Initializing Model With Random Weights')


    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits


    def __load_pretrained_weights(self):

        p_dict = torch.load(self.pretrained_path)
        p_dict_keys = list(p_dict.keys())[0:-2]

        s_dict = self.state_dict()
        s_dict_keys = list(s_dict)[0:-2]

        for p_dict_key, s_dict_key in zip(p_dict_keys, s_dict_keys):
            s_dict[s_dict_key] = p_dict[p_dict_key]

        self.load_state_dict(s_dict)


    def __resume(self):
        checkpoint = torch.load(self.res_ckpt_path)
        self.load_state_dict(checkpoint['model_state'])
        self.load_state_dict(checkpoint['optim_state'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

        return best_acc, start_epoch


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    net = C3D(num_classes=10, pretrained_path="/Users/ethan/Documents/Pycharm Projects/Pretrained/c3d-pretrained.pth", pretrained=False)
    # summary(net, input_size=(3, 16, 112, 112))
    x = torch.ones((2, 3, 16, 112, 112))
    y = net(x)
    print(y.shape)