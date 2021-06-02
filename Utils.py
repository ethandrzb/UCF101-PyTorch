import os, sys
import numpy as np
import cv2
import random
import torch
from datetime import datetime

def build_paths(print_paths=True):
    cwd = os.getcwd()
    print('\nYour cwd:', cwd, '\n')

    win_cwd = 'C:\\Users\\ethan\\Documents\\Pycharm Projects\\UCF101-PyTorch'
    # linux_cwd = '/home/robert/PycharmProjects/UCF101'
    cluster_cwd = '/lustre/fs0/home/crcvreu.student9/PycharmProjects/UCF101-PyTorch'

    if cwd == win_cwd:
        print('==> Using Windows Paths...\n')
        print("We'll come back for you!!")
        exit()
        root = 'C:\\Users\\ethan\\Documents\\Pycharm Projects\\UCF101-PyTorch'
        # pretrained_model_path = '/Users/robertbrowning/Desktop/UCF/CRCV/Data/Pretrained/C3D_Sports1M_Pretrained.pickle'
        pretrained_model_path = '/Users/robertbrowning/Desktop/UCF/CRCV/Data/Pretrained/c3d-pretrained.pth'

    # elif cwd == linux_cwd:
    #     print('==> Using Linux Paths...\n')
    #     root = '/home/robert/Desktop/Data/UCF101'
    #     pretrained_model_path = '/home/robert/Desktop/Data/PretrainedActionWeights/c3d-pretrained.pth'

    elif cwd == cluster_cwd:
        print('==> Using Cluster Paths...\n')
        root = '/datasets/UCF-101'
        # pretrained_model_path = '/home/robert/Datasets/PretrainedActionWeights/C3D_Sports1M_Pretrained.pickle'
        pretrained_model_path = '/home/crcvreu.student9/Data/Pretrained/c3d-pretrained.pth'

    else:
        print('Where in the F*** are you!?')
        sys.exit()

    class_idxs = os.path.join(root, 'ActionRecognitionSplits/classInd.txt')
    train_split = os.path.join(root, 'ActionRecognitionSplits/trainlist01.txt')
    test_split = os.path.join(root, 'ActionRecognitionSplits/testlist01.txt')
    frames_root = os.path.join(root, 'Frames/frames-128x128')

    assert os.path.exists(class_idxs), 'class_idxs path DNE!'
    assert os.path.exists(train_split), 'train_split path DNE!'
    assert os.path.exists(test_split), 'test_split path DNE!'

    if print_paths:
        print('Class Index Path: %s' % class_idxs,
              '\nTrain Split Path: %s' % train_split,
              '\nTest Split Path: %s' % test_split,
              '\nFrames Root Dir: %s' % frames_root)
              #'\nPretrained Path: %s' % pretrained_model_path)

    return class_idxs, train_split, test_split, frames_root, pretrained_model_path


def print_time(msg: str):
    assert isinstance(msg, str), 'Argument "msg" must be a string.'
    now = datetime.now()
    current_time = now.strftime("%A, %B %d %Y at %I:%M%p")
    print('\n%s: %s\n' % (msg ,current_time))


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)





# Computes Mean and Std Dev, across RGB channels, of all training images in a Dataset & returns averages
# Set Pytorch Transforms to None for this function
def calc_mean_and_std(dataset):
    mean = np.zeros((3,1))
    std = np.zeros((3,1))
    print('==> Computing mean and std...')
    for img in dataset:
        scaled_img = np.array(img[0])/255
        mean_tmp, std_tmp = cv2.meanStdDev(scaled_img)
        mean += mean_tmp
        std += std_tmp
    mean = mean/len(dataset)
    std = std/len(dataset)

    return mean, std


def cv2_imshow(img_path, wait_key=0, window_name='Test'):
    img = cv2.imread(img_path)
    cv2.imshow(window_name, img)
    cv2.waitKey(wait_key)
    cv2.destroyAllWindows()