import os, sys, select
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

from Utils import build_paths


class UCF101(Dataset):
    """
    Args:
        class_idxs (string): Path to list of class names and corresponding label (.txt)
        split (string): Path to train or test split (.txt)
        frames_root (string): Directory (root) of directories (classes) of directories (vid_id) of frames.

        UCF101
        ├── ApplyEyeMakeup
        │   ├── v_ApplyEyeMakeup_g01_c01
        │   │   ├── 000.jpg
        │   │   ├── 001.jpg
        │   │   └── ...
        │   └── ...
        │
        ├── ApplyLipstick
        │   ├── v_ApplyLipstick_g01_c01
        │   │   ├── 000.jpg
        │   │   ├── 001.jpg
        │   │   └── ...
        │   └── ...
        │
        ├── Archery
        │   │   ├── 000.jpg
        │   │   ├── 001.jpg
        │   │   └── ...
        │   └── ...
        │   └── ...
        │

        clip_len (int): Number of frames per sample, i.e. depth of Model input.
        train (bool): Training vs. Testing model. Default is True
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, class_idxs, split, frames_root, train, clip_len=16):

        self.class_idxs = class_idxs
        self.split_path = split
        self.frames_root = frames_root
        self.train = train
        self.clip_len = clip_len

        self.class_dict = self.read_class_ind()
        self.paths = self.read_split()
        self.data_list = self.build_data_list()

        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112


    # Reads .txt file w/ each line formatted as "1 ApplyEyeMakeup" and returns dictionary {'ApplyEyeMakeup': 0, ...}
    def read_class_ind(self):
        class_dict = {}
        with open(self.class_idxs) as f:
            for line in f:
                label, class_name_key = line.strip().split()
                if class_name_key not in class_dict:
                    class_dict[class_name_key] = []
                class_dict[class_name_key] = int(label) - 1  # .append(line.strip())

        return class_dict


    # Reads train or test split.txt file and returns list [('v_ApplyEyeMakeup_g08_c01', array(0)), ...]
    def read_split(self):
        paths = []
        with open(self.split_path) as f:
            for line in f:
                rel_vid_path = line.strip().split()[0][:-4]
                # ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01
                class_name, vid_id = rel_vid_path.split('/')
                # ApplyEyeMakeup, v_ApplyEyeMakeup_g08_c01
                vid_dir = os.path.join(self.frames_root, rel_vid_path)
                # /datasets/UCF-101/Frames/frames-128x128/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01
                assert os.path.exists(vid_dir), 'Directory %s does not exist!' % vid_dir
                paths.append((vid_dir, class_name))

        return paths


    def build_data_list(self):
        paths = self.paths
        class_dict = self.class_dict
        data_list = []
        for vid_dir, class_name in paths:
            label = np.array(class_dict[class_name], dtype=int)
            frame_count = len([frame for frame in os.listdir(vid_dir) if not frame.startswith('.')])
            data_list.append((vid_dir, label, frame_count, class_name))

        return data_list


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        vid_dir, label, frame_count, class_name = self.data_list[index]
        buffer = self.load_frames(vid_dir, frame_count)
        buffer = self.spatial_crop(buffer, self.crop_size)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        return buffer, label


    def load_frames(self, vid_dir, frame_count):
        time_index = np.random.randint(0, frame_count - self.clip_len - 1)
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        # (16, 128, 171, 3)
        for i in range(self.clip_len):
            frame_name = os.path.join(vid_dir, str(time_index + i).zfill(3) + '.jpg')
            # time_index = 11, i = 0
            # frame_name = /datasets/UCF-101/Frames/frames-128x128/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/011.jpg
            assert os.path.exists(frame_name), 'Path %s does not exist!' % frame_name
            try:
                frame = cv2.imread(frame_name)
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            except:
                print('The image %s is potentially corrupt!\nDo you wish to proceed? [y/n]\n' % frame_name)
                response, _, _ = select.select([sys.stdin], [], [], 15)
                if response == 'n':
                    sys.exit()
                else:
                    frame = np.zeros((buffer.shape[1:]))

            frame = np.array(frame).astype(np.float32)
            buffer[i] = frame

        return buffer


    @staticmethod
    def spatial_crop(buffer, crop_size):
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)
        # Spatial crop is performed on the entire array, so each frame is cropped in the same location.
        buffer = buffer[:, height_index:height_index + crop_size, width_index:width_index + crop_size, :]

        return buffer


    @staticmethod
    def normalize(buffer):
        for i, frame in enumerate(buffer):
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            frame -= np.array([[[90.0, 98.0, 102.0]]])  # BGR means
            buffer[i] = frame

        return buffer


    @staticmethod
    def to_tensor(buffer):
        # (clip_len, height, width, channels)
        #     0        1       2       3
        # buffer = buffer.transpose((3, 0, 1, 2))
        # # (channels, clip_len, height, width)

        return torch.from_numpy(buffer)

if __name__ == '__main__':

    batch_size = 2
    class_idx, train_split, test_split, frames_root, pret_path = build_paths()
    trainset = UCF101(class_idx, train_split, frames_root, train=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    for i, (batch, labels) in enumerate(trainloader):
        labels = np.array(labels)
        for j, clip in enumerate(batch):
            # (channels, clip_len, height, width)
            # clip = np.array(clip).transpose((1, 2, 3, 0))

            # (clip_len, channels, height, width)
            clip = np.array(clip).transpose((0, 2, 3, 1))

            # (clip_len, height, width, channels)

            clip += np.array([[[90.0, 98.0, 102.0]]])
            print(clip.shape)
            # for img in clip:
            #     img = img.astype('uint8')
            #     name = 'Class: %s' % str(labels[j] + 1)
            #     cv2.imshow(name, img)
            #     cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            #     cv2.resizeWindow(name, 140, 112)
            #     cv2.moveWindow(name, 10, 30)
            #     cv2.waitKey(50)
            #     cv2.destroyAllWindows()

