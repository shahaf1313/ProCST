import numpy as np
from core.constants import IGNORE_LABEL, IMG_CROP_SIZE_SEMSEG
from core.functions import GeneratePyramid
import os.path as osp
from torch.utils import data
from torchvision import transforms

class domainAdaptationDataSet(data.Dataset):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False):
        self.root = root
        if images_list_path != None:
            self.images_list_file = osp.join(images_list_path, '%s.txt' % set)
            self.img_ids = [image_id.strip() for image_id in open(self.images_list_file)]
        self.scale_factor = scale_factor
        self.num_scales = num_scales
        self. curr_scale = curr_scale
        self.set = set
        self.trans = transforms.ToTensor()
        self.crop_size = IMG_CROP_SIZE_SEMSEG
        self.ignore_label = IGNORE_LABEL
        self.get_image_label = get_image_label

    def __len__(self):
        return len(self.img_ids)

    def SetEpochSize(self, epoch_size):
        if (epoch_size > len(self.img_ids)):
            self.img_ids = self.img_ids * int(np.ceil(float(epoch_size) / len(self.img_ids)))
        self.img_ids = self.img_ids[:epoch_size]

    def convert_to_class_ids(self, label_image):
        label = np.asarray(label_image, np.float32)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def GeneratePyramid(self, image, is_label=False):
        scales_pyramid = GeneratePyramid(image, self.num_scales, self.curr_scale, self.scale_factor, is_label=is_label)
        return scales_pyramid

