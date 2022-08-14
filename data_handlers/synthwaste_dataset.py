import os
from data_handlers.domain_adaptation_dataset import domainAdaptationDataSet
from PIL import Image
import numpy as np
from core.constants import RESIZE_SHAPE

class SynthWasteDataset(domainAdaptationDataSet):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False, get_image_label_pyramid=False, get_filename=False, get_original_image=False):
        super(SynthWasteDataset, self).__init__(root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=get_image_label)
        self.domain_resize = RESIZE_SHAPE['synthwaste']
        self.get_filename = get_filename
        self.get_original_image = get_original_image
        self.get_image_label =get_image_label
        self.get_image_label_pyramid = get_image_label_pyramid
        self.sem_seg_root = os.path.join(self.root, set, "sem_seg")
        self.img_root = os.path.join(self.root, set, "data")
        self.img_ids = os.listdir(self.img_root)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(os.path.join(self.img_root, name)).convert('RGB')
        if self.get_original_image:
            pass
        else: # use random crop:
            image = image.resize(self.domain_resize, Image.BICUBIC)
            left = self.domain_resize[0]-self.crop_size[0]
            upper= self.domain_resize[1]-self.crop_size[1]
            left = np.random.randint(0, high=left)
            upper= np.random.randint(0, high=upper)
            right= left + self.crop_size[0]
            lower= upper+ self.crop_size[1]
            image = image.crop((left, upper, right, lower))

        label, label_copy, labels_pyramid = None, None, None
        if self.get_image_label or self.get_image_label_pyramid:
            label = Image.open(os.path.join(self.sem_seg_root, name))
            if self.get_original_image:
                pass
            else: # use random crop
                label = label.resize(self.domain_resize, Image.NEAREST)
                label = label.crop((left, upper, right, lower))
            label = np.asarray(label, np.uint8)
            label_copy = np.copy(label)

        scales_pyramid = self.GeneratePyramid(image)
        if self.get_image_label:
            return scales_pyramid, label_copy
        elif self.get_filename:
            return scales_pyramid, self.img_ids[index]
        else:
            return scales_pyramid