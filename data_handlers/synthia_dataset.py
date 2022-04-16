import os.path as osp
from PIL import Image
import numpy as np
import imageio
imageio.plugins.freeimage.download()
from .domain_adaptation_dataset import domainAdaptationDataSet
from core.constants import RESIZE_SHAPE

class SynthiaDataSet(domainAdaptationDataSet):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False, get_image_label_pyramid=False, get_filename=False, get_original_image=False):
        super(SynthiaDataSet, self).__init__(root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=get_image_label)
        self.domain_resize = RESIZE_SHAPE['synthia']
        self.get_image_label_pyramid = get_image_label_pyramid
        self.get_filename = get_filename
        self.get_original_image = get_original_image
        self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(self.root, "RGB/%s" % name)).convert('RGB')
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
            label_path = osp.join(self.root, "GT/LABELS/%s" % name)
            label = np.asarray(imageio.imread(label_path, format='PNG-FI'))[:,:,0]
            label = Image.fromarray(label)
            if self.get_original_image:
                pass
            else: # use random crop
                label = label.resize(self.domain_resize, Image.NEAREST)
                label = label.crop((left, upper, right, lower))
            if self.get_image_label_pyramid:
                labels_pyramid =  self.GeneratePyramid(label, is_label=True)
                labels_pyramid = [self.convert_to_class_ids(label_scale) for label_scale in labels_pyramid]
            else:
                label = self.convert_to_class_ids(label)


        scales_pyramid = self.GeneratePyramid(image)
        if self.get_image_label:
            return scales_pyramid, label
        elif self.get_image_label_pyramid:
            return scales_pyramid, labels_pyramid
        else:
            return scales_pyramid if not self.get_filename else scales_pyramid, self.img_ids[index]