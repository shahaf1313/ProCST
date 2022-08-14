from torch.utils import data
from data_handlers.gta5_dataset import GTA5DataSet
from data_handlers.cityscapes_dataset import cityscapesDataSet
from data_handlers.synthia_dataset import SynthiaDataSet
from data_handlers.zerowaste_dataset import ZeroWastedataset
from data_handlers.synthwaste_dataset import SynthWasteDataset


def CreateSrcDataLoader(opt, set='train', get_image_label=False, get_image_label_pyramid=False, get_filename=False, get_original_image=False):
    if opt.source == 'gta5':
        source_dataset = GTA5DataSet(opt.src_data_dir,
                                     opt.src_data_list,
                                     opt.scale_factor,
                                     opt.num_scales,
                                     opt.curr_scale,
                                     set,
                                     get_image_label=get_image_label,
                                     get_image_label_pyramid=get_image_label_pyramid,
                                     get_filename=get_filename,
                                     get_original_image=get_original_image)
    elif opt.source == 'synthia':
        source_dataset = SynthiaDataSet(opt.src_data_dir,
                                        opt.src_data_list,
                                        opt.scale_factor,
                                        opt.num_scales,
                                        opt.curr_scale,
                                        set,
                                        get_image_label=get_image_label,
                                        get_image_label_pyramid=get_image_label_pyramid,
                                        get_filename=get_filename,
                                        get_original_image=get_original_image)
    elif opt.source == 'zerowastev1' or opt.source == 'zerowastev2':
        source_dataset = ZeroWastedataset(opt.src_data_dir,
                                          None,
                                          opt.scale_factor,
                                          opt.num_scales,
                                          opt.curr_scale,
                                          set,
                                          get_image_label=get_image_label,
                                          get_image_label_pyramid=get_image_label_pyramid,
                                          get_filename=get_filename,
                                          get_original_image=get_original_image)
    elif opt.source == 'synthwaste' or opt.source == 'synthwaste_aug':
        source_dataset = SynthWasteDataset(opt.src_data_dir,
                                          None,
                                          opt.scale_factor,
                                          opt.num_scales,
                                          opt.curr_scale,
                                          set,
                                          get_image_label=get_image_label,
                                          get_image_label_pyramid=get_image_label_pyramid,
                                          get_filename=get_filename,
                                          get_original_image=get_original_image)
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')

    source_dataloader = data.DataLoader(source_dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        num_workers=opt.num_workers,
                                        pin_memory=True,
                                        drop_last=not opt.no_drop_last)
    return source_dataloader


def CreateTrgDataLoader(opt, set='train', get_image_label=False, get_scales_pyramid=False):
    if opt.target == 'cityscapes':
        target_dataset = cityscapesDataSet(opt.trg_data_dir,
                                       opt.trg_data_list,
                                       opt.scale_factor,
                                       opt.num_scales,
                                       opt.curr_scale,
                                       set,
                                       get_image_label=get_image_label,
                                       get_scales_pyramid=get_scales_pyramid,
                                       get_original_image= set == 'val' or set == 'test')
    elif opt.target == 'zerowastev2':
        target_dataset = ZeroWastedataset(opt.trg_data_dir,
                                          None,
                                          opt.scale_factor,
                                          opt.num_scales,
                                          opt.curr_scale,
                                          set,
                                          get_image_label=get_image_label,
                                          get_image_label_pyramid=False,
                                          get_filename=False,
                                          get_original_image= set == 'val' or set == 'test')
    else:
        raise ValueError('The target dataset mush be either cityscapes or zerowastev2')

    target_dataloader = data.DataLoader(target_dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        num_workers=opt.num_workers,
                                        pin_memory=True,
                                        drop_last=not opt.no_drop_last)
    return target_dataloader
