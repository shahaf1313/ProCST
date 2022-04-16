def main(opt):
    opt.source_loaders, opt.target_loaders = segmentation_train(opt)
    opt.target_validation_loader = CreateTrgDataLoader(opt, set='val', get_image_label=True, get_scales_pyramid=False)

    print('########################### ProCST Configuration ##############################')
    for arg in vars(opt):
        print(arg + ': ' + str(getattr(opt, arg)))
    print('##################################################################################')
    train(opt)
    print('Finished Training.')

def segmentation_train(opt):
    opt.batch_size = 1
    opt.curr_scale = 0
    source_loader, target_loader = CreateSrcDataLoader(opt), CreateTrgDataLoader(opt)
    opt.epoch_size = np.maximum(len(target_loader.dataset), len(source_loader.dataset))

    source_loaders, target_loaders, num_epochs =[], [], []
    for i in range(opt.num_scales+1):
        opt.batch_size = opt.batch_size_list[i]
        opt.curr_scale = i
        source_loader, target_loader = CreateSrcDataLoader(opt, get_image_label=True), CreateTrgDataLoader(opt, get_scales_pyramid=True)
        source_loader.dataset.SetEpochSize(opt.epoch_size)
        target_loader.dataset.SetEpochSize(opt.epoch_size)
        source_loaders.append(source_loader)
        target_loaders.append(target_loader)

    return source_loaders, target_loaders

if __name__ == '__main__':
    from core.config import get_arguments, post_config
    parser = get_arguments()
    opt = parser.parse_args()
    opt = post_config(opt)
    from data_handlers import CreateSrcDataLoader
    from data_handlers import CreateTrgDataLoader
    from core.training import train
    import numpy as np
    main(opt)

