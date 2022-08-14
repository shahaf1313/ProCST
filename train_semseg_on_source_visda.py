def main(opt):
    opt.num_scales = 0
    opt.curr_scale = opt.num_scales
    opt.num_steps = 250e3
    source_train_loader = CreateSrcDataLoader(opt, 'train', get_image_label=True)
    source_val_loader = CreateSrcDataLoader(opt, 'val', get_image_label=True)
    opt.epoch_size = len(source_train_loader.dataset)
    opt.save_pics_rate = set_pics_save_rate(opt.pics_per_epoch, opt.batch_size, opt)
    if opt.continue_train_from_path != '':
        _, semseg_optimizer = CreateSemsegModel(opt)
        semseg_net = torch.nn.DataParallel(torch.load(opt.continue_train_from_path))
        semseg_schedualer = PolynomialLR(semseg_optimizer, max_iter=opt.num_steps, gamma=0.9)
        semseg_schedualer.step(opt.resume_step)
    else:
        semseg_net, semseg_optimizer = CreateSemsegModel(opt)
        semseg_net = torch.nn.DataParallel(semseg_net)
        semseg_schedualer = PolynomialLR(semseg_optimizer, max_iter=opt.num_steps, gamma=0.9)
    print('########################### Configuration ##############################')
    for arg in vars(opt):
        print(arg + ': ' + str(getattr(opt, arg)))
    print('########################################################################')
    print('Architecture of Semantic Segmentation network:\n' + str(semseg_net.module))
    opt.tb = SummaryWriter(os.path.join(opt.tb_logs_dir, '%sGPU%d' % (datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S'), opt.gpus[0])))

    best_miou = 0
    steps = 0 if opt.continue_train_from_path == '' else opt.resume_step
    print_int = 0
    save_pics_int = 0
    epoch_num = 1
    start = time.time()
    keep_training = True

    while keep_training:
        print('semeg train: starting epoch %d...' % (epoch_num))
        semseg_net.train()

        for batch_num, (source_scales, source_label) in enumerate(source_train_loader):
            if steps > opt.num_steps:
                keep_training = False
                break

            semseg_optimizer.zero_grad()
            source_image = source_scales[opt.curr_scale].to(opt.device)
            source_label = source_label.to(opt.device)
            output_softs, semseg_loss = semseg_net(source_image, source_label)
            semseg_loss = semseg_loss.mean()
            output_label = output_softs.argmax(1)
            opt.tb.add_scalar('TrainSemseg/loss', semseg_loss.item(), steps)
            semseg_loss.backward()
            semseg_optimizer.step()
            semseg_schedualer.step()


            if int(steps/opt.print_rate) >= print_int or steps == 0:
                elapsed = time.time() - start
                print('train semseg:[%d/%d] ; elapsed time = %.2f secs per step' %
                      (print_int*opt.print_rate, opt.num_steps, elapsed/opt.print_rate))
                start = time.time()
                print_int += 1

            if int(steps/opt.save_pics_rate) >= save_pics_int or steps == 0:
                s       = denorm(source_image[0])
                s_lbl   = colorize_mask(source_label[0], palette=PALETTE_ZEROWASTE)
                pred_lbl = colorize_mask(output_label[0], palette=PALETTE_ZEROWASTE)
                opt.tb.add_image('TrainSemseg/source', s, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_label', s_lbl, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/pred_label', pred_lbl, save_pics_int*opt.save_pics_rate)
                save_pics_int += 1
            steps += 1

        #Validation:
        print('train semseg: starting validation after epoch %d.' % epoch_num)
        iou, miou, cm = calculte_validation_accuracy(semseg_net, source_val_loader, opt, epoch_num)
        save_epoch_accuracy(opt.tb, 'Validtaion', iou, miou, epoch_num)
        if epoch_num > 15 and miou > best_miou:
            best_miou = miou
            torch.save(semseg_net.module, '%s/semseg_trained_on_%s_miou_%.2f.pth' % (opt.out_folder, opt.source, miou))
        epoch_num += 1

    opt.tb.close()
    print('Finished training.')

def save_epoch_accuracy(tb, set, iou, miou, epoch):
    for i in range(NUM_CLASSES_ZEROWASTE):
        tb.add_scalar('%sAccuracy/%s class accuracy' % (set, id_to_name_zerowaste[i]), iou[i], epoch)
    tb.add_scalar('%sAccuracy/Accuracy History [mIoU]' % set, miou, epoch)


def calculte_validation_accuracy(semseg_net, val_loader, opt, epoch_num):
    semseg_net.eval()
    rand_samp_inds = np.random.randint(0, len(val_loader.dataset), 5)
    rand_batchs = np.floor(rand_samp_inds/opt.batch_size).astype(np.int)
    cm = torch.zeros((NUM_CLASSES_ZEROWASTE, NUM_CLASSES_ZEROWASTE)).cuda()
    for batch_num, (images, labels) in enumerate(val_loader):
        images = images[opt.curr_scale].to(opt.device)
        labels = labels[opt.curr_scale].to(opt.device)
        with torch.no_grad():
            pred_softs = semseg_net(images)
            pred_labels = torch.argmax(pred_softs, dim=1)
            cm += compute_cm_batch_torch(pred_labels, labels, IGNORE_LABEL_ZEROWASTE, NUM_CLASSES_ZEROWASTE)
            if batch_num in rand_batchs:
                t        = denorm(images[0])
                t_lbl    = colorize_mask(labels[0], palette=PALETTE_ZEROWASTE)
                pred_lbl = colorize_mask(pred_labels[0], palette=PALETTE_ZEROWASTE)
                opt.tb.add_image('Validtaion/Epoch%d/target' % (epoch_num), t, batch_num)
                opt.tb.add_image('Validtaion/Epoch%d/target_label' % (epoch_num), t_lbl, batch_num)
                opt.tb.add_image('Validtaion/Epoch%d/prediction_label' % (epoch_num), pred_lbl, batch_num)
    iou, miou = compute_iou_torch(cm)
    return iou, miou, cm

def set_pics_save_rate(pics_per_epoch, batch_size, opt):
    return np.maximum(2, int(opt.epoch_size / batch_size / pics_per_epoch))


if __name__ == "__main__":
    from core.config import get_arguments, post_config
    parser = get_arguments()
    opt = parser.parse_args()
    opt = post_config(opt)
    from torch.optim.lr_scheduler import _LRScheduler
    from semseg_models import CreateSemsegModel
    from core.constants import NUM_CLASSES_ZEROWASTE, IGNORE_LABEL_ZEROWASTE, id_to_name_zerowaste, PALETTE_ZEROWASTE
    from core.functions import compute_cm_batch_torch, compute_iou_torch
    from data_handlers import CreateSrcDataLoader
    import torch
    from core.config import get_arguments, post_config
    from core.functions import denorm, colorize_mask
    import numpy as np
    import time
    import os
    from torch.utils.tensorboard import SummaryWriter
    import datetime
    class PolynomialLR(_LRScheduler):
        def __init__(self, optimizer, max_iter, decay_iter=1,
                     gamma=0.9, last_epoch=-1):
            self.decay_iter = decay_iter
            self.max_iter = max_iter
            self.gamma = gamma
            super(PolynomialLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            factor = max(factor, 0)
            return [base_lr * factor for base_lr in self.base_lrs]
    main(opt)

