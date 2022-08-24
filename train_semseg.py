"""
Author: Benny
Date: Nov 2019
"""
import argparse
from operator import mod
import os
from xml.sax.saxutils import prepare_input_source
from data_utils.SemanticKittiDataset import SemanticKITTI, ConfigSemanticKITTI
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from models.focal_loss import FocalLoss
from data_utils.metric import IoUCalculator
from functools import partialmethod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
# disable tqdm progressing bar
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

seg_label_to_cat = {
  1: "car",
  2: "bicycle",
  3: "motorcycle",
  4: "truck",
  5:  "other-vehicle",
  6: "person",
  7: "bicyclist",
  8: "motorcyclist",
  9: "road",
  10:  "parking",
  11: "sidewalk",
  12: "other-ground",
  13: "building",
  14: "fence",
  15: "vegetation",
  16: "trunk",
  17: "terrain",
  18: "pole",
  19: "traffic-sign",
}
ignored_labels = [0]
num_classes = 19
cfg = ConfigSemanticKITTI()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--epoch', default=80, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=40960, help='Point Number [default: 4096]')
    parser.add_argument('--syn', type=int, default=0, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--batch_size', type=int, default=4, help='Decay step for lr decay [default: every 10 epochs]')
    # parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    return parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = '/data/gpfs/projects/punim1650/Chaoyinc/data/SemanticKitti/sequences'
    NUM_CLASSES = 19
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    # get_dataset & dataloader
    if args.syn==1:
        seq_l = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10', '100', '101', '102', '103']
    elif args.syn==2:
        seq_l = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212']
    else:
        seq_l = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    TRAIN_DATASET = SemanticKITTI(mode='train',seq_list=seq_l)
    # TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = SemanticKITTI(mode='validation')
    # TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=5,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=5,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.get_class_weights()).to(device)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).to(device)
    criterion = FocalLoss(weights, 2)
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        acc_sum = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            # print(seg_pred.shape)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            target = target.view(-1, 1)[:, 0]

            valid_pred, valid_target, loss = compute_loss(seg_pred, target, criterion)

            loss.backward()
            optimizer.step()

            acc = compute_acc(valid_pred, valid_target)

            loss_sum += loss.item()
            acc_sum += acc.item()

            
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (acc_sum / num_batches))
        # quit()
        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        iou_calc = IoUCalculator(cfg)
        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)

            loss_sum = 0
            acc_sum = 0
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().to(device), target.long().to(device)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                target = target.view(-1, 1)[:, 0]

                valid_pred, valid_target, loss = compute_loss(seg_pred, target, criterion)

                acc = compute_acc(valid_pred, valid_target)

                loss_sum += loss.item()
                acc_sum += acc.item()
                iou_calc.add_data(valid_pred,valid_target)
        mean_iou, iou_list = iou_calc.compute_iou()
        logger.info('mean IoU:{:.1f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        logger.info(s)
        print("Valid Epoch: {}, Loss: {}, Acc: {}, mIoU: {}".format(epoch, loss_sum/num_batches, acc_sum/num_batches, mean_iou))
        
        if mean_iou >= best_iou:
            best_iou = mean_iou
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'class_avg_iou': mean_iou,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1

def compute_acc(valid_pred, valid_target):

    logits = valid_pred.max(dim=1)[1]
    acc = (logits == valid_target).sum().float() / float(valid_target.shape[0])
    return acc

def compute_loss(preds, targets, criterion):
    ignored_labels = [0] 

    # Boolean mask of points that should be ignored
    ignored_bool = (targets == 0)

    for ign_label in ignored_labels:
        ignored_bool = ignored_bool | (targets == ign_label)

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits = preds[valid_idx, :]
    valid_labels_init = targets[valid_idx]
    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, 19).long().to(preds.device)
    inserted_value = torch.zeros((1,)).long().to(preds.device)
    for ign_label in ignored_labels:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    loss = criterion(valid_logits, valid_labels).mean()
    return valid_logits, valid_labels, loss

if __name__ == '__main__':
    args = parse_args()
    main(args)
