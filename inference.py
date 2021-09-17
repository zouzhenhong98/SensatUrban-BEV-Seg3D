import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from dataloaders import make_data_loader
from modeling.unet import *

class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        self.model = Unet(n_channels=4, n_classes=14)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        # Resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def inferrence(self, mode, nclass):
        """inferrence and save images, mode: grey, color"""
        self.model.eval()

        save_dir = '/home/user/disk4T/dataset/SensatUrban/BEV/data_11/pred5697/'
        label_color_map13 = [[255,248,220], [220,220,220], [139, 71, 38], 
                            [238,197,145], [ 70,130,180], [179,238, 58], 
                            [110,139, 61], [105,105,105], [  0,  0,128], 
                            [205, 92, 92], [244,164, 96], [147,112,219], 
                            [255,228,225]]
        label_color_map14 = [[0,0,0], [255,248,220], [220,220,220], [139, 71, 38], 
                            [238,197,145], [ 70,130,180], [179,238, 58], 
                            [110,139, 61], [105,105,105], [  0,  0,128], 
                            [205, 92, 92], [244,164, 96], [147,112,219], 
                            [255,228,225]]

        tbar = tqdm(self.test_loader, desc='\r')
        for j, sample in enumerate(tbar):
            image, fname = sample['image'], sample['fname']
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # colorize prediction
            edge = self.args.base_size
            for i in range(self.args.batch_size):
                if i>pred.shape[0]-1:
                    continue
                if mode=='grey':
                    out = int(pred[i])
                elif mode=='color':
                    if nclass==13:
                        label_color_map = label_color_map13
                    elif nclass==14:
                        label_color_map = label_color_map14
                    # provide two types of color prediction output methods
                    for x in range(edge):
                        for y in range(edge):
                            out[x, y, 2] = label_color_map[int(pred[0, x, y])][0] # R
                            out[x, y, 1] = label_color_map[int(pred[0, x, y])][1] # G
                            out[x, y, 0] = label_color_map[int(pred[0, x, y])][2] # B
                    # fill = lambda x: label_color_map[x]
                    # grey = int(pred[i]).reshape(1,-1)[0]
                    # out = np.array(list(map(fill, grey))).reshape(edge, edge, 3)
                    # out = cv2.cvtColor(cv2.COLOR_RGB2BGR, out)
                cv2.imwrite(os.path.join(save_dir, fname[i]), out)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Unet Training")
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=500,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=500,
                        help='crop image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos','elr'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 4e-5)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                         help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0,1',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 400,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.05,
        }
        args.lr = lrs[args.dataset.lower()] / (2 * len(args.gpu_ids)) * args.batch_size

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.inferrence()

if __name__ == "__main__":
   main()
