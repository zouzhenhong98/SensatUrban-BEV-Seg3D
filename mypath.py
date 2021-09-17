class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/user/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'cityscapes':
            return '/home/user/disk4T/dataset/SensatUrban/BEV/data_11/'
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
