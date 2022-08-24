from logging.config import valid_ident
import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

from os.path import join
import pickle

# some util functions
def get_file_list(dataset_path, seq_list):
        data_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            new_data = [(seq_id, f[:-4]) for f in np.sort(os.listdir(pc_path))]
            data_list.extend(new_data)

        return data_list

def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]



class ConfigSemanticKITTI:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 4096 * 10  # Number of input points
    num_classes = 19  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 8  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    # sub_sampling_ratio = [1, 1, 1, 1]
    d_out = [16, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

cfg = ConfigSemanticKITTI()


class SemanticKITTI(Dataset):
    def __init__(self, mode, data_list=None, seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']):
        self.name = 'SemanticKITTI'
        self.dataset_path = '/data/gpfs/projects/punim1650/Chaoyinc/data/SemanticKitti/sequences_0.06'
        self.num_classes = cfg.num_classes
        self.ignored_labels = np.sort([0])
        self.mode = mode
        if data_list is None:
            if mode == 'validation':
                seq_list = ['08']
            self.data_list = get_file_list(self.dataset_path, seq_list)
        else:
            self.data_list = data_list
        self.data_list = sorted(self.data_list)
    def __getitem__(self, index):
        cloud_ind = index
        pc_path = self.data_list[cloud_ind]
        pc, tree, labels = self.get_data(pc_path)
        
        # crop a small point cloud
        pick_idx = np.random.choice(len(pc), 1)
        selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)
        # selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(index, self.data_list)

        return selected_pc, selected_labels

    def __len__(self):
        return len(self.data_list)


    def get_class_weights(self):
        # pre-calculate the number of points in each category
        num_per_class = [0 for _ in range(self.num_classes)]
        for file_path in tqdm(self.data_list, total=len(self.data_list)):
            label_path = join(self.dataset_path, file_path[0], 'labels', file_path[1] + '.npy')
            label = np.load(label_path).reshape(-1)
            inds, counts = np.unique(label, return_counts=True)
            for i, c in zip(inds, counts):
                if i == 0:      # 0 : unlabeled
                    continue
                else:
                    num_per_class[i-1] += c
        '''
        if dataset_name == 'S3DIS':
            num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                        650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
        elif dataset_name == 'Semantic3D':
            num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                        dtype=np.int32)
        elif dataset_name == 'SemanticKITTI':
            num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                        240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                        9833174, 129609852, 4506626, 1168181])
        '''
        num_per_class = np.array(num_per_class)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        # return np.expand_dims(ce_label_weight, axis=1)
        return ce_label_weight

    def get_data(self, file_path):
        seq_id = file_path[0]
        frame_id = file_path[1]
        kd_tree_path = join(self.dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
        # read pkl with search tree
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)
        # load labels
        label_path = join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
        labels = np.squeeze(np.load(label_path))
        return points, search_tree, labels
    @staticmethod
    def crop_pc(points, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
        select_idx = shuffle_idx(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx



if __name__ == '__main__':
    data_root = '/mnt/chaoyinc/dataset/semantickitti/sequences'


    point_data = SemanticKITTI(mode='train')
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()