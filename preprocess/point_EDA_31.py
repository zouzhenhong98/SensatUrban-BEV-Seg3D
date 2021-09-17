# define a class for SensatUrban Points EDA

"""
    TODO:
     - [x] multiprocessing -> ~10 Process
     - [x] complexity -> optimize
     - [x] finally reach 800x acceleration in point clouds statistics
     - [x] add BEV projection function
     - [x] add class-overlay (pillar) analysis function
     - [x] add projection cover rate function
     - [x] add Bev2Point mapping function
     - [x] reconstruct the code to be well-organized
     - [x] overall point clouds mIoU calculation
     - [x] seg2points mapping
     - [x] overlapped sampling
     - [x] offline mapping: map img to pts
     - [x] pixel nn completion: image, edge
     - [x] Multi-scale Sampling (grid=25,50)
     - [ ] split the code (experiment part)

     - overlay ratio count: for (0.04, 25, 25) setting
        class: over 97.7%
        miou: over 93.7%
     - details for completion
        using 0.05 may results better in completion, 
        makinge more consisting annotation, and multi-scale referrence point
"""

from typing import Iterator, Tuple
import cv2
import math
import numpy as np
import os
import random

from helper_ply import read_ply
from helper_metrics import batch_mIoU, mIoU
import helper_image as hi
from tqdm import tqdm


class SensatUrbanEDA(object):
    """Exploraing Data Analysis (EDA) code for SensatUrban dataset

    annotation categories:    
        - Ground: including impervious surfaces, grass, terrain
        - Vegetation: including trees, shrubs, hedges, bushes
        - Building: including commercial / residential buildings
        - Wall: including fence, highway barriers, walls
        - Bridge: road bridges
        - Parking: parking lots
        - Rail: railroad tracks
        - Traffic Road: including main streets, highways
        - Street Furniture: including benches, poles, lights
        - Car: including cars, trucks, HGVs
        - Footpath: including walkway, alley
        - Bike: bikes / bicyclists
        - Water: rivers / water canals

    points coordinate range:
        z_min in [0.0, 37.78]
        z_max in [18.33, 105.81]
        cambridge
            x_max = 2152.9375
            x_min = 0
            y_max = 2136.9062
            y_min = 0
            S_eta -> 4600625
        briminghan
            x_max = 1273.6875
            x_min = 0
            y_max = 1306.75
            y_min = 0
            S_eta -> 1664391
        S_total_eta -> 6265016
        total points: 2702883679

    target image to generate:
        size: 512*512
        num: 20000
        valid pixel ratio: 50%
        valid pixel count: 125000
    
    target mapped points:
        lower bound > 239/m2
        mapped ratio > 92.49% (overlayed < 7.5%)

    Attributes:
        altitude_difference:
        grid_generator:
        grid_point_overlay:
        grid_class_overlay:
        load_points:
        map_offline_img2pts:
        miou_ply:
        miou_gird:
        project_3d_bev_zc:
        project_3d_bev_img:

    Analyse tools:
        evaluate:
        evaluate_batch:
        single_ply_analysis:
        batch_ply_analysis:
            exp_class_overlay_count:
            exp_gen_bev_projection:
            exp_point_overlay_count:
    """
    def __init__(self):
        self.root_dir = '/home/user/disk4T/dataset/SensatUrban/'
        self.split = 'all'
        self.grids_scale = 0.05
        self.grids_size = 25
        self.grids_step = 25
        self.label_color_map = [[255,248,220], [220,220,220], [139, 71, 38], 
                                [238,197,145], [ 70,130,180], [179,238, 58], 
                                [110,139, 61], [105,105,105], [  0,  0,128], 
                                [205, 92, 92], [244,164, 96], [147,112,219], 
                                [255,228,225]]
        self.label_color_map_new = [[0,0,0], [255,248,220], [220,220,220], [139, 71, 38], 
                                [238,197,145], [ 70,130,180], [179,238, 58], 
                                [110,139, 61], [105,105,105], [  0,  0,128], 
                                [205, 92, 92], [244,164, 96], [147,112,219], 
                                [255,228,225]]
        super().__init__()

    # ===========================================================================
    #
    #                                 IO functions
    #
    # ===========================================================================

    def load_points(self, ply_path: str, reformat: bool = False) -> np.array:
        """load points from ply to array

        Args:
            ply_path:
            reformat: whether to transfer np.void to np.array
        Returns: 
            points in np.array
        Raise: 
            Exception error
        """
        if os.path.exists(ply_path):
            _ply_data = read_ply(ply_path)
            if reformat:
                _ply_data = read_ply(ply_path)
                x = _ply_data["x"]
                y = _ply_data["y"]
                z = _ply_data["z"]
                r = _ply_data["red"]
                g = _ply_data["green"]
                b = _ply_data["blue"]
                c = _ply_data["class"]
                _ply_data = np.vstack((x,y,z,r,g,b,c)).T # shape of (N,7)
                del x,y,z,r,g,b,c
            return _ply_data
        else:
            raise Exception("{} does not exists".format(ply_path))

    def grid_generator(self, ply_data: np.array, grids_size: float, grids_step: int, margin: bool) -> Iterator[Tuple[int, int, np.array]]:
        """generate grid points iterally

        Args:
            ply_data:
            grids_size:
            grids_step:
        Returns:
            tuple(x_idx, y_idx, points) in sliding grid iterally
        Raises:
            Exception when get an empty ply_data
        """
        if ply_data is None:
            raise Exception("get an empty ply input, expected an array with shape of (N,7)")
        margin = 1 if margin else 0
        # sort from x-axis
        idx_sort_x = np.argsort(ply_data[:, 0])
        ply_sort_x = ply_data[idx_sort_x, :]
        x_max = math.ceil(ply_sort_x[-1, 0])
        x_min = math.floor(ply_sort_x[0, 0])

        for idx_start_x in tqdm(range(x_min, x_max+margin, grids_step)):
            idx_grid_x = np.where(ply_sort_x[:, 0] < idx_start_x + grids_size)[0]
            grid_sort_x = ply_sort_x[idx_grid_x, :] # 1-sampling for x-grid
            if grid_sort_x is None or grid_sort_x.size == 0: continue

            # sort from y-axis
            idx_sort_y = np.argsort(grid_sort_x[:, 1])
            grid_sort_x = grid_sort_x[idx_sort_y, :]
            y_max = math.ceil(grid_sort_x[-1, 1])
            y_min = math.floor(grid_sort_x[0, 1])

            for idx_start_y in range(y_min, y_max+margin, grids_step):
                idx_grid_xy = np.where(grid_sort_x[:, 1] < idx_start_y + grids_size)[0]
                grid_sort_xy = grid_sort_x[idx_grid_xy, :].copy() # 2-sampling for xy-grid
                if grid_sort_xy is None or grid_sort_xy.size==0:
                    print("empty grid at ({},{}), ({},{})".format(idx_start_x, idx_start_y, \
                        idx_start_x + grids_step, idx_start_y + grids_step))
                    continue
                else:
                    grid_sort_xy[:, :2] -= np.array([idx_start_x, idx_start_y]).T
                    yield (idx_start_x, idx_start_y, grid_sort_xy) # shape of (N_grid,7)

                grid_sort_x = grid_sort_x[idx_grid_xy[-1]+1:, :]
            ply_sort_x = ply_sort_x[idx_grid_x[-1]+1:, :]

    # ===========================================================================
    #
    #                           Pointcloud Analyse functions
    #
    # ===========================================================================

    def grid_point_overlay(self, grid_data: np.array, grid_scale: float, grid_size: float):
        """overlay ratio of the single grid

        Args:
            grid_data: array with shape of (?,N)
            grid_scale: value <= 0.05 (meter): float
            grid_size: 
        Returns:
            overlay ratio, total num of points
        """
        grid_size_scale = int(grid_size / grid_scale) # size of flag
        num = grid_data.shape[1] # num of pts
        bev = np.zeros((grid_size_scale, grid_size_scale)) # (grid_scale, grid_scale)
        bev -= 1000 # ensure smaller than z
        xs = (grid_data[0] / grid_scale).astype(np.int32)
        ys = (grid_data[1] / grid_scale).astype(np.int32)
        overlay = 0 # points overlayed on z-axis
        for i in range(num):
            if bev[xs[i], ys[i]]==-1000:
                bev[xs[i], ys[i]] = 1
            else:
                overlay += 1

        return overlay, num
    
    def grid_class_overlay(self, grid_data: np.array, grid_scale: float, grid_size: float):
        """calculate the points overlay ratio on class

        Args: grid_data: points array with shape of (7,N)
        Returns: Overall Accuracy, total point number
        """
        bev, xs, ys, num = self.project_3d_bev_zc(grid_data, grid_scale, grid_size)
        acc = num
        for i in range(num):
            if grid_data[6, i]!=bev[xs[i], ys[i], 1]: acc -= 1
        OAcc = acc / num

        return OAcc, num
    
    def project_3d_bev_zc(self, grid_data: np.array, grid_scale: float, grid_size: float):
        """fetch top-z bev projection, and map bev pixels to 3d points

        Args:
            grid_data: array <- points with shape of (?,N)
            grid_scale: float
            grid_size: float
        Returns:
            bev: bev-view projection with z and class
            xs,ys: scaled int for x and y
            num: total number of 3d points
        """
        grid_size_scale = int(grid_size / grid_scale) # size of projection
        num = grid_data.shape[1] # num of pts
        bev = np.zeros((grid_size_scale, grid_size_scale, 2)) # <- 2 (z,c)
        bev[:, :, 0] -= 1000 # ensure smaller than z
        xs = (grid_data[0] / grid_scale).astype(np.int32)
        ys = (grid_data[1] / grid_scale).astype(np.int32)
        for i in range(num):
            if bev[xs[i], ys[i], 0] < grid_data[2, i]:
                bev[xs[i], ys[i], 0] = grid_data[2, i] # update points
                bev[xs[i], ys[i], 1] = grid_data[6, i] # update class

        return bev, xs, ys, num

    # ===========================================================================
    #
    #                               Experiment tools
    #
    # ===========================================================================

    def exp_point_overlay_count(self, grids_data: Iterator[Tuple[int, int, np.array]]) -> list:
        """calculate the points overlay ratio on coordinate"""
        content = [('x','y','num','Overlay')]
        for grid in grids_data:
            x_idx, y_idx, pts = grid
            Overlay, num = self.grid_point_overlay(pts.T, self.grids_scale, self.grids_size)
            content.append((x_idx, y_idx, num, Overlay))
        return content
    
    def exp_class_overlay_count(self, grids_data: Iterator[Tuple[int, int, np.array]]) -> list:
        """calculate the points overlay ratio on class"""
        content = [('x','y','num','OAcc')]
        for grid in grids_data:
            x_idx, y_idx, pts = grid
            OAcc, num = self.grid_class_overlay(pts.T, self.grids_scale, self.grids_size)
            content.append((x_idx, y_idx, num, OAcc))
        return content

    def exp_gen_bev_projection(self, grids_data: Iterator[Tuple[int, int, np.array]], ply_name) -> list:
        """generate BEV projection for altitude(z), alt_canny, RGB, class, class_vis
        Notice: we add invalid point as unlabeled annotation
        """
        for grid in grids_data:
            x_idx, y_idx, pts = grid
            # self.project_3d_bev_img(pts.T, self.grids_scale, self.grids_step, x_idx+'_'+y_idx)
            imgid = ply_name[:-4]+'_'+str(x_idx)+'_'+str(y_idx) # city_block_id_x_y_type.png
            alt, rgb, cla = hi.project_3d_bev_img3(pts.T, self.grids_scale, self.grids_step, self.label_color_map, imgid)
            
            # completion
            n_loop = 3
            alt = hi.complete2d(alt, n_loop)
            cla = hi.complete2d(cla, n_loop)
            for c in range(3): rgb[:, :, c] = hi.complete2d(rgb[:, :, c], n_loop)

            # relabel and make vis
            cla[cla==-1000] = -1
            cla = cla + 1
            vis = np.zeros_like(rgb) - 1000
            for x in range(rgb.shape[0]):
                for y in range(rgb.shape[1]):
                    vis[x, y, 2] = self.label_color_map_new[int(cla[x, y])][0] # R
                    vis[x, y, 1] = self.label_color_map_new[int(cla[x, y])][1] # G
                    vis[x, y, 0] = self.label_color_map_new[int(cla[x, y])][2] # B
            
            # edge analysis
            alt_cp = alt.copy()
            alt_cp[alt==-1000] = 0
            alt_canny = hi.canny2d(alt_cp.astype(np.uint8), blur=False, lb=2, hb=5)
            
            store = hi.write_root2
            cv2.imwrite(store+'/alt/{}_alt50.png'.format(imgid), alt)
            cv2.imwrite(store+'/canny/{}_alt50.png'.format(imgid), alt_canny)
            cv2.imwrite(store+'/rgb/{}_rgb50.png'.format(imgid), rgb)
            cv2.imwrite(store+'/cls/{}_cls50.png'.format(imgid), cla)
            cv2.imwrite(store+'/vis/{}_vis50.png'.format(imgid), vis)
        return None

    def single_ply_analysis(self, func):
        """experiment with func for single ply file
        
        Args: func: experiment function, class method as default
        """
        data_dir = os.path.join(self.root_dir, self.split)
        ply_list = sorted(os.listdir(data_dir))
        print("loading {} files".format(len(ply_list)))

        ply_name = random.choice(ply_list)
        ply_path = os.path.join(data_dir, ply_name)
        print("loading file {}".format(ply_path))

        ply_data = self.load_points(ply_path, reformat=True)
        grids_data = self.grid_generator(ply_data, self.grids_size, self.grids_step, False)
        content = func(grids_data) # main function for experiment
        if content is not None:
            with open('log_{}_{}_scale={}_size={}_step={}.txt'.format(\
                func.__name__, ply_name, self.grids_scale, self.grids_size, self.grids_step), 'w') as file:
                for cont in content: file.write(str(cont)+'\n')

    def batch_ply_analysis(self, func):
        """experiment with func for all ply files

        Args: func: experiment function, class method as default
        """
        data_dir = os.path.join(self.root_dir, self.split)
        ply_list = sorted(os.listdir(data_dir))
        print("loading {} files".format(len(ply_list)))

        for ply_name in ply_list:
            ply_path = os.path.join(data_dir, ply_name)
            print("loading file {}".format(ply_path))

            ply_data = self.load_points(ply_path, reformat=True)
            grids = self.grid_generator(ply_data, self.grids_size, self.grids_step, False)
            content = func(grids, ply_name) # main function for experiment
            # content = func(grids)
            if content is not None:
                with open('log_{}_{}_scale={}_size={}_step={}.txt'.format(\
                    func.__name__, ply_name, self.grids_scale, self.grids_size, self.grids_step), 'w') as file:
                    for cont in content: file.write(str(cont)+'\n')

    def map_offline_img2pts(self, ply_data, ply_size, ply_name):
        """write bev projection to img, reload img and map with raw ply"""
        # initialization
        grids_data = self.grid_generator(ply_data, self.grids_size, self.grids_step, False)
        ply_size_scale = int(ply_size / self.grids_scale)
        plymap = np.zeros((ply_size_scale, ply_size_scale))
        store = 'gen_bev'
        if not os.path.exists(store): os.mkdir(store)

        # generate offline images
        for grid in grids_data:
            x_idx, y_idx, pts = grid
            x_idx = math.ceil(x_idx / self.grids_scale)
            y_idx = math.ceil(y_idx / self.grids_scale)
            bev, xs, ys, num = self.project_3d_bev_zc(pts.T, self.grids_scale, self.grids_size)
            imgid = ply_name[:-4]+'_'+str(x_idx)+'_'+str(y_idx) # city_block_id_x_y_type.png
            cv2.imwrite(store+'/{}_cls.png'.format(imgid), bev[:, :, 1])

        # reload offline images
        files = np.array(os.listdir(store))
        split_name = lambda x: ''.join(x.split('_')[:3])
        files_name = list(map(split_name, files))
        file_idx = np.where(np.array(files_name)==split_name(ply_name)[:-4]) # select corresponding images
        files_target = files[file_idx]
        print('retrieve {} images'.format(files_target.shape[0]))
        for fname in files_target:
            _city, _block, _bid, _x, _y, _type = fname.split('_')
            fpath = os.path.join(store, fname)
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            # remap fullmap
            xl, yl = map(int, (_x, _y))
            xh = xl + img.shape[0] # if xl + img.shape[0] < plymap.shape[0] else plymap.shape[0]
            yh = yl + img.shape[1] # if yl + img.shape[1] < plymap.shape[1] else plymap.shape[1]
            plymap[xl:xh, yl:yh] = img # [:xh-xl, :yh-yl]

        # remap raw pts
        xlst = (ply_data[:, 0] / self.grids_scale).astype(np.int32)
        ylst = (ply_data[:, 1] / self.grids_scale).astype(np.int32)
        pts_num = xlst.shape[0]
        pred = np.zeros(pts_num)
        for i in range(pts_num): pred[i] = plymap[xlst[i], ylst[i]]

        OAcc = np.sum(ply_data[:, 6]==pred)
        print('OAcc: {:3.2%}'.format(OAcc / pts_num))

        # assert False

        return pred

    def eval_offline_img2pts(self, ply_data, ply_size, ply_name, mini):
        """write bev projection to img, reload img and map with raw ply"""
        # initialization
        ply_size_scale = int(ply_size / self.grids_scale)
        plymap = np.zeros((ply_size_scale, ply_size_scale))
        store = '/home/user/disk4T/dataset/SensatUrban/BEV/data_11/pred5990/'
        # store = 'gen_bev'
        if not os.path.exists(store):
            raise ValueError("empty offline prediction path at {}".format(store))

        # reload offline images
        files = np.array(os.listdir(store))
        split_name = lambda x: ''.join(x.split('_')[:3])
        files_name = list(map(split_name, files))
        file_idx = np.where(np.array(files_name)==split_name(ply_name)[:-4]) # select corresponding images
        files_target = files[file_idx]
        print('retrieve {} images'.format(files_target.shape[0]))
        for fname in files_target:
            _city, _block, _bid, _x, _y, _type = fname.split('_')
            fpath = os.path.join(store, fname)
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            img = img - 1
            img[img<0] = 0
            # remap fullmap
            xl = math.ceil((int(_x) - int(mini[0])) / self.grids_scale)
            yl = math.ceil((int(_y) - int(mini[1])) / self.grids_scale)
            xh = xl + img.shape[0] # if xl + img.shape[0] < plymap.shape[0] else plymap.shape[0]
            yh = yl + img.shape[1] # if yl + img.shape[1] < plymap.shape[1] else plymap.shape[1]
            # print(_x, _y, xl, xh, yl, yh, ply_size_scale)
            plymap[xl:xh, yl:yh] = img # [:xh-xl, :yh-yl]

        # remap raw pts
        xlst = ((ply_data[:, 0] - int(mini[0])) / self.grids_scale).astype(np.int32)
        ylst = ((ply_data[:, 1] - int(mini[1])) / self.grids_scale).astype(np.int32)
        pts_num = xlst.shape[0]
        pred = np.zeros(pts_num)
        for i in range(pts_num): pred[i] = plymap[xlst[i], ylst[i]]

        OAcc = np.sum(ply_data[:, 6]==pred)
        print('OAcc: {:3.2%}'.format(OAcc / pts_num))

        # assert False

        return pred, OAcc, pts_num


    def miou_ply(self, ply_data, grid_size):
        """calculate the mIoU for single ply file in one time"""
        bev, xs, ys, num = self.project_3d_bev_zc(ply_data.T, self.grids_scale, grid_size)
        pred = ply_data[:, 6].copy()
        for i in range(num): pred[i] = bev[xs[i], ys[i], 1]
        return pred

    def miou_grid(self, ply_data, grid_size):
        """calculate the mIoU for single ply file in iterable grids"""
        # generate empty map for whole ply
        fullmap_size_scale = int(grid_size / self.grids_scale)
        fullmap = np.zeros((fullmap_size_scale, fullmap_size_scale))

        # use grid map to fill the whole map
        grids_data = self.grid_generator(ply_data, self.grids_size, self.grids_step, True)
        for grid in grids_data:
            x_idx, y_idx, pts = grid
            x_idx = math.ceil(x_idx / self.grids_scale)
            y_idx = math.ceil(y_idx / self.grids_scale)
            bev, xs, ys, num = self.project_3d_bev_zc(pts.T, self.grids_scale, self.grids_size)
            for i in range(num): fullmap[xs[i]+x_idx, ys[i]+y_idx] = bev[xs[i], ys[i], 1]

        # remap the whole map with pts
        pred = ply_data[:, 6].copy() # (N,7)
        pts_num = ply_data.shape[0]
        xlst = (ply_data[:, 0] / self.grids_scale).astype(np.int32)
        ylst = (ply_data[:, 1] / self.grids_scale).astype(np.int32)
        for i in range(pts_num): pred[i] = fullmap[xlst[i], ylst[i]]
        return pred

    def evaluate(self, mode, func):
        """calculate the mIoU for ply file(s)
        
        Args:
            mode: 
                eval: eval mIoU index with different project methods
                eval_batch: generate grid data for batch file evaluation
                offline: generate offline imgs and reload for evaluation
            func: project functions for mIoU 
        """
        data_dir = os.path.join(self.root_dir, self.split)
        ply_list = sorted(os.listdir(data_dir))
        print("loading {} files".format(len(ply_list)))

        for ply_name in tqdm(ply_list):
            ply_path = os.path.join(data_dir, ply_name)
            print("loading file {}".format(ply_path))
            ply_data = self.load_points(ply_path, reformat=True)

            # uniform coordinates
            x_min = ply_data[:, 0].min()
            y_min = ply_data[:, 1].min()
            ply_data[:, 0] -= x_min
            ply_data[:, 1] -= y_min
            x_max = ply_data[:, 0].max()
            y_max = ply_data[:, 1].max()
            ply_size = math.ceil(max(x_max, y_max) + self.grids_size)
            print(ply_size)

            if mode=='eval':
                pred = func(ply_data, ply_size) # call project function
                print('start evaluation')
                miou, iou_list, gt_list = mIoU(pred, ply_data[:, 6])
                print("mIoU: {:3.2%}".format(miou))
                print("iou", np.round(iou_list, 4))
                print("GT pts num", gt_list)
            # elif mode=='eval_batch':
            #     pred = func(ply_data, ply_size)
            #     yield (pred, ply_data[:, 6])
            elif mode=='offline':
                pred = func(ply_data, ply_size, ply_name)
                print('start evaluation')
                miou, iou_list, gt_list = mIoU(pred, ply_data[:, 6])
                print("mIoU: {:3.2%}".format(miou))
                print("iou", np.round(iou_list, 4))
                print("GT pts num", gt_list)

    def evaluate_nn(self, func):
        """calculate the mIoU for ply file(s)
        
        Args:
            mode: 
                eval: eval mIoU index with different project methods
                eval_batch: generate grid data for batch file evaluation
                offline: generate offline imgs and reload for evaluation
            func: project functions for mIoU 
        """
        data_dir = os.path.join(self.root_dir, self.split)
        ply_list = sorted(os.listdir(data_dir))
        print("loading {} files".format(len(ply_list)))

        for ply_name in tqdm(ply_list):
            ply_path = os.path.join(data_dir, ply_name)
            print("loading file {}".format(ply_path))
            ply_data = self.load_points(ply_path, reformat=True)

            # uniform coordinates
            x_min = ply_data[:, 0].min()
            y_min = ply_data[:, 1].min()
            x_max = ply_data[:, 0].max()
            y_max = ply_data[:, 1].max()
            ply_size = math.ceil(max(x_max-x_min, y_max-y_min) + self.grids_size)
            pred = func(ply_data, ply_size, ply_name, (x_min, y_min))
            print('start evaluation')
            miou, iou_list, gt_list = mIoU(pred, ply_data[:, 6])
            print("mIoU: {:3.2%}".format(miou))
            print("iou", np.round(iou_list, 4))
            print("GT pts num", gt_list)
    
    def evaluate_batch_nn(self, func):
        """calculate the mIoU for ply file(s)
        
        Args:
            mode: 
                eval: eval mIoU index with different project methods
                eval_batch: generate grid data for batch file evaluation
                offline: generate offline imgs and reload for evaluation
            func: project functions for mIoU 
        """
        data_dir = os.path.join(self.root_dir, self.split)
        ply_list = sorted(os.listdir(data_dir))
        print("loading {} files".format(len(ply_list)))

        for ply_name in tqdm(ply_list):
            ply_path = os.path.join(data_dir, ply_name)
            print("loading file {}".format(ply_path))
            ply_data = self.load_points(ply_path, reformat=True)

            # uniform coordinates
            x_min = ply_data[:, 0].min()
            y_min = ply_data[:, 1].min()
            x_max = ply_data[:, 0].max()
            y_max = ply_data[:, 1].max()
            ply_size = math.ceil(max(x_max-x_min, y_max-y_min) + self.grids_size)
            pred = func(ply_data, ply_size, ply_name, (x_min, y_min))
            yield (pred, ply_data[:, 6])
            print('start evaluation')
            # miou, iou_list, gt_list = mIoU(pred, ply_data[:, 6])
            # print("mIoU: {:3.2%}".format(miou))
            # print("iou", np.round(iou_list, 4))
            # print("GT pts num", gt_list)

    def evaluate_batch(self, batch_iterator: Iterator):
        """calculate the mIoU for multiple ply files"""
        miou, iou_list, gt_list, oacc, macc = batch_mIoU(batch_iterator)
        print("OA: {:3.2%}".format(oacc))
        print("mAcc: ", np.round(macc, 4))
        print("overall mIoU: {:3.2%}".format(miou))
        print("iou", np.round(iou_list, 4))
        print("GT pts num",gt_list)


if __name__=='__main__':
    Sensat = SensatUrbanEDA()
    Sensat.root_dir = '/home/user/disk4T/dataset/SensatUrban/'
    Sensat.split = 'train' # 'train', 'test', 'all'
    Sensat.grids_scale = 0.05
    Sensat.grids_size = 25
    Sensat.grids_step = 25
    # Sensat.evaluate('offline', Sensat.map_offline_img2pts)
    # Sensat.evaluate_nn(Sensat.eval_offline_img2pts)
    Sensat.evaluate_batch(Sensat.evaluate_batch_nn(Sensat.eval_offline_img2pts))
    # Sensat.evaluate_batch(Sensat.evaluate('offline', Sensat.eval_offline_img2pts))
    # Sensat.evaluate('eval', Sensat.miou_grid)
    # Sensat.evaluate_batch(Sensat.evaluate('eval_batch', Sensat.miou_grid))
    # Sensat.single_ply_analysis(Sensat.exp_class_overlay_count) # single file demo
    # Sensat.batch_ply_analysis()

    # Sensat.batch_ply_analysis(Sensat.exp_gen_bev_projection) # generate data
