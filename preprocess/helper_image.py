'''
Author: your name
Date: 2021-09-07 18:12:53
LastEditTime: 2021-09-08 10:22:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /sensat-tools/helper_image.py
'''

import cv2
import math
import numpy as np
import os
from point_EDA_31 import SensatUrbanEDA as SU
import matplotlib.pyplot as plt

cmap = plt.cm.get_cmap('hsv', 256)
cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

write_root0 = '/home/user/disk4T/dataset/SensatUrban/BEV/data_0'
write_root1 = '/home/user/disk4T/dataset/SensatUrban/BEV/data_1'
write_root11 = '/home/user/disk4T/dataset/SensatUrban/BEV/data_11'
write_root2 = '/home/user/disk4T/dataset/SensatUrban/BEV/data_2'


def project_3d_bev_img3(grid_data, grid_scale, grid_size, label_color_map, imgid=None):
    """project 3d points to 2d bev pixels (BEV Projection)"""
    grid_size_scale = int(grid_size / grid_scale) # size of projection
    num = grid_data.shape[1]
    # project elements
    bev = np.zeros((grid_size_scale, grid_size_scale, 4)) - 1000 # (grid_scale, grid_scale, 1)
    vis = np.zeros((grid_size_scale, grid_size_scale, 3)) - 1000
    cla = np.zeros((grid_size_scale, grid_size_scale)) - 1000

    xs = (grid_data[0] / grid_scale).astype(np.int32)
    ys = (grid_data[1] / grid_scale).astype(np.int32)
    for i in range(num):
        if bev[xs[i], ys[i], 0] < grid_data[2, i]:
            # update points
            bev[xs[i], ys[i], 0] = grid_data[2, i]
            # update RGB:
            bev[xs[i], ys[i], 3] = grid_data[3, i] # R
            bev[xs[i], ys[i], 2] = grid_data[4, i] # G
            bev[xs[i], ys[i], 1] = grid_data[5, i] # B
            # update class (label)
            cla[xs[i], ys[i]] = grid_data[6, i]
    alt = bev[:, :, 0] # altitude(z)
    # alt_df = Sensat.altitude_difference(alt, grid_size_scale-1)
    rgb = bev[:, :, 1:]

    return alt, rgb, cla

def project_3d_bev_img4(grid_data, grid_scale, grid_size, label_color_map, imgid=None):
    """project 3d points to 2d bev pixels (BEV Projection)"""
    grid_size_scale = int(grid_size / grid_scale) # size of projection
    num = grid_data.shape[1]
    # project elements
    bev = np.zeros((grid_size_scale, grid_size_scale, 4)) - 1000 # (grid_scale, grid_scale, 1)
    vis = np.zeros((grid_size_scale, grid_size_scale, 3)) - 1000
    cla = np.zeros((grid_size_scale, grid_size_scale)) - 1000

    xs = (grid_data[0] / grid_scale).astype(np.int32)
    ys = (grid_data[1] / grid_scale).astype(np.int32)
    for i in range(num):
        if bev[xs[i], ys[i], 0] < grid_data[2, i]:
            # update points
            bev[xs[i], ys[i], 0] = grid_data[2, i]
            # update RGB:
            bev[xs[i], ys[i], 3] = grid_data[3, i] # R
            bev[xs[i], ys[i], 2] = grid_data[4, i] # G
            bev[xs[i], ys[i], 1] = grid_data[5, i] # B
            # update class (label)
            cla[xs[i], ys[i]] = grid_data[6, i]
            # update class_vis (label in color)
            vis[xs[i], ys[i], 2] = label_color_map[int(cla[xs[i], ys[i]])][0] # R
            vis[xs[i], ys[i], 1] = label_color_map[int(cla[xs[i], ys[i]])][1] # G
            vis[xs[i], ys[i], 0] = label_color_map[int(cla[xs[i], ys[i]])][2] # B
    alt = bev[:, :, 0] # altitude(z)
    # alt_df = Sensat.altitude_difference(alt, grid_size_scale-1)
    rgb = bev[:, :, 1:]

    return alt, rgb, cla, vis
    # return alt, alt_df, rgb, cla, vis

    # alt = cv2.cvtColor(alt, cv2.COLOR_GRAY2BGR)
    # alt_df = cv2.cvtColor(alt_df, cv2.COLOR_GRAY2BGR)

    # store = 'gen_bev_test'
    # if not os.path.exists(store): os.mkdir(store)
    # cv2.imwrite(store+'/{}_alt.png'.format(imgid), alt)
    # cv2.imwrite(store+'/{}_adf.png'.format(imgid), alt_df)
    # cv2.imwrite(store+'/{}_img.png'.format(imgid), rgb)
    # cv2.imwrite(store+'/{}_cla.png'.format(imgid), cla)
    # cv2.imwrite(store+'/{}_vis.png'.format(imgid), vis)


def max_count_pool(arr):
    val, idx = np.unique(arr, return_counts=True)
    return val[np.argsort(idx)[-1]]
    

def pooling2d(inputMap, poolSize, poolStride, pool_func):
    """Completion by pooling on sliding windows for 2d array
    Args:
        inputMap: input array of the pooling layer
        poolSize: X-size(equivalent to Y-size) of receptive field
        poolStride: the stride size between successive pooling squares
        pool_func: pooling methods: np.max, np.mean, max_count_pool

    Returns:
        outputMap: output array of the pooling layer
    """
    # inputMap sizes
    in_row, in_col = np.shape(inputMap)
    
    # outputMap sizes
    out_row, out_col = int(np.floor(in_row/poolStride)), int(np.floor(in_col/poolStride))
    row_remainder, col_remainder = np.mod(in_row, poolStride), np.mod(in_col, poolStride)
    if row_remainder != 0:
        out_row += 1
    if col_remainder != 0:
        out_col += 1
    outputMap = np.zeros((out_row, out_col))
    
    # padding
    temp_map = np.lib.pad(inputMap, ((0, poolSize-row_remainder), (0, poolSize-col_remainder)), 'edge')
    
    for r_idx in range(poolSize, out_row-poolSize):
        for c_idx in range(poolSize, out_col-poolSize):
            startX = c_idx * poolStride
            startY = r_idx * poolStride
            poolField = temp_map[startY-poolSize:startY+poolSize+1, startX-poolSize:startX+poolSize+1]
            poolOut = pool_func(poolField)
            outputMap[r_idx, c_idx] = poolOut
    
    return outputMap


# completion for items by max pooling
def complete2d(src_map, loops, keep_margin=False):
    """completion for 2d image (repeat for multi-channel objects)"""
    for l in range(loops):
        if keep_margin:
            poolfunc = np.max if l<1 else max_count_pool
        else:
            poolfunc = np.max
        comp_map = pooling2d(src_map, 1, 1, poolfunc) # 3x3 operation
        # comp_map = fastcompletion(src_map)
        invalid_idx = (src_map==-1000) # invalid condition
        # print(np.sum(invalid_idx) / (src_map.shape[0]**2), comp_map.min())
        src_map = comp_map * invalid_idx + src_map * (1 - invalid_idx)
    # edge completion
    src_map = complete_edge(src_map)
    return src_map


# complete images on the edges
def complete_edge(src_map):
    out_map = src_map.copy()
    # edge
    out_map[0, :] = src_map[1, :]
    out_map[-1, :] = src_map[-2, :]
    out_map[:, 0] = src_map[:, 1]
    out_map[:, -1] = src_map[:, -2]
    # corner
    out_map[0, 0] = src_map[1, 1]
    out_map[0, -1] = src_map[1, -2]
    out_map[-1, 0] = src_map[-2, 1]
    out_map[-1, -1] = src_map[-2, -2]
    # merge source map
    invalid_idx = (src_map<-100)
    out_map = out_map * invalid_idx + src_map * (1 - invalid_idx)
    return out_map


# fast-completion for items without pooling
def fastcompletion(inputMap):
    outputMap = np.zeros((500, 500))
    temp_map = np.lib.pad(inputMap, ((0, 1), (0, 1)), 'edge')
    for r_idx in range(1, 499):
        for c_idx in range(1, 499):
            outputMap[r_idx, c_idx] = np.mean(temp_map[r_idx-1:r_idx+2, c_idx-1:c_idx+2])
    return outputMap


# ISP operators for features analysis
def laplacian2d(src, **kwargs):
    return cv2.Laplacian(src, cv2.CV_16S, kwargs['ksize'])


def sobel2d(src, **kwargs):
    x = cv2.Sobel(src, cv2.CV_16S, 1, 0, kwargs['ksize'])
    y = cv2.Sobel(src, cv2.CV_16S, 0, 1, kwargs['ksize'])
    ax = cv2.convertScaleAbs(x)
    ay = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(ax, 1, ay, 1, 0)
    return sobel


def canny2d(src, **kwargs):
    if kwargs['blur']:
        src = cv2.GaussianBlur(src, (3,3), 0)
    canny = cv2.Canny(src, kwargs['lb'], kwargs['hb']) # optimal 5,10
    return canny


def scharr2d(src):
    grad_x = cv2.Scharr(src, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(src, cv2.CV_32F, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    scharr = cv2.addWeighted(gradx, 1, grady, 1, 0)
    return scharr


def map_grey_color(src):
    # normalization
    src = src - src.min() 
    src = src / src.max() * 255
    # re-colorize
    xlen, ylen = src.shape
    out = np.zeros((xlen, ylen, 3))
    for x in range(xlen):
        for y in range(ylen):
            if src[x, y] >= 0:
                out[x, y, :] = cmap[int(src[x, y]), :]
    return out


def altitude_dif_func_v1(src_map, border):
    """calculate the curvature for altitude map from 4 nearest points

    NOTICE: this function can be applied within/without a loop
    
    Args: 
        src_map: the altitude map, expect shape of (H,W)
        border: size of calculation area
        calc_size: size of single operation area
    Returns:
        difference (curvature) map of the input: an array
    """
    src_df = src_map.copy()
    idx_valid = np.where(src_map>-1000) # (?,N)

    for ii in range(len(idx_valid[0])):
        cx = idx_valid[0][ii]
        cy = idx_valid[1][ii]

        if cx > 0 and cx < border-1:
            if cy > 0 and cy < border-1:
                idx_lst = [(cx+1,cy), (cx-1,cy), (cx,cy+1), (cx,cy-1)]
                idx_able = []
                for idx in idx_lst:
                    if idx[0] in idx_valid[0] and idx[1] in idx_valid[1]:
                        idx_able.append(idx)
                df = [src_map[cx, cy]]
                for idx in idx_able:
                    z = src_map[cx, cy]
                    zN = src_map[idx[0], idx[1]]
                    df.append(abs(z-zN))
                src_df[x, y] = np.mean(df)
    return src_df


def altitude_dif_func_v2(src_map, radius=1, valid_key=-1000):
    """calculate the curvature for altitude map from a square with length of 2*radius+1

    NOTICE: this function can ONLY be applied within a loop
    
    Args: 
        src_map: the altitude map, expect shape of (H,W)
        radius: edge length of the calculation square
        valid_key: key indicates points to ignore
    Returns:
        difference (curvature) value of the input data: a value
    """
    invalid_idx = (src_map==valid_key)
    valid_idx = 1 - invalid_idx
    valid_num = np.sum(valid_idx)

    if src_map is not None and valid_num > 0:
        valid_idx[radius, radius] = False
        src_df = np.zeros((2*radius+1, 2*radius+1)) - 1 # avoid lower bound at 0
        nalt = src_map[radius, radius]
        for nx in range(0, 2*radius+1):
            for ny in range(0, 2*radius+1):
                if valid_idx[nx, ny]:
                    src_df[nx, ny] = (src_map[nx, ny] - nalt) / np.linalg.norm([nx-radius, ny-radius])
        src_df = (np.sum(src_df) + src_df.shape[0]**2) / valid_num - 1
    else:
        src_df = 0
    return src_df


if __name__=='__main__':
    # initialization
    Sensat = SU()
    Sensat.root_dir = '/home/user/disk4T/dataset/SensatUrban/'
    Sensat.split = 'train' # 'train', 'test', 'all'
    Sensat.grids_scale = 0.05
    Sensat.grids_size = 25
    Sensat.grids_step = 25

    # load data
    data_dir = os.path.join(Sensat.root_dir, Sensat.split)
    ply_list = sorted(os.listdir(data_dir))
    print("loading {} files".format(len(ply_list)))
    # ply_name = random.choice(ply_list)
    ply_name = ply_list[0]
    ply_path = os.path.join(data_dir, ply_name)
    print("loading file {}".format(ply_path))

    # generate grids
    ply_data = Sensat.load_points(ply_path, reformat=True)
    grids_data = Sensat.grid_generator(ply_data, Sensat.grids_size, Sensat.grids_step, False)
    grid = grids_data.__next__()
    x_idx, y_idx, pts = grid[0], grid[1], grid[2]
    x_idx = math.ceil(x_idx / Sensat.grids_scale)
    y_idx = math.ceil(y_idx / Sensat.grids_scale)
    name_id = str(x_idx)+'_'+str(y_idx)

    # generate key items
    alt, rgb, cla, vis = project_3d_bev_img4(pts.T, Sensat.grids_scale, Sensat.grids_step, name_id)
    
    # completion
    n_loop = 3
    # channel = 1
    alt = complete2d(alt, n_loop)
    cla = complete2d(cla, n_loop)
    # channel = 3
    for c in range(3):
        vis[:, :, c] = complete2d(vis[:, :, c], n_loop) # to be optimize by colorize cla-comp
        rgb[:, :, c] = complete2d(rgb[:, :, c], n_loop)
    
    # edge analysis
    alt_cp = alt.copy()
    alt_cp[alt==-1000] = 0
    alt_canny = canny2d(alt_cp.astype(np.uint8), blur=False, lb=2, hb=5)
    
    # src = alt.copy()
    # src *= (src!=-1000) # remove dirty values
    # src = src.astype(np.uint8)

    # out = Laplacian2d(src, ksize=3) * 255
    # cv2.imwrite('lp_alt.png', out)

    # out = Scharr2d(src) * 255
    # cv2.imwrite('scharr_alt.png', out)

    # out = Sobel2d(src, ksize=3) * 255
    # cv2.imwrite('sobel_alt.png', out)

    # out = Canny2d(src, blur=True, lb=2, hb=5)
    # cv2.imwrite('canny_alt.png', out)


    