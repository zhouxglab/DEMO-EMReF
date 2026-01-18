# Copyright (C) 2024  Hong Cao, Jiahua He, Tao Li, Sheng-You Huang and Huazhong University of Science and Technology

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import mrcfile
import numpy as np
from math import ceil

from model.interp3d import interp3d

# ===================== 工具函数 =====================
def hann1d(n):
    if n <= 1: return np.ones((n,), dtype=np.float32)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))
    return np.maximum(w.astype(np.float32), 1e-3)

def center_window_1d(n, margin):
    if margin <= 0: return np.ones((n,), dtype=np.float32)
    r = np.ones(n, dtype=np.float32)
    m = min(margin, n//2)
    if m > 0:
        x = np.linspace(0, np.pi/2, m, dtype=np.float32)
        e = np.sin(x)**2
        r[:m]  = e
        r[-m:] = e[::-1]
    return r

def fusion_window_3d(box_size, valid_margin):
    h = hann1d(box_size)
    c = center_window_1d(box_size, valid_margin)
    w1d = h * c
    W = w1d[:,None,None] * w1d[None,:,None] * w1d[None,None,:]
    return W.astype(np.float32)


def split_map_into_overlapped_chunks(map, box_size, stride, dtype=np.float32, padding=0.0):
    assert stride <= box_size
    map_shape = np.shape(map)
    padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), padding, dtype=dtype)
    padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map
    chunk_list = list()
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while (cur_z + stride < map_shape[2] + box_size):
        next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        cur_x += stride
        if (cur_x + stride >= map_shape[0] + box_size):
            cur_y += stride
            cur_x = start_point # Reset X
            if (cur_y + stride  >= map_shape[1] + box_size):
                cur_z += stride
                cur_y = start_point # Reset Y
                cur_x = start_point # Reset X
        chunk_list.append(next_chunk)
    n_chunks = len(chunk_list)
    ncx, ncy, ncz = [ceil(map_shape[i] / stride) for i in range(3)]
    assert(n_chunks == ncx * ncy * ncz)
    chunks = np.asarray(chunk_list, dtype=dtype)
    return chunks, ncx, ncy, ncz

# get a map from all of the overlapped chunks
def get_map_from_overlapped_chunks(chunks, ncx, ncy, ncz, box_size, stride, nxyz, dtype=np.float32):
    map = np.zeros(((ncx - 1) * stride + box_size, \
                    (ncy - 1) * stride + box_size, \
                    (ncz - 1) * stride + box_size), dtype=dtype)
    denominator = np.zeros(((ncx - 1) * stride + box_size, \
                            (ncy - 1) * stride + box_size, \
                            (ncz - 1) * stride + box_size), dtype=dtype) # should clip to 1
    i = 0
    for z_steps in range(ncz):
        for y_steps in range(ncy):
            for x_steps in range(ncx):
                map[x_steps * stride : x_steps * stride + box_size,
                    y_steps * stride : y_steps * stride + box_size,
                    z_steps * stride : z_steps * stride + box_size] += chunks[i]
                denominator[x_steps * stride : x_steps * stride + box_size,
                            y_steps * stride : y_steps * stride + box_size,
                            z_steps * stride : z_steps * stride + box_size] += 1
                    
                i += 1
    return (map / denominator.clip(min=1))[stride : nxyz[2] + stride, stride : nxyz[1] + stride, stride : nxyz[0] + stride]

# ===================== 改写后的拼回函数 =====================
def get_map_from_overlapped_chunks_weighted(chunks, ncx, ncy, ncz, box_size, stride, nxyz, valid_margin=0, dtype=np.float32):
    """
    将切块拼回整图，使用 Hann×Center 加权平均
    chunks: numpy array, shape (N_chunks, box_size, box_size, box_size)
    ncx/ncy/ncz: 切块数
    box_size: 每个切块大小
    stride: 步长
    nxyz: 原图尺寸 (X,Y,Z)
    valid_margin: Center 窗 margin
    """
    W = fusion_window_3d(box_size, valid_margin)

    map_out = np.zeros(((ncx - 1) * stride + box_size, 
                        (ncy - 1) * stride + box_size, 
                        (ncz - 1) * stride + box_size), dtype=dtype)
    denominator = np.zeros_like(map_out, dtype=dtype)

    i = 0
    for z_steps in range(ncz):
        for y_steps in range(ncy):
            for x_steps in range(ncx):
                map_out[x_steps * stride : x_steps * stride + box_size,
                        y_steps * stride : y_steps * stride + box_size,
                        z_steps * stride : z_steps * stride + box_size] += chunks[i] * W
                denominator[x_steps * stride : x_steps * stride + box_size,
                            y_steps * stride : y_steps * stride + box_size,
                            z_steps * stride : z_steps * stride + box_size] += W
                i += 1
    # 防止除以0
    denominator[denominator==0] = 1.0

    # 裁回原图尺寸
    return map_out / denominator, \
           map_out[stride : nxyz[2] + stride,
                   stride : nxyz[1] + stride,
                   stride : nxyz[0] + stride]

def pad_map(map, box_size, dtype=np.float32, padding=0.0):
    map_shape = np.shape(map)
    padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), padding, dtype=dtype)
    padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map
    return padded_map

# generator version
def chunk_generator(padded_map, maximum, box_size, stride):
    assert stride <= box_size
    padded_map_shape = np.shape(padded_map)
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while (cur_z + stride < padded_map_shape[2] - box_size):
        next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        cur_x0, cur_y0, cur_z0 = cur_x, cur_y, cur_z
        cur_x += stride
        if (cur_x + stride >= padded_map_shape[0] - box_size):
            cur_y += stride
            cur_x = start_point # Reset X
            if (cur_y + stride  >= padded_map_shape[1] - box_size):
                cur_z += stride
                cur_y = start_point # Reset Y
                cur_x = start_point # Reset X

        if next_chunk.max() <= 0.0:
            continue
        else:
            yield cur_x0, cur_y0, cur_z0, next_chunk.clip(min=0.0, max=maximum) / maximum *1

def chunk_generator100(padded_map, maximum, box_size, stride):
    assert stride <= box_size
    padded_map_shape = np.shape(padded_map)
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while (cur_z + stride < padded_map_shape[2] - box_size):
        next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        cur_x0, cur_y0, cur_z0 = cur_x, cur_y, cur_z
        cur_x += stride
        if (cur_x + stride >= padded_map_shape[0] - box_size):
            cur_y += stride
            cur_x = start_point # Reset X
            if (cur_y + stride  >= padded_map_shape[1] - box_size):
                cur_z += stride
                cur_y = start_point # Reset Y
                cur_x = start_point # Reset X

        if next_chunk.max() <= 0.0:
            continue
        else:
            yield cur_x0, cur_y0, cur_z0, next_chunk.clip(min=0.0, max=maximum) / maximum *100
# get a batch of chunks from generator
def get_batch_from_generator(generator, batch_size, dtype=np.float32):
    positions = list()
    batch = list()
    for _ in range(batch_size):
        try:
            output = next(generator)
            positions.append(output[:3])
            batch.append(output[3])
        except StopIteration:
            break
    return positions, np.asarray(batch, dtype=dtype)

# map the batch of chunks to the map
def map_batch_to_map(pred_map, denominator, positions, batch, box_size):
    for position, chunk in zip(positions, batch):
        pred_map[position[0]:position[0] + box_size, position[1]:position[1] + box_size, position[2]:position[2] + box_size] += chunk
        denominator[position[0]:position[0] + box_size, position[1]:position[1] + box_size, position[2]:position[2] + box_size] += 1
    return pred_map, denominator

# ===================== 改写后的批处理累加 =====================
def map_batch_to_map_weighted(pred_map, denominator, positions, batch, box_size, valid_margin=0):
    """
    批处理版本：把一个 batch 的切块累加到 map
    pred_map: 已累加 map
    denominator: 权重和
    positions: 每个切块左上角坐标列表
    batch: 切块数组
    """
    W = fusion_window_3d(box_size, valid_margin)
    for position, chunk in zip(positions, batch):
        x0, y0, z0 = position
        pred_map[x0:x0 + box_size, y0:y0 + box_size, z0:z0 + box_size] += chunk * W
        denominator[x0:x0 + box_size, y0:y0 + box_size, z0:z0 + box_size] += W
    return pred_map, denominator




# ===================== 核心融合函数 =====================
def map_batch_to_map_optimal(pred_map, denominator, positions, batch, box_size, 
                              valid_margin=None, edge_weight=0.7):
    """
    最佳实践的3D切块融合方案
    
    参数:
        pred_map: 累加的预测图 (H, W, D)
        denominator: 权重累加图 (H, W, D)
        positions: 切块位置列表 [(x, y, z), ...]
        batch: 切块数据数组 (N, box_size, box_size, box_size)
        box_size: 切块大小
        valid_margin: 边缘过渡区宽度，默认为 box_size//8
        edge_weight: 边缘最小权重 [0.5-0.8]，越大越保留边缘信号
    
    返回:
        pred_map, denominator
    """
    if valid_margin is None:
        valid_margin = max(1, box_size // 8)  # 自适应margin
    
    # 生成优化的融合窗口
    W = create_optimal_fusion_window(box_size, valid_margin, edge_weight)
    
    for position, chunk in zip(positions, batch):
        x0, y0, z0 = position
        pred_map[x0:x0 + box_size, y0:y0 + box_size, z0:z0 + box_size] += chunk * W
        denominator[x0:x0 + box_size, y0:y0 + box_size, z0:z0 + box_size] += W
    
    return pred_map, denominator


# ===================== 窗口生成函数 =====================
def create_optimal_fusion_window(box_size, valid_margin, edge_weight=0.7):
    """
    创建优化的3D融合窗口
    
    设计原则:
    1. 中心区域权重=1.0 (完整保留)
    2. 边缘区域权重=edge_weight (保留大部分信号)
    3. 使用余弦平滑过渡 (避免硬边界)
    """
    if valid_margin == 0:
        return np.ones((box_size, box_size, box_size))
    
    # 创建1D窗口函数
    def create_1d_window(size, margin, min_weight):
        w = np.ones(size)
        if margin > 0 and margin < size // 2:
            # 余弦平滑过渡: cos从π到0映射到[0,1]
            t = np.linspace(0, 1, margin)
            ramp = 0.5 * (1 - np.cos(np.pi * t))  # [0, 1]
            ramp = ramp * (1 - min_weight) + min_weight  # [min_weight, 1]
            
            w[:margin] = ramp
            w[-margin:] = ramp[::-1]
        return w
    
    # 生成三个维度的1D窗口
    wx = create_1d_window(box_size, valid_margin, edge_weight)
    wy = create_1d_window(box_size, valid_margin, edge_weight)
    wz = create_1d_window(box_size, valid_margin, edge_weight)
    
    # 3D窗口 = 三个1D窗口的外积
    window_3d = np.einsum('i,j,k->ijk', wx, wy, wz)
    
    return window_3d


# ===================== 自适应版本 =====================
def map_batch_to_map_adaptive(pred_map, denominator, positions, batch, box_size, stride):
    """
    自适应版本：根据重叠率自动调整参数
    
    参数:
        stride: 滑动窗口步长
    """
    # 计算重叠率
    overlap_ratio = 1.0 - stride / box_size
    
    # 根据重叠率自适应设置参数
    if overlap_ratio >= 0.5:
        # 高重叠: 可以用更激进的平滑
        valid_margin = box_size // 6
        edge_weight = 0.6
    elif overlap_ratio >= 0.25:
        # 中等重叠: 平衡方案
        valid_margin = box_size // 8
        edge_weight = 0.7
    else:
        # 低重叠: 保守方案，最大化保留边缘
        valid_margin = box_size // 12
        edge_weight = 0.8
    
    W = create_optimal_fusion_window(box_size, valid_margin, edge_weight)
    
    for position, chunk in zip(positions, batch):
        x0, y0, z0 = position
        pred_map[x0:x0 + box_size, y0:y0 + box_size, z0:z0 + box_size] += chunk * W
        denominator[x0:x0 + box_size, y0:y0 + box_size, z0:z0 + box_size] += W
    
    return pred_map, denominator


def map_batch_to_map_frequency_adaptive(pred_map, denominator, positions, batch, 
                                        box_size, stride, global_map_std=None):
    """
    根据信号频率内容自适应调整融合权重
    - 高频区域(细节多): 用更强的平滑 → 提高Q-score
    - 低频区域(平坦): 保留更多信号 → 提高CC
    """
    # 计算全局标准差(如果未提供)
    if global_map_std is None:
        global_map_std = np.std([chunk.std() for chunk in batch])
    
    overlap_ratio = 1.0 - stride / box_size
    
    for position, chunk in zip(positions, batch):
        # 1. 分析当前切块的频率特征
        chunk_std = chunk.std()
        chunk_grad = np.gradient(chunk)
        edge_strength = np.sqrt(sum(g**2 for g in chunk_grad)).mean()
        
        # 2. 自适应参数
        if edge_strength > 1.5 * global_map_std:
            # 高频区域(边缘多、细节丰富)
            valid_margin = box_size // 6
            edge_weight = 0.6  # 更强平滑，优先Q-score
        elif edge_strength > 0.8 * global_map_std:
            # 中频区域
            valid_margin = box_size // 8
            edge_weight = 0.75  # 平衡
        else:
            # 低频区域(平坦区域)
            valid_margin = box_size // 10
            edge_weight = 0.85  # 保留信号，优先CC
        
        # 3. 生成自适应窗口
        W = create_optimal_fusion_window(box_size, valid_margin, edge_weight)
        
        # 4. 累加
        x0, y0, z0 = position
        pred_map[x0:x0 + box_size, y0:y0 + box_size, z0:z0 + box_size] += chunk * W
        denominator[x0:x0 + box_size, y0:y0 + box_size, z0:z0 + box_size] += W
    
    return pred_map, denominator


def parse_map(map_file, ignorestart, apix=None, origin_shift=None):

    ''' parse mrc '''
    mrc = mrcfile.open(map_file, mode='r')

    map = np.asarray(mrc.data.copy(), dtype=np.float32)
    voxel_size = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    ncrsstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float32)
    ncrs = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma], dtype=np.float32)

    ''' check orthogonal '''
    try:
        assert(angle[0] == angle[1] == angle[2] == 90.0)
    except AssertionError:
        print("# Input grid is not orthogonal. EXIT.")
        mrc.close()
        exit()

    ''' reorder axes '''
    mapcrs = np.subtract([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], 1)
    sort = np.asarray([0, 1, 2], dtype=np.int64)
    for i in range(3):
        sort[mapcrs[i]] = i
    nxyzstart = np.asarray([ncrsstart[i] for i in sort])
    nxyz = np.asarray([ncrs[i] for i in sort])
    nxyz_old = nxyz

    map = np.transpose(map, axes=2-sort[::-1])
    mrc.close()

    ''' shift origin according to n*start '''
    if not ignorestart:
        origin += np.multiply(nxyzstart, voxel_size)

    ''' interpolate grid interval '''
    if apix is not None:
        try:
            assert(voxel_size[0] == voxel_size[1] == voxel_size[2] == apix and origin_shift is None)
        except AssertionError:
            interp3d.del_mapout()
            target_voxel_size = np.asarray([apix, apix, apix], dtype=np.float32)
            print("# Rescale voxel size from {} to {}".format(voxel_size, target_voxel_size))
            if origin_shift is not None:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, origin_shift[2], origin_shift[1], origin_shift[0], nxyz[2], nxyz[1], nxyz[0])
                origin += origin_shift
            else:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, 0.0, 0.0, 0.0, nxyz[2], nxyz[1], nxyz[0])
                
            map = interp3d.mapout
            nxyz = np.asarray([interp3d.pextx, interp3d.pexty, interp3d.pextz], dtype=np.int64)
            voxel_size = target_voxel_size

    assert(np.all(nxyz == np.asarray([map.shape[2], map.shape[1], map.shape[0]], dtype=np.int64)))

    return map, origin, nxyz, voxel_size, nxyz_old

def inverse_map(map_pred, nxyz, origin, voxel_size, old_voxel_size, origin_shift):
    interp3d.del_mapout()
    interp3d.inverse_cubic(map_pred, voxel_size[2], voxel_size[1], voxel_size[0], old_voxel_size[2], old_voxel_size[1], old_voxel_size[0], origin_shift[2], origin_shift[1], origin_shift[0], nxyz[2], nxyz[1], nxyz[0])
    origin += origin_shift
    nxyz = np.asarray([interp3d.pextx, interp3d.pexty, interp3d.pextz], dtype=np.int64)
    map_pred = interp3d.mapout
    voxel_size = old_voxel_size
    return map_pred, origin, nxyz, voxel_size


def write_map(file_name, map, voxel_size, origin=(0.0, 0.0, 0.0), nxyzstart=(0, 0, 0)):
    mrc = mrcfile.new(file_name, overwrite=True)
    mrc.set_data(map)
    (mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart) = nxyzstart
    (mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z) = origin
    mrc.voxel_size = [voxel_size[i] for i in range(3)]

    mrc.close()
