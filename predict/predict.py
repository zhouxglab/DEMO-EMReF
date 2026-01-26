import warnings
from Bio import BiopythonWarning
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', BiopythonWarning)
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os
import torch
import warnings
import numpy as np
from torch import nn
from math import ceil
from tqdm import tqdm
from Bio.PDB import PDBParser, MMCIFParser
from Bio import BiopythonWarning
from model.utils import parse_map, pad_map, chunk_generator,chunk_generator100,get_batch_from_generator,map_batch_to_map_optimal, map_batch_to_map,map_batch_to_map_weighted, write_map, inverse_map
from model.hma import HMA
from ops.argparser import argparser

import textwrap

def main(params):
    params = argparser()
    config = params['config']
    # Extract parameters from config
    in_map = params['F']
    out_map = params['output']
    mask_map = params['mask_map']
    mask_contour = params['contour']
    mask_str = params['mask_str']
    inverse_mask = params['inverse_mask'] 
    gpu_id = params.get('gpu', "0")
    batch_size = params["batch_size"]
    stride = params['stride']
    use_gpu = params.get('gpu', True)
    interp_back = params.get('interp_back', True)
    model_dir = params['model']['path']
    BOX_SIZE = params['model']['length']['box_size']  
    PERCENTILE = 99.999
    # ===== Print welcome banner and user-provided args =====
    print(textwrap.dedent(r"""
    ======================================
    ╔════════════════════════════════════╗
    ║       Welcome to DEMO-EMReF        ║
    ║ Cryo-EM / ET Map Refinement Tool   ║
    ╚════════════════════════════════════╝
    ======================================
    Type the following command to view full help and parameter descriptions:
    ./DEMO-EMReF -h
    ========================================================================
    """).strip())

    # Set CUDA environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"# Running on {n_gpus} GPU(s)")
        else:
            raise RuntimeError("CUDA not available")
    else:
        n_gpus = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("# Running on CPU")

    # Check stride range
    if not (48 >= stride >= 6):
        raise ValueError("`--stride` (`-s`) must be in the range of [6, 48]")

    if mask_map is not None and mask_str is not None:
        raise ValueError("`--mask_map` (`-m`) and `--mask_str` (`-p`) cannot be provided at the same time")

    _, _, _, voxel_size, _ = parse_map(in_map, ignorestart=False)
    print(f"# Voxel size of the input map: {voxel_size}")

    resume_model_path = os.path.abspath(model_dir)
    if use_gpu:
        apix = 1.0
        model_state_dict = torch.load(resume_model_path)
    else:
        apix = 1.0
        model_state_dict = torch.load(resume_model_path, map_location=torch.device('cpu'))
    
    hma_runner = HMA(params)
    model = hma_runner.netG
    model.load_state_dict(model_state_dict['state_dict'], strict=False)
    msg=model.load_state_dict(model_state_dict['state_dict'], strict=False)

    if use_gpu:
        torch.cuda.empty_cache()
        model = model.cuda()
        if n_gpus > 1:
            model = nn.DataParallel(model)
    model.eval()

    print("# Loading the input map...")

    # Load input map
    map, origin, nxyz, voxel_size, nxyz_origin = parse_map(in_map, ignorestart=False, apix=apix)
    print(f"# Original map dimensions: {nxyz_origin}")

    try:
        assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)

    except AssertionError:
        origin_shift =  ( np.round(origin / voxel_size) - origin / voxel_size ) * voxel_size
        map, origin, nxyz, voxel_size, _ = parse_map(in_map, ignorestart=False, apix=1, origin_shift=origin_shift)
        assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)

    nxyzstart = np.round(origin / voxel_size).astype(np.int64)

    print(f"# Map dimensions at {apix} Angstrom grid size: {nxyz}")

    map_volume = map.copy()

    del map

    _, _, _, old_voxel_size, _ = parse_map(in_map, ignorestart=False, apix=None)

    # Process mask map if provided
    if mask_map is not None:
        map_mask = map_volume.copy()
        del map_volume

        print("# Loading the mask map...")

        mask, origin_mask, nxyz_mask, voxel_size_mask, _ = parse_map(mask_map, ignorestart=False, apix=apix)

        try:
            assert np.all(np.abs(np.round(origin_mask / voxel_size_mask) - origin_mask / voxel_size_mask) < 1e-4)

        except AssertionError:
            origin_shift_mask =  ( np.round(origin_mask / voxel_size_mask) - origin_mask / voxel_size_mask ) * voxel_size_mask
            mask, origin_mask, nxyz_mask, voxel_size_mask, _ = parse_map(mask_map, ignorestart=False, apix=apix, origin_shift=origin_shift_mask)
            assert np.all(np.abs(np.round(origin_mask / voxel_size_mask) - origin_mask / voxel_size_mask) < 1e-4)

        nxyzstart_mask = np.round(origin_mask / voxel_size_mask).astype(np.int64)

        print(f"# Mask map dimensions: {nxyz_mask}")

        assert np.all(nxyz_mask <= nxyz)

        try:
            assert np.all(nxyz_mask == nxyz)

        except AssertionError:
            pad_mask = np.zeros(nxyz[::-1]).astype(np.float32)
            nxyz_shift = nxyzstart_mask - nxyzstart
            pad_mask[nxyz_shift[2]:nxyz_shift[2]+nxyz_mask[2], nxyz_shift[1]:nxyz_shift[1]+nxyz_mask[1], nxyz_shift[0]:nxyz_shift[0]+nxyz_mask[0]] = mask
            mask = pad_mask
            origin_mask = origin
            nxyz_mask = nxyz
            nxyzstart_mask = nxyzstart

        if inverse_mask:
            map_volume = np.where(mask < mask_contour, map_mask, 0).astype(np.float32)
        else:
            map_volume = np.where(mask >= mask_contour, map_mask, 0).astype(np.float32)

        if mask_out is not None:
            if inverse_mask:
                mask_o = np.where(mask < mask_contour, 1, 0).astype(np.float32)
            else:
                mask_o = np.where(mask >= mask_contour, 1, 0).astype(np.float32)

            print(f"# Saving the binary mask map to {mask_out}")
            write_map(mask_out, mask_o, voxel_size_mask, nxyzstart=nxyzstart_mask)

        del map_mask, mask

    if mask_str is not None:
        map_mask = map_volume.copy()
        del map_volume

        if mask_str.split(".")[-1][-3:] == "pdb" or mask_str.split(".")[-1][-4:] == "pdb1":
            parser = PDBParser()
        elif mask_str.split(".")[-1][-3:] == "cif":
            parser = MMCIFParser()
        else:
            raise RuntimeError("Unknown type for structure file:", mask_str[-3:])
        structures = parser.get_structure("str", mask_str)
        coords = []

        structure = structures[0]
        for atom in structure.get_atoms():
            if atom.element == 'H':
                continue
            coords.append(atom.get_coord())
        atoms = np.asarray(coords, dtype=np.float32)
        del coords

        print(f"# Generating the mask map from the structure file {mask_str}...")
        map_volume = np.zeros(nxyz[::-1], dtype=np.float32)
        mask = np.zeros(nxyz[::-1], dtype=np.int16)
        for atom in atoms:
            atom_shifted = atom - origin
            lower = np.floor((atom_shifted - mask_str_radius) / voxel_size).astype(np.int32)
            upper = np.ceil ((atom_shifted + mask_str_radius) / voxel_size).astype(np.int32)
            for x in range(lower[0], upper[0] + 1):
                for y in range(lower[1], upper[1] + 1):
                    for z in range(lower[2], upper[2] + 1):
                        if 0 <= x < nxyz[0] and 0 <= y < nxyz[1] and 0 <= z < nxyz[2]:
                            if mask[z, y, x] == 0:
                                vector = np.array([x, y, z], dtype=np.float32) * voxel_size - atom_shifted
                                dist = np.sqrt(vector@vector)
                                if dist < mask_str_radius:
                                    mask[z, y, x] = 1

        if inverse_mask:
            mask = 1 - mask
        map_volume = map_mask * mask.astype(np.float32)

        if mask_out is not None:
            print(f"# Saving the binary mask map to {mask_out}")
            write_map(mask_out, mask.astype(np.float32), voxel_size, nxyzstart=nxyzstart)

        del map_mask, mask

    map = map_volume.copy()
    del map_volume
    
    padded_map = pad_map(map, BOX_SIZE, dtype=np.float32, padding=0.0)

    maximum = np.percentile(map[map > 0], PERCENTILE)
    del map

    map_pred = np.zeros_like(padded_map, dtype=np.float32)
    denominator = np.zeros_like(padded_map, dtype=np.float32)

    # Run inference
    print("# Start processing...")

    if "infer_MR" in config:
        generator = chunk_generator(padded_map, maximum, BOX_SIZE, stride)

        cached_chunks = list(chunk_generator(padded_map, maximum, BOX_SIZE, stride))
    elif "infer_HR" or "infer_ET" in config:
        generator = chunk_generator100(padded_map, maximum, BOX_SIZE, stride)

        cached_chunks = list(chunk_generator100(padded_map, maximum, BOX_SIZE, stride))
    else:
        raise ValueError(f"Unsupported config path: {config}")
    total_steps = len(cached_chunks) 

    acc_steps, acc_steps_x, l_bar = 0.0, 0, 0

    with torch.inference_mode():
      
        with tqdm(total=total_steps, desc="Inference", unit="chunk") as pbar:
            while True:
                positions, chunks = get_batch_from_generator(generator, batch_size, dtype=np.float32)
                if len(positions) == 0:
                    break

                X = torch.tensor(
                    chunks,
                    dtype=torch.float32,
                    device='cuda' if use_gpu else 'cpu'
                ).view(-1, 1, BOX_SIZE, BOX_SIZE, BOX_SIZE)

                with torch.no_grad():
                    y_pred = model(X, istrain=False)

                y_pred = y_pred.view(-1, BOX_SIZE, BOX_SIZE, BOX_SIZE)
                y_pred = y_pred.cpu().numpy()

                map_pred, denominator = map_batch_to_map(map_pred, denominator, positions, y_pred, BOX_SIZE)

                pbar.update(len(positions))

    map_pred = (map_pred / denominator.clip(min=1))[BOX_SIZE:BOX_SIZE + nxyz[2], BOX_SIZE:BOX_SIZE + nxyz[1], BOX_SIZE:BOX_SIZE + nxyz[0]]


    if interp_back:
        out_map_nointerp = f"{out_map[:-4]}_grid_size_{apix}{out_map[-4:]}"
        print(f"# Saving the processed map that has not been interpolated back to the original grid size to {out_map_nointerp}")
        write_map(out_map_nointerp, map_pred, voxel_size, nxyzstart=nxyzstart)

        print(f"# Interpolating the voxel size from {voxel_size} back to {old_voxel_size}")
        origin = nxyzstart * voxel_size
        origin_shift = [0.0, 0.0, 0.0]
        try:
            assert np.all(np.abs(np.round(origin / old_voxel_size) - origin / old_voxel_size) < 1e-4)
        except AssertionError:
            origin_shift = ( np.round(origin / old_voxel_size) - origin / old_voxel_size ) * old_voxel_size
        map_pred, origin, nxyz, voxel_size = inverse_map(map_pred, nxyz, origin, voxel_size, old_voxel_size, origin_shift)
        assert np.all(np.abs(np.round(origin / old_voxel_size) - origin / old_voxel_size) < 1e-4)
        nxyzstart = np.round(origin / voxel_size).astype(np.int64)

        print(f"# Saving the processed map that has been interpolated back to the original grid size to {out_map}")
        write_map(out_map, map_pred, voxel_size, nxyzstart=nxyzstart)

    else:
        print(f"# Saving the processed map that has not been interpolated back to the original grid size to {out_map}")
        write_map(out_map, map_pred, voxel_size, nxyzstart=nxyzstart)

if __name__ == "__main__":
    # 使用配置参数
    params = argparser()
    main(params)
