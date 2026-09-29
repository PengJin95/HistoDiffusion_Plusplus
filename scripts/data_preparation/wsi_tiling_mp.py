# Multi-processing tiling of whole slide images (WSIs) into patches

from os import path, makedirs, environ
from os.path import join
import shutil
import openslide as slide
import numpy as np
import pandas as pd
from skimage import io, transform
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from skimage import img_as_ubyte
import time, datetime, warnings, glob
import argparse
import sys

warnings.simplefilter('ignore')

def thres_saturation(img, t=15): # t: threshold of saturation, 0-255
    img = rgb2hsv(img)
    h, w, _ = img.shape
    sat_img = img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return ave_sat >= t


def crop_slide(img, save_slide_path, position=(0, 0), step=(0, 0), patch_size=224, down_scale=1, save_original=False): 
    # position given as (x, y) at nx scale (target)
    patch_name = "{}_{}".format(step[0], step[1])
    p0 = int(position[0] * down_scale)
    p1 = int(position[1] * down_scale)
    psize_original = int(patch_size * down_scale)
    _psize = patch_size if not save_original else psize_original

    img_nx_path = join(save_slide_path, f"{patch_name}-tile-r{p1}-c{p0}-{_psize}x{_psize}.jpg")
    if path.exists(img_nx_path):
        print('%s file already exists.' % img_nx_path)
        return 1

    try: 
        img_x = img.read_region(( int(position[0] * down_scale), int(position[1] * down_scale) ), 0, 
                                ( int(patch_size * down_scale), int(patch_size * down_scale) ))
    except Exception as e:
        print(e)
        return
    img_x = np.array(img_x)[..., :3]
    if not save_original:
        img_x = transform.resize(img_x, (patch_size, patch_size), order=1,  anti_aliasing=False)
    if thres_saturation(img_x, 30): # -1 for all
        try:
            io.imsave(img_nx_path, img_as_ubyte(img_x))
        except Exception as e:
            print(e)


def slide_to_patch(out_base, img_slides, start_idx, patch_size, step_size, scale, 
                   replace, save_original, rough_scale):
    makedirs(out_base, exist_ok=True)
    for idx, img_slide in enumerate(img_slides):
        img_name = img_slide.split(path.sep)[-1].split('.')[0]
        bag_path = join(out_base, img_name)
        if replace:
            shutil.rmtree(bag_path, ignore_errors=True)
        makedirs(bag_path)
        img = slide.OpenSlide(img_slide)

        if 'openslide.objective-power' in img.properties.keys():
            obj_power = int(img.properties['openslide.objective-power'])
            if obj_power < 10:
                print(img_name, "Objective power <10x, too small for resizing", file=sys.stderr)
                continue
            down_scale = obj_power // scale
        elif 'openslide.mpp-x' in img.properties.keys():
            target_mpp = 0.50 * (20 / scale)
            origin_mpp = float(img.properties['openslide.mpp-x'])
            down_scale = target_mpp / origin_mpp
            if rough_scale:
                down_scale = round(down_scale)
        else:
            print(img_name, "No properties 'openslide.objective-power' or 'openslide.mpp-x'", file=sys.stderr)
            continue

        dimension = img.level_dimensions[0]
        # dimension and step at given scale
        step_y_max = int(np.floor(dimension[1]/(step_size*down_scale))) # rows
        step_x_max = int(np.floor(dimension[0]/(step_size*down_scale))) # columns
        num =  step_x_max*step_y_max
        print("slide_to_patch --> ", img_name, num, img.dimensions, img.level_dimensions, img.level_count)
        start_time = time.time()
        for j in range(step_y_max):
            for i in range(step_x_max):
                crop_slide(img, bag_path, (i*step_size, j*step_size), step=(j, i), 
                           patch_size=patch_size, down_scale=down_scale, save_original=save_original)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{idx + start_idx} completed, includes {num} patches, time: {total_time_str}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop the WSIs into patches')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap pixels between adjacent patches')
    parser.add_argument('--patch_size', type=int, default=1024, help='Patch size')
    parser.add_argument('--scale', type=int, default=20, help='20x 10x 5x')
    parser.add_argument('--dataset', type=str, default='./dataset', help='Dataset folder name')
    parser.add_argument('--output', type=str, default='./result/tiled_patches', help='Output folder name')
    parser.add_argument('--resume', type=str, default=None, help='resume file contains rest of the files path')
    parser.add_argument('--replace', action='store_true', help='replace files if already exist')
    parser.add_argument('--save_original', action='store_true', help='save level 0 patches without resizing')
    parser.add_argument('--rough_scale', action='store_true', help='use approximation when computing mpp')
    args = parser.parse_args()
    
    num_processes = int(environ['SLURM_NTASKS'])
    process_idx = int(environ['SLURM_PROCID'])
    print(f'#{process_idx} process in {num_processes} processes')

    # print('Cropping patches, this could take a while for big dataset, please be patient')
    step = args.patch_size - args.overlap

    if args.resume is not None:
        with open(args.resume,'r') as f:
            all_slides = f.readlines()
            all_slides = [s[:-1] for s in all_slides] 
    else:
        path_base = args.dataset
        if path.isdir(path_base):
            if 'tcga' in path_base:
                all_slides = glob.glob(f"{path_base}/*/*.svs") + \
                            glob.glob(f"{path_base}/*/*.tif") + \
                            glob.glob(f"{path_base}/*/*.tiff") + \
                            glob.glob(f"{path_base}/*/*.mrxs") + \
                            glob.glob(f"{path_base}/*/*.ndpi")
            else: # cptac, paip
                all_slides = glob.glob(f"{path_base}/*.svs") + \
                            glob.glob(f"{path_base}/*.tif") + \
                            glob.glob(f"{path_base}/*.tiff") + \
                            glob.glob(f"{path_base}/*.mrxs") + \
                            glob.glob(f"{path_base}/*.ndpi")
        elif path.isfile(path_base):
            df = pd.read_csv(path_base)
            all_slides = df.Slide_Path.values.tolist()
        else:
            raise ValueError(f'Please check dataset folder {path_base}')

    all_slides = sorted(all_slides)
    per_process = int(np.ceil(len(all_slides)/num_processes))
    start_idx = per_process*process_idx
    end_idx =  per_process*(process_idx+1) 
    if end_idx > len(all_slides):
        end_idx = len(all_slides)

    out_base = args.output
    slide_to_patch(out_base, all_slides[start_idx:end_idx], start_idx, args.patch_size, 
                   step, args.scale, args.replace, args.save_original, args.rough_scale)

"""
Example usage:

    #SBATCH --ntasks=96
    #SBATCH --cpus-per-task=1

    srun python -u wsi_tiling_mp.py --patch_size 1024 \
        --dataset /path_to_your_dataset/tcga_slides/acc \
        --output /path_to_your_output/tcga_patches3/acc --scale 20 --save_original 
"""