import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download TCGA dataset according to manifest file')
    parser.add_argument('--project', type=str, help='Slides of certain project to download')
    parser.add_argument('--type', type=str, options=['tissue', 'diagnostic'], default='tissue',
                        help='Tissue slides or diagnostic slides')
    args = parser.parse_args()

    rfolder = '' if args.type == 'tissue' else '2' # storage folder for diagnostic slides is different from tissue slides
    root = '/path_to_your_root_directory/tcga_slides%s' % rfolder

    project_dir = f'{root}/{args.project}'
    os.makedirs(project_dir, exist_ok=True)
    os.chdir(project_dir)
    print('Start download project: ', args.project)
    manifest_dir = '/path_to_your_manifest_directory/tcga_misc/manifest'
    os.system(f'gdc-client download -m {manifest_dir}/{args.project}_{args.type}_manifest.txt')

"""
Example usage:
    python -u download_files_single.py --project acc --type tissue
""" 
