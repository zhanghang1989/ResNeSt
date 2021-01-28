"""Prepare MS COCO datasets"""
import os
import shutil
import argparse
import zipfile
from resnest.utils import download, mkdir

_TARGET_DIR = os.path.expanduser('./coco')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize MS COCO dataset.',
        epilog='Example: python mscoco.py --download-dir ~/mscoco',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args

def download_coco(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('http://images.cocodataset.org/zips/train2017.zip',
         '10ad623668ab00c62c096f0ed636d6aff41faca5'),
        ('http://images.cocodataset.org/zips/val2017.zip',
         '4950dc9d00dbe1c933ee0170f5797584351d2a41'),
        ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
         '8551ee4bb5860311e79dace7e79cb91e432e78b3'),
        ('https://hangzh.s3.amazonaws.com/encoding/data/coco/train_ids.pth',
         '12cd266f97c8d9ea86e15a11f11bcb5faba700b6'),
        ('https://hangzh.s3.amazonaws.com/encoding/data/coco/val_ids.pth',
         '4ce037ac33cbf3712fd93280a1c5e92dae3136bb'),
    ]
    mkdir(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=path, overwrite=overwrite, sha1_hash=checksum)
        # extract
        if os.path.splitext(filename)[1] == '.zip':
            with zipfile.ZipFile(filename) as zf:
                zf.extractall(path=path)
        else:
            shutil.move(filename, os.path.join(path, 'annotations/'+os.path.basename(filename)))


def install_coco_api():
    repo_url = "https://github.com/cocodataset/cocoapi"
    os.system("git clone " + repo_url)
    os.system("cd cocoapi/PythonAPI/ && python setup.py install")
    shutil.rmtree('cocoapi')
    try:
        import pycocotools
    except Exception:
        print("Installing COCO API failed, please install it manually %s"%(repo_url))


if __name__ == '__main__':
    args = parse_args()
    mkdir(os.path.expanduser('~/.encoding/data'))
    if args.download_dir is not None:
        if os.path.isdir(_TARGET_DIR):
            os.remove(_TARGET_DIR)
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)
    else:
        download_coco(_TARGET_DIR, overwrite=False)
    install_coco_api()
