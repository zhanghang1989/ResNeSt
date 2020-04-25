##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Model store which provides pretrained models."""
from __future__ import print_function

__all__ = ['get_model_file', 'purge']

import os
import zipfile

from ..utils import download, check_sha1

_model_sha1 = {name: checksum for checksum, name in [
    ('bcfefe1dd1dd1ef5cfed5563123c1490ea37b42e', 'resnest50'),
    ('5da943b3230f071525a98639945a6b3b3a45ac95', 'resnest101'),
    ('0c5d117df664ace220aa6fc2922c094bb079d381', 'resnest200'),
    ('11ae7f5da2bcdbad05ba7e84f9b74383e717f3e3', 'resnest269'),
    ('5e16dbe56f1fba8e1bc2faddd91f874bfbd74193', 'resnest50_fast_1s1x64d'),
    ('85eb779a5e313d74b5e5390dae02aa8082a0f469', 'resnest50_fast_2s1x64d'),
    ('3f215532c6d8e07a10df116309993d4479fc3e4b', 'resnest50_fast_4s1x64d'),
    ('af3514c2ec757a3a9666a75b82f142ed47d55bee', 'resnest50_fast_1s2x40d'),
    ('2db13245aa4967cf5e8617cb4911880dd41628a4', 'resnest50_fast_2s2x40d'),
    ('b24d515797832e02da4da9c8a15effd5e44cfb56', 'resnest50_fast_4s2x40d'),
    ('7318153ddb5e542a20cc6c58192f3c6209cff9ed', 'resnest50_fast_1s4x24d'),
    ]}

encoding_repo_url = 'https://hangzh.s3-us-west-1.amazonaws.com/'
_url_format = '{repo_url}gluon/models/{file_name}.zip'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, root=os.path.join('~', '.encoding', 'models')):
    r"""Return location for the pretrained on local file system.
    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.
    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    if name not in _model_sha1:
        import gluoncv as gcv
        return gcv.model_zoo.model_store.get_model_file(name, root=root)
    file_name = '{name}-{short_hash}'.format(name=name, short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.params')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            print('Mismatch in the content of model file {} detected.' +
                  ' Downloading again.'.format(file_path))
    else:
        print('Model file {} is not found. Downloading.'.format(file_path))

    if not os.path.exists(root):
        os.makedirs(root)

    zip_file_path = os.path.join(root, file_name+'.zip')
    repo_url = os.environ.get('ENCODING_REPO', encoding_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')

def purge(root=os.path.join('~', '.encoding', 'models')):
    r"""Purge all pretrained model files in local file store.
    Parameters
    ----------
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".params"):
            os.remove(os.path.join(root, f))

def pretrained_model_list():
    return list(_model_sha1.keys())

