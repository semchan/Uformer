import random
from os import PathLike
from pathlib import Path
from typing import Any, List, Dict

import h5py
import numpy as np
import yaml
import re
import os

def get_ovp_user_summary(ovp_user_summary_path,video_name,n_frames):
    # ovp_user_summary_path = '/home/chenys/VideoSummary/UFormer20211212/splits/ovp_usersummary_org/'
    # video_name = 'video_50'
    # n_frames = 1400

    regex = re.compile(r'\d+')
    num = int(max(regex.findall(video_name)))
    user_summary_folder_name = 'v'+str(20+num)
    user_summary_folder_path = ovp_user_summary_path+user_summary_folder_name
    user_summary = np.zeros((5,n_frames), dtype=np.float32)
    for dirpath, dirnames, filenames in os.walk(user_summary_folder_path):
        for dirname in dirnames:
            user_index = int(max(regex.findall(dirname)))
            for dirpath2, dirnames2, filenames2 in os.walk(dirpath+'/'+dirname):
                # user_summary_image_index=[]
                for file_name in filenames2:
                    file_name_index = int(max(regex.findall(file_name)))
                    if file_name_index>n_frames:
                        continue
                    else:
                        user_summary[user_index-1][file_name_index-1]=1.0
                    # user_summary_image_index.append(file_name_index)
                # path = [os.path.join(dirpath2, names) for names in filenames2]
                # print(dirpath, dirnames, filenames)
    
    return user_summary

class VideoDataset(object):
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.datasets = self.get_datasets(keys)

    def __getitem__(self, index):
        key = self.keys[index]
        video_path = Path(key)
        dataset_name = str(video_path.parent)
        video_name = video_path.name
        video_file = self.datasets[dataset_name][video_name]

        seq = video_file['features'][...].astype(np.float32)
        seq_size = seq.shape

        seqdiff = []
        seq_temp = 0
        for j in range((seq_size[0])):
            if j==0:
                seq_current = seq[j]
                # seq_before = seq[j]
            else:
                seq_current = seq[j]
                seq_temp = seq_current-seq_before
                seqdiff.append(seq_temp)
            seq_before = seq_current
        seqdiff.append(seq_temp)
        seqdiff = np.array(seqdiff)




        gtscore = video_file['gtscore'][...].astype(np.float32)
        cps = video_file['change_points'][...].astype(np.int32)
        n_frames = video_file['n_frames'][...].astype(np.int32)
        nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)
        user_summary = None
        # if 'ovp' in dataset_name:
        #     ovp_user_summary_path = '/home/chenys/VideoSummary/UFormer20211212/splits/ovp_usersummary_org/'
        #     user_summary=get_ovp_user_summary(ovp_user_summary_path,video_name,n_frames)
        # else:
        #     if 'user_summary' in video_file:
        #         user_summary = video_file['user_summary'][...].astype(np.float32)#（20，4463）


        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)#（20，4463）

        gtscore -= gtscore.min()
        gtscore /= gtscore.max()#461

        return key, seq, seqdiff, gtscore, cps, n_frames, nfps, picks, user_summary

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def get_datasets(keys: List[str]):#  -> Dict[str, h5py.File]:
        dataset_paths = {str(Path(key).parent) for key in keys}
        datasets = {path: h5py.File(path, 'r') for path in dataset_paths}
        return datasets


class DataLoader(object):
    def __init__(self, dataset: VideoDataset, shuffle: bool):
        self.dataset = dataset
        self.shuffle = shuffle
        self.data_idx = list(range(len(self.dataset)))

    def __iter__(self):
        self.iter_idx = 0
        if self.shuffle:
            random.shuffle(self.data_idx)
        return self

    def __next__(self):
        if self.iter_idx == len(self.dataset):
            raise StopIteration
        curr_idx = self.data_idx[self.iter_idx]
        batch = self.dataset[curr_idx]
        self.iter_idx += 1
        return batch


class AverageMeter(object):
    def __init__(self, *keys: str):
        self.totals = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self._check_attr(key)
            self.totals[key] += value
            self.counts[key] += 1

    def __getattr__(self, attr: str):#  -> float:
        self._check_attr(attr)
        total = self.totals[attr]
        count = self.counts[attr]
        return total / count if count else 0.0

    def _check_attr(self, attr: str):#  -> None:
        assert attr in self.totals and attr in self.counts


def get_ckpt_dir(model_dir: PathLike):#  -> Path:
    return Path(model_dir) / 'checkpoint'


def get_ckpt_path(model_dir: PathLike,
                  split_path: PathLike,
                  split_index: int):#  -> Path:
    split_path = Path(split_path)
    return get_ckpt_dir(model_dir) / f'{split_path.name}.{split_index}.pt'


def load_yaml(path: PathLike):#  -> Any:
    with open(path) as f:
        obj = yaml.safe_load(f)
    return obj


def dump_yaml(obj: Any, path: PathLike):#  -> None:
    with open(path, 'w') as f:
        yaml.dump(obj, f)



if __name__ == '__main__':
    ovp_user_summary_path = '/home/chenys/VideoSummary/UFormer20211212/splits/ovp_usersummary_org/'
    video_name = 'video_50'
    n_frames = 1400

    outsummary = get_ovp_user_summary(ovp_user_summary_path,video_name,n_frames)
        
