#import d4rl
import os
import numpy as np
import itertools
import h5py
import copy
from spirl.utils.pytorch_utils import RepeatedDataLoader
from spirl.components.data_loader import Dataset
from spirl.utils.general_utils import AttrDict


class D4RLSequenceSplitDataset(Dataset):
    SPLIT = AttrDict(train=0.9, val=0.1, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 0
        self.shuffle = shuffle
        self.dataset = self._get_filenames()
        
        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))
            start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
            if not isinstance(self.spec.filter_indices[0], list):
                self.spec.filter_indices = [self.spec.filter_indices]
            self.seqs = list(itertools.chain.from_iterable([\
                list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                               for x in self.seqs[fi[0] : fi[1]+1])) for fi in self.spec.filter_indices]))
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        output = AttrDict(
            states=seq.states[start_idx:start_idx+self.subseq_len],
            actions=seq.actions[start_idx:start_idx+self.subseq_len-1],
            pad_mask=np.ones((self.subseq_len,)),
        )
        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output

    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)

    def _get_filenames(self):
        filenames = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".hdf5"):
                    file_path = os.path.join(root, file)
                    with h5py.File(file_path, 'r') as f:
                        print(f"data_load : {file_path}")
                        # HDF5 파일 내용을 깊은 복사하여 저장
                        filenames = self.deep_copy_hdf5_content(f)
        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        return filenames
    
    def get_data_loader(self, batch_size, n_repeat):
        print('len {} dataset {}'.format(self.phase, len(self)))
        assert self.device in ['cuda', 'cpu']  # Otherwise the logic below is wrong
        return RepeatedDataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.n_worker,
                                  drop_last=False, n_repeat=n_repeat, pin_memory=self.device == 'cuda',
                                  worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x))
    
    def deep_copy_hdf5_content(self,file):
        data = {}
        def recursive_copy(group, dest):
            for key, item in group.items():
                if isinstance(item, h5py.Group):
                    dest[key] = {}
                    recursive_copy(item, dest[key])
                elif isinstance(item, h5py.Dataset):
                    dest[key] = item[:]
                    
        recursive_copy(file, data)
        return data
