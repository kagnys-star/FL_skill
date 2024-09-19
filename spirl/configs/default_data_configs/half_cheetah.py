from spirl.utils.general_utils import AttrDict
from spirl.data.half.src.half_cheetah_dataloader import D4RLSequenceSplitDataset


data_spec = AttrDict(
    dataset_class=D4RLSequenceSplitDataset,
    n_actions=6,
    state_dim=21,
    env_name="HalfCheetah-v3",
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 1000
