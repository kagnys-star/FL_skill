from spirl.utils.general_utils import AttrDict
from spirl.data.metaworld.src.metaworld_data_loader import D4RLSequenceSplitDataset


data_spec = AttrDict(
    dataset_class=D4RLSequenceSplitDataset,
    n_actions=4,
    state_dim=49,
    env_name="metaworlds",
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 500
