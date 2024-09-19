from spirl.utils.general_utils import AttrDict
from spirl.components.data_loader import GlobalSplitVideoDataset
from spirl.data.maze.src.maze_data_loader import MazeStateSequenceDataset

data_spec = AttrDict(
    dataset_class=MazeStateSequenceDataset,
    n_actions=2,
    state_dim=4,
    env_name="maze2d-large-v1",
    res=32,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 300
