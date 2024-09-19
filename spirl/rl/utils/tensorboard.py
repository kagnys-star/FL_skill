import torch
import numpy as np
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from spirl.utils.general_utils import flatten_dict, prefix_dict
from spirl.utils.vis_utils import plot_graph

class TensorBoardLogger:
    """Logs to TensorBoard."""
    N_LOGGED_SAMPLES = 3  # how many examples should be logged in each logging step

    def __init__(self, log_dir, conf, exclude=None):
        """
        :param log_dir: path to which TensorBoard log files will be written
        :param conf: hyperparam config that will get logged to TensorBoard
        :param exclude: (optional) list of (flattened) hyperparam names that should not get logged
        """
        if exclude is None: exclude = []
        flat_config = flatten_dict(conf)
        filtered_config = {k: v for k, v in flat_config.items() if (k not in exclude and not inspect.isclass(v))}
        
        self.writer = SummaryWriter(log_dir=log_dir)
        for key, value in filtered_config.items():
            self.writer.add_text(key, str(value))

    def log_scalar_dict(self, d, prefix='', step=None):
        """Logs all entries from a dict of scalars. Optionally can prefix all keys in dict before logging."""
        if prefix: d = prefix_dict(d, prefix + '_')
        for k, v in d.items():
            self.writer.add_scalar(k, v, step)

    def log_scalar(self, v, k, step=None, phase=''):
        if phase:
            k = phase + '/' + k
        self.log_scalar_dict({k: v}, step=step)

    def log_histogram(self, array, name, step=None, phase=''):
        if phase:
            name = phase + '/' + name
        if isinstance(array, torch.Tensor):
            array = array.cpu().detach().numpy()
        self.writer.add_histogram(name, array, step)

    def log_videos(self, vids, name, step=None, fps=20):
        """Logs videos to TensorBoard in mp4 format.
        Assumes list of numpy arrays as input with [time, channels, height, width]."""
        assert len(vids[0].shape) == 4 and vids[0].shape[1] == 3
        assert isinstance(vids[0], np.ndarray)
        if vids[0].max() <= 1.0: vids = [np.asarray(vid * 255.0, dtype=np.uint8) for vid in vids]
        for i, vid in enumerate(vids):
            self.writer.add_video(f'{name}_{i}', vid, step, fps=fps)

    def log_gif(self, v, k, step=None, phase='', fps=20):
        if phase:
            k = phase + '/' + k
        if len(v[0].shape) != 4:
            v = v.unsqueeze(0)
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        self.log_videos(v, k, step=step, fps=fps)

    def log_images(self, images, name, step=None, phase=''):
        if phase:
            name = phase + '/' + name
        if len(images.shape) == 4:  # batch of images
            for i, img in enumerate(images):
                self.writer.add_image(f'{name}_{i}', img, step)
        else:  # single image
            self.writer.add_image(name, images, step)

    def log_graph(self, v, name, step=None, phase=''):
        img = plot_graph(v)
        if phase:
            name = phase + '/' + name
        self.writer.add_image(name, img, step)

    def log_plot(self, fig, name, step=None):
        """Logs matplotlib graph to TensorBoard.
        fig is a matplotlib figure handle."""
        self.writer.add_figure(name, fig, step)

    @property
    def n_logged_samples(self):
        return self.N_LOGGED_SAMPLES

    def visualize(self, *args, **kwargs):
        """Subclasses can implement this method to visualize training results."""
        pass

