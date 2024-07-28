import os
import torch
import numpy as np

from Network.stgcn import OneStream_STGCN

from Network.oneshot_stgcn_nano import OSA_STGCN_nano_1S
from Network.oneshot_stgcn_small import OSA_STGCN_small_1S
from Network.oneshot_stgcn_medium import OSA_STGCN_medium_1S
from Network.oneshot_stgcn_large import OSA_STGCN_large_1S
from Network.exponential_dense_stgcn import Exp_DenseSTGCN_1S
from pose_utils import normalize_points_with_size, scale_pose


class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                weight_file = 'Result/URFD_3classes/Exp_DenseSTGCN_1S_20240511134826/Exp_DenseSTGCN_1S_best.pth',
                device='cuda'):
        self.graph_args = {'strategy': 'spatial'}
        # self.class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
        #                     'Stand up', 'Sit down', 'Fall Down']
        self.class_names = ['Not Fall', 'Falling', 'Fall Detected']
        self.num_class = len(self.class_names)
        self.device = device
        
        self.model = Exp_DenseSTGCN_1S(num_class = self.num_class, graph_args = self.graph_args).to(self.device)
        # self.model = OSA_STGCN_small_1S(num_class = self.num_class, graph_args = self.graph_args).to(self.device)
        self.model.load_state_dict(torch.load(weight_file),  strict=False)
        self.model.eval()

    def predict(self, pts, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: (numpy array) points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).,
                v : number of graph node (body parts).,
                c : channel (x, y, score).,
            image_size: (tuple of int) width, height of image frame.
        Returns:
            (numpy array) Probability of each class actions.
        """
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)

        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :]


        pts = pts.to(self.device)

        out = self.model(pts)

        return out.detach().cpu().numpy()
