import torch
from .base_mesh import BaseReposableMesh
from smplx import SMPL
import numpy as np

class SMPLMesh(BaseReposableMesh):
    def __init__(self, *args, **kwargs) -> None:
        self.model_name = 'smpl'
        self.bg_prompt = "forest"
        self.object_prompt = f"A photo of a clothed person wearing pants and tshirt in front of a {self.bg_prompt} , photorealistic, 4k, DLSR"
        self.object_negative_prompt = "grey, gray, monochrome, distorted, disfigured, naked, nude"

        self.bg_negative_prompt = None

        self.keypoint_kernel = np.ones((50, 50), np.uint8)

        super().__init__(*args, **kwargs)
        
    def init_camera(self):
        self.dist = 2.1
        self.elev = 20
        self.azim_offset = 40
        self.y_offset = [0,0.25,0]

    def init_deformation_model_custom(self):
        self.body_pose = self.model.body_pose.clone()
        self.body_pose[0, 16*3 + 2] = 1.2
        self.body_pose[0, 15*3 + 2] = -1.2
        self.global_orient = self.model.global_orient.clone()
        self.transl = self.model.transl.clone()
        self.scale = torch.tensor([[1.]]).cuda()
        
    def init_model(self, model_path):
        # default_body_pose = torch.zeros((1, 32 * 3), dtype=torch.float32)
        self.model = SMPL(
            model_path=model_path, 
            # body_pose=default_body_pose,
            device='cuda',
            use_hands=False,
            use_feet_keypoints=False)
        self.model.cuda()
        self.vertices = self.model.v_template[None].detach().cuda()
        self.faces = torch.tensor(self.model.faces.astype(int))[None].cuda()
        
    def get_vertices_custom(self, params):
        params[2] = 0.1 * params[2]
        transl = params[2]
        scale = params[3]
        params = params[:3]
        return (self.model(None, *params).vertices)

    def get_keypoints_custom(self, params, camera, two2d=True):
        all_landmarks = []
        for i in range(len(params[0])): # TODO: Don't iterate
            landmarks = self.model(None, *[param[i][None] for param in params]).joints
            all_landmarks.append(landmarks)
        all_landmarks = torch.cat(all_landmarks, dim=0)
        if two2d:
            camera.transform_points_screen(all_landmarks)[:, :, :2] - 0.5 
        return all_landmarks

    def get_zero_pose_custom(self):
        return [self.body_pose, self.global_orient, self.transl, self.scale]
    
    def set_zero_pose_custom(self, params):
        self.body_pose = params[0]
        self.global_orient = params[1]
        self.transl = params[2]
        self.scale = params[3]

    def get_grad_for_pose_custom(self):
        return [True, True, True, True]
    
    def get_pose_multipliers_custom(self):
        return [1e-1, 1e-2, 1e-2, 1e-2]