import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SMALify'))
from smal_model.smal_torch import SMAL

from .base_mesh import BaseReposableMesh
import torch

class SMALMesh(BaseReposableMesh):
    def __init__(self, *args, **kwargs) -> None:
        animal_num = int(args[0]) # Hijack model_path for animal number as SMAL has no model path
        assert animal_num in [0, 1, 2, 3, 4], "Animal number must be between 0 and 4"
        num_to_animal = {
            0: 'big cat',
            1: 'dog',
            2: 'horse',
            3: 'cow',
            4: 'hippo'
        }

        self.animal_num = animal_num
        self.model_name = num_to_animal[animal_num]
        self.bg_prompt = "garden"
        self.object_prompt = f"A photo of a {self.model_name} in front of a {self.bg_prompt}, photorealistic, 4k, DLSR"
        self.object_negative_prompt = "grey, gray, monochrome, distorted, disfigured, render"

        self.bg_negative_prompt = None

        super().__init__(*args, **kwargs)
        
    def init_camera(self):
        self.dist = 2.1
        self.elev = 20
        self.azim_offset = 45
        self.y_offset = [0,0,0]

    def init_deformation_model_custom(self):
        self.theta = torch.zeros(1, 35 * 3).cuda()
        self.trans = torch.zeros(1, 3).cuda()
        self.betas = torch.tensor([[]]).cuda()
        
    def init_model(self, model_path):
        self.model = SMAL('cuda', shape_family_id=self.animal_num)
        self.model.cuda()
        self.vertices = self.model.v_template[None].detach().cuda()
        self.faces = self.model.faces[None].cuda()
        # SMAL has a different coordinate system
        self.rot = torch.tensor([
            [0, 0, 1], 
            [1, 0, 0], 
            [0, 1, 0]], dtype=torch.float32)[None].cuda()
        
    def get_vertices_custom(self, params):
        params[1] = 0.1 * params[1]
        vertices = self.model(self.betas, *params)[0]
        return torch.bmm(vertices, self.rot)

    def get_keypoints_custom(self, params, camera, two2d=True):
        all_landmarks = []
        for i in range(len(params[0])): # TODO: Don't iterate
            landmarks = self.model(self.betas, *[param[i][None] for param in params])[1]
            landmarks = torch.bmm(landmarks, self.rot)
            all_landmarks.append(landmarks)
        all_landmarks = torch.cat(all_landmarks, dim=0)
        if two2d:
            camera.transform_points_screen(all_landmarks)[:, :, :2] - 0.5
        return all_landmarks

    def get_zero_pose_custom(self):
        return [self.theta, self.trans]
    
    def get_pose_multipliers_custom(self):
        return [1e-1, 1e-2]

    def get_grad_for_pose_custom(self):
        return [True, True]