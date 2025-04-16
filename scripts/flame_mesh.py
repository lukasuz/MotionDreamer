import numpy as np
import torch
from .base_mesh import BaseReposableMesh
from smplx import FLAME
import cv2


class FLAMEMesh(BaseReposableMesh):
    def __init__(self, *args, **kwargs) -> None:
        self.model_name = "flame"
        self.fg_kernel = np.ones((15, 15), np.uint8)

        self.bg_prompt = "forest"

        self.object_prompt = f"A portrait photo a face in front of a {self.bg_prompt}, photorealistic, 4k, DLSR, bokeh"
        self.object_negative_prompt = "grey, gray, monochrome, distorted, disfigured, render, teeth, hat"

        self.bg_negative_prompt = None

        super().__init__(*args, **kwargs)
        
    def init_camera(self):
        self.dist = 0.25
        self.elev = 5
        self.y_offset = [0,0.01,0]
        self.azim_offset = 0

    def init_deformation_model_custom(self):    
        self.global_orient = self.model.global_orient.clone().cuda()
        self.neck_pose = self.model.neck_pose.clone().cuda()
        self.transl = self.model.transl.clone().cuda()
        self.expression = self.model.expression.clone().cuda()
        self.jaw_pose = self.model.jaw_pose.clone().cuda()
        self.leye_pose = self.model.leye_pose.clone().cuda()
        self.reye_pose = self.model.reye_pose.clone().cuda()
        
    def init_model(self, model_path):
        self.model = FLAME(model_path, num_expression_coeffs=10)
        self.model.cuda()
        self.vertices = self.model.v_template[None].detach().cuda()
        self.faces = torch.tensor(self.model.faces.astype(int))[None].cuda()
        
    def get_vertices_custom(self, params):
        params[3] = params[3] * 5 # Scale expression
        params[2] = 0.1 * params[2]
        return self.model(None, *params).vertices
    
    def draw_eye_brow(self, img, left_eyebrow, right_eyebrow, ebd = 5, eye_brow_thickness = [20, 25, 30, 40]):
        eye_brow_img = np.zeros_like(img).astype(np.int8)

        num = left_eyebrow.shape[1]
        right_eyebrow = np.flip(right_eyebrow, axis=1)
        right_eyebrow[:,:,1] = right_eyebrow[:,:,1] + np.array(eye_brow_thickness)[None,:] // 2
        left_eyebrow[:,:,1] = left_eyebrow[:,:,1] + np.array(eye_brow_thickness)[None,:] // 2
        for i in range(num-1):
            le = left_eyebrow[:,i:i+2]
            re = right_eyebrow[:,i:i+2]
            eye_brow_img = cv2.polylines(eye_brow_img, re, False, (ebd, ebd, ebd), eye_brow_thickness[i])
            eye_brow_img = cv2.polylines(eye_brow_img, le, False, (ebd, ebd, ebd), eye_brow_thickness[i])

        eye_brow_img[eye_brow_img == ebd] += (np.random.randn(*(eye_brow_img[eye_brow_img == ebd]).shape)).astype(np.int8)
        eye_brow_img = cv2.GaussianBlur(eye_brow_img.astype(np.uint8), (9,9), 20)

        return eye_brow_img

    def get_zero_pose_custom(self):
        return [self.global_orient, self.neck_pose, self.transl, self.expression, self.jaw_pose, self.leye_pose, self.reye_pose]
    
    def get_pose_multipliers_custom(self):
        return [1e-2, 1e-2, 1e-2, 1e-1, 1e-2, 1e-2, 1e-2]

    def get_grad_for_pose_custom(self):
        return [True, True, True, True, True, False, False]