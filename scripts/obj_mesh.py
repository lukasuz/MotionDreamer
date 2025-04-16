import numpy as np
import torch
from .base_mesh import BaseReposableMesh
import os
import json
import mesh2sdf
import pymeshlab
from pytorch3d.io import load_obj

class OBJMesh(BaseReposableMesh):
    def __init__(self, *args, **kwargs) -> None:
        self.model_name = "obj"
        self.fg_kernel = np.ones((0, 0), np.uint8)
        self.keypoint_kernel = np.ones((0, 0), np.uint8)

        param_path = os.path.join(args[0], 'params.json')
        self.params_json = json.load(open(param_path))

        self.bg_prompt = f"{self.params_json['background']}"

        self.object_prompt = f"A photo of a {self.params_json['object']} in front of a {self.bg_prompt}, photorealistic, 4k, DLSR"
        self.object_negative_prompt = "grey, gray, monochrome, distorted, disfigured, render"

        
        self.bg_negative_prompt = None

        try:
            kwargs['light_location'] = self.params_json['light_location']
        except:
            pass
        super().__init__(*args, **kwargs)
        
    def init_camera(self):
        self.dist = self.params_json['dist']
        self.elev = self.params_json['elev']
        self.y_offset = self.params_json['y_offset']
        self.azim_offset = self.params_json['azim_offset']
        
    def init_model(self, model_path, target_num=8000):
        print("Loading OBJ")
        try:
            obj_path = os.path.join(model_path, 'new_data.obj')
            vertices, faces, _ = load_obj(obj_path)
            preprocess = False
        except:
            obj_path = os.path.join(model_path, 'data.obj')
            vertices, faces, _ = load_obj(obj_path)
            preprocess = True
        vertices = vertices.cpu().numpy()
        faces = faces.verts_idx.cpu().numpy()

        changed = False
        if self.params_json['connect'] & preprocess:
            print("Turning mesh into SDF")
            # Create a mesh that is connected and watertight, necessary for NJF
            # https://github.com/wang-ps/mesh2sdf/tree/master?tab=readme-ov-file
            size = 128
            level = 2 / size
            vertices = (vertices - np.min(vertices)) / ((np.max(vertices) - np.min(vertices)))
            vertices = (vertices - 0.5) * 1.9 # Give a bit of a buffer for mesh2sdf

            _, mesh = mesh2sdf.compute(vertices, faces, size, fix=True, level=level, return_mesh=True)
            vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)
            changed = True

        m = pymeshlab.Mesh(vertices, faces)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m)
        
        if self.params_json['retriangulate'] & preprocess:
            print("Retriangulating mesh")
            ms.meshing_isotropic_explicit_remeshing()
            changed = True
        
        # Decimate mesh, if has more than target_num vertices
        if (ms.current_mesh().vertex_number() > target_num) & preprocess:
            print("Decimating mesh")
            changed = True
        
            numFaces = 100 + 2 * target_num
            while (ms.current_mesh().vertex_number() > target_num):
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=numFaces, preservenormal=True)
                numFaces = numFaces - (ms.current_mesh().vertex_number() - target_num)
        
        if changed:
            ms.save_current_mesh(os.path.join(model_path, 'new_data.obj'))
        m = ms.current_mesh()
        vertices = torch.tensor(np.array(m.vertex_matrix()), dtype=torch.float32)
        faces = torch.tensor(np.array(m.face_matrix()))

        # Shift mesh
        def shift_dim(vert, dim):
            y_min = vert[:,dim].min()
            y_max = vert[:,dim].max()
            vert[:,dim] -= y_min 
            vert[:,dim] -= (y_max - y_min) / 2
            return vert
        vertices = (vertices - vertices.min()) / (vertices.max() - vertices.min())
        vertices = shift_dim(vertices, 0)
        vertices = shift_dim(vertices, 1)
        vertices = shift_dim(vertices, 2)
        self.vertices = vertices[None].cuda()
        self.faces = faces[None].cuda()
        

    def init_deformation_model_custom(self):
        raise NotImplementedError("init_deformation_model_custom not implemented")
    
    def get_zero_pose_custom(self):
        raise NotImplementedError("get_zero_pose_custom not implemented")
    
    def get_vertices_custom(self, params):
        raise NotImplementedError("get_vertices_custom not implemented")
    
    def get_grad_for_pose_custom(self):
        raise NotImplementedError("get_grad_for_pose_custom not implemented")

    def get_keypoints_custom(self, params, camera):
        raise NotImplementedError("get_keypoints_custom not implemented")