import torch
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (TexturesUV,PointsRenderer)
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, AutoPipelineForInpainting
from PIL import Image

import open3d as o3d
import roma
import torchvision
from abc import ABC, abstractmethod
from time import time

from .render import *
from .mlp import *

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'NeuralJacobianFields', 'source_njf'))
import SourceMesh


class BaseReposableMesh(ABC):
    def __init__(self,
                 model_path,
                 img_size,
                 steps=25, 
                 num_frames=14,
                 tensorboard_path=None,
                 skip_texture_init=False,
                 deformation_model='njf',
                 feature_size=(88, 160),
                 seed=2024,
                 use_depth_process_custom=False,
                 white_background=False,
                 light_location=None,
                 camera=None,
                 azim_offset=None,
                 **kwargs) -> None:
        self.seed = seed
        self.model_path = model_path
        self.img_size = img_size
        self.steps = steps
        self.num_frames = num_frames
        self.tensorboard_path = tensorboard_path
        self.skip_texture_init = skip_texture_init
        self.deformation_model = deformation_model
        self.deformation_model = self.deformation_model
        self.use_depth_process_custom = use_depth_process_custom
        self.feature_size = feature_size
        self.feature_render_size = (feature_size[0] * 1, feature_size[1] * 1)
        self.white_background = white_background
        self.bg = torch.ones((1, *img_size, 3)).cuda()
        self.white_bg = self.bg.clone()

        self.fg_kernel = np.ones((0, 0), np.uint8)

        self.init_camera()
        if camera is None:
            azim_offset = self.azim_offset if azim_offset is None else azim_offset
            self.camera = sample_circular_cams(1, self.dist, self.elev, azim_offset=azim_offset, y_offset=self.y_offset, device='cuda')
        else:
            self.camera = camera
        self.camera_opposite = sample_circular_cams(1, 1.5 * self.dist, self.elev, azim_offset=self.azim_offset - 70, y_offset=self.y_offset, device='cuda')
        self.camera_experiment = sample_circular_cams(1, 1.8 * self.dist, 15, azim_offset=self.azim_offset - 60, y_offset=self.y_offset, device='cuda')
        self.camera.image_size = img_size
        self.init_model(model_path)

        self.renderer, self.rasterizer = get_simple_renderer(img_size=img_size, device='cuda', type='point', location=light_location)
        self.renderer_diffuse, _ = get_simple_renderer(img_size=img_size, device='cuda', type='ambient', location=light_location)
        self.faces_per_pixel_features = 1
        self.feature_rasterizer = get_simple_rasterizer(img_size=self.feature_render_size, faces_per_pixel=self.faces_per_pixel_features, cull_backfaces=False, blur_radius=0)

        self.texture = torch.ones((1, *img_size, 3)).cuda() * 0.5
        self.grey_texture = self.texture.clone()
        self.uv_map = torch.zeros((1, self.vertices.shape[1], 2)).cuda()
        
        self.implicit_override = False
        self.implicit = False
        if self.deformation_model == 'njf':
            print(f"Using {model_path} for load/saving NJF data.")
            self.init_njf(self.njf_path if hasattr(self, 'njf_path') else model_path)
        elif self.deformation_model == 'delta_explicit':
            self.init_delta_explicit_deformation()
        elif self.deformation_model == 'delta_implicit':
            self.implicit = True
            self.init_delta_implicit_deformation()
        elif self.deformation_model == 'custom':
            self.init_deformation_model_custom()
        else:
            raise NotImplementedError("Deformation model can only njf or custom")
        
        if not skip_texture_init:
            self.set_texture(self.seed, texture = self.imgmesh_img if hasattr(self, 'imgmesh_img') else None)
        
        # Map xyz to rgb color
        vertex_features = self.vertices.clone()[0]
        vertex_features = (vertex_features - vertex_features.min(dim=0)[0]) / (vertex_features.max(dim=0)[0] - vertex_features.min(dim=0)[0])
        self.vertex_features = vertex_features[None].cuda()
        self.bg_features = torch.linspace(0,1, self.img_size[1])[None, None,...,None].repeat(1, self.img_size[0], 1, 3).cuda()
        self.feature_texture_map = self.bg_features

        self.writer = None
        if tensorboard_path is not None:
            self.init_tensorboard(tensorboard_path)

    def do_implicit_override(self):
        self.implicit_override = True
        if self.implicit_override:
            num_params = 0
            zero_pose = self.get_zero_pose()
            for param in zero_pose:
                num_params += param.numel()
            
            self.mlp = ParamNet(num_params).cuda()


    @abstractmethod
    def init_camera(self):
        raise NotImplementedError("init_camera not implemented")

    @abstractmethod
    def init_model(self, model_path):
        raise NotImplementedError("init_model not implemented")

    @abstractmethod
    def init_deformation_model_custom(self):
        raise NotImplementedError("init_deformation_model_custom not implemented")
    
    @abstractmethod
    def get_zero_pose_custom(self):
        raise NotImplementedError("get_zero_pose_custom not implemented")
    
    @abstractmethod
    def get_vertices_custom(self, params):
        raise NotImplementedError("get_vertices_custom not implemented")
    
    @abstractmethod
    def get_grad_for_pose_custom(self):
        raise NotImplementedError("get_grad_for_pose_custom not implemented")

    def init_tensorboard(self, tensorboard_path):
        layout = {
            "Fitting": {
                "temporal_smoothness_loss": ["Multiline", [f"temporal_smoothness_loss/{step}" for step in range(self.steps)]],
                "fidelity_loss": ["Multiline", [f"fidelity_loss/{step}" for step in range(self.steps)]],
                "render_loss": ["Multiline", [f"render_loss/{step}" for step in range(self.steps)]],
            },
        }

        self.tensorboard_path = tensorboard_path
        self.writer = SummaryWriter(tensorboard_path)
        self.writer.add_custom_scalars(layout)
        os.makedirs(tensorboard_path, exist_ok=True)
        os.chmod(tensorboard_path, 0o755)

    def init_njf(self, source_dir):
        # Only keep biggest component for NJF, otherwise it breaks, e.g. removes the eyes from FLAME
        vertices = o3d.utility.Vector3dVector(self.vertices.cpu().numpy()[0])
        triangles = o3d.utility.Vector3iVector(self.faces.cpu().numpy()[0])
        mesh = o3d.geometry.TriangleMesh(vertices, triangles)
        
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
        if len(cluster_n_triangles) > 1:
            max_i = np.argmax(cluster_n_triangles)
            visible_faces = np.where(np.array(triangle_clusters) == max_i)[0]
            vertices, faces, _ = self.prune_mesh(self.vertices[0].cpu().numpy(), self.faces[0].cpu().numpy(), visible_faces)

            vertices = vertices / vertices.max()
        else:
            vertices = self.vertices[0].cpu().numpy()
            faces = self.faces[0].cpu().numpy()

        self.jacobian_source = SourceMesh.SourceMesh(0, source_dir, {}, 1, ttype=torch.float)
        self.jacobian_source.load(source_v=vertices, source_f=faces)
        self.jacobian_source.to('cuda')
        self.vertices = torch.tensor(vertices)[None].cuda()
        self.faces = torch.tensor(faces)[None].cuda()
        with torch.no_grad():
            self.gt_jacobians = self.jacobian_source.jacobians_from_vertices(self.vertices)

        # Account fot NJF shift
        vertices = self.jacobian_source.vertices_from_jacobians(self.gt_jacobians)
        self.vertex_offset = (self.vertices - vertices).mean(dim=1, keepdim=True)

        self.transl = torch.zeros(1, 3).cuda()
        self.center = torch.zeros(1, 3).cuda()
        self.rot = torch.zeros(1, 3).cuda()

    def init_delta_explicit_deformation(self):
        self.delta_zero = torch.zeros(1, self.vertices.shape[1], 3).cuda()
        self.transl = torch.zeros(1, 3).cuda()
        self.rot = torch.zeros(1, 3).cuda()

    def init_delta_implicit_deformation(self):
        self.deform_net = DeformNet().cuda()
        self.delta_zero = torch.zeros(1, self.vertices.shape[1], 3).cuda()

    def prune_mesh(self, verts, faces,  visible_faces):
        new_faces = []
        new_verts = []
        v_map = [None] * len(verts)                                      
        for i, face_i in enumerate(visible_faces):
            vertices = faces[face_i]
            for vertex in vertices:
                if v_map[vertex] is None:
                    v_map[vertex] = len(new_verts)
                    new_verts.append(verts[vertex])

            new_face = torch.tensor([v_map[vertices[0]], v_map[vertices[1]], v_map[vertices[2]]])
            new_faces.append(new_face)

        return np.array(new_verts), np.array(new_faces), v_map

    def visibility_map(self, params=None, camera=None):
        # https://github.com/facebookresearch/pytorch3d/issues/126
        camera = self.camera if camera is None else camera
        params = self.get_zero_pose() if params is None else params
        vertices_og = self.get_vertices(params)

        # Get the output from rasterization
        _, _, meshes, fragments = self.render(params, camera, return_mesh=True)
        mesh = meshes[0]
        fragments = fragments[0]

        pix_to_face = fragments.pix_to_face  
        packed_faces = mesh.faces_packed() 
        packed_verts = mesh.verts_packed() 
        visible_faces = pix_to_face.unique()[1:]   # (num_visible_faces )

        new_faces, new_verts, v_map = self.prune_mesh(packed_verts.cpu().numpy(), packed_faces.cpu().numpy(),  visible_faces.cpu().numpy())                               

        return torch.tensor(new_faces).cuda(), torch.tensor(new_verts).cuda(), vertices_og, v_map
    
    def get_visibility(self, params=None, camera=None):
        # https://github.com/facebookresearch/pytorch3d/issues/126
        camera = self.camera if camera is None else camera
        params = self.get_zero_pose() if params is None else params

        # Get the output from rasterization
        _, _, meshes, fragments = self.render(params, camera, return_mesh=True)

        visibility = torch.zeros((len(meshes), len(meshes[0].verts_list()[0])), dtype=torch.bool).cuda()
        for i, (mesh, fragment) in enumerate(zip(meshes, fragments)):
            pix_to_face = fragment.pix_to_face  
            packed_faces = mesh.faces_packed() 
            visible_faces = pix_to_face.unique()[1:]   # (num_visible_faces )

            visible_vertices = packed_faces[visible_faces].unique()
            visibility[i, visible_vertices] = True
        
        return visibility
    
    @torch.no_grad()
    def set_texture(self, seed=2024, texture=None):
        print("Generating textures")
        pipe, controlnet = None, None
        generator = torch.manual_seed(seed)
        if texture is None:
            _, depth = self.render()
            bg_mask = depth == -1

            bg_depth = depth.max() + depth.max() / 2
            depth[torch.where(depth==-1)] = bg_depth
            depth = (1 - ((depth - depth.min()) / (depth.max() - depth.min()))) * 255
            depth = depth[0].detach().cpu().numpy().astype(np.uint8)
            depth = np.concatenate([depth, depth, depth], axis=2)

            if self.use_depth_process_custom:
                depth = self.depth_process_custom(depth)

            depth_pil = Image.fromarray(depth)
            depth_bg = Image.fromarray(np.zeros_like(depth))

            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()

            print("  Generating object image")
            face_pil = pipe(
                self.object_prompt,
                depth_pil, 
                num_inference_steps=100, 
                negative_prompt = self.object_negative_prompt,
                generator=generator
                ).images
            
            self.texture = torch.tensor(np.array(face_pil), dtype=torch.float32).cuda() / 255.0

            # Process texture map such that background of image is less likely to bleed into the object
            bg_mask = cv2.dilate(bg_mask[0].detach().cpu().numpy().astype(np.uint8), self.fg_kernel, iterations=1)
            img = self.texture[0].detach().mul(255).cpu().numpy().astype(np.uint8)
            self.texture = torch.tensor(cv2.inpaint(img, bg_mask, 3, cv2.INPAINT_TELEA), dtype=torch.float32)[None].cuda() / 255.0
        else:
            _, depth = self.render()
            bg_mask = depth == -1
            bg_mask = cv2.dilate(bg_mask[0].detach().cpu().numpy().astype(np.uint8), self.fg_kernel, iterations=1)
            self.texture = texture
            face_pil = [Image.fromarray((self.texture[0].detach().mul(255).cpu().numpy().astype(np.uint8)))]

        if not self.white_background:
            del pipe, controlnet
            controlnet = None
            torch.cuda.empty_cache()

            pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
            )
            pipe.enable_model_cpu_offload()
            
            face_pil = Image.fromarray(cv2.inpaint(np.array(face_pil[0]), 1-bg_mask, 3, cv2.INPAINT_TELEA))
            # Completely remove silhouette to avoid biasing the diffusion model
            fg_mask = (bg_mask == 0)
            fg_where = np.where(fg_mask)
            y_min, y_max = fg_where[0].min(), fg_where[0].max()
            x_min, x_max = fg_where[1].min(), fg_where[1].max()

            fg_mask[y_min:y_max, x_min:x_max] = 1
            fg_mask = (cv2.GaussianBlur(fg_mask.astype(np.float32), (31,31), 0) * 255).astype(np.uint8)
            mask_pil = Image.fromarray(((fg_mask)[...,None]).repeat(3,-1))
            
            print("  Inpainting background image")
            bg_pil = pipe(
                prompt=f"Background image of a {self.bg_prompt}",
                negative_prompt="Person, face, animal, object",
                image=face_pil, 
                mask_image=mask_pil, 
                generator=generator).images
            
            bg_pil[0] = torchvision.transforms.Resize((self.img_size))(bg_pil[0])
            self.bg = torch.tensor(np.array(bg_pil), dtype=torch.float32).cuda() / 255.0

        ## Sample vertex RGB from image
        camera = self.camera
        vertices = self.get_vertices()

        textures = TexturesUV(self.texture, faces_uvs=self.faces, verts_uvs=self.uv_map)
        meshes = Meshes(verts=[*vertices], faces=[*self.faces], textures=textures)

        fragments = self.rasterizer(meshes, cameras=camera)
        pix_to_face = fragments.pix_to_face  
        packed_faces = meshes.faces_packed()
        packed_verts = meshes.verts_packed()
        visible_faces = pix_to_face.unique()[1:]   # (num_visible_faces )

        visible_vertices = packed_faces[visible_faces].unique()
        visible_vertices_3D = packed_verts[visible_vertices]

        vertices_2D = camera.transform_points_screen(visible_vertices_3D)[:, :2] - 0.5 # Account for offset
        vertices_2D = vertices_2D / (torch.tensor(self.texture.shape[1:3], dtype=torch.float32).flip(-1).cuda() - 1)[None]
        vertices = vertices_2D.clamp(0, 1)
        vertices[:,1] = 1 - vertices[:,1]

        self.uv_map[0, visible_vertices] = vertices

        # Assign texture to invisible vertices based on nearest vertex based on euclidean distance instead of geodesic, simple but effective
        invisible_vertices = torch.tensor([i for i in range(self.vertices.shape[1]) if i not in visible_vertices]).cuda()
        invisible_vertices_3D = packed_verts[invisible_vertices]
        nn = (visible_vertices_3D[None] - invisible_vertices_3D[:, None]).norm(dim=-1).argmin(dim=1)
        self.uv_map[0, invisible_vertices] = vertices[nn]
        
        del pipe, controlnet
        torch.cuda.empty_cache()

    def get_vertices(self, params=None):
        params = self.get_zero_pose() if params is None else params

        all_vertices = []
        all_dx = []
        for i in range(len(params[0])): 
            if self.deformation_model == 'custom':
                vertices = self.get_vertices_custom([param[i][None] for param in params])
                all_vertices.append(vertices)

            elif self.deformation_model == 'njf':
                vertices = self.jacobian_source.vertices_from_jacobians(params[0][i][None])
                vertices = vertices + self.vertex_offset
                rot = roma.rotvec_to_rotmat(params[1][i][None])
                transl = 0.1 * params[2][i][None]
                center = params[3][i][None]
                all_vertices.append(torch.bmm(vertices - center, rot.transpose(1,2)) + center + transl)

            elif self.deformation_model == 'delta_explicit':
                vertices = self.vertices + params[0][i][None]
                rot = roma.rotvec_to_rotmat(params[1][i][None])
                transl = params[2][i][None]
                all_vertices.append(vertices)
            
            elif self.deformation_model == 'delta_implicit':
                dx = self.deform_net(self.vertices[0], torch.tensor([i / len(params[0])]).cuda().repeat(self.vertices.shape[1], 1))[None]
                all_dx.append(dx)
                vertices = self.vertices + dx
                all_vertices.append(vertices)
        
        if self.deformation_model == 'delta_implicit':
            self.dx = torch.cat(all_dx, dim=0)

        return torch.cat(all_vertices, dim=0)

    def get_zero_pose(self):
        if self.deformation_model == 'njf':
             return [self.gt_jacobians, self.rot, self.transl,  self.center]
        elif self.deformation_model == 'delta_explicit':
            return [self.delta_zero, self.rot, self.transl]
        elif self.deformation_model == 'delta_implicit':
            return [self.delta_zero]
        else:
            return self.get_zero_pose_custom()
        
    def get_pose_multipliers(self):
        if self.deformation_model == 'njf':
             return [1e-4, 1e-2, 1e-2, 1e-2]
        else:
            return self.get_pose_multipliers_custom()
        
    def get_grad_for_pose(self):
        if self.deformation_model == 'njf':
            return [True, True, True, True]
        elif self.deformation_model == 'delta_explicit':
            return [True, True, True]
        elif self.deformation_model == 'delta_implicit':
            return [False]
        else:
            return self.get_grad_for_pose_custom()
        
    def render(self, params=None, camera=None, return_mesh=False, untextured=False, no_background=False):
        camera = self.camera if camera is None else camera
        params = self.get_zero_pose() if params is None else params
        texture = self.grey_texture if untextured else self.texture
        all_vertices = self.get_vertices(params)

        imgs = []
        zbufs = []
        meshes = []
        fragments = []
        for i in range(len(params[0])):
            vertices = all_vertices[i][None]

            if type(self.renderer) == PointsRenderer:
                point_cloud = Pointclouds(points=vertices, features=self.vertex_features)
                img = self.renderer(point_cloud)
                fragment = self.rasterizer(point_cloud)
                zbuf = fragment.zbuf.max(dim=-1).values[...,None]
            else:
                textures = TexturesUV(texture, faces_uvs=self.faces, verts_uvs=self.uv_map)
                mesh = Meshes(verts=[*vertices], faces=[*self.faces], textures=textures)
                specular = untextured or self.skip_texture_init
                renderer = self.renderer if specular else self.renderer_diffuse
                img = renderer(mesh, cameras=camera)       
                fragment = self.rasterizer(mesh, cameras=camera)
                zbuf = fragment.zbuf
            
            if not no_background:
                alpha = (zbuf > -1).float()
                img = img[..., :3] * alpha + (1 - alpha) * self.bg
            else:
                img[...,-1][torch.where(img[...,-1] > 0)] = 1.

            imgs.append(img)
            zbufs.append(zbuf)
            if return_mesh:
                meshes.append(mesh)
                fragments.append(fragment)

        if return_mesh:
            return torch.cat(imgs, dim=0), torch.cat(zbufs, dim=0), meshes, fragments

        return torch.cat(imgs, dim=0), torch.cat(zbufs, dim=0)
    
    def latent_heuristic(self, _latents, model_i=0):
        latents = _latents.clone()
        mat = torch.zeros(len(latents), len(latents))
        for i in range(len(latents)):
            for j in range(len(latents)):
                if i == j:
                    continue
                mat[i,j] = torch.nn.CosineSimilarity(dim=-1)(latents[i], latents[j]).mean().item()

        latent_comparison = mat.mean(dim=0)
        outliers = latent_comparison <= 0.1
        mat[model_i] = False
        latent_swap = {} 
        if outliers.sum() > 0:
            print("Outliers detected")
            print(latent_comparison)
            print("Replace with nearest neighbour:")

            nn = (torch.where(outliers)[0][None] - torch.where(torch.logical_not(outliers))[0][:,None]).argmin(dim=0)

            for i, o in enumerate(torch.where(outliers)[0]):
                print(f"  Copying latent values at index {nn[i]} to {o}")
                latent_swap[nn[i]] = o
                latents[o] = latents[nn[i]]

        return latents, latent_swap, latent_comparison


    @torch.enable_grad()
    def fit_pose(self, step, latents, lr=0.0005, log_iter=100, verbose=True, model_i=0, iters=1001, pred_depth=None, use_heuristic=True, weights=None):
        time_start = time()
        if weights is None:
            if self.deformation_model == 'njf':
                weights=[10, 5e-2, 1e-2, 1e-2, 5e-1]
            else:
                weights=[10, 5e-2, 1e-2, 1e-2, 0]

        if type(model_i) != int:
            model_i = model_i.item()
 
        feature_cam = self.camera.clone()
        latents = latents.permute(0,2,3,1)
        
        latents = latents / latents.norm(dim=-1, keepdim=True)
        if use_heuristic:
            latents, latent_swap, latent_comparison = self.latent_heuristic(latents, model_i)
        else:
            latent_swap, latent_comparison = {}, torch.zeros(len(latents))
        latents_zero = latents[model_i][None]

        with torch.no_grad():
            self.set_vertex_features(latents_zero, camera=feature_cam)
            _, _, model_i_vertices, _ = self.render_vertex_features(camera=feature_cam)

        
            dist_per_vertex = feature_cam.get_world_to_view_transform().transform_points(model_i_vertices)[:, :, 2:]
            mean_dist = dist_per_vertex.mean()
            zero_delta_dist_per_vertex = (mean_dist - dist_per_vertex)


        cos = lambda x, y: 1 - torch.nn.CosineSimilarity(dim=-1)(x,y)

        
        req_grads = self.get_grad_for_pose()
        params_zero = self.get_zero_pose()
        num_optim_params = len([req_grad for req_grad in req_grads if req_grad])
        if self.implicit:
            optimizer = torch.optim.Adam(self.deform_net.parameters(), lr=lr)
            og_params = [[0] * self.num_frames] # Dummy list, so get_vertices works
        elif self.implicit_override:
            optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
            _og_params = [torch.nn.Parameter(param.clone().repeat(self.num_frames, * [1] * (param.ndim - 1)), requires_grad=req_grad) for param, req_grad in zip(params_zero, req_grads)]
        else:
            og_params = [torch.nn.Parameter(param.clone().repeat(self.num_frames, * [1] * (param.ndim - 1)), requires_grad=req_grad) for param, req_grad in zip(params_zero, req_grads)]
            optimizer = torch.optim.Adam(og_params, lr=lr)
        
        render_losses = []
        temporal_smoothness_losses = []
        fidetity_losses = []
        depth_losses = []
        zero_losses = []

        max_i = model_i + 1
        min_i = model_i

        increases = torch.linspace(0, iters * 0.5, (self.num_frames - 1)).round()
        mlp_input = torch.arange(self.num_frames).float().cuda() / self.num_frames
        for iter in range(iters):
            optimizer.zero_grad()

            if iter in increases:
                max_increased = False
                if max_i < self.num_frames:
                    max_i += 1
                    max_increased = True
                else:
                    min_i -= 1
            
            
            if self.implicit_override:
                param_pred = self.mlp(mlp_input[...,None])
                og_params = []
                _cum = 0

                for i, (param, req_grad) in enumerate(zip(_og_params, req_grads)):
                    if req_grad:
                        _param = params_zero[i].clone().detach() + 1e-2 * param_pred[:,_cum:_cum+param.shape[1:].numel()].view(len(param), *params_zero[i].shape[1:])
                        og_params.append(_param)
                        _cum += param.shape[1]
                    else:
                        og_params.append(param)

            else:
                if iter in increases:
                    # Copy over parameters from previous frame
                    with torch.no_grad():
                        for i in range(len(params_zero)):
                            if not req_grads[i]:
                                continue
                            if max_increased:
                                og_params[i][max_i-1] = og_params[i][max_i-2]
                            else:
                                og_params[i][min_i] = og_params[i][min_i+1]

            params = [param[min_i:max_i] for param in og_params]
            
            latents_render, depth_render, vertices_render, visibility_render = self.render_vertex_features(camera=feature_cam, params=params, set_only=True)
            

            # Depth Reg 
            dist_per_vertex = feature_cam.get_world_to_view_transform().transform_points(vertices_render)[:, :, 2:]
            mean_dist = dist_per_vertex.mean(dim=1)
            delta_dist_per_vertex = (mean_dist[:,None] - dist_per_vertex)

            depth_loss = weights[3] * (delta_dist_per_vertex - zero_delta_dist_per_vertex).abs().mean()
            depth_losses.append(depth_loss.item())


            # Render loss
            mask_render = depth_render >= 0

            latents_masked = latents[min_i:max_i][:,None].clone().repeat(1, self.faces_per_pixel_features, 1, 1, 1)
            # Overwrite features with background if occluded from previous layer
            fg_mask = mask_render[:,0]
            for k in range(1, self.faces_per_pixel_features):
                latents_masked[:,k][fg_mask[...,0]] = self.bg_features_keep.expand(len(fg_mask),-1,-1,-1)[fg_mask[...,0]]

            alignment = cos(latents_render, latents_masked)
            render_loss = weights[0] * alignment.mean()
            render_losses.append(render_loss.item())

            zero_losses.append(0)


            # Temporal smoothness loss
            temporal_smoothness_loss = 0
            if self.implicit:
                temporal_smoothness_loss = weights[1] * (self.dx[1:] - self.dx[:-1]).abs().mean()
            else:
                for i in range(len(params_zero)):
                    if not req_grads[i]:
                        continue
                    temporal_smoothness_loss += (params[i][1:] - params[i][:-1]).abs().mean()
                temporal_smoothness_loss = weights[1] * temporal_smoothness_loss / num_optim_params
            temporal_smoothness_losses.append(temporal_smoothness_loss.item())
            
            # Fidelity loss
            fidetity_loss = 0
            _fidetity_loss = 0
            if self.deformation_model == 'njf':
                diff = params[0] - torch.eye(3, 3).cuda()
                _fidetity_loss = weights[4] * ((diff).pow(2).mean() + (diff).abs().mean()) / 2

            for i in range(len(params_zero)):
                if not req_grads[i] or (self.deformation_model == 'njf' and i == 0):
                    continue
                fidetity_loss += (params[i]).abs().mean()
            fidetity_loss = weights[2] * fidetity_loss / (num_optim_params - 1)
            fidetity_loss += _fidetity_loss
            fidetity_losses.append(fidetity_loss.item())


            loss = render_loss + temporal_smoothness_loss + fidetity_loss + depth_loss

            loss.backward()
            optimizer.step()

            if iter % log_iter == 0:
                mean_temporal_smoothness_loss = np.mean(temporal_smoothness_losses)
                mean_render_loss = np.mean(render_losses)
                mean_fidetity_loss = np.mean(fidetity_losses)
                mean_depth_loss = np.mean(depth_losses)
                mean_zero_loss = np.mean(zero_losses)
                
                render_losses = []
                temporal_smoothness_losses = []
                fidetity_losses = []
                depth_losses = []
                zero_losses = []
                if self.writer is not None:
                    self.writer.add_scalar(f"temporal_smoothness_loss/{step}", mean_temporal_smoothness_loss, iter)
                    self.writer.add_scalar(f"render_loss/{step}", mean_render_loss, iter)
                    self.writer.add_scalar(f"fidelity_loss/{step}", mean_fidetity_loss, iter)
                    self.writer.add_scalar(f"depth_loss/{step}", mean_depth_loss, iter)
                    self.writer.add_scalar(f"zero_loss/{step}", mean_zero_loss, iter)


                    with torch.no_grad():
                        video, _ = self.render(og_params)
                        video = video.permute(0,3,1,2)[None]
                    
                    depth_render[torch.logical_not(mask_render)] = depth_render.max() + depth_render.max() / 10
 
                    self.writer.add_video(f"videos/step:{step}", video, global_step=iter, fps=2)
                    self.writer.add_video(f"depth/step:{step}", (depth_render  / depth_render.max()).repeat(1,3,1,1,1)[None,...,0], global_step=iter, fps=2)

                    for k in range(mask_render.shape[1]):
                        _m = mask_render[:,k].permute(0,3,1,2).float().repeat(1,3,1,1)[None]
                        self.writer.add_video(f"mask_render_{k}/step:{step}", _m, global_step=iter, fps=2)
                               
                if verbose:
                    print(f"Step {step} Iter {iter}: Render: {mean_render_loss}, Temporal: {mean_temporal_smoothness_loss}, Fidelity: {mean_fidetity_loss} depth: {mean_depth_loss} zero: {mean_zero_loss} range: {min_i}:{max_i}")
        

        time_end = time()
        time_fitting = time_end - time_start
        print(f"Fitting {step} took {time_fitting} seconds")

        with torch.no_grad():
            latents_render, depth_render, _, _ = self.render_vertex_features(camera=feature_cam, params=og_params)
            latents_render = latents_render[:,0]
            mask_render = depth_render[:,0,...,0] >= 0

        fg_mask = mask_render[:,0]
        fitting_errors = []
        for k in range(0, len(mask_render)):
            fitting_errors.append(cos(latents_render[k][mask_render[k]], latents[k][mask_render[k]]).mean().item())

        out = {}
        out['fitting_errors'] = fitting_errors
        out['time_fitting'] = time_fitting
        out['latent_swap'] = latent_swap
        out['latent_comparison'] = latent_comparison
        out['params'] = og_params

        with torch.no_grad():
            # Render standard setup
            video, _ = self.render(params)
            out['full'] = video.permute(3,0,1,2)[None]

            # Render no background, with texture, standard camera
            video, _ = self.render(params, no_background=True)
            out['no_bg'] = video.permute(3,0,1,2)[None]

            # Render experiment image
            videos = self.render_experiment_image(params)
            out['no_tex_view1'] = videos[0].permute(3,0,1,2)[None]
            out['no_tex_view2'] = videos[1].permute(3,0,1,2)[None]

        del optimizer
        if self.implicit_override:
            try:
                del self.mlp
            except:
                pass
            self.mlp = None
            self.implicit_override = False

        torch.cuda.empty_cache()
        return out

    def render_experiment_image(self, params=None, all_vertices=None, all_faces=None):
        params = self.get_zero_pose() if params is None else params        
        
        _all_imgs = []
        if all_vertices is None:
            all_vertices = self.get_vertices(params)
        for camera in [self.camera, self.camera_experiment]:
            all_imgs = []
            for i in range(len(all_vertices)):
                vertices = all_vertices[i][None]
                if all_faces is None:
                    faces = self.faces
                else:
                    faces = all_faces[i][None]
                mesh = Meshes(verts=[*vertices], faces=[*faces])
                mesh._compute_vertex_normals()
                fragments = self.rasterizer(mesh, cameras=camera)
                vertex_features = mesh.verts_normals_list()[0][None]

                pix_to_face = fragments.pix_to_face
                bary_coords = fragments.bary_coords
                
                mask = pix_to_face >= 0
                mask = mask.squeeze(-1)
                
                feat_dim = vertex_features.shape[-1]
                img = torch.zeros((self.faces.shape[0], self.img_size[0], self.img_size[1], feat_dim), dtype=torch.float32, device='cuda')

                vertices_b = self.faces[0][pix_to_face[0][mask[0]]]
                length = len(vertices_b)
                vertices_b = vertices_b.view(-1)

                feature_b = vertex_features[0][vertices_b].view(length, 3, vertex_features.shape[-1])

                blended_features = (feature_b * bary_coords[0][mask[0]].permute(0,2,1)).sum(dim=1)
                img[0][mask[0]] = blended_features
                alignment = (-torch.nn.functional.cosine_similarity(img[0], (camera.R @ torch.tensor([[0,0,1.]]).T.cuda()).permute(0,2,1), dim=-1)).clamp(0, 1)[...,None]
                img = alignment * torch.tensor([[[0.65, 0.65, 0.65]]]).cuda()
                img[torch.logical_not(mask[0])] = 1.
                img = torch.concat([img, mask.float().permute(1,2,0)], dim=-1)

                all_imgs.append(img[None])
            _all_imgs.append(torch.cat(all_imgs, dim=0))
        return _all_imgs

    def render_vertex_features(self, params=None, camera=None, set_only=False):
        camera = self.camera if camera is None else camera
        params = self.get_zero_pose() if params is None else params        

        all_imgs = []
        all_zbuffs = []
        all_vertices = self.get_vertices(params)
        # visibility = torch.zeros((len(params[0]), self.vertices.shape[1]), dtype=torch.bool).cuda()
        for i in range(len(params[0])):
            vertices = all_vertices[i][None]
            if set_only:
                faces = self.set_faces
            else:
                faces = self.faces

            mesh = Meshes(verts=[*vertices], faces=[*faces])
            fragments = self.feature_rasterizer(mesh, cameras=camera)

            imgs_layer = []
            zbuffs_layer = []  
            for j in range(self.faces_per_pixel_features):
        
                pix_to_face = fragments.pix_to_face[...,j,None]
                zbuff = fragments.zbuf[...,j,None]
                bary_coords = fragments.bary_coords[...,j,None,:]
                
                mask = pix_to_face >= 0
                alpha = mask.float()
                mask = mask.squeeze(-1)
                
                feat_dim = self.vertex_features.shape[-1]
                img = torch.zeros((faces.shape[0], self.feature_render_size[0], self.feature_render_size[1], feat_dim), dtype=torch.float32, device='cuda')

                vertices_b = faces[0][pix_to_face[0][mask[0]]]
                length = len(vertices_b)
                vertices_b = vertices_b.view(-1)

                feature_b = self.vertex_features[0][vertices_b].view(length, 3, self.vertex_features.shape[-1])

                blended_features = (feature_b * bary_coords[0][mask[0]].permute(0,2,1)).sum(dim=1)
                img[0][mask[0]] = blended_features

                bg_features = torchvision.transforms.Resize((self.feature_render_size[0], self.feature_render_size[1]))(self.bg_features.permute(0,3,1,2)).permute(0,2,3,1)
                
                img = img * alpha + (1 - alpha) * bg_features

                # Downscale instead of rendering at low resolutions, Pytorch3D gets weird at small resolutions here
                if self.feature_size != self.feature_render_size:
                    img = torchvision.transforms.Resize((self.feature_size[0], self.feature_size[1]))(img.permute(0,3,1,2)).permute(0,2,3,1)
                    zbuff = torchvision.transforms.Resize((self.feature_size[0], self.feature_size[1]))(zbuff.permute(0,3,1,2)).permute(0,2,3,1)
                    zbuff[zbuff < 0] = -1

                imgs_layer.append(img)
                zbuffs_layer.append(zbuff)

            all_imgs.append(torch.cat(imgs_layer, dim=0)[None])
            all_zbuffs.append(torch.cat(zbuffs_layer, dim=0)[None])

        all_imgs = torch.cat(all_imgs, dim=0)
        all_zbuffs = torch.cat(all_zbuffs, dim=0)            
        return all_imgs, all_zbuffs, all_vertices, None

    def set_vertex_features(self, images, camera=None, params=None):
        camera = self.camera if camera is None else camera
        assert len(images) == len(camera), "Number of images and cameras must be equal"
        params = self.get_zero_pose() if params is None else params

        self.feature_texture_map = images

        all_visible_vertices = []
        all_visible_features = []
        all_visible_background = []
        
        for i in range(len(camera)):
            cam = camera[i].clone()
            cam.image_size = tuple(self.img_size)

            vertices_og = self.get_vertices(params)
            meshes = Meshes(verts=[*vertices_og], faces=[*self.faces])
        
            fragments = self.rasterizer(meshes, cameras=cam)
            pix_to_face = fragments.pix_to_face 
           
            packed_faces = meshes.faces_packed()
            packed_verts = meshes.verts_packed()
            visible_faces = pix_to_face.unique()[1:]   # (num_visible_faces )

            visible_vertices = packed_faces[visible_faces].unique()
            visible_vertices_3D = packed_verts[visible_vertices]

            vertices_2D = cam.transform_points_screen(visible_vertices_3D)[:, :2] - 0.5 # Account for offset
            vertices_2D = vertices_2D / (torch.tensor(self.img_size, dtype=torch.float32).cuda().flip(-1) - 1)[None]
            vertices = vertices_2D * 2 - 1

            features = torch.nn.functional.grid_sample(
                images[i][None].permute(0,3,1,2), 
                vertices[None, None].to(images.dtype), 
                mode='bilinear', 
                align_corners=True)
            features = features.squeeze([0,2]).permute(1,0)
            all_visible_vertices.append(visible_vertices)
            all_visible_features.append(features)

            # Create background features (fill fg with mean)
            bg = torchvision.transforms.Resize((self.img_size[0], self.img_size[1]))(images[i][None].permute(0,3,1,2)).permute(0,2,3,1).clone()[0]
            fg_mask = fragments.zbuf > -1
            bg[fg_mask[0,...,0]] = bg.mean()
            all_visible_background.append(bg[None])         
        
        verts_features = torch.zeros(len(vertices_og[0]), images.shape[-1]).cuda()
        verts_count = torch.zeros(len(vertices_og[0])).cuda()
        for i in range(len(camera)):
            vertices = all_visible_vertices[i].cuda()
            features = all_visible_features[i].cuda()
            verts_features[vertices] += features
            verts_count[vertices] += 1
        visible_vertices = verts_count > 0
        verts_features[visible_vertices] /= verts_count[visible_vertices][...,None]

        self.vertex_features = verts_features[None]
        self.globally_visible_vertices = visible_vertices[None]
        self.bg_features = torch.cat(all_visible_background, dim=0).cuda().mean(dim=0)[None]
        self.bg_features = torchvision.transforms.Resize((self.feature_render_size[0], self.feature_render_size[1]))(self.bg_features.permute(0,3,1,2)).permute(0,2,3,1)
        self.bg_features_keep = self.bg_features.clone()

        non_visible_vertices = torch.where(torch.logical_not(self.globally_visible_vertices[0]))[0]
        new_faces = []
        for face in self.faces[0]:
            if torch.any(torch.isin(face, non_visible_vertices)):
                continue
            new_faces.append(face)
        self.set_faces = torch.stack(new_faces)[None]

        return self.vertex_features, self.globally_visible_vertices