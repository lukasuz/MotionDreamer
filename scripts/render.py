import torch
from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer, PointLights, AmbientLights,
    SoftPhongShader, MeshRenderer,
    FoVPerspectiveCameras, look_at_view_transform, PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor)

import os
import re
import json

def get_simple_rasterizer(img_size=512, faces_per_pixel=1, cull_backfaces=False, blur_radius=0):
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=blur_radius, 
        faces_per_pixel=faces_per_pixel,
        cull_backfaces=cull_backfaces ,
        bin_size = None,
    )
    return MeshRasterizer(raster_settings=raster_settings)

def get_simple_renderer(img_size=512, device='cuda', type='point', location=None):
    location = [0., 1., 4.0] if location is None else location
    if type == 'point':
        lights = PointLights(device=device, location=[location])
    else:
        lights = AmbientLights(device=device)
    rasterizer = get_simple_rasterizer(img_size=img_size)
    shader = SoftPhongShader(device=device, lights=lights)
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )

    return renderer, rasterizer

def get_point_renderer(cameras, img_size=512, radius=0.009, points_per_pixel=10, device='cuda'):
    raster_settings = PointsRasterizationSettings(
        image_size=img_size, 
        radius = radius,
        points_per_pixel = points_per_pixel
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    return renderer, rasterizer

def random_camera(num, dist_range, elev_range, azim_range, device='cuda'):
    dist = torch.rand(num) * (dist_range[1] - dist_range[0]) + dist_range[0]
    elev = torch.rand(num) * (elev_range[1] - elev_range[0]) + elev_range[0]
    azim = torch.rand(num) * (azim_range[1] - azim_range[0]) + azim_range[0]
    R, T = look_at_view_transform(dist, elev, azim)
    return FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.1, zfar=1000)

def sample_circular_cams(num, dist, elev, azim_offset=0, y_offset=[0,0,0], device='cuda'):
    azim = torch.linspace(0, 360, num + 1)[:-1]
    azim = (azim + azim_offset) % 360
    dist = torch.tensor([dist] * num).float()
    elev = torch.tensor([elev] * num).float()
    R, T = look_at_view_transform(dist, elev, azim)
    T = T + torch.tensor(y_offset)[None]
    return FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.1, zfar=1000)

def uniform_sample_mesh(num, pos, face):
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/sample_points.html#SamplePoints
    assert pos is not None
    assert face is not None

    assert pos.size(1) == 3 and face.size(0) == 3

    pos_max = pos.abs().max()
    pos = pos / pos_max

    area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
    area = area.norm(p=2, dim=1).abs() / 2

    prob = area / area.sum()
    sample = torch.multinomial(prob, num, replacement=True)
    face = face[:, sample]

    frac = torch.rand(num, 2, device=pos.device)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    vec1 = pos[face[1]] - pos[face[0]]
    vec2 = pos[face[2]] - pos[face[0]]

    pos_sampled = pos[face[0]]
    pos_sampled += frac[:, :1] * vec1
    pos_sampled += frac[:, 1:] * vec2

    pos_sampled = pos_sampled * pos_max

    return pos_sampled