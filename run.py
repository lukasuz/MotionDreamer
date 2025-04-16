import argparse
import torch
import numpy as np
import os
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
from datetime import datetime
from PIL import Image
import random
import torchvision
import imageio

# Quick so that SMAL and SMPL work with newer versions of numpy
if not hasattr(np, "bool"):
    np.bool = np.bool_
    np.int = np.int_
    np.float = np.float_
    np.complex = np.complex_
    np.object = np.object_
    np.unicode = np.unicode_
    np.str = np.str_

from scripts.flame_mesh import FLAMEMesh
from scripts.obj_mesh import OBJMesh
from scripts.smal_mesh import SMALMesh
from scripts.smpl_mesh import SMPLMesh

from modified_i2vgen_xl import I2VGenXLPipeline
from modified_dynamicrafter import get_DC_args, run_inference, init_model

def save_video(tensor, path, fps, loop=True):
    if tensor.max() <= 1:
        tensor = tensor.mul(255)
    video = tensor.cpu().numpy().astype(np.uint8)
    # video_writer = imageio.get_writer(path, mode='I', fps=fps, codec='libx264', bitrate='1M')
    video_writer = imageio.get_writer(path, mode='I', fps=fps, codec='h264', quality=9)
    for j in range(len(video)):
        video_writer.append_data(video[j])
    if loop:
        for j in range(len(video)-1,-1,-1):
            video_writer.append_data(video[j])
    video_writer.close()

def _callback(pipe, i, kwargs, path='', fps=7, img_size=(1024, 576), activation=None, model=None, model_i=0, iters=[], raise_on_last_iter=False, coords=None, matching_size=(128, 72), read_and_delete=False, weights=None):
    if read_and_delete:
        features = torch.load(f"{path}/features_{i}.pt")
    else:
        features = activation['out'] # [F, 640, 88, 160]
    features = features.to(torch.float32)

    if model is not None and i in iters:
        # Manually off-load DC to CPU during fitting
        if not 'diffusers' in str(type(pipe)):
            pipe.to('cpu')
            torch.cuda.empty_cache()

        if model_i is None:
            model_i = activation['current_model_i']
            with open(f"{path}/cfg.txt", "a") as f:
                f.write(f"current_model_i: {model_i}\n")

        with torch.enable_grad():
            model.do_implicit_override()
            out = model.fit_pose(i, features, model_i=model_i, iters=1001, weights=weights, use_heuristic=False)
            # del model
            torch.cuda.empty_cache()

        for key, value in out.items():
            if key in ['full', 'no_tex_view1', 'no_tex_view2', 'no_bg']:
                os.makedirs(f"{path}/{i}_{key}", exist_ok=True)
                for j in range(value.shape[2]):
                    img = value[0,:,j]
                    Image.fromarray(img.cpu().mul(255.).numpy().transpose(1,2,0).astype(np.uint8)).save(f'{path}/{i}_{key}/{j}.png')
            try:
                save_video(value[0].permute(1,2,3,0), f"{path}/{i}_{key}.mp4", fps)
            except:
                pass

        if read_and_delete:
            os.remove(f"{path}/features_{i}.pt")
                    
        if raise_on_last_iter and i == iters[-1]:
            raise ValueError("Last iteration reached")

        if not 'diffusers' in str(type(pipe)):
            torch.cuda.empty_cache()
            pipe.to('cuda')


    return kwargs


def save_final_video(frames, path, args):
    save_video(frames, f'{path}/diff_output.mp4', args.fps)
    try:
        mask_paths = [f"{path}/{args.iters[0]}_masks_{i}.png" for i in range(args.frames)]
        filter =  torchvision.transforms.GaussianBlur(9, sigma=3.0)
        for i, mask_path in enumerate(mask_paths):
            mask = load_image(mask_path)
            mask = (torch.tensor(np.array(mask)) == 0).all(dim=2)
            mask = mask[None,...].float().cuda()
            mask = filter(mask)[0,...,None]
            frames[i] = frames[i] * (1 - mask) + mask * 255
        save_video(frames, f'{path}/video_final_masked.mp4', args.fps)
        Image.fromarray(frames[-1].cpu().numpy()).save(f"{path}/masked_last.png")
    except:
        pass

def main(args):
    from tensorboard import program

    generator = torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model_i = None
    if args.model == 'SVD':
        img_size = (1024, 576)
        matching_size = (128, 72)
        white_background = True
        skip_texture_init = True

    elif args.model == 'VC':
        img_size = (1280, 704)
        matching_size = (160, 88)
        # img_size = (1024, 576) # (512, 320)
        # matching_size = (128, 72) # (64, 40)
        white_background = False
        skip_texture_init = False
    
    elif args.model == 'DC':
        if args.steps < 30:
            raise ValueError("DC outputs nans if too few steps")
        img_size = (1024, 576) # (512, 320)
        matching_size = (128, 72) # (64, 40)
        white_background = True
        skip_texture_init = False

    elif args.model == 'DC_small':
        if args.steps < 30:
            raise ValueError("DC outputs nans if too few steps")
        img_size = (512, 320)
        matching_size = (64, 40)
        white_background = True
        skip_texture_init = False
        args.model = 'DC'
    else:
        raise ValueError("Model must be 'SVD': Stable Video Diffusion, 'VC': VideoComposer or 'DC': DynamiCrafter")
    
    if args.white_background is not None:
        white_background = args.white_background

    if skip_texture_init is not None:
        skip_texture_init = args.skip_texture_init

    if "smpl:" in args.name:
        model_path = args.name.replace("smpl:", "")
        name = "smpl"
        model_type = SMPLMesh

    elif "flame:" in args.name:
        model_path = args.name.replace("flame:", "")
        name = "flame"
        model_type = FLAMEMesh
        
    elif "obj:" in args.name:
        model_path = args.name.replace("obj:", "")
        name = "obj"
        model_type = OBJMesh
    
    elif "smal:" in args.name:
        model_path = args.name.replace("smal:", "")
        name = "smal"
        model_type = SMALMesh

    else:
        raise ValueError("Name must start with 'smpl:', 'flame:', 'smal:' ,'obj:'")
    
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    path = f"outputs/{name}/{args.model}/{args.prompt.replace(',', '').replace(' ','_')}/{date_time}_{args.seed}"
    os.makedirs(path, exist_ok=True)

    print("Running experiment:")
    with open(f"{path}/cfg.txt", "a") as f:
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
            f.write(f"{key}: {value}\n")

    model = model_type(
        model_path, 
        img_size[::-1],
        seed=args.seed,
        feature_size=matching_size[::-1],
        white_background=white_background,
        skip_texture_init=skip_texture_init,
        num_frames = args.frames,
        deformation_model = args.deformation_model
    )

    if args.save_mesh:
        import pymeshlab
        with torch.no_grad():
            vertices = model.get_vertices().detach().cpu().numpy()[0]
        m = pymeshlab.Mesh(
            vertices, 
            model.faces.detach().cpu().numpy()[0])
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m)
        ms.save_current_mesh(os.path.join(path, 'mesh.obj'))

    if args.model == 'SVD':
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16").to('cuda')

    elif args.model == 'VC':
        pipe = I2VGenXLPipeline.from_pretrained(
            "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16").to('cuda')
    
    elif args.model == 'DC':
        dc_args = get_DC_args(img_size=img_size, steps=args.steps, fps=args.fps, guidance=args.guidance, frames=args.frames, seed=args.seed)
        pipe = init_model(dc_args, gpu_no = 0)

    coords = None
    # coords = model.get_keypoints()
    # coords = coords.detach().cpu().numpy()[0]

    ## Render initial images
    with torch.no_grad():
        # Input image in experiment configuration for diffusion model
        _image, _ = model.render()
        image = _image[0,...,:3].cpu().numpy()
        image = Image.fromarray((image * 255).astype(np.uint8))
        image.save(f"{path}/img_input.png")

        # Img without background and no texture
        # no_bg_no_tex, _ = model.render(no_background=True, untextured=True)
        # no_bg_no_tex = Image.fromarray((no_bg_no_tex[0] * 255).cpu().numpy().astype(np.uint8))
        # no_bg_no_tex.save(f"{path}/img_blank.png")

        # Img without background and texture (if specified)
        no_bg_tex, _ = model.render(no_background=True, untextured=False)
        no_bg_tex = Image.fromarray((no_bg_tex[0] * 255).cpu().numpy().astype(np.uint8))
        no_bg_tex.save(f"{path}/img_input_no_bg.png")

        # Save background alone
        bg = Image.fromarray((model.bg[0] * 255).cpu().numpy().astype(np.uint8))
        bg.save(f"{path}/bg.png")

    if args.tensorboard:
        model.init_tensorboard(f"{path}/tensorboard")
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', f"{path}/tensorboard"])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")

    activation = {}
    if args.model == 'SVD' or args.model == 'VC':
        pipe.enable_model_cpu_offload()
        # Create hook to get activations of necessary block
        def get_activation(name, cond=True):
            def hook(model, input, output):
                noise_pred_uncond, noise_pred_cond = output.chunk(2)
                noise_pred = noise_pred_cond if cond else noise_pred_uncond
                activation[name] = noise_pred.detach()
            return hook
        
        pipe.unet.up_blocks[args.level].register_forward_hook(get_activation('out'))

        model_i = 0
    else:
        model_i = None

    # Create callback
    callback = lambda pipe, i, t, kwargs: _callback(
        pipe, i, kwargs, path=path, fps=args.fps, img_size=img_size,
        activation=activation, model=model, model_i=model_i, iters=args.iters,
        coords=coords, matching_size=matching_size, raise_on_last_iter=args.early_stopping,
        weights=args.weights)
    
    if args.model == 'DC':
        callback = lambda _, i: _callback(
            pipe, i, None, path=path, fps=args.fps, img_size=img_size,
            activation=activation, model=model, model_i=model_i, iters=args.iters,
            coords=coords, matching_size=matching_size, raise_on_last_iter=args.early_stopping,
            weights=args.weights)
    

    try:
        if args.model == 'SVD':
            out = pipe(
                image, num_inference_steps=args.steps, decode_chunk_size=8, 
                generator=generator, #noise_aug_strength=0, 
                fps=args.fps, callback_on_step_end=callback,
                max_guidance_scale=args.guidance, num_frames=args.frames,
                callback_on_step_end_tensor_inputs=['latents']) # motion_bucket_id=motion,
            
            frames = [torch.tensor(np.array(frame)[None]).cuda() for frame in out.frames[0]]
            frames = torch.cat(frames, dim=0)

        elif args.model == 'VC':
            out = pipe(
                image=image, num_inference_steps=args.steps, decode_chunk_size=8, 
                generator=generator, target_fps=args.fps,
                guidance_scale=args.guidance,
                num_frames=args.frames,
                prompt=args.prompt,
                negative_prompt="Distorted, motionless, static, disfigured",
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=['latents'])

            frames = [torch.tensor(np.array(frame)[None]).cuda() for frame in out.frames[0]]
            frames = torch.cat(frames, dim=0)
        
        elif args.model == 'DC':
            model_i, dc_out, pipe = run_inference(dc_args, image, args.prompt, activation, callback=callback, level=args.level, pipe=pipe)

            video = torch.clamp(dc_out.squeeze().float(), -1., 1.).permute(1,0,2,3).cuda()
            _image = _image.permute(0, 3, 1, 2).float().cuda() * 2 - 1
            frames = (video.permute(0,2,3,1) + 1) / 2

            comparison_img = (torch.cat([video[model_i], _image[0]], dim=1) + 1) / 2
            save_final_video(frames, path, args)

            comparison_img = (comparison_img.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
            Image.fromarray(comparison_img).save(f"{path}/comparison.png")
            frames = frames * 255

    except ValueError as e:
        print(e)
        pass
    
    if not args.early_stopping:
        save_final_video(frames, path, args)
        os.makedirs(f'{path}/diff_output', exist_ok=True)
        for j in range(frames.shape[0]):
            img = frames[j]
            Image.fromarray(img.cpu().numpy().astype(np.uint8)).save(f'{path}/diff_output/final_{j}.png')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Image name', default='smpl1.png', type=str)
    parser.add_argument('--guidance', help='guidance scales', type=float, default=9)
    parser.add_argument('--model', help='SVD, VC, DC', type=str, default='VC')
    parser.add_argument('--steps', help='Num of inference steps', default=50, type=int)
    parser.add_argument('--frames', help='number of frames', default=16, type=int)
    parser.add_argument('--tensorboard', help='Logging', action='store_true')
    parser.add_argument('--early_stopping', help='Whether to stop after fitting', action='store_true')
    parser.add_argument('--save_mesh', help='wheter save the mesh', action='store_true')

    parser.add_argument('--deformation_model', default='custom', type=str)
    parser.add_argument('-p', '--prompt', default=' ', type=str)
    parser.add_argument('-f', '--fps', help='Generation fps', default=16, type=int)
    parser.add_argument('-l','--level',  help='Which Unet-levels to use', default=2, type=int)
    parser.add_argument('-s','--seed', help='Which random seed to use', default=1, type=int)
    parser.add_argument('-i', '--iters', help='Fitting iters', nargs='+', type=int, default=[])
    parser.add_argument('--weights', default=[5, 1e-1, 1e-2, 1e-2, 5e-1], type=list)

    parser.add_argument('--white_background', help='White background', default=None, type=bool)
    parser.add_argument('--skip_texture_init', help='Skip texture init', default=None, type=bool)
    
    args = parser.parse_args()

    main(args)