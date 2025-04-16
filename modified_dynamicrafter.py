# This file is a modified version stemming from the original code of 
# "DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors"
# (https://github.com/Doubiiu/DynamiCrafter), which is licensed under the 
# Apache License 2.0. The original code's copyright remains with its 
# respective owners.
#
# Modifications by the CGV group, TU Delft.
#
# This file os licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0


import sys, os

sys.path.insert(1, os.path.join(sys.path[0], 'DynamiCrafter'))
from DynamiCrafter.scripts.evaluation.inference import *

@torch.no_grad()
def init_model(args, gpu_no = 0):
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
        
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    #model.half()
    model.eval()
    return model

@torch.no_grad()
def run_inference(args, img, prompt, activation, callback=None, level=2, pipe=None):
    """
    img must be pil image of shape (h,w,c)
    """
    seed_everything(args.seed) # DynamiCrafter breaks without manually using pytorch lightning seeding 
    if callback is None:
        callback = lambda img, i: 0

    gpu_no = 0
    ## model config
    config = OmegaConf.load(args.config)
    if pipe is None:
        model = init_model(args, gpu_no)
    else:
        model = pipe

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]


    ## prompt file setting
    video_size = (args.height, args.width)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    image_tensor = transform(img).unsqueeze(1) # [c,1,h,w]
    video = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=n_frames).cuda() #.to(torch.float16)

    ## Add callback
    def get_activation(name, _counter = [0]):
        def hook(model, input, output):
            if _counter[0] == 0: # skip unconditional latents (cfg)
                try:
                    batch_means = torch.nn.functional.cosine_similarity(activation[name], output).mean(dim=-1).mean(dim=-1)
                    mean = batch_means.mean()
                    std = batch_means.std()
                    batch_means = (batch_means - mean) / (std + 1e-8)
                    activation['similarities'] = activation['similarities'] + batch_means
                except:
                    activation['similarities'] = torch.zeros(len(output)).cuda()
                activation[name] = output.detach()
                # print(activation['similarities'], torch.argmax(activation['similarities']))
                activation['current_model_i'] = torch.argmax(activation['similarities'])
            _counter[0] = (_counter[0] + 1) % 2
        return hook
    
    # level = 2
    block_num = (1 + level) * 3 - 1
    model.model.diffusion_model.output_blocks[block_num].register_forward_hook(get_activation('out'))


    batch_samples = image_guided_synthesis(model, [prompt], video[None], noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                        args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, args.text_input, args.multiple_cond_cfg, \
                        args.loop, args.interp, args.timestep_spacing, args.guidance_rescale, img_callback=callback)
    
    return torch.argmax(activation['similarities']), batch_samples, model


def get_DC_args(img_size=(512, 320), steps=30, fps=20, guidance=7.5, frames=16, seed=1):
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    width = img_size[0]
    if width == 512:
        assert img_size[1] == 320

    elif width == 1024:
        assert img_size[1] == 576

    else:
        raise ValueError("Invalid image size")


    return Namespace(
        ckpt_path = f'DynamiCrafter/checkpoints/dynamicrafter_{width}_v1/model.ckpt', 
        config = f'DynamiCrafter/configs/inference_{width}_v1.0.yaml', 
        n_samples = 1, 
        ddim_steps = steps, 
        ddim_eta = 1.0, 
        bs = 1, 
        height = img_size[1], 
        width = img_size[0], 
        frame_stride = fps, 
        unconditional_guidance_scale = guidance, 
        seed = seed, 
        video_length = frames, 
        negative_prompt = False, 
        text_input = True, 
        multiple_cond_cfg = False, 
        cfg_img = None, 
        timestep_spacing = 'uniform_trailing', 
        guidance_rescale = 0.7, 
        perframe_ae = True, 
        loop = False, 
        interp = False)
