# MotionDreamer: Exploring Semantic Video Diffusion features for Zero-Shot 3D Mesh Animation

![Header animation](./static/imgs/header.gif)


### 3DV 2025

#### [🌐 Project Page](https://lukas.uzolas.com/MotionDreamer/) | [📝 Paper](https://arxiv.org/abs/2405.20155) 

**[Lukas Uzolas](https://lukas.uzolas.com/), 
[Elmar Eisemann](https://graphics.tudelft.nl/~eisemann/),
[Petr Kellnhofer](https://kellnhofer.xyz/)**
<br>
[Delft University of Technology](https://graphics.tudelft.nl/)
<br>

Animation techniques bring digital 3D worlds and characters to life. However, manual animation is tedious and automated techniques are often specialized to narrow shape classes. In our work, we propose a technique for automatic re-animation of arbitrary 3D shapes based on a motion prior extracted from a video diffusion model. Unlike existing 4D generation methods, we focus solely on the motion, and we leverage an explicit mesh-based representation compatible with existing computer-graphics pipelines. Furthermore, our utilization of diffusion features enhances accuracy of our motion fitting. We analyze efficacy of these features for animation fitting and we experimentally validate our approach for two different diffusion models and four animation models. Finally, we demonstrate that our time-efficient zero-shot method achieves a superior performance re-animating a diverse set of 3D shapes when compared to existing techniques in a user study.


## Setup
1. Set Up Environment
    1. ```conda env create -f environment.yaml```
    2. ```conda activate motiondreamer```
    3. ```pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"```
        * For more information consult [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
    4. ```pip install torch-scatter torch-sparse``` (Needed for NJF)
        - Fore more information consult [Neural Jacobian Fields](https://github.com/ThibaultGROUEIX/NeuralJacobianFields)
2. Get DynamiCrafter weights and put into "./DynamiCrafter/checkpoints/dynamicrafter_1024_v1/model.ckpt"
    * Consult the [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter) repository
3. Obtain the [FLAME](https://flame.is.tue.mpg.de/) model
4. Obtain the [SMPL](https://smpl.is.tue.mpg.de/) model
5. ```git submodule update --init --recursive```, needed for SMALIFY to 

## Run
Check the ```examples.sh``` for examples of how to generate animations for the different models.


## License

This project is licensed under the GNU General Public License v3.0 (see [LICENSE.txt](./LICENSE.txt)).

### Exceptions

Some files use other licenses, as noted below and within their file headers:

- `modified_i2vgen_xl.py`: Apache License 2.0 — [Link to license](https://www.apache.org/licenses/LICENSE-2.0)
- `modified_dynamicrafter.py`: Apache License 2.0 — [Link to license](https://www.apache.org/licenses/LICENSE-2.0)
- `config.py`: Licensed under the MIT License  — [Link to license](https://opensource.org/licenses/MIT)
- `objs/bunny/data.obj`: Stanford Scan License  — [Link to license](https://graphics.stanford.edu/data/3Dscanrep/)
- `objs/lego_truck/data.obj`: Licensed under Creative Commons Attribution 3.0 Unported (CC BY 3.0) — [Link to license](https://creativecommons.org/licenses/by/3.0/)

Please refer to each file’s header for specific licensing and copyright details.


## Citation

```
@misc{uzolas2024motiondreamerexploringsemanticvideo,
      title={MotionDreamer: Exploring Semantic Video Diffusion features for Zero-Shot 3D Mesh Animation}, 
      author={Lukas Uzolas and Elmar Eisemann and Petr Kellnhofer},
      year={2024},
      eprint={2405.20155},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.20155}, 
}
```
