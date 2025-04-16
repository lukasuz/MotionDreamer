#!/bin/bash
python run.py --name obj:./objs/bunny/ -l 2 -f 16 --frames 20 -s 2 --steps 50 -p "A bunny shaking its ears" --guidance 6 --deformation_model njf --model VC --iters 20

## Requires download of DynamiCrafter weights. See README.txt.
python run.py --name obj:./objs/lego_truck/ -l 2 -f 16 --frames 20 -s 2 --steps 50 -p "A yellow bulldozer moving its blade up and down" --guidance 6 --deformation_model njf --model DC --iters 40
python run.py --name smal:2 -p "A horse walking" --guidance 6 --deformation_model custom --model DC --iters 40

## Requires you to download the FLAME and SMPl model. See README.txt.
# Add the path to the  folder containing the .pkl. not only the pkl itself, as it contains multiple important files
python run.py --name flame:<PATH_TO_FOLDER_CONTAINING_flame2023_no_jaw.pkl> -p "A person laughing" --guidance 6 --deformation_model custom --model VC --iters 20
# Just add the path to the .pkl directly for SMPL
python run.py --name smpl:<PATH_TO_SMPL_MALE.pkl> -s 2 -p "A person laughing" --guidance 6 --deformation_model custom --model DC --iters 40

