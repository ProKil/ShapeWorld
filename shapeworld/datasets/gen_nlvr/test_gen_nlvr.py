from io import BytesIO
import os

from typing import List

import numpy as np
from shapeworld.world.world import World
from shapeworld.datasets.gen_nlvr.gen_nlvr import GenNLVRDataset
import tqdm

if __name__ == '__main__':
    dataset = GenNLVRDataset(world_size=64,
                             world_colors=('white',),
                             entity_counts=(3, 4, 5),
                             shapes=('square', 'rectangle', 'triangle'))
    generated = dataset.generate(100000)
    world_models_l: List[World] = generated['world_model_l']
    world_models_r: List[World] = generated['world_model_r']
    worlds_l = generated['world_l']
    worlds_r = generated['world_r']
    captions = generated["caption"]
    directory = "/data/hzhu2/synthetic-nlvr-64"
    
    for index in tqdm.tqdm(range(100000)):
        filename_l = 'world_{}_l.png'.format(index)
        filename_r = 'world_{}_r.png'.format(index)
        image_bytes_l = BytesIO()
        image_bytes_r = BytesIO()
        world_array_l = worlds_l[index]
        world_array_r = worlds_r[index]
        World.get_image(world_array=world_array_l).save(image_bytes_l, format='png')
        World.get_image(world_array=world_array_r).save(image_bytes_r, format='png')
        with open(os.path.join(directory, filename_l), 'wb') as filehandle:
            filehandle.write(image_bytes_l.getvalue())
        with open(os.path.join(directory, filename_r), 'wb') as filehandle:
            filehandle.write(image_bytes_r.getvalue())
        image_bytes_l.close()
        image_bytes_r.close()
    
    with open(os.path.join(directory, "captions.txt"), "w") as f:
        for caption in captions:
            f.write(f"{caption}\n")

    with open(os.path.join(directory, "specs_l.txt"), "w") as f:
        for world_model_l in world_models_l:
            line = []
            for entity in world_model_l.entities:
                line.append(f"{entity.shape.name}:{entity.color.name}")
            f.write(" ".join(line) + "\n")
    
    with open(os.path.join(directory, "specs_r.txt"), "w") as f:
        for world_model_r in world_models_r:
            line = []
            for entity in world_model_r.entities:
                line.append(f"{entity.shape.name}:{entity.color.name}")
            f.write(" ".join(line) + "\n")
