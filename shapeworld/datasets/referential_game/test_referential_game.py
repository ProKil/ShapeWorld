from io import BytesIO
import os
import pickle
import sys

import numpy as np
from shapeworld.world.world import World
from shapeworld.datasets.referential_game.referential_game import ReferentialGameDataset
from shapeworld.util import draw

if __name__ == '__main__':
    dataset = ReferentialGameDataset(world_size=224)
    generated = dataset.generate(1000000)
    worlds = generated['world']
    world_models = generated['world_model']
    target_ids = generated['target_id']
    captions = generated["caption"]
    generated.pop("world")
    directory = "/data/hzhu2/referential-game/train/"
    # directory = "./"

    with open(os.path.join(directory, f"generated_1M_vol{sys.argv[1]}.pkl"), "wb") as f:
        pickle.dump(generated, f)
    
    # for index in range(20000):
    #     filename = 'world_{}.png'.format(index)
    #     image_bytes = BytesIO()
    #     center = list(world_models[index]["entities"][target_ids[index]]["center"].values())
    #     world_array = np.transpose(worlds[index], (1, 0, 2))
    #     World.get_image(
    #         world_array=draw(
    #             world_array,
    #             shape="circle",
    #             border_color=np.ones(3),
    #             center=center,
    #             border_weight=5,
    #             size=20)).save(image_bytes, format='png')
    #     with open(os.path.join(directory, filename), 'wb') as filehandle:
    #         filehandle.write(image_bytes.getvalue())
    #     image_bytes.close()
    
    # with open(os.path.join(directory, "captions.txt"), "w") as f:
    #     for caption in captions:
    #         f.write(f"{caption}\n")

    # with open(os.path.join(directory, "worlds.pkl"), "wb") as f:
    #     pickle.dump(world_models, f)