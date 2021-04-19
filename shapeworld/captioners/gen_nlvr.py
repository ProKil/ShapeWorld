from random import choice, random
from typing import List
from shapeworld.world.entity import Entity
from shapeworld.world.world import World
from shapeworld import util
# from shapeworld.captions.referential import Referential
# from shapeworld.captioners import WorldCaptioner


# Not inherenting WorldCaptioner for now
class GenNLVRCaptioner:
    def __init__(self):
        pass

    def __call__(self, world_l: World, world_r: World) -> List[str]:
        entities_l: List[Entity] = world_l.entities
        entities_r: List[Entity] = world_r.entities
        
        shapes = ["object"] + [i.shape.name for i in entities_l] + [i.shape.name for i in entities_r]
        shapes = set(shapes)

        target_shape = choice(list(shapes))
        if target_shape == "object":
            return f"There are {len(entities_l) + len(entities_r)} objects."
        else:
            n = sum(1 if i.shape.name == target_shape else 0 for i in entities_l + entities_r)
            return f"There are {n} {target_shape}."