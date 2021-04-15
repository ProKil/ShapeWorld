import random
from shapeworld.world.point import Point
from typing import Dict, Tuple

from shapeworld.generators import WorldGenerator

class MetaGenerator(WorldGenerator):
    def sample_range(self, left: float, right: float, length: float) -> Tuple[float, float]:
        if length < right - left:
            start = random.uniform(left, right - length)
            end = start + length
        else:
            start, end = left, right

        return start, end
    
    def generate_world(self, size_of_size_range: float=None, n_shapes: int=None, n_colors: int=None, size_of_location_range: float=None) -> Dict:
        world = dict()
        if size_of_size_range is not None:
            world["size_range"] = self.sample_range(*self.size_range, size_of_size_range)
        else:
            world["size_range"] = self.size_range

        if n_shapes is not None and n_shapes < len(self.shapes):
            world["shapes"] = random.sample(self.shapes, k=n_shapes)
        else:
            world["shapes"] = self.shapes

        if n_colors is not None and n_colors < len(self.colors):
            world["colors"] = random.sample(self.colors, k=n_colors)
        else:
            world["colors"] = self.colors

        if size_of_location_range is not None:
            x_left, x_right = self.sample_range(0, 1, size_of_location_range)
            y_left, y_right = self.sample_range(0, 1, size_of_location_range)
            world["location_range"] = {"topleft": Point(x_left, y_left),
                                       "bottomright": Point(x_right, y_right)}
        else:
            world["location_range"] = {"topleft": Point.zero,
                                       "bottomright": Point.one}
        
        return world