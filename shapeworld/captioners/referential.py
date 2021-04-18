from random import choice, random
from typing import List
from shapeworld.world.entity import Entity
from shapeworld.world.world import World
from shapeworld import util
# from shapeworld.captions.referential import Referential
# from shapeworld.captioners import WorldCaptioner


# Not inherenting WorldCaptioner for now
class ReferentialCaptioner:

    # caption syntax:
    # <size> <color> [shape] <loc>
    SIZE_DELIMITER = {(0, 0.2): "small", (0.2, 1.1): "big"}
    CENTER_BOUNDARY = (0.4, 0.6)

    def __init__(
        self,
        size_word_prob: float=0.3,
        color_word_prob: float=0.3,
        shape_word_prob: float=0.7,
        location_word_prob: float=0.3,
        absolute_location_prob: float=0.7
        ):
        self.size_word_prob = size_word_prob
        self.color_word_prob = color_word_prob
        self.shape_word_prob = shape_word_prob
        self.location_word_prob = location_word_prob
        self.absolute_location_prob = absolute_location_prob

    def realizer(
        self,
        *,
        size_word: str=None,
        color_word: str=None,
        shape_word: str="object",
        absolute_word: str=None,
        relative_word: str=None
        ) -> List[str]:
        caption = ['the']
        if relative_word is not None:
            caption.append(relative_word)
        if size_word is not None:
            caption.append(size_word)
        if color_word is not None:
            caption.append(color_word)
        caption.append(shape_word)
        if absolute_word is not None:
            caption.extend(["on", "the", absolute_word])
        return caption

    def __call__(self, world: World, target_id: int) -> List[str]:
        entities: List[Entity] = world.entities
        target_entity = entities[target_id]
        features = dict()

        # size feature
        if random() < self.size_word_prob:
            target_size = target_entity.shape.size
            size_word = None
            for i, j in self.__class__.SIZE_DELIMITER.items():
                if i[0] <= target_size < i[1]:
                    size_word = j
            features["size_word"] = size_word
        
        # color feature
        if random() < self.color_word_prob:
            target_color = target_entity.color
            features["color_word"] = target_color.name.lower()
        

        # shape feature
        if random() < self.shape_word_prob:
            target_shape = target_entity.shape
            features["shape_word"] = target_shape.name.lower()

        # location
        if random() < self.location_word_prob:
            location = target_entity.center
            if random() < self.absolute_location_prob:
                ## absolute location
                vertical = "top" if location[0] < 0.5 else "bottom"
                horizontal = "left" if location[1] < 0.5 else "right"
                absolute_choices = [vertical, horizontal, vertical+horizontal]
                if self.CENTER_BOUNDARY[0] <= location[0] < self.CENTER_BOUNDARY[1] and \
                self.CENTER_BOUNDARY[0] <= location[1] < self.CENTER_BOUNDARY[1]:
                    absolute_choices.append("center")
                features["absolute_word"] = choice(absolute_choices)
            else:
                ## relative location
                ### most
                relative_choices = list()
                other_locations = [i.center for i in entities]
                if all(location[0] <= i[0] for i in other_locations):
                    relative_choices.append('topmost')
                if all(location[0] >= i[0] for i in other_locations):
                    relative_choices.append('bottommost')
                if all(location[1] <= i[1] for i in other_locations):
                    relative_choices.append('leftmost')
                if all(location[1] >= i[1] for i in other_locations):
                    relative_choices.append('rightmost')
                try:
                    features["relative_word"] = choice(relative_choices)
                except IndexError:
                    features["relative_word"] = None

        return self.realizer(**features)