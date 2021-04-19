from random import randint

import tqdm
from shapeworld.captioners.gen_nlvr import GenNLVRCaptioner
from shapeworld.generators.meta_attributes import MetaGenerator
from shapeworld import Dataset
from shapeworld.generators import RandomAttributesGenerator

class GenNLVRDataset(Dataset):
    GENERATOR_INIT_FREQUENCY = 10

    def __init__(
        self,
        world_size=64,
        world_colors=('black',),
        shapes=('square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'),
        colors=('red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray'),
        textures=('solid',),
        rotation=True,
        size_range=(0.1, 0.25),
        distortion_range=(2.0, 3.0),
        shade_range=0.4,
        collision_tolerance=0.25,
        collision_shade_difference=0.5,
        boundary_tolerance=None,
        entity_counts=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=None,
        validation_entity_counts=None,
        test_entity_counts=None,
        validation_count_rate=0.5,
        test_count_rate=0.5,
        validation_combinations=None,
        test_combinations=None,
        validation_space_rate_range=(0.0, 1.0),
        test_space_rate_range=(0.0, 1.0),
        validation_combination_rate=0.5,
        test_combination_rate=0.5,
        caption_size=31,
        # vocabulary=('.', 'a', 'above', 'all', 'an', 'and', 'are', 'as', 'at', 'behind', 'below', 'besides', 'bigger', 'biggest', 'blue', 'but', 'circle', 'circles', 'closer', 'closest', 'color', 'cross', 'crosses', 'cyan', 'darker', 'darkest', 'different', 'eight', 'either', 'ellipse', 'ellipses', 'exactly', 'exists', 'farther', 'farthest', 'few', 'five', 'four', 'from', 'front', 'gray', 'green', 'half', 'if', 'in', 'is', 'least', 'left', 'leftmost', 'less', 'lighter', 'lightest', 'lower', 'lowermost', 'magenta', 'many', 'more', 'most', 'no', 'none', 'not', 'of', 'one', 'only', 'or', 'pentagon', 'pentagons', 'quarter', 'quarters', 'rectangle', 'rectangles', 'red', 'right', 'rightmost', 'same', 'semicircle', 'semicircles', 'seven', 'shape', 'shapes', 'six', 'smaller', 'smallest', 'square', 'squares', 'than', 'the', 'there', 'third', 'thirds', 'three', 'to', 'triangle', 'triangles', 'twice', 'two', 'upper', 'uppermost', 'yellow', 'zero'),
        pixel_noise_stddev=None,
    ):
        values = dict(world_l='world', world_model_l='model', 
                      world_r='world', world_model_r='model',
                      caption='language', caption_length='alternatives(int)')
        vectors = dict(caption=caption_size)
        super(GenNLVRDataset, self).__init__(values=values, world_size=world_size, vectors=vectors, pixel_noise_stddev=pixel_noise_stddev)
        
        world_generator = RandomAttributesGenerator(
            world_size=world_size,
            world_colors=world_colors,
            shapes=shapes,
            colors=colors,
            textures=textures,
            rotation=rotation,
            size_range=size_range,
            distortion_range=distortion_range,
            shade_range=shade_range,
            collision_tolerance=collision_tolerance,
            collision_shade_difference=collision_shade_difference,
            boundary_tolerance=boundary_tolerance,
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            validation_count_rate=validation_count_rate,
            test_entity_counts=test_entity_counts,
            test_count_rate=test_count_rate,
            validation_combinations=validation_combinations,
            validation_space_rate_range=validation_space_rate_range,
            validation_combination_rate=validation_combination_rate,
            test_combinations=test_combinations,
            test_space_rate_range=test_space_rate_range,
            test_combination_rate=test_combination_rate,
            max_provoke_collision_rate=0. # set to zero here
        )
        self.world_generator = world_generator
        self.meta_generator = MetaGenerator(
            world_size=world_size,
            world_colors=world_colors,
            shapes=shapes,
            colors=colors,
            textures=textures,
            rotation=rotation,
            size_range=size_range,
            distortion_range=distortion_range,
            shade_range=shade_range,
            collision_tolerance=collision_tolerance,
            collision_shade_difference=collision_shade_difference,
            boundary_tolerance=boundary_tolerance
        )
        self.captioner = GenNLVRCaptioner()

    @property
    def type(self):
        return 'referential game'

    def specification(self):
        specification = super(Dataset, self).specification()
        # TODO: add specification
        return specification

    def set_world_generator_space(self, size_range=None, shapes=None, colors=None, location_range=None):
        if size_range is not None:
            self.world_generator.size_range = size_range
        if shapes is not None:
            self.world_generator.shapes = shapes
        if colors is not None:
            self.world_generator.colors = colors
        if location_range is not None:
            self.sample_entity_kwargs = {"location_range": location_range}

    def generate(self, n, mode=None, include_model=True, alternatives=False):
        # if mode == 'none':
        #     mode = None
        # if mode == 'train':
        #     correct_ratio = self.train_correct_ratio
        # elif mode == 'validation':
        #     correct_ratio = self.validation_correct_ratio
        # elif mode == 'test':
        #     correct_ratio = self.test_correct_ratio
        # else:
        #     correct_ratio = self.correct_ratio

        # pn2id = self.vocabularies['pn']
        # unknown = pn2id['[UNKNOWN]']
        # pn_size = self.vector_shape('caption_pn')[0]

        batch = self.zero_batch(n, include_model=include_model, alternatives=alternatives)
        for i in tqdm.tqdm(range(n)):

            resample = 0
            caption = None
            while True:

                if resample % self.__class__.GENERATOR_INIT_FREQUENCY == 0:
                    if resample // self.__class__.GENERATOR_INIT_FREQUENCY >= 1:
                        pass
                    while not self.meta_generator.initialize(mode=mode):
                        pass
                    while not self.world_generator.initialize(mode=mode):
                        pass

                # if resample % self.__class__.CAPTIONER_INIT_FREQUENCY == 0:
                #     if resample // self.__class__.CAPTIONER_INIT_FREQUENCY >= 1:
                #         pass
                #     if self.worlds_per_instance > 1:
                #         correct = True
                #         while not self.world_captioner.initialize(mode=mode, correct=False):
                #             pass
                #     else:
                #         while not self.world_captioner.initialize(mode=mode, correct=correct):
                #             pass
                #         assert self.world_captioner.incorrect_possible()

                resample += 1

                # Step 1: generate range
                world_restrictions = self.meta_generator()

                # Step 2: generate world
                self.set_world_generator_space(**world_restrictions)
                world_l = self.world_generator()
                if world_l is None:
                    continue
                world_r = self.world_generator()
                if world_r is None:
                    continue
                
                # Step 4: generate caption
                caption = self.captioner(world_l, world_r)

                if caption is None:
                    continue # unreachable actually

                break

            # store game in batch
            batch['caption'][i] = caption

            batch['world_l'][i] = self.apply_pixel_noise(world=world_l.get_array(world_array=batch['world_l'][i]))
            batch['world_r'][i] = self.apply_pixel_noise(world=world_r.get_array(world_array=batch['world_r'][i]))
            if include_model:
                batch['world_model_l'][i] = world_l
                batch['world_model_r'][i] = world_r

        # word2id = self.vocabularies['language']
        # unknown = word2id['[UNKNOWN]']
        # caption_size = self.vector_shape('caption')[0]

        # unused_words = set(word2id)  # for assert
        # unused_words.remove('')
        # unused_words.remove('[UNKNOWN]')
        # missing_words = set()  # for assert
        # max_caption_size = caption_size  # for assert

        # assert len(captions) == n * self.captions_per_instance if alternatives else len(captions) == n
        # captions = self.caption_realizer.realize(captions=captions)

        # for i, caption in enumerate(captions):
        #     caption = util.sentence2tokens(sentence=caption)

        #     if len(caption) > caption_size:
        #         if len(caption) > max_caption_size:
        #             max_caption_size = len(caption)
        #         continue

        #     if alternatives and self.captions_per_instance > 1:
        #         j = i % self.captions_per_instance
        #         i = i // self.captions_per_instance
        #         batch['caption_length'][i].append(len(caption))
        #         caption_array = batch['caption'][i][j]
        #     else:
        #         batch['caption_length'][i] = len(caption)
        #         caption_array = batch['caption'][i]

        #     for k, word in enumerate(caption):
        #         if word in word2id:
        #             unused_words.discard(word)
        #         else:
        #             missing_words.add(word)
        #         caption_array[k] = word2id.get(word, unknown)

        # if util.debug() and len(unused_words) > 0:
        #     print('Words unused in vocabulary: \'{}\''.format('\', \''.join(sorted(unused_words))))
        # if util.debug() and max_caption_size < caption_size:
        #     print('Caption size smaller than max size: {} < {}'.format(max_caption_size, caption_size))
        # if len(missing_words) > 0:
        #     print('Words missing in vocabulary: \'{}\''.format('\', \''.join(sorted(missing_words))))
        # if max_caption_size > caption_size:
        #     print('Caption size exceeds max size: {} > {}'.format(max_caption_size, caption_size))
        # assert not missing_words, missing_words
        # assert max_caption_size <= caption_size, (max_caption_size, caption_size)

        return batch

    def get_html(self, generated, image_format='bmp', image_dir=''):
        id2word = self.vocabulary(value_type='language')
        worlds = generated['world']
        captions = generated['caption']
        caption_lengths = generated['caption_length']
        agreements = generated['agreement']

        data_html = list()
        for n, (world, caption, caption_length, agreement) in enumerate(zip(worlds, captions, caption_lengths, agreements)):

            if self.worlds_per_instance > 1 or self.captions_per_instance > 1:
                data_html.append('<div class="instance">')
            else:
                if agreement == 1.0:
                    agreement = 'correct'
                elif agreement == 0.0:
                    agreement = 'incorrect'
                else:
                    agreement = 'ambiguous'
                data_html.append('<div class="{agreement}">'.format(agreement=agreement))

            if self.worlds_per_instance > 1:
                for i, agreement in enumerate(agreement):
                    if agreement == 1.0:
                        agreement = 'correct'
                    elif agreement == 0.0:
                        agreement = 'incorrect'
                    else:
                        agreement = 'ambiguous'
                    data_html.append('<div class="{agreement}" style="padding: 5px;"><div class="world"><img src="{image_dir}world-{world}-{alt}.{format}" alt="world-{world}-{alt}.{format}"></div></div>'.format(
                        agreement=agreement,
                        image_dir=image_dir,
                        world=n,
                        format=image_format,
                        alt=i
                    ))
            else:
                data_html.append('<div class="world"><img src="{image_dir}world-{world}.{format}" alt="world-{world}.{format}"></div>'.format(image_dir=image_dir, world=n, format=image_format))

            data_html.append('<div class="num"><b>({num})</b></div>'.format(num=(n + 1)))

            if self.captions_per_instance > 1:
                data_html.append('<div class="caption">')
                for caption, caption_length, agreement in zip(caption, caption_length, agreement):
                    if agreement == 1.0:
                        agreement = 'correct'
                    elif agreement == 0.0:
                        agreement = 'incorrect'
                    else:
                        agreement = 'ambiguous'
                    data_html.append('<div class="{agreement}">{caption}</div>'.format(
                        agreement=agreement,
                        caption=util.tokens2sentence(id2word[word] for word in caption[:caption_length])
                    ))
                data_html.append('</div>')
            else:
                data_html.append('<div class="caption">{caption}</div>'.format(
                    caption=util.tokens2sentence(id2word[word] for word in caption[:caption_length])
                ))

            data_html.append('</div>')

        html = '<!DOCTYPE html><html><head><title>{dtype} {name}</title><style>.data{{width: 100%; height: 100%;}} .instance{{width: 100%; display: flex; margin-top: 1px; margin-bottom: 1px; background-color: #DDEEFF; vertical-align: middle; align-items: center;}} .world{{height: {world_height}px; display: inline-block; flex-grow: 0; vertical-align: middle;}} .num{{width: 50px; display: inline-block; flex-grow: 0; text-align: center; vertical-align: middle; margin-left: 10px;}} .caption{{display: inline-block; flex-grow: 1; vertical-align: middle; margin-left: 10px;}} .correct{{margin-top: 1px; margin-bottom: 1px; background-color: #BBFFBB;}} .incorrect{{margin-top: 1px; margin-bottom: 1px; background-color: #FFBBBB;}} .ambiguous{{margin-top: 1px; margin-bottom: 1px; background-color: #FFFFBB;}}</style></head><body><div class="data">{data}</div></body></html>'.format(
            dtype=self.type,
            name=self.name,
            world_height=self.world_shape()[0],
            data=''.join(data_html)
        )
        return html
