import math

import numpy as np


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    max_width, max_height = max_reso
    max_area = max_width * max_height
    resos = set()

    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))
        width += divisible

    return sorted(resos)


class BucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps):
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], "max_size must be >= max_reso width"
                assert max_size >= max_reso[1], "max_size must be >= max_reso height"
            if min_size is not None:
                assert max_size >= min_size, "max_size must be >= min_size"

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []

    def add_image(self, reso, image_or_info):
        self.buckets[self.reso_to_id[reso]].append(image_or_info)

    def set_predefined_resos(self, resos):
        self.predefined_resos = list(resos)
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def make_buckets(self):
        self.set_predefined_resos(make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps))

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            self.reso_to_id[reso] = len(self.resos)
            self.resos.append(reso)
            self.buckets.append([])

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            reso = (image_width, image_height)
            if reso not in self.predefined_resos_set:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
        else:
            if image_width * image_height > self.max_area:
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
            else:
                resized_size = (image_width, image_height)

            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)
        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error
