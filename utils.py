import os
import random

def load_files_with_given_extension(path, ext=[".jpg", ".png", ".jpeg"], name_format = "path"):
    images = []
    valid_images = ext
    for root, dirs, files in os.walk(path):
        for name in iter(files):
            ext = os.path.splitext(name)[1]
            if ext.lower() not in valid_images:
                continue
            # full path
            if name_format == "path":
                images.append(os.path.join(path, name))
            # jast its name
            elif name_format == "name":
                images.append(name)
            else:
                raise ValueError("wrong format for parameter : name_format")
    return images

def random_idx_with_exclude(exclude, idx_range):
    randInt = random.randint(idx_range[0], idx_range[1])
    return random_idx_with_exclude(exclude, idx_range) if randInt in exclude else randInt 