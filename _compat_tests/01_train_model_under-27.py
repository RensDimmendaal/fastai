# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastai==2.7.19",
#     "fastcore==1.7.29",
# ]
# ///

import fastai, fastcore
assert fastai.__version__ == "2.7.19"
assert fastcore.__version__ == "1.7.29"

from fastai.vision.all import *
def label_func(f): return f[0].isupper()
path = untar_data(URLs.PETS)
files = get_image_files(path/"images")
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
learn = vision_learner(dls, resnet34, metrics=error_rate, path=".")
learn.fine_tune(1)
learn.export("model_27.pkl")