# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastai==2.7.19",
#     "fastcore==1.7.29",
# ]
# ///

import fastai, fastcore

assert fastai.__version__ == "2.7.19", f"fastai.__version__ == {fastai.__version__}"
assert (
    fastcore.__version__ == "1.7.29"
), f"fastcore.__version__ == {fastcore.__version__}"

from fastai.vision.all import *

def label_func(f):
    return f[0].isupper()

model = load_learner("model_27.pkl")
assert model is not None
print(f"✅ Successfully loaded model under {fastai.__version__=}")