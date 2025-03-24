import fastai, fastcore

assert fastai.__version__ == "2.8.1", f"fastai.__version__ == {fastai.__version__}"
assert (
    fastcore.__version__ == "1.8.1"
), f"fastcore.__version__ == {fastcore.__version__}"

from fastai.vision.all import *


def label_func(f):
    return f[0].isupper()

model = load_learner("model_27.pkl", backward_compat=True)
assert model is not None
print("✅ Successfully loaded model with backward compatibility flag set to True")