import fastai, fastcore

assert fastai.__version__ == "2.8.1", f"fastai.__version__ == {fastai.__version__}"
assert (
    fastcore.__version__ == "1.8.1"
), f"fastcore.__version__ == {fastcore.__version__}"

from fastai.vision.all import *


def label_func(f):
    return f[0].isupper()

try:
    model = load_learner("model_27.pkl")
    assert (
        False
    ), "Should not be able to load model under 2.8 without manual backward compatibility flag set to True"
except RuntimeError as e:
    assert (
        "Your Learner was most likely trained under `fastai<2.8`, you can load such Learners by setting load_learner(...,backward_compat=True). But you should retrain it under `fastai>=2.8` for continued compatibility. This compatibility layer will be removed in `fastai>=2.9`"
        in str(e)
    )
    print("✅ Successfully raised error as expected")