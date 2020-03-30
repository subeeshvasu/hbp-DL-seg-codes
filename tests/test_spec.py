from io import BytesIO
from pathlib import Path

import imageio
import numpy
import torch

from pybio.core.transformations import apply_transformations
from pybio.spec import load_model
from pybio.spec.utils import get_instance


def test_UNetDA(cache_path):
    spec_path = (Path(__file__).parent / "../UNetDA.model.yaml").resolve()
    assert spec_path.exists(), spec_path
    pybio_model = load_model(str(spec_path), cache_path=cache_path)

    assert isinstance(pybio_model.spec.prediction.weights.source, BytesIO)
    assert pybio_model.spec.test_input is not None
    assert pybio_model.spec.test_output is not None

    model: torch.nn.Module = get_instance(pybio_model)
    if torch.cuda.is_available():
        test_device = torch.device("cuda")
        model = model.to(device=test_device)
    else:
        test_device = torch.device("cpu")

    model.eval()

    model_weights = torch.load(pybio_model.spec.prediction.weights.source, map_location=test_device)
    model.load_state_dict(model_weights)
    pre_transformations = [get_instance(trf) for trf in pybio_model.spec.prediction.preprocess]
    post_transformations = [get_instance(trf) for trf in pybio_model.spec.prediction.postprocess]

    test_ipt = numpy.load(str(pybio_model.spec.test_input))
    test_out = numpy.load(str(pybio_model.spec.test_output))

    assert hasattr(model, "forward")
    preprocessed_inputs = apply_transformations(pre_transformations, test_ipt)
    assert isinstance(preprocessed_inputs, list)
    assert len(preprocessed_inputs) == 1
    out = model.forward(*[t.to(test_device) for t in preprocessed_inputs])
    postprocessed_outputs = apply_transformations(post_transformations, out)
    assert isinstance(postprocessed_outputs, list)
    assert len(postprocessed_outputs) == 1
    out = postprocessed_outputs[0]
    assert numpy.allclose(test_out, out)


def test_2sUNetDA(cache_path):
    spec_path = (Path(__file__).parent / "../2sUNetDA.model.yaml").resolve()
    assert spec_path.exists(), spec_path
    pybio_model = load_model(str(spec_path), cache_path=cache_path)

    assert isinstance(pybio_model.spec.prediction.weights.source, BytesIO)
    assert pybio_model.spec.test_input is not None
    assert pybio_model.spec.test_output is not None

    model: torch.nn.Module = get_instance(pybio_model)
    if torch.cuda.is_available():
        test_device = torch.device("cuda")
        model = model.to(device=test_device)
    else:
        test_device = torch.device("cpu")

    model.eval()

    model_weights = torch.load(pybio_model.spec.prediction.weights.source, map_location=test_device)
    model.load_state_dict(model_weights)
    pre_transformations = [get_instance(trf) for trf in pybio_model.spec.prediction.preprocess]
    post_transformations = [get_instance(trf) for trf in pybio_model.spec.prediction.postprocess]

    test_ipt = numpy.load(str(pybio_model.spec.test_input))
    test_out = numpy.load(str(pybio_model.spec.test_output))

    assert hasattr(model, "forward")
    preprocessed_inputs = apply_transformations(pre_transformations, test_ipt)
    assert isinstance(preprocessed_inputs, list)
    assert len(preprocessed_inputs) == 1
    out = model.forward(*[t.to(test_device) for t in preprocessed_inputs])
    postprocessed_outputs = apply_transformations(post_transformations, out)
    assert isinstance(postprocessed_outputs, list)
    assert len(postprocessed_outputs) == 1
    out = postprocessed_outputs[0]
    assert numpy.allclose(test_out, out)
