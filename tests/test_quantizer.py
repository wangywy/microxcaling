import torch
from torchmo.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
from torchmo.quantization.config.config import QuantizationSpec
from torchmo.quantization.observer.observer import PerTensorMinMaxObserver, PerChannelMinMaxObserver, PerTensorHistogramObserver

def test_calculate_int_quant_params():
    # Test Symmetric
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                        qscheme=QSchemeType.per_tensor,
                                                        observer_cls=PerTensorMinMaxObserver,
                                                        symmetric=True,
                                                        scale_type=ScaleType.float,
                                                        round_method=RoundType.half_even,
                                                        is_dynamic=False)
    observer = PerTensorMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    min_val = torch.Tensor([0.0, 0.0])
    max_val = torch.Tensor([1.0, 1.0])
    scale, zero_point = observer.calculate_int_quant_params(min_val, max_val)
    assert torch.allclose(scale, torch.Tensor([0.00784314, 0.00784314]), atol=1e-6)
    assert torch.equal(zero_point, torch.Tensor([0, 0]))

    # Test Asymmetric
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(dtype=Dtype.uint4,
                                                        qscheme=QSchemeType.per_tensor,
                                                        observer_cls=PerTensorMinMaxObserver,
                                                        symmetric=False,
                                                        scale_type=ScaleType.float,
                                                        round_method=RoundType.half_even,
                                                        is_dynamic=False)
    observer = PerTensorMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    min_val = torch.Tensor([-1.0, -1.0])
    max_val = torch.Tensor([10.0, 10.0])
    scale, zero_point = observer.calculate_int_quant_params(min_val, max_val)
    assert torch.allclose(scale, torch.Tensor([0.733333333333, 0.733333333333]), atol=1e-6)
    assert torch.equal(zero_point, torch.Tensor([1, 1
	
def test_calculate_fp8_quant_parameters():
    FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                           qscheme=QSchemeType.per_tensor,
                                           observer_cls=PerTensorMinMaxObserver,
                                           is_dynamic=False)
    observer = PerTensorMinMaxObserver(FP8_PER_TENSOR_SPEC)
    min_val = torch.Tensor([0.0])
    max_val = torch.Tensor([1.0])
    scale, zero_point = observer.calculate_fp8_quant_parameters(min_val, max_val)
    assert torch.allclose(scale, torch.Tensor([1 / 448]), atol=1e-6)
    assert torch.equal(zero_point, torch.Tensor([0]))

def test_PerTensorMinMaxObserver():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                        qscheme=QSchemeType.per_tensor,
                                                        observer_cls=PerTensorMinMaxObserver,
                                                        symmetric=True,
                                                        scale_type=ScaleType.float,
                                                        round_method=RoundType.half_even,
                                                        is_dynamic=False)
    observer = PerTensorMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    x_orig = torch.Tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
    assert torch.allclose(observer.max_val, torch.Tensor([3.0]), atol=1e-6)
    assert torch.allclose(observer.min_val, torch.Tensor([-1.0]), atol=1e-6)
	
def test_PerChannelMinMaxObserver():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                        qscheme=QSchemeType.per_tensor,
                                                        observer_cls=PerChannelMinMaxObserver,
                                                        symmetric=True,
                                                        scale_type=ScaleType.float,
                                                        round_method=RoundType.half_even,
                                                        is_dynamic=False,
                                                        ch_axis=1)
    observer = PerChannelMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    x_orig = torch.Tensor([[-1.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
    assert torch.allclose(observer.max_val, torch.Tensor([3.0, 4.0]), atol=1e-6)
    assert torch.allclose(observer.min_val, torch.Tensor([-1.0, 0.0]), atol=1e-6)

def test_PerTensorHistogramObserver():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                        qscheme=QSchemeType.per_tensor,
                                                        observer_cls=PerTensorHistogramObserver,
                                                        symmetric=True,
                                                        scale_type=ScaleType.float,
                                                        round_method=RoundType.half_even,
                                                        is_dynamic=False)
    observer = PerTensorHistogramObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    x_orig = torch.Tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
    assert torch.allclose(observer.calib_bin_edges.sum(), torch.Tensor([3073.5000]), atol=1e-6)
    assert torch.allclose(observer.calib_hist.sum(), torch.Tensor([5]), atol=1e-6)
