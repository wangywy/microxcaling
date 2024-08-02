import pytest

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchmo import ModelQuantizer
from torchmo.quantization.config.type import Dtype, QSchemeType
from torchmo.quantization.observer.observer import PerBlockMXObserver
from torchmo.quantization.config.config import Config, QuantizationSpec, QuantizationConfig

from torchmo.quantization.utils import reshape_to_blocks, get_dtype_params, calculate_qmin_qmax
from torchmo.kernel.hw_emulation.hw_emulation_interface import fake_quantize_mx

import math

class SimpleNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleNetwork, self).__init__()
        self.fc = nn.Linear(in_features=4, out_features=num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

input_tensor = torch.ones(1, 4, 4)
class SimpleDataset(Dataset):

    def __init__(self):
        return

    def __len__(self):
        return 2

    def __getitem__(self, _):
        return input_tensor

def create_quantize_run_simple_network(qconfig : QuantizationConfig):
    model = SimpleNetwork()
    model.fc.weight = torch.nn.Parameter(torch.randn([2, 4]))
    model(input_tensor)
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    quant_config = Config(global_quant_config=qconfig)
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(model, dataloader)
    quantized_model(input_tensor)

valid_configs = [
    (Dtype.mx, False, PerBlockMXObserver, QSchemeType.per_group, Dtype.fp8_e4m3, 1, 4),
]

@pytest.mark.parametrize("dtype, is_dynamic, observer_cls, qscheme, mx_element_dtype, ch_axis, group_size", valid_configs)
def text_mx_valid_config_verification(dtype, is_dynamic, observer_cls, qscheme, mx_element_dtype, ch_axis, group_size):
    MXFP8_PER_BLOCK_SPEC = QuantizationSpec(dtype=dtype, is_dynamic=is_dynamic, observer_cls=observer_cls, qscheme=qscheme, mx_element_dtype=mx_element_dtype, ch_axis=ch_axis, group_siz
e=group_size)
    TEST_WEIGHT = QuantizationConfig(weight=MXFP8_PER_BLOCK_SPEC)
    create_quantize_run_simple_network(TEST_WEIGHT)

def test_mx_block_size_larger_than_tensor():
    MXFP8_PER_BLOCK_SPEC = QuantizationSpec(dtype=Dtype.mx, is_dynamic=False, observer_cls=PerBlockMXObserver, qscheme=QSchemeType.per_group, mx_element_dtype=Dtype.fp8_e4m3, ch_axis=1,
 group_size=32)
    TEST_WEIGHT = QuantizationConfig(weight=MXFP8_PER_BLOCK_SPEC)
    create_quantize_run_simple_network(TEST_WEIGHT)

invalid_spec_args = [
    (Dtype.int8, 1, 32, Dtype.fp8_e4m3),
    (Dtype.mx, None, 32, Dtype.fp8_e4m3),
    (Dtype.mx, 1, None, Dtype.fp8_e4m3),
    (Dtype.mx, 1, 32, None)
]

@pytest.mark.parametrize("dtype, ch_axis, group_size, mx_element_dtype", invalid_spec_args)
def test_per_block_observer_invalid_arguments(dtype, ch_axis, group_size, mx_element_dtype):
    spec = QuantizationSpec(dtype=dtype, ch_axis=ch_axis, group_size=group_size, mx_element_dtype=mx_element_dtype)
    with pytest.raises(Exception):
	    PerBlockMXObserver(qspec=spec)

reshape_args = [
    (torch.Size([10, 10]), 32, 0, torch.Size([10, 1, 32])),  # the output shape is the same as the output is reshaped to have the blocked dimension last
    (torch.Size([10, 10]), 32, 1, torch.Size([10, 1, 32])),
    (torch.Size([10, 10]), 32, 2, None),  # axis is greater than number of axes in tensor
    (torch.Size([10, 10]), 5, 0, torch.Size([10, 2, 5])),  # block size smaller than axis so needs to be tiled - however
    (torch.Size([10, 10]), 5, 1, torch.Size([10, 2, 5]))
]

@pytest.mark.parametrize("tensor_shape, block_size, axis, expected_shape", reshape_args)
def test_mx_reshape_to_blocks(tensor_shape, block_size, axis, expected_shape):
    a = torch.ones(tensor_shape)

    if expected_shape is None:
        with pytest.raises(IndexError):
            reshaped = reshape_to_blocks(a, block_size, axis)
    else:
        reshaped = reshape_to_blocks(a, block_size, axis)
        assert reshaped.shape == expected_shape

def test_mx_reshape_to_blocks_axis0():
    a = torch.ones(10, 10)
    for row_idx in range(10):
        a[row_idx] = a[row_idx] * row_idx

    block_size = 10
    block_a = reshape_to_blocks(a, block_size, 0)

    assert block_a.shape == torch.Size([10, 1, block_size])

    for row_idx in range(10):
        for cell_idx in range(10):
            assert block_a[row_idx, 0, cell_idx] == cell_idx

def test_mx_reshape_to_blocks_axis1():
    a = torch.ones(10, 10)
    for row_idx in range(10):
	        a[row_idx] = a[row_idx] * row_idx

    block_size = 10
    block_a = reshape_to_blocks(a, block_size, 1)

    assert block_a.shape == torch.Size([10, 1, block_size])

    for row_idx in range(10):
        for cell_idx in range(10):
            assert block_a[row_idx, 0, cell_idx] == row_idx

def test_mx_reshape_to_blocks_more_detail():
    a = torch.zeros(2, 10)
    for i in range(10):
        a[0, i] = -5 + i
        a[1, i] = 5 - i

    # 'a' should look like this
    # [
    #    [ -5, -4, -3, -2, -1, 0, 1, 2, 3, 4],
    #    [ 5, 4, 3, 2, 1, 0 , -1, -2, -3, -4]
    # ]
    block_size = 5
    reshaped_a = reshape_to_blocks(a, block_size, 1)
    # 'reshaped_a' should look like
    # [
    #    [[ -5, -4, -3, -2, -1], [0, 1, 2, 3, 4]],
    #    [[ 5, 4, 3, 2, 1], [0, -1, -2, -3, -4]]
    # ]
    assert reshaped_a.dim() == 3
    assert reshaped_a.shape[0] == 2
    assert reshaped_a.shape[1] == 2, 'Incorrect number of block tiles'
    assert reshaped_a.shape[2] == block_size

    # first block
    for idx, val in enumerate([-5, -4, -3, -2, -1]):
        assert reshaped_a[0, 0, idx] == val
	
	# second block
    for idx, val in enumerate([0, 1, 2, 3, 4]):
        assert reshaped_a[0, 1, idx] == val

    # third block
    for idx, val in enumerate([5, 4, 3, 2, 1]):
        assert reshaped_a[1, 0, idx] == val

    # fourth block
    for idx, val in enumerate([0, -1, -2, -3, -4]):
        assert reshaped_a[1, 1, idx] == val

def create_4d_tensor_with_interesting_pattern():
    # let's create a tensor with a nice pattern we can inspect
    # tensor([[[[ 10.,  20.,  30.,  40.],
    #           [ 20.,  40.,  60.,  80.],
    #           [ 30.,  60.,  90., 120.]],
    #           [[110., 120., 130., 140.],
    #           [120., 140., 160., 180.],
    #           [130., 160., 190., 220.]]]])
    result = torch.zeros(1, 2, 3, 4)
    for x0 in range(result.shape[0]):
        for x1 in range(result.shape[1]):
            for x2 in range(result.shape[2]):
                for x3 in range(result.shape[3]):
                    result[x0, x1, x2, x3] = (x3 + 1.0) * (10 * (x2 + 1.0)) + 100.0 * x1
    return result

def test_per_block_simple_scale():
    element_dtype = Dtype.fp8_e4m3

    spec = QuantizationSpec(dtype=Dtype.mx, ch_axis=1, group_size=32, mx_element_dtype=element_dtype)
    observer = PerBlockMXObserver(qspec=spec)

    a = torch.zeros(10, 10)
    for i in range(10):
        a[i, i] = -5 + i

    observer(a)

    _, _, emax = get_dtype_params(Dtype.fp8_e4m3)

    for i in range(10):
        amax = abs(-5 + i)
        if amax != 0:
            scale_val = math.pow(2.0, math.floor(math.log2(amax)) - emax)
        else:
            scale_val = observer.eps
        # these values should be directly representable by floating point so direct comparison is valid here
        scale, _ = observer.calculate_qparams()
        assert scale[i, 0, 0] == scale_val

def test_per_block_scale_tiled():
    a = torch.zeros(2, 10)
    for i in range(10):
        a[0, i] = -5 + i
        a[1, i] = 5 - i

    # a should look like this
    # [
    #    [ -5, -4, -3, -2, -1, 0, 1, 2, 3, 4],
    #    [ 5, 4, 3, 2, 1, 0 , -1, -2, -3, -4]
    # ]
    element_dtype = Dtype.fp8_e4m3
    spec = QuantizationSpec(dtype=Dtype.mx, ch_axis=1, group_size=5, mx_element_dtype=element_dtype)
    observer = PerBlockMXObserver(qspec=spec)
    observer(a)

    _, _, emax = get_dtype_params(Dtype.fp8_e4m3)
    scale, _ = observer.calculate_qparams()
    assert scale[0, 0, 0] == math.pow(2.0, math.floor(math.log2(5)) - emax)
    assert scale[0, 1, 0] == math.pow(2.0, math.floor(math.log2(4)) - emax)
    assert scale[1, 0, 0] == math.pow(2.0, math.floor(math.log2(5)) - emax)
    assert scale[1, 1, 0] == math.pow(2.0, math.floor(math.log2(4)) - emax)

per_block_to_quantize_mx = [
    (1, 8)
]

@pytest.mark.parametrize("axis, block_size", per_block_to_quantize_mx)
def test_per_block_to_fake_quantize_mx(axis, block_size):
    x_orig = create_4d_tensor_with_interesting_pattern()

    element_dtype = Dtype.fp8_e4m3

    spec = QuantizationSpec(dtype=Dtype.mx, ch_axis=axis, group_size=block_size, mx_element_dtype=element_dtype)
    observer = PerBlockMXObserver(qspec=spec)
    observer(x_orig)
    scale, _ = observer.calculate_qparams()
    fake_quantize_mx(x_orig, scale, element_dtype, axis, block_size)

def test_simple_network():
    MXFP8_PER_BLOCK_SPEC = QuantizationSpec(dtype=Dtype.mx, is_dynamic=False, observer_cls=PerBlockMXObserver, qscheme=QSchemeType.per_group, mx_element_dtype=Dtype.fp8_e4m3, ch_axis=1,
 group_size=32)
    TEST_WEIGHT = QuantizationConfig(weight=MXFP8_PER_BLOCK_SPEC)
    create_quantize_run_simple_network(TEST_WEIGHT)

def generate_test_case_input_normal():
    normal_input = torch.tensor([[-406., -881., 227., -676., 404., 291., 267., 286.],
                                [-557., 505., -175., -252., -518., -136., -134., 990.],
                                [-148., 413., -346., -954., 748., -877., -201., 338.],
                                [-262., 437., 755., 756., 772., 767., 968., -989.],
                                [336., -38., 601., -162., 585., -686., 610., 98.],
                                [-718., 957., -854., -985., -570., 989., -774., 695.],
                                [708., -71., 863., 260., -252., 558., -750., -849.],
                                [-396., -986., -329., 814., -340., -291., 141., 784.]], dtype=torch.float32)
    return normal_input
	
def generate_test_case_input():
    test_data = {
        'normal': generate_test_case_input_normal(),
        'zeros': torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
                               [-557., 505., -175., -252., -518., -136., -134., 990.],
                               [-148., 413., -346., -954., 748., -877., -201., 338.],
                               [-262., 437., 755., 756., 772., 767., 968., -989.],
                               [336., -38., 601., -162., 585., -686., 610., 98.],
                               [-718., 957., -854., -985., -570., 989., -774., 695.],
                               [708., -71., 863., 260., -252., 558., -750., -849.],
                               [-396., -986., -329., 814., -340., -291., 141., 784.]], dtype=torch.float32),
        'nan': torch.tensor([[-406., torch.nan, 227., -676., 404., 291., 267., 286.],
                             [-557., 505., -175., -252., -518., -136., -134., 990.],
                             [-148., 413., -346., -954., 748., -877., -201., 338.],
                             [-262., 437., 755., 756., 772., 767., 968., -989.],
                             [336., -38., 601., -162., 585., -686., 610., 98.],
                             [-718., 957., -854., -985., -570., 989., -774., 695.],
                             [708., -71., 863., 260., -252., 558., -750., -849.],
                             [-396., -986., -329., 814., -340., -291., 141., 784.]], dtype=torch.float32),
        'inf': torch.tensor([[-406., torch.inf, 227., -676., 404., 291., 267., 286.],
                             [-557., 505., -175., -252., -518., -136., -134., 990.],
                             [-148., 413., -346., -954., 748., -877., -201., 338.],
                             [-262., 437., 755., 756., 772., 767., 968., -989.],
                             [336., -38., 601., -162., 585., -686., 610., 98.],
                             [-718., 957., -854., -985., -570., 989., -774., 695.],
                             [708., -71., 863., 260., -252., 558., -750., -849.],
                             [-396., -986., -329., 814., -340., -291., 141., 784.]], dtype=torch.float32),
        'maximum': torch.tensor([[-406., math.e**60, 227., -676., 404., 291., 267., 286.],
                                 [-557., 505., -175., -252., -518., -136., -134., 990.],
                                 [-148., 413., -346., -954., 748., -877., -201., 338.],
                                 [-262., 437., 755., 756., 772., 767., 968., -989.],
                                 [336., -38., 601., -162., 585., -686., 610., 98.],
                                 [-718., 957., -854., -985., -570., 989., -774., 695.],
                                 [708., -71., 863., 260., -252., 558., -750., -849.],
                                 [-396., -986., -329., 814., -340., -291., 141., 784.]], dtype=torch.float32),
    }
    return test_data
	
def load_test_case_result():
    result = {
        'input': generate_test_case_input_normal(),
        'fp8_e4m3': {
            'torchao_result': torch.tensor([[-416., -896., 224., -704., 416., 288., 256., 288.],
                                            [-576., 512., -176., -256., -512., -128., -128., 896.],
                                            [-144., 416., -352., -896., 768., -896., -208., 352.],
                                            [-256., 448., 768., 768., 768., 768., 896., -896.],
                                            [320., -40., 576., -160., 576., -704., 640., 96.],
                                            [-704., 896., -832., -896., -576., 896., -768., 704.],
                                            [704., -72., 832., 256., -256., 576., -768., -832.],
                                            [-384., -896., -320., 832., -352., -288., 144., 768.]], dtype=torch.float32),
            'MX_result': torch.tensor([[-416., -896., 224., -704., 416., 288., 256., 288.],
                                       [-576., 512., -176., -256., -512., -128., -128., 896.],
                                       [-144., 416., -352., -896., 768., -896., -208., 352.],
                                       [-256., 448., 768., 768., 768., 768., 896., -896.],
                                       [320., -40., 576., -160., 576., -704., 640., 96.],
                                       [-704., 896., -832., -896., -576., 896., -768., 704.],
                                       [704., -72., 832., 256., -256., 576., -768., -832.],
                                       [-384., -896., -320., 832., -352., -288., 144., 768.]], dtype=torch.float32),
        },
		        'fp6_e3m2': {
            'torchao_result': torch.tensor([[-384., -896., 224., -640., 384., 320., 256., 256.],
                                            [-512., 512., -160., -256., -512., -128., -128., 896.],
                                            [-160., 384., -320., -896., 768., -896., -192., 320.],
                                            [-256., 448., 768., 768., 768., 768., 896., -896.],
                                            [320., -40., 640., -160., 640., -640., 640., 96.],
                                            [-768., 896., -896., -896., -512., 896., -768., 640.],
                                            [768., -64., 896., 256., -256., 512., -768., -896.],
                                            [-384., -896., -320., 768., -320., -320., 128., 768.]], dtype=torch.float32),
            'MX_result': torch.tensor([[-384., -896., 224., -640., 384., 320., 256., 256.],
                                       [-512., 512., -160., -256., -512., -128., -128., 896.],
                                       [-160., 384., -320., -896., 768., -896., -192., 320.],
                                       [-256., 448., 768., 768., 768., 768., 896., -896.],
                                       [320., -40., 640., -160., 640., -640., 640., 96.],
                                       [-768., 896., -896., -896., -512., 896., -768., 640.],
                                       [768., -64., 896., 256., -256., 512., -768., -896.],
                                       [-384., -896., -320., 768., -320., -320., 128., 768.]], dtype=torch.float32),
        },
        'fp6_e2m3': {
            'torchao_result': torch.tensor([[-416., -896., 224., -704., 416., 288., 256., 288.],
                                            [-576., 512., -176., -256., -512., -128., -128., 960.],
                                            [-144., 416., -352., -960., 768., -896., -208., 352.],
                                            [-256., 448., 768., 768., 768., 768., 960., -960.],
                                            [320., -32., 576., -160., 576., -704., 640., 96.],
                                            [-704., 960., -832., -960., -576., 960., -768., 704.],
                                            [704., -64., 832., 256., -256., 576., -768., -832.],
                                            [-384., -960., -320., 832., -352., -288., 144., 768.]], dtype=torch.float32),
            'MX_result': torch.tensor([[-416., -896., 224., -704., 416., 288., 256., 288.],
                                       [-576., 512., -176., -256., -512., -128., -128., 960.],
                                       [-144., 416., -352., -960., 768., -896., -208., 352.],
                                       [-256., 448., 768., 768., 768., 768., 960., -960.],
                                       [320., -32., 576., -160., 576., -704., 640., 96.],
                                       [-704., 960., -832., -960., -576., 960., -768., 704.],
                                       [704., -64., 832., 256., -256., 576., -768., -832.],
                                       [-384., -960., -320., 832., -352., -288., 144., 768.]], dtype=torch.float32),
        },
		'fp4': {
            'torchao_result': torch.tensor([[-384., -768., 256., -768., 384., 256., 256., 256.],
                                            [-512., 512., -192., -256., -512., -128., -128., 768.],
                                            [-128., 384., -384., -768., 768., -768., -192., 384.],
                                            [-256., 384., 768., 768., 768., 768., 768., -768.],
                                            [384., -64., 512., -192., 512., -768., 512., 128.],
                                            [-768., 768., -768., -768., -512., 768., -768., 768.],
                                            [768., -64., 768., 256., -256., 512., -768., -768.],
                                            [-384., -768., -384., 768., -384., -256., 128., 768.]], dtype=torch.float32),
            'MX_result': torch.tensor([[-384., -768., 256., -768., 384., 256., 256., 256.],
                                       [-512., 512., -192., -256., -512., -128., -128., 768.],
                                       [-128., 384., -384., -768., 768., -768., -192., 384.],
                                       [-256., 384., 768., 768., 768., 768., 768., -768.],
                                       [384., -64., 512., -192., 512., -768., 512., 128.],
                                       [-768., 768., -768., -768., -512., 768., -768., 768.],
                                       [768., -64., 768., 256., -256., 512., -768., -768.],
                                       [-384., -768., -384., 768., -384., -256., 128., 768.]], dtype=torch.float32),
        }
    }
    return result
	
torchmo_mx_dtype_lst = [Dtype.fp8_e4m3, Dtype.fp6_e3m2, Dtype.fp6_e2m3, Dtype.fp4, Dtype.int8]

@pytest.mark.parametrize("torchmo_mx_dtype", torchmo_mx_dtype_lst)
def test_fake_quantize_mx(torchmo_mx_dtype):
    test_data = generate_test_case_input()
    for scene, test_tensor in test_data.items():
        block_size = 32
        axis = 1
        mx_element_dtype = torchmo_mx_dtype
        _, _, emax = get_dtype_params(mx_element_dtype)

        block_x = reshape_to_blocks(test_tensor, block_size, axis)
        scale, _ = torch.max(torch.abs(block_x), dim=axis + 1, keepdim=True)
        scale = torch.pow(2, torch.floor(torch.log2(scale)) - emax)
        _, q_max = calculate_qmin_qmax(mx_element_dtype)

        fake_quantize_mx(input_tensor=test_tensor.clone(), scale=scale, mx_element_dtype=mx_element_dtype, axis=axis, block_size=block_size, quant_max=q_max)

torchmo_supported_elem_dtype = {
    "fp8_e4m3": Dtype.fp8_e4m3,
    "fp6_e3m2": Dtype.fp6_e3m2,
    "fp6_e2m3": Dtype.fp6_e2m3,
    "fp4": Dtype.fp4,
    "int8": Dtype.int8,
}
elem_dtype_lst = ["fp8_e4m3", "fp6_e3m2", "fp6_e2m3", "fp4"]

@pytest.mark.parametrize("elem_dtype", elem_dtype_lst)
def test_compare_torchmo_ao_mx_repo(elem_dtype):
    """
    compare the performance of torchmo, torchao, MX
    torchmo: 0.1.0+559df62f
    torchao: 0.3.1
    """
    result = load_test_case_result()
    test_tensor = result["input"]

    block_size, axis = 32, 1
    mx_element_dtype = torchmo_supported_elem_dtype[elem_dtype]
    _, _, emax = get_dtype_params(mx_element_dtype)
    block_x = reshape_to_blocks(test_tensor, block_size, axis)
    scale, _ = torch.max(torch.abs(block_x), dim=axis + 1, keepdim=True)
    scale = torch.pow(2, torch.floor(torch.log2(scale)) - emax)
    _, q_max = calculate_qmin_qmax(mx_element_dtype)

    torchmo_output_tensor = fake_quantize_mx(input_tensor=test_tensor.clone(), scale=scale, mx_element_dtype=mx_element_dtype, axis=axis, block_size=block_size, quant_max=q_max)
    torchao_result = result[elem_dtype]["torchao_result"]
    MX_result = result[elem_dtype]["MX_result"]

    max_diff_ao = torch.max(abs(torchmo_output_tensor - torchao_result))
    max_diff_MX = torch.max(abs(torchmo_output_tensor - MX_result))
    assert max_diff_ao == 0, f"The {elem_dtype} quantization result of torchmo and torchao is different"
    assert max_diff_MX == 0, f"The {elem_dtype} quantization result of torchmo and MX is different"

if __name__ == '__main__':
    # test_per_block_to_fake_quantize_mx(1, 8)
    test_per_block_simple_scale()
