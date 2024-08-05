import sys
sys.path.append("..")
import torchmo.kernel  # noqa
import torch
# import torchmo.quantization.nn.modules.quantize_conv as quant_conv
import torchmo.quantization.nn.modules.quantize_conv_bn_fused as conv_bn_fused
from torchmo.quantization.config.config import QuantizationSpec, QuantizationConfig
from torchmo.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from torchmo.quantization.observer.observer import PerTensorMinMaxObserver
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_QuantizedConvBatchNorm2d():
    conv_map = {
        torch.nn.Conv2d: (torch.nn.BatchNorm2d, (1, 3, 16, 16), conv_bn_fused.QuantizedConvBatchNorm2d),
        # TODO haoliang
        # torch.nn.Conv3d: (torch.nn.BatchNorm3d, (1, 3, 16, 16, 16), conv_bn_fused.QuantizedConvBatchNorm3d),
        # torch.nn.ConvTranspose2d: (torch.nn.BatchNorm2d, (1, 3, 16, 16), conv_bn_fused.QuantizedConvTransposeBatchNorm2d),
        # torch.nn.ConvTranspose3d: (torch.nn.BatchNorm3d, (1, 3, 16, 16, 16), conv_bn_fused.QuantizedConvTransposeBatchNorm3d),
    }
    empty_config = QuantizationConfig()
    for conv, (bn, input_size, q_conv) in conv_map.items():
        input = torch.randn(input_size).to(device)
        float_conv = conv(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=False).to(device)
        float_bn = bn(16).to(device)
        quantized_conv = q_conv.from_float(float_conv, float_bn, empty_config).to(device)
        float_out = float_bn(float_conv(input))
        quantized_out = quantized_conv(input)
        assert torch.allclose(float_out.mean(), quantized_out.mean()), "{} vs {} diffs in mean".format(float_conv.__class__.__name__, quantized_conv.__class__.__name__)
        assert torch.allclose(float_out.std(), quantized_out.std()), "{} vs {} diffs in std".format(float_conv.__class__.__name__, quantized_conv.__class__.__name__)
    
	# test forward
    INT8_PER_TENSER_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                            qscheme=QSchemeType.per_tensor,
                                            observer_cls=PerTensorMinMaxObserver,
                                            symmetric=True,
                                            scale_type=ScaleType.float,
                                            round_method=RoundType.half_even,
                                            is_dynamic=False)
    quant_config = QuantizationConfig(weight=INT8_PER_TENSER_SPEC, bias=INT8_PER_TENSER_SPEC)
    for conv, (bn, input_size, q_conv) in conv_map.items():
        input = torch.randn(input_size).to(device)
        float_conv = conv(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=True).to(device)
        float_bn = bn(16).to(device)
        quantized_conv = q_conv.from_float(float_conv, float_bn, quant_config)
        quantized_out = quantized_conv(input)
