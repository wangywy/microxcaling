import os
import sys
import torch
from torch.utils.data import DataLoader
from dataclasses import replace
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchmo import ModelQuantizer
from torchmo.quantization.config.config import Config, QuantizationSpec, QuantizationConfig, AWQConfig, SmoothQuantConfig, GPTQConfig
from torchmo.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from torchmo.quantization.observer.observer import PlaceholderObserver, PerTensorMinMaxObserver, PerChannelMinMaxObserver, PerGroupMinMaxObserver

# Quant_Spec
FLOAT16_SPEC = QuantizationSpec(dtype=Dtype.float16, observer_cls=PlaceholderObserver)

BFLOAT16_SPEC = QuantizationSpec(dtype=Dtype.bfloat16, observer_cls=PlaceholderObserver)

FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                       qscheme=QSchemeType.per_tensor,
                                       observer_cls=PerTensorMinMaxObserver,
                                       is_dynamic=False)

INT4_PER_TENSER_SPEC = QuantizationSpec(dtype=Dtype.int4,
                                        qscheme=QSchemeType.per_tensor,
                                        observer_cls=PerTensorMinMaxObserver,
                                        symmetric=True,
                                        scale_type=ScaleType.float,
                                        round_method=RoundType.half_even,
                                        is_dynamic=False)
										
INT4_PER_CHANNEL_SPEC = QuantizationSpec(dtype=Dtype.int4,
                                         observer_cls=PerChannelMinMaxObserver,
                                         symmetric=True,
                                         scale_type=ScaleType.float,
                                         round_method=RoundType.half_even,
                                         qscheme=QSchemeType.per_channel,
                                         ch_axis=0,
                                         is_dynamic=False)

INT4_PER_GROUP_SYM_SPEC = QuantizationSpec(dtype=Dtype.int4,
                                           observer_cls=PerGroupMinMaxObserver,
                                           symmetric=True,
                                           scale_type=ScaleType.float,
                                           round_method=RoundType.half_even,
                                           qscheme=QSchemeType.per_group,
                                           ch_axis=1,
                                           is_dynamic=False,
                                           group_size=128)

DEFAULT_UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(dtype=Dtype.uint4,
                                                     observer_cls=PerGroupMinMaxObserver,
                                                     symmetric=False,
                                                     scale_type=ScaleType.float,
                                                     round_method=RoundType.half_even,
                                                     qscheme=QSchemeType.per_group,
                                                     ch_axis=1,
                                                     is_dynamic=False,
                                                     group_size=128)

INT8_PER_TENSER_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                        qscheme=QSchemeType.per_tensor,
                                        observer_cls=PerTensorMinMaxObserver,
                                        symmetric=True,
                                        scale_type=ScaleType.float,
                                        round_method=RoundType.half_even,
                                        is_dynamic=False)
										
INT8_PER_TENSER_DYNAMIC_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                qscheme=QSchemeType.per_tensor,
                                                observer_cls=PerTensorMinMaxObserver,
                                                symmetric=True,
                                                scale_type=ScaleType.float,
                                                round_method=RoundType.half_even,
                                                is_dynamic=True)

# Float16 config
DEFAULT_FLOAT16_CONFIG = QuantizationConfig(input_tensors=FLOAT16_SPEC, weight=FLOAT16_SPEC)

# Fp8(e4m3) config
DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                           weight=FP8_PER_TENSOR_SPEC)

DEFAULT_W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                                weight=FP8_PER_TENSOR_SPEC,
                                                                output_tensors=FP8_PER_TENSOR_SPEC)

# Per tensor config
DEFAULT_W_INT4_PER_TENSOR_CONFIG = QuantizationConfig(weight=INT4_PER_TENSER_SPEC)

DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSER_SPEC,
                                                             weight=INT8_PER_TENSER_SPEC)

DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSER_DYNAMIC_SPEC,
                                                                     weight=INT8_PER_TENSER_DYNAMIC_SPEC)

# Per Channel Config
DEFAULT_W_INT4_PER_CHANNEL_CONFIG = QuantizationConfig(weight=INT4_PER_CHANNEL_SPEC)

# Per Group Config
DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=INT4_PER_GROUP_SYM_SPEC)

DEFAULT_W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)

DEFAULT_W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG = QuantizationConfig(input_tensors=BFLOAT16_SPEC,
                                                                 weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)
																 
# Default AWQ Config
DEFAULT_AWQ_CONFIG = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=AWQConfig())

# Default SmoothQuant Config
DEFAULT_SMOOTH_QUANT_CONFIG = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG,
                                     algo_config=SmoothQuantConfig())

# Default GPTQ Config
DEFAULT_GPTQ_CONFIG = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=GPTQConfig())

# mutil_gpu should disable lm_head replacement in opt, qwen, llama, and it is needed in algos.
EXCLUDE_LAYERS = ["lm_head"]

sys.path.append("..")
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(model_name="facebook/opt-125m", device=default_device):
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs['input_ids'].to(device))
    return calib_dataloader
	
def quantize_model(quant_config, model_name="facebook/opt-125m", multi_gpu=False):

    # Get quantizer
    quantizer = ModelQuantizer(quant_config)

    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        model = model.to(default_device)
    # Get dataloader, if multi_gpu, give the first layer's device
    calib_dataloader = get_dataloader(model_name, model.device)

    quant_model = quantizer.quantize_model(model, calib_dataloader)
    # Inference with quantized model
    for i in calib_dataloader:
        quant_model(i)

    return quant_model
	
def test_smoke_basic_quantization():
    '''
    Test Features:
        Data Type:                Float16 / Bfloat16 / Int4 / Uint4 / Int8 / FP8(e4m3fn)
        Quantization Strategies:  Post Training Weight-Only Quantization / Post Training Dynamic Quantization / Post Training Static Quantization
        Quantization Scheme:      Per tensor / Per channel / Per group
        Symmetric:                Symmetric / Asymmetric
        In-Place Replace OP:      nn.Linear
    '''
    quant_config_list = [
        DEFAULT_FLOAT16_CONFIG,
        DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG,
        DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG,
        DEFAULT_W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG,
        DEFAULT_W_INT4_PER_TENSOR_CONFIG,
        DEFAULT_W_INT4_PER_CHANNEL_CONFIG,
        DEFAULT_W_UINT4_PER_GROUP_CONFIG,
        DEFAULT_W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG,
        DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG,
        DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG,
    ]

    for quant_config in quant_config_list:
        quant_config = Config(global_quant_config=quant_config, exclude=EXCLUDE_LAYERS)
        quantize_model(quant_config)
        # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after device support multi-GPU
		
def test_smoke_kv_cache_quant():
    '''
    Test Features:
        KV-Cache Quant:           FP8 KV-Cache Quant
    '''
    quant_config = Config(global_quant_config=DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG)
    KV_CACHE_CFG = {
                    "*.v_proj":
                    QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                       output_tensors=FP8_PER_TENSOR_SPEC,
                                       weight=FP8_PER_TENSOR_SPEC),
                    "*.k_proj":
                    QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                       output_tensors=FP8_PER_TENSOR_SPEC,
                                       weight=FP8_PER_TENSOR_SPEC),
                }
    quant_config = replace(quant_config, layer_quant_config=KV_CACHE_CFG, exclude=EXCLUDE_LAYERS)

    quantize_model(quant_config)
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after device support multi-GPU

def test_smoke_calibration_method():
    '''
    Test Features:
        Calibration method:       MinMax / Percentile / MSE
    '''
    quant_config = Config(global_quant_config=DEFAULT_W_INT4_PER_TENSOR_CONFIG, exclude=EXCLUDE_LAYERS)

    from torchmo.quantization.observer.observer import PerTensorPercentileObserver, PerTensorMSEObserver

    supported_observers = [PerTensorPercentileObserver, PerTensorMSEObserver]

    for observer in supported_observers:
        quant_config.global_quant_config.weight = replace(quant_config.global_quant_config.weight, observer_cls=observer)
        quantize_model(quant_config)
        # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after device support multi-GPU
		
def test_smoke_export_torchmo_fp8_safetensors():
    '''
    Test Features:
        Export Format:             FP8 SafeTensors
    '''
    quant_config = Config(global_quant_config=DEFAULT_W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG,
                          exclude=EXCLUDE_LAYERS)

    with torch.inference_mode():
        export_path = "./smoke_test_output"
        from torchmo import ModelExporter
        from torchmo.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
        exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
#        for multi_gpu in [False, True]:# TODO: uncomment after device support multi-GPU
        for multi_gpu in [False]:
            model = quantize_model(quant_config, model_name="Qwen/Qwen1.5-0.5B", multi_gpu=multi_gpu)
            exporter.export_model_info(model, quant_config=quant_config)
        if os.path.isdir(export_path):
            try:
                os.rmdir(export_path)
            except OSError as e:
                pass
				
def test_smoke_export_torchmo_pergroup_safetensors():
    '''
    Test Features:
        Export Format:             Per_group SafeTensors
    '''
    quant_config = Config(global_quant_config=DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG,
                          exclude=EXCLUDE_LAYERS)

    with torch.inference_mode():
        export_path = "./smoke_test_output"
        from torchmo import ModelExporter
        from torchmo.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
        exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
#        for multi_gpu in [False, True]:# TODO: uncomment after device support multi-GPU
        for multi_gpu in [False]:
            model = quantize_model(quant_config, model_name="Qwen/Qwen1.5-0.5B", multi_gpu=multi_gpu)
            exporter.export_model_info(model, quant_config=quant_config)
        if os.path.isdir(export_path):
            try:
                os.rmdir(export_path)
            except OSError as e:
                pass
				
def test_smoke_export_torchmo_pertensor_safetensors():
    '''
    Test Features:
        Export Format:             Per_tensor SafeTensors
    '''
    quant_config = Config(global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG,
                          exclude=EXCLUDE_LAYERS)

    with torch.inference_mode():
        export_path = "./smoke_test_output"
        from torchmo import ModelExporter
        from torchmo.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
        exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
#        for multi_gpu in [False, True]:# TODO: uncomment after device support multi-GPU
        for multi_gpu in [False]:
            model = quantize_model(quant_config, model_name="Qwen/Qwen1.5-0.5B", multi_gpu=multi_gpu)
            exporter.export_model_info(model, quant_config=quant_config)
        if os.path.isdir(export_path):
            try:
                os.rmdir(export_path)
            except OSError as e:
                pass
				
def test_smoke_export_torchmo_perchannel_safetensors():
    '''
    Test Features:
        Export Format:             Per_channel SafeTensors
    '''
    quant_config = Config(global_quant_config=DEFAULT_W_INT4_PER_CHANNEL_CONFIG,
                          exclude=EXCLUDE_LAYERS)

    with torch.inference_mode():
        export_path = "./smoke_test_output"
        from torchmo import ModelExporter
        from torchmo.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
        exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
#        for multi_gpu in [False, True]:# TODO: uncomment after device support multi-GPU
        for multi_gpu in [False]:
            model = quantize_model(quant_config, model_name="Qwen/Qwen1.5-0.5B", multi_gpu=multi_gpu)
            exporter.export_model_info(model, quant_config=quant_config)
        if os.path.isdir(export_path):
            try:
                os.rmdir(export_path)
            except OSError as e:
                pass
				
def test_smoke_export_onnx():
    '''
    Test Features:
        Export Format:            ONNX
    '''
    quant_config = Config(global_quant_config=DEFAULT_W_INT4_PER_TENSOR_CONFIG, exclude=EXCLUDE_LAYERS)
    with torch.inference_mode():
        calib_dataloader = get_dataloader()
        batch_iter = iter(calib_dataloader)
        input_args = next(batch_iter)

        from torchmo import ModelExporter
        from torchmo.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
        export_path = "./smoke_test_output"
        exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
        # for multi_gpu in [False, True]:# TODO: uncomment after device support multi-GPU
        for multi_gpu in [False]:
            model = quantize_model(quant_config, multi_gpu=multi_gpu)
            exporter.export_onnx_model(model, input_args.to(model.device))
        if os.path.isdir(export_path):
            try:
                os.rmdir(export_path)
            except OSError as e:
                pass
				
def test_smoke_eager_save_load():
    '''
    Test Features:
        Eager mode save and load functions
    '''
    quant_config = Config(global_quant_config=DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG,
                          exclude=EXCLUDE_LAYERS)

    with torch.inference_mode():
        calib_dataloader = get_dataloader()
        batch_iter = iter(calib_dataloader)
        input_args = next(batch_iter)

        export_path = "./smoke_test_output"
#        for multi_gpu in [False, True]:# TODO: uncomment after device support multi-GPU
        for multi_gpu in [False]:
            quant_model = quantize_model(quant_config, model_name="facebook/opt-125m", multi_gpu=multi_gpu)
            saved_out = quant_model(input_args)
            from torchmo import save_params
            save_params(quant_model, model_type="opt", export_dir=export_path)

            from torchmo import load_params
            model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
            model.eval()
            model = model.to(default_device)
            json_path = export_path + "/opt.json"
            safetensors_path = export_path + "/opt.safetensors"
            loaded_model = load_params(model, json_path=json_path, safetensors_path=safetensors_path)
            loaded_out = loaded_model(input_args)

            assert torch.allclose(saved_out['logits'], loaded_out['logits'], atol=1e-2)

        if os.path.isdir(export_path):
            try:
                os.rmdir(export_path)
            except OSError as e:
                pass
				
def set_config_for_quant_algo(quant_config):
    quant_config.algo_config.scaling_layers = [
        {"prev_op": "self_attn_layer_norm", "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "inp": "self_attn.q_proj", "module2inspect": "self_attn", "has_kwargs
": True, "help": "attention input"},
        {"prev_op": "self_attn.v_proj", "layers": ["self_attn.out_proj"], "inp": "self_attn.out_proj", "module2inspect": None, "has_kwargs": False, "help": "attention out"},
        {"prev_op": "final_layer_norm", "layers": ["fc1"], "inp": "fc1", "module2inspect": None, "has_kwargs": False, "help": "linear 1"},
        {"prev_op": "fc1", "layers": ["fc2"], "inp": "fc2", "module2inspect": None, "has_kwargs": False, "help": "linear 2"}
    ]
    quant_config.algo_config.model_decoder_layers = "model.decoder.layers"
    quant_config.algo_config.embedding_layers = ["model.decoder.embed_tokens", "model.decoder.embed_positions"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
    return quant_config

def test_smoke_smooth_quant_quantization():
    '''
    Test Features:
        Pre-Quant Optimization:   SmoothQuant
    '''
    quant_config = DEFAULT_SMOOTH_QUANT_CONFIG
    quant_config = set_config_for_quant_algo(quant_config)
    quantize_model(quant_config)
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after device support multi-GPU

def test_smoke_awq_quantization():
    '''
    Test Features:
        Quant Algorithm:          AWQ
    '''
    quant_config = DEFAULT_AWQ_CONFIG
    quant_config = set_config_for_quant_algo(quant_config)
    quantize_model(quant_config)
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after device support multi-GPU
	
def test_smoke_gptq_quantization():
    '''
    Test Features:
        Quant Algorithm:          GPTQ
    '''
    quant_config = DEFAULT_GPTQ_CONFIG
    quant_config.algo_config.inside_layer_modules = ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.out_proj", "fc1", "fc2"]
    quant_config.algo_config.model_decoder_layers = "model.decoder.layers"
    quant_config.algo_config.embedding_layers = ["model.decoder.embed_tokens", "model.decoder.embed_positions"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
    quantize_model(quant_config)
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after device support multi-GPU
