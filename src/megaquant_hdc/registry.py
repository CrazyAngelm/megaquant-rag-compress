from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

from .efficient_methods import EFFICIENT_FUNCTIONS
from .methods import HDC_FUNCTIONS

ALL_FUNCTIONS = {**HDC_FUNCTIONS, **EFFICIENT_FUNCTIONS}


@dataclass(frozen=True)
class HDCMethodSpec:
    name: str
    bits: float
    family: str
    title: str
    description: str
    status: str = 'canonical'
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def apply(self, x, **kwargs):
        call_kwargs = dict(self.default_kwargs)
        call_kwargs.update(kwargs)
        return ALL_FUNCTIONS[self.name](x, **call_kwargs)

    def effective_bits_per_dim(self, dim: int, seq_len: int | None = None) -> float:
        meta = self.metadata
        scale_bits = float(meta.get('scale_bits', 0.0))
        mean_bits = float(meta.get('mean_bits', 0.0))
        zero_bits = float(meta.get('zero_bits', 0.0))
        if not meta:
            return float(self.bits)

        def metadata_range_overhead(groups: int) -> float:
            # Simulated meta4/meta8 paths quantize metadata using a shared
            # min/range pair per metadata field. Account for those shared
            # parameters when seq_len is known. This remains a modeled
            # accounting convention, not measured packed storage.
            if not seq_len:
                return 0.0
            param_bits = float(meta.get('meta_range_param_bits', 16.0))
            fields = 0
            for bits in (scale_bits, mean_bits, zero_bits):
                if 0.0 < bits < 16.0:
                    fields += 1
            return groups * fields * 2.0 * param_bits / (float(seq_len) * float(dim))

        mode = meta.get('mode')
        if mode == 'token_group':
            group = max(1, int(meta.get('group', dim)))
            groups = (dim + group - 1) // group
            return float(self.bits) + groups * (scale_bits + mean_bits + zero_bits) / float(dim) + metadata_range_overhead(groups)
        if mode == 'channel':
            if not seq_len:
                return float('nan')
            return float(self.bits) + (scale_bits + mean_bits + zero_bits) / float(seq_len)
        if mode == 'vector':
            return float(self.bits) + (scale_bits + mean_bits + zero_bits) / float(dim)
        if mode == 'sink1':
            base_bits = float(meta.get('base_bits', self.bits))
            base_mode = meta.get('base_mode', 'token_group')
            sink_bits = float(meta.get('sink_bits', 16.0))
            if base_mode == 'channel':
                if not seq_len:
                    return float('nan')
                # Channelwise scale is shared across the prefix, not per token.
                # A preserved sink token adds FP16 payload, but it does not remove
                # the shared channel-scale overhead.
                sink_tokens = min(float(meta.get('sink_tokens', 1.0)), float(seq_len))
                shared = (scale_bits + mean_bits + zero_bits) / float(seq_len)
                return base_bits + shared + sink_tokens * (sink_bits - base_bits) / float(seq_len)
            base_group = int(meta.get('group', dim))
            groups = (dim + base_group - 1) // base_group
            per_token_base = base_bits + groups * (scale_bits + mean_bits + zero_bits) / float(dim)
            shared_range = metadata_range_overhead(groups)
            base = per_token_base + shared_range
            if not seq_len:
                return base
            sink_tokens = min(float(meta.get('sink_tokens', 1.0)), float(seq_len))
            # Per-token code/metadata for sink tokens is replaced by FP16 payload;
            # shared metadata-range parameters remain prefix-level overhead.
            return per_token_base + shared_range + sink_tokens * (sink_bits - per_token_base) / float(seq_len)
        if mode == 'sparse_topk':
            base_bits = float(meta.get('base_bits', self.bits))
            base_mode = meta.get('base_mode', 'token_group')
            if base_mode == 'channel':
                if not seq_len:
                    base = float('nan')
                else:
                    base = base_bits + (scale_bits + mean_bits + zero_bits) / float(seq_len)
            else:
                group = max(1, int(meta.get('group', dim)))
                groups = (dim + group - 1) // group
                base = base_bits + groups * (scale_bits + mean_bits + zero_bits) / float(dim) + metadata_range_overhead(groups)
            k = float(meta.get('k', 1.0))
            index_bits = float(meta.get('index_bits', max(1, (dim - 1).bit_length())))
            value_bits = float(meta.get('value_bits', 16.0))
            return base + k * (index_bits + value_bits) / float(dim)
        return float(self.bits)


HDC_METHODS: Dict[str, HDCMethodSpec] = {
    'centered_sign_1bit': HDCMethodSpec(
        name='centered_sign_1bit',
        bits=1.0,
        family='megaquant',
        title='Centered Sign 1-bit',
        description='Blockwise 1-bit quantizer using per-block mean plus signed residual magnitude.',
        metadata={'mode': 'token_group', 'group': 4, 'scale_bits': 16, 'mean_bits': 16},
    ),
    'blockwise_ternary_2bit': HDCMethodSpec(
        name='blockwise_ternary_2bit',
        bits=2.0,
        family='megaquant',
        title='Blockwise Ternary 2-bit',
        description='Early aggressive 2-bit baseline with blockwise ternary-style reconstruction.',
        metadata={'mode': 'token_group', 'group': 4, 'scale_bits': 16},
    ),
    'blockwise_four_level_2bit': HDCMethodSpec(
        name='blockwise_four_level_2bit',
        bits=2.0,
        family='megaquant',
        title='Blockwise Four-Level 2-bit',
        description='Strict 2-bit descendant of the 7-level design tuned on a small attention proxy.',
        default_kwargs={'split': 0.75},
        metadata={'mode': 'token_group', 'group': 4, 'scale_bits': 16},
    ),
    'centered_four_level_2bit': HDCMethodSpec(
        name='centered_four_level_2bit',
        bits=2.0,
        family='megaquant',
        title='Centered Four-Level 2-bit',
        description='Centered 2-bit four-level quantizer using per-block mean plus quantized residuals.',
        default_kwargs={'split': 0.75},
        metadata={'mode': 'token_group', 'group': 4, 'scale_bits': 16, 'mean_bits': 16},
    ),
    'mixed_precision_avg_2bit': HDCMethodSpec(
        name='mixed_precision_avg_2bit',
        bits=2.0,
        family='megaquant',
        title='Mixed-Precision Average-2-bit',
        description='Mixed-precision average-2-bit variant combining 7-level, 4-level and sign-only paths.',
        default_kwargs={'qsplit': 0.75},
        metadata={'mode': 'token_group', 'group': 4, 'scale_bits': 16},
    ),
    'mixed_precision_avg_2bit_centered_sign': HDCMethodSpec(
        name='mixed_precision_avg_2bit_centered_sign',
        bits=2.0,
        family='megaquant',
        title='Mixed-Precision Average-2-bit with Centered Sign',
        description='Average-2-bit mixed-precision variant that replaces the 1-bit branch with centered sign quantization.',
        default_kwargs={'qsplit': 0.75},
        metadata={'mode': 'token_group', 'group': 4, 'scale_bits': 16, 'mean_bits': 4},
    ),
    'five_level_3bit': HDCMethodSpec(
        name='five_level_3bit',
        bits=3.0,
        family='megaquant',
        title='Five-Level 3-bit',
        description='Simple sign-preserving 5-level 3-bit quantizer.',
        metadata={'mode': 'vector', 'scale_bits': 16},
    ),
    'blockwise_seven_level_3bit': HDCMethodSpec(
        name='blockwise_seven_level_3bit',
        bits=3.0,
        family='megaquant',
        title='Blockwise Seven-Level 3-bit',
        description='Main 3-bit MegaQuant method with blockwise robust scaling and 7 signed levels.',
        metadata={'mode': 'token_group', 'group': 4, 'scale_bits': 16},
    ),
    'centered_seven_level_3bit': HDCMethodSpec(
        name='centered_seven_level_3bit',
        bits=3.0,
        family='megaquant',
        title='Centered Seven-Level 3-bit',
        description='Centered 3-bit seven-level quantizer using per-block mean plus quantized residuals.',
        metadata={'mode': 'token_group', 'group': 4, 'scale_bits': 16, 'mean_bits': 16},
    ),
    'channelwise_four_level_2bit': HDCMethodSpec(
        name='channelwise_four_level_2bit',
        bits=2.0,
        family='megaquant_efficient',
        title='Channelwise Four-Level 2-bit',
        description='Experimental 2-bit four-level quantizer with one FP16 scale per channel across tokens.',
        status='experimental',
        default_kwargs={'split': 0.75},
        metadata={'mode': 'channel', 'scale_bits': 16},
    ),
    'tokenwise_four_level_2bit_g32': HDCMethodSpec(
        name='tokenwise_four_level_2bit_g32',
        bits=2.0,
        family='megaquant_efficient',
        title='Tokenwise Four-Level 2-bit G32',
        description='Experimental 2-bit four-level quantizer using group=32 to reduce scale overhead.',
        status='experimental',
        default_kwargs={'group': 32, 'split': 0.75},
        metadata={'mode': 'token_group', 'group': 32, 'scale_bits': 16},
    ),
    'global_centered_four_level_2bit': HDCMethodSpec(
        name='global_centered_four_level_2bit',
        bits=2.0,
        family='megaquant_efficient',
        title='Global-Centered Four-Level 2-bit',
        description='Experimental 2-bit centered quantizer with one mean and one scale per vector.',
        status='experimental',
        default_kwargs={'split': 0.75},
        metadata={'mode': 'vector', 'scale_bits': 16, 'mean_bits': 16},
    ),
    'channelwise_seven_level_3bit': HDCMethodSpec(
        name='channelwise_seven_level_3bit',
        bits=3.0,
        family='megaquant_efficient',
        title='Channelwise Seven-Level 3-bit',
        description='Experimental 3-bit seven-level quantizer with one FP16 scale per channel across tokens.',
        status='experimental',
        metadata={'mode': 'channel', 'scale_bits': 16},
    ),
    'affine_four_level_2bit_g16': HDCMethodSpec(
        name='affine_four_level_2bit_g16',
        bits=2.0,
        family='megaquant_efficient',
        title='Affine Four-Level 2-bit G16',
        description='Experimental asymmetric uniform 2-bit affine quantizer with group=16.',
        status='experimental',
        metadata={'mode': 'token_group', 'group': 16, 'scale_bits': 16, 'zero_bits': 16},
    ),
    'affine_four_level_2bit_g32': HDCMethodSpec(
        name='affine_four_level_2bit_g32',
        bits=2.0,
        family='megaquant_efficient',
        title='Affine Four-Level 2-bit G32',
        description='Experimental asymmetric uniform 2-bit affine quantizer with group=32.',
        status='experimental',
        metadata={'mode': 'token_group', 'group': 32, 'scale_bits': 16, 'zero_bits': 16},
    ),
    'tokenwise_seven_level_3bit_g32': HDCMethodSpec(
        name='tokenwise_seven_level_3bit_g32',
        bits=3.0,
        family='megaquant_efficient',
        title='Tokenwise Seven-Level 3-bit G32',
        description='Experimental 3-bit seven-level quantizer using group=32 to reduce scale overhead.',
        status='experimental',
        default_kwargs={'group': 32},
        metadata={'mode': 'token_group', 'group': 32, 'scale_bits': 16},
    ),
    'global_centered_seven_level_3bit': HDCMethodSpec(
        name='global_centered_seven_level_3bit',
        bits=3.0,
        family='megaquant_efficient',
        title='Global-Centered Seven-Level 3-bit',
        description='Experimental 3-bit centered quantizer with one mean and one scale per vector.',
        status='experimental',
        metadata={'mode': 'vector', 'scale_bits': 16, 'mean_bits': 16},
    ),
    'affine_seven_level_3bit_g32': HDCMethodSpec(
        name='affine_seven_level_3bit_g32',
        bits=3.0,
        family='megaquant_efficient',
        title='Affine Seven-Level 3-bit G32',
        description='Experimental asymmetric uniform 3-bit affine quantizer with group=32.',
        status='experimental',
        metadata={'mode': 'token_group', 'group': 32, 'scale_bits': 16, 'zero_bits': 16},
    ),
    'affine_four_level_2bit_g64': HDCMethodSpec(
        name='affine_four_level_2bit_g64',
        bits=2.0,
        family='megaquant_efficient',
        title='Affine Four-Level 2-bit G64',
        description='Experimental asymmetric uniform 2-bit affine quantizer with group=64.',
        status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 16, 'zero_bits': 16},
    ),
    'affine_four_level_2bit_g64_meta8': HDCMethodSpec(
        name='affine_four_level_2bit_g64_meta8',
        bits=2.0,
        family='megaquant_efficient',
        title='Affine Four-Level 2-bit G64 Meta8',
        description='Experimental asymmetric uniform 2-bit affine quantizer with group=64 and simulated int8 scale/zero metadata.',
        status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 8, 'zero_bits': 8},
    ),
    'affine_four_level_2bit_g64_meta4': HDCMethodSpec(
        name='affine_four_level_2bit_g64_meta4',
        bits=2.0,
        family='megaquant_efficient',
        title='Affine Four-Level 2-bit G64 Meta4',
        description='Experimental asymmetric uniform 2-bit affine quantizer with group=64 and simulated int4 scale/zero metadata.',
        status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 4, 'zero_bits': 4},
    ),
    'tokenwise_four_level_2bit_g64': HDCMethodSpec(
        name='tokenwise_four_level_2bit_g64',
        bits=2.0,
        family='megaquant_efficient',
        title='Tokenwise Four-Level 2-bit G64',
        description='Experimental 2-bit signed four-level quantizer using group=64.',
        status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 16},
    ),
    'affine_seven_level_3bit_g64': HDCMethodSpec(
        name='affine_seven_level_3bit_g64',
        bits=3.0,
        family='megaquant_efficient',
        title='Affine Seven-Level 3-bit G64',
        description='Experimental asymmetric uniform 3-bit affine quantizer with group=64.',
        status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 16, 'zero_bits': 16},
    ),
    'affine_seven_level_3bit_g64_meta8': HDCMethodSpec(
        name='affine_seven_level_3bit_g64_meta8',
        bits=3.0,
        family='megaquant_efficient',
        title='Affine Seven-Level 3-bit G64 Meta8',
        description='Experimental asymmetric uniform 3-bit affine quantizer with group=64 and simulated int8 scale/zero metadata.',
        status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 8, 'zero_bits': 8},
    ),
    'affine_seven_level_3bit_g64_meta4': HDCMethodSpec(
        name='affine_seven_level_3bit_g64_meta4',
        bits=3.0,
        family='megaquant_efficient',
        title='Affine Seven-Level 3-bit G64 Meta4',
        description='Experimental asymmetric uniform 3-bit affine quantizer with group=64 and simulated int4 scale/zero metadata.',
        status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 4, 'zero_bits': 4},
    ),
    'tokenwise_seven_level_3bit_g64': HDCMethodSpec(
        name='tokenwise_seven_level_3bit_g64',
        bits=3.0,
        family='megaquant_efficient',
        title='Tokenwise Seven-Level 3-bit G64',
        description='Experimental 3-bit seven-level quantizer using group=64.',
        status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 16},
    ),
    'sink1_affine_four_level_2bit_g32': HDCMethodSpec(
        name='sink1_affine_four_level_2bit_g32', bits=2.0, family='megaquant_efficient',
        title='Sink1 Affine Four-Level 2-bit G32', description='Protect first token in FP16, quantize rest with affine 2-bit group=32.', status='experimental',
        metadata={'mode': 'sink1', 'base_bits': 2, 'group': 32, 'scale_bits': 16, 'zero_bits': 16, 'sink_tokens': 1, 'sink_bits': 16},
    ),
    'sink1_affine_four_level_2bit_g64_meta8': HDCMethodSpec(
        name='sink1_affine_four_level_2bit_g64_meta8', bits=2.0, family='megaquant_efficient',
        title='Sink1 Affine Four-Level 2-bit G64 Meta8', description='Protect first token in FP16, quantize rest with affine 2-bit group=64 meta8.', status='experimental',
        metadata={'mode': 'sink1', 'base_bits': 2, 'group': 64, 'scale_bits': 8, 'zero_bits': 8, 'sink_tokens': 1, 'sink_bits': 16},
    ),
    'sink1_channelwise_four_level_2bit': HDCMethodSpec(
        name='sink1_channelwise_four_level_2bit', bits=2.0, family='megaquant_efficient',
        title='Sink1 Channelwise Four-Level 2-bit', description='Protect first token in FP16, quantize rest with channelwise 2-bit.', status='experimental',
        metadata={'mode': 'sink1', 'base_mode': 'channel', 'base_bits': 2, 'scale_bits': 16, 'sink_tokens': 1, 'sink_bits': 16},
    ),
    'sink1_affine_seven_level_3bit_g64_meta8': HDCMethodSpec(
        name='sink1_affine_seven_level_3bit_g64_meta8', bits=3.0, family='megaquant_efficient',
        title='Sink1 Affine Seven-Level 3-bit G64 Meta8', description='Protect first token in FP16, quantize rest with affine 3-bit group=64 meta8.', status='experimental',
        metadata={'mode': 'sink1', 'base_bits': 3, 'group': 64, 'scale_bits': 8, 'zero_bits': 8, 'sink_tokens': 1, 'sink_bits': 16},
    ),
    'sparse1_affine_four_level_2bit_g64_meta8': HDCMethodSpec(
        name='sparse1_affine_four_level_2bit_g64_meta8', bits=2.0, family='megaquant_efficient',
        title='Sparse1 Affine Four-Level 2-bit G64 Meta8', description='Restore top-1 absolute value per vector sparsely, quantize dense remainder with affine2 g64 meta8.', status='experimental',
        metadata={'mode': 'sparse_topk', 'base_bits': 2, 'group': 64, 'scale_bits': 8, 'zero_bits': 8, 'k': 1, 'value_bits': 16},
    ),
    'sparse1_affine_four_level_2bit_g32': HDCMethodSpec(
        name='sparse1_affine_four_level_2bit_g32', bits=2.0, family='megaquant_efficient',
        title='Sparse1 Affine Four-Level 2-bit G32', description='Restore top-1 absolute value per vector sparsely, quantize dense remainder with affine2 g32.', status='experimental',
        metadata={'mode': 'sparse_topk', 'base_bits': 2, 'group': 32, 'scale_bits': 16, 'zero_bits': 16, 'k': 1, 'value_bits': 16},
    ),
    'sparse1_channelwise_four_level_2bit': HDCMethodSpec(
        name='sparse1_channelwise_four_level_2bit', bits=2.0, family='megaquant_efficient',
        title='Sparse1 Channelwise Four-Level 2-bit', description='Restore top-1 absolute value per vector sparsely, quantize dense remainder with channelwise2.', status='experimental',
        metadata={'mode': 'sparse_topk', 'base_mode': 'channel', 'base_bits': 2, 'scale_bits': 16, 'k': 1, 'value_bits': 16},
    ),
    'sparse1_affine_seven_level_3bit_g64_meta8': HDCMethodSpec(
        name='sparse1_affine_seven_level_3bit_g64_meta8', bits=3.0, family='megaquant_efficient',
        title='Sparse1 Affine Seven-Level 3-bit G64 Meta8', description='Restore top-1 absolute value per vector sparsely, quantize dense remainder with affine3 g64 meta8.', status='experimental',
        metadata={'mode': 'sparse_topk', 'base_bits': 3, 'group': 64, 'scale_bits': 8, 'zero_bits': 8, 'k': 1, 'value_bits': 16},
    ),
    'hadamard_affine_four_level_2bit_g64_meta8': HDCMethodSpec(
        name='hadamard_affine_four_level_2bit_g64_meta8', bits=2.0, family='megaquant_efficient',
        title='Hadamard Affine Four-Level 2-bit G64 Meta8', description='Fixed Hadamard/RHT preconditioning plus affine2 g64 meta8.', status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 8, 'zero_bits': 8},
    ),
    'hadamard_affine_four_level_2bit_g32': HDCMethodSpec(
        name='hadamard_affine_four_level_2bit_g32', bits=2.0, family='megaquant_efficient',
        title='Hadamard Affine Four-Level 2-bit G32', description='Fixed Hadamard/RHT preconditioning plus affine2 g32.', status='experimental',
        metadata={'mode': 'token_group', 'group': 32, 'scale_bits': 16, 'zero_bits': 16},
    ),
    'hadamard_affine_seven_level_3bit_g64_meta8': HDCMethodSpec(
        name='hadamard_affine_seven_level_3bit_g64_meta8', bits=3.0, family='megaquant_efficient',
        title='Hadamard Affine Seven-Level 3-bit G64 Meta8', description='Fixed Hadamard/RHT preconditioning plus affine3 g64 meta8.', status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 8, 'zero_bits': 8},
    ),
    'nf2_g64_meta8': HDCMethodSpec(
        name='nf2_g64_meta8', bits=2.0, family='megaquant_efficient',
        title='NF2 G64 Meta8', description='Fixed nonuniform symmetric 2-bit codebook with group=64 int8 scale.', status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 8},
    ),
    'nf3_g64_meta8': HDCMethodSpec(
        name='nf3_g64_meta8', bits=3.0, family='megaquant_efficient',
        title='NF3 G64 Meta8', description='Fixed nonuniform symmetric 3-bit codebook with group=64 int8 scale.', status='experimental',
        metadata={'mode': 'token_group', 'group': 64, 'scale_bits': 8},
    ),
}

def get_method_spec(name: str) -> HDCMethodSpec:
    return HDC_METHODS[name]


def list_method_specs(bits: float | None = None) -> list[HDCMethodSpec]:
    methods = list(HDC_METHODS.values())
    if bits is None:
        return methods
    return [spec for spec in methods if float(spec.bits) == float(bits)]
