import pytest
import torch

from transformer_nmt.config import ModelSettings
from transformer_nmt.model import (
    DecoderBlock,
    EncoderBlock,
    EncoderDecoder,
    MultiHeadAttention,
    create_padding_mask,
    make_causal_mask,
    make_target_mask,
)


@pytest.fixture
def small_config() -> ModelSettings:
    return ModelSettings(
        _env_file=None,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_position_embeddings=16,
    )


def test_causal_mask_is_lower_triangular():
    mask = make_causal_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0].tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]


def test_padding_and_target_masks():
    tokens = torch.tensor([[2, 5, 6, 0]])
    padding = create_padding_mask(tokens, pad_idx=0)
    assert padding.shape == (1, 1, 1, 4) and padding[0, 0, 0].tolist() == [True, True, True, False]
    target = make_target_mask(tokens, pad_idx=0)
    assert target.shape == (1, 1, 4, 4) and target[0, 0, 3].tolist() == [True, True, True, False]


def test_attention_fused_path_matches_manual_path_and_respects_mask():
    torch.manual_seed(0)
    attention = MultiHeadAttention(d_model=16, nhead=4).eval()
    x = torch.randn(2, 5, 16)
    mask = torch.ones(2, 1, 1, 5, dtype=torch.bool)
    mask[:, :, :, 3:] = False
    fused, _ = attention(x, x, x, mask)
    manual, weights = attention(x, x, x, mask, need_weights=True)
    assert torch.allclose(fused, manual, atol=1e-5)
    assert weights is not None and torch.all(weights[..., 3:] == 0)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 5))


def test_attention_paths_agree_on_fully_masked_rows():
    torch.manual_seed(0)
    attention = MultiHeadAttention(d_model=16, nhead=4).eval()
    x = torch.randn(2, 3, 16)
    mask = torch.ones(2, 1, 1, 3, dtype=torch.bool)
    mask[1] = False
    fused, _ = attention(x, x, x, mask)
    manual, weights = attention(x, x, x, mask, need_weights=True)
    assert torch.isfinite(manual).all() and torch.allclose(fused, manual, atol=1e-5)
    assert weights is not None and torch.all(weights[1] == 0)


def test_blocks_preserve_shape():
    x = torch.randn(2, 5, 32)
    memory = torch.randn(2, 7, 32)
    assert EncoderBlock(32, 4, 64)(x).shape == x.shape
    assert DecoderBlock(32, 4, 64)(x, memory).shape == x.shape


def test_forward_shape_blocks_and_parameter_count(
    small_config,
):
    model = EncoderDecoder(50, 60, small_config)
    logits = model(torch.randint(1, 50, (3, 7)), torch.randint(1, 60, (3, 6)))
    assert logits.shape == (3, 6, 60)
    assert len(model.encoder.blocks) == 2 and len(model.decoder.blocks) == 2
    assert all(isinstance(block, EncoderBlock) for block in model.encoder.blocks)
    assert all(isinstance(block, DecoderBlock) for block in model.decoder.blocks)
    unique = {id(p): p for p in model.parameters()}
    assert model.count_parameters() == sum(p.numel() for p in unique.values())
    assert model.decoder.output_projection.weight is model.decoder.token_emb.weight


def test_decoder_is_causal(
    small_config,
):
    torch.manual_seed(0)
    model = EncoderDecoder(50, 60, small_config).eval()
    source = torch.randint(1, 50, (1, 6))
    target = torch.randint(1, 60, (1, 6))
    altered = target.clone()
    altered[0, 4:] = (altered[0, 4:] + 1) % 59 + 1
    with torch.no_grad():
        assert torch.allclose(
            model(source, target)[:, :4], model(source, altered)[:, :4], atol=1e-5
        )


def test_padding_tokens_do_not_change_outputs(
    small_config,
):
    torch.manual_seed(0)
    model = EncoderDecoder(50, 60, small_config).eval()
    target = torch.tensor([[2, 7, 8]])
    with torch.no_grad():
        plain = model(torch.tensor([[4, 5, 6]]), target)
        padded = model(torch.tensor([[4, 5, 6, 0, 0]]), target)
    assert torch.allclose(plain, padded, atol=1e-5)


def test_untied_output_projection(
    small_config,
):
    config = small_config.model_copy(update={"tie_target_embeddings": False})
    model = EncoderDecoder(20, 30, config)
    assert model.decoder.output_projection.weight is not model.decoder.token_emb.weight
    assert model(torch.randint(1, 20, (2, 4)), torch.randint(1, 30, (2, 3))).shape == (2, 3, 30)


def test_greedy_decode_stops_at_eos_and_pads(
    small_config,
):
    model = EncoderDecoder(20, 30, small_config).eval()

    def staggered_eos(
        tgt,
        memory,
        memory_padding_mask,
    ):
        step = tgt.size(1)  # decoder input length == 1-based decoding step
        logits = torch.zeros(tgt.size(0), tgt.size(1), 30)
        logits[0, -1, 3 if step == 1 else 7] = 1.0  # row 0 finishes immediately
        logits[1, -1, 3 if step == 3 else 7] = 1.0  # row 1 finishes on step 3
        return logits

    model.train()
    model.decode = staggered_eos  # type: ignore[method-assign]
    generated = model.greedy_decode(torch.randint(1, 20, (2, 5)), sos_idx=2, eos_idx=3, max_len=6)
    assert generated.tolist() == [[3, 0, 0], [7, 7, 3]]
    assert model.training  # mode restored


def test_sequence_longer_than_positional_table_raises(
    small_config,
):
    model = EncoderDecoder(20, 30, small_config)
    with pytest.raises(ValueError):
        model(torch.randint(1, 20, (1, 17)), torch.randint(1, 30, (1, 3)))
