import pytest
import torch

from transformer_nmt.config import ModelSettings
from transformer_nmt.model import (
    MultiHeadAttention,
    Seq2SeqTransformer,
    make_causal_mask,
    make_padding_mask,
    make_target_mask,
)


def test_causal_mask_is_lower_triangular():
    mask = make_causal_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0].tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]


def test_padding_and_target_masks():
    tokens = torch.tensor([[2, 5, 6, 0]])
    padding = make_padding_mask(tokens, pad_index=0)
    assert padding.shape == (1, 1, 1, 4) and padding[0, 0, 0].tolist() == [True, True, True, False]
    target = make_target_mask(tokens, pad_index=0)
    assert target.shape == (1, 1, 4, 4)
    assert target[0, 0, 3].tolist() == [True, True, True, False]


def test_attention_fast_path_matches_reference_and_respects_mask():
    torch.manual_seed(0)
    attention = MultiHeadAttention(d_model=16, num_heads=4).eval()
    x = torch.randn(2, 5, 16)
    mask = torch.ones(2, 1, 1, 5, dtype=torch.bool)
    mask[:, :, :, 3:] = False
    fast, _ = attention(x, x, x, mask)
    reference, weights = attention(x, x, x, mask, return_weights=True)
    assert torch.allclose(fast, reference, atol=1e-5)
    assert weights is not None and torch.all(weights[..., 3:] == 0)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 5))


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


def test_forward_shape_and_parameter_count(small_config):
    model = Seq2SeqTransformer(50, 60, small_config, pad_index=0)
    logits = model(torch.randint(1, 50, (3, 7)), torch.randint(1, 60, (3, 6)))
    assert logits.shape == (3, 6, 60)
    assert model.count_parameters() == sum(
        p.numel() for p in {id(p): p for p in model.parameters()}.values()
    )
    assert model.decoder.output_projection.weight is model.decoder.token_embedding.weight


def test_decoder_is_causal(small_config):
    torch.manual_seed(0)
    model = Seq2SeqTransformer(50, 60, small_config, pad_index=0).eval()
    source = torch.randint(1, 50, (1, 6))
    target = torch.randint(1, 60, (1, 6))
    altered = target.clone()
    altered[0, 4:] = (altered[0, 4:] + 1) % 59 + 1
    with torch.no_grad():
        assert torch.allclose(
            model(source, target)[:, :4], model(source, altered)[:, :4], atol=1e-5
        )


def test_padding_tokens_do_not_change_outputs(small_config):
    torch.manual_seed(0)
    model = Seq2SeqTransformer(50, 60, small_config, pad_index=0).eval()
    source = torch.tensor([[4, 5, 6]])
    padded = torch.tensor([[4, 5, 6, 0, 0]])
    target = torch.tensor([[2, 7, 8]])
    with torch.no_grad():
        assert torch.allclose(model(source, target), model(padded, target), atol=1e-5)


def test_post_norm_variant_runs(small_config):
    config = small_config.model_copy(update={"norm_first": False, "tie_target_embeddings": False})
    model = Seq2SeqTransformer(20, 30, config, pad_index=0)
    assert model(torch.randint(1, 20, (2, 4)), torch.randint(1, 30, (2, 3))).shape == (2, 3, 30)


def test_greedy_decode_stops_at_eos_and_pads(small_config):
    model = Seq2SeqTransformer(20, 30, small_config, pad_index=0).eval()

    def always_eos(target_tokens, memory, memory_mask):
        logits = torch.zeros(target_tokens.size(0), target_tokens.size(1), 30)
        logits[..., 3] = 1.0
        return logits

    model.decode = always_eos  # type: ignore[method-assign]
    generated = model.greedy_decode(
        torch.randint(1, 20, (2, 5)), sos_index=2, eos_index=3, max_len=6
    )
    assert generated.shape == (2, 1) and torch.all(generated == 3)


def test_sequence_longer_than_positional_table_raises(small_config):
    model = Seq2SeqTransformer(20, 30, small_config, pad_index=0)
    with pytest.raises(ValueError):
        model(torch.randint(1, 20, (1, 17)), torch.randint(1, 30, (1, 3)))
