import flash_attn
import torch
from accelerate.logging import get_logger
from diffusers.models.embeddings import apply_rotary_emb
from flash_attn.flash_attn_interface import _flash_attn_forward, flash_attn_varlen_func

logger = get_logger(__name__)


def get_cu_seqlens(attention_mask):
    """Calculate cu_seqlens_q, cu_seqlens_kv

    Args:
        attention_mask (torch.Tensor): boolean attention mask of shape: [B, 1, 1, N]

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = attention_mask.shape[0]
    text_len = attention_mask.sum(dim=-1, dtype=torch.int)
    max_len = attention_mask.shape[-1]

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i]
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


class HunyuanVideoFlashAttnProcessor:
    def __init__(self):
        pass

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        batch_size = hidden_states.shape[0]
        img_seq_len = hidden_states.shape[1]
        txt_seq_len = 0

        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

            txt_seq_len = encoder_hidden_states.shape[1]

        max_seqlen_q = max_seqlen_kv = img_seq_len + txt_seq_len
        cu_seqlens_q = cu_seqlens_kv = get_cu_seqlens(attention_mask)

        query = query.transpose(1, 2).reshape(-1, query.shape[1], query.shape[3])
        key = key.transpose(1, 2).reshape(-1, key.shape[1], key.shape[3])
        value = value.transpose(1, 2).reshape(-1, value.shape[1], value.shape[3])
        hidden_states = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            softmax_scale=None,
            dropout_p=0.0,
            causal=False,
        )

        hidden_states = hidden_states.reshape(batch_size, max_seqlen_q, -1)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
