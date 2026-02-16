"""Codificación de prompts largos (>77 tokens) por chunking + concatenación.

Estilo lpw_stable_diffusion_xl: divide tokens en bloques de 75 contenido + BOS + EOS,
encodea cada chunk por separado y concatena hidden states para mantener fidelidad.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from config.constants import CONTENT_TOKENS_PER_CHUNK, DEFAULT_CLIP_SKIP

if TYPE_CHECKING:
    import torch
    from diffusers import StableDiffusionXLPipeline


def encode_long_prompt_sdxl(
    pipe: StableDiffusionXLPipeline,
    full_prompt: str,
    full_negative: str,
    clip_skip: int = DEFAULT_CLIP_SKIP,
    device: Optional[str] = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Codifica prompt y negative largos por chunking + concat.

    Devuelve (prompt_embeds, negative_embeds, pooled_positive, pooled_negative).
    """
    import torch

    device = device or pipe._execution_device
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    enc1, enc2 = pipe.text_encoder, pipe.text_encoder_2
    eos_id1 = tok1.eos_token_id
    eos_id2 = tok2.eos_token_id
    bos_id1 = tok1.bos_token_id if tok1.bos_token_id is not None else eos_id1
    bos_id2 = tok2.bos_token_id if tok2.bos_token_id is not None else eos_id2

    def _tokenize_no_trunc(tokenizer, text: str) -> list[int]:
        out = tokenizer(text, truncation=False, return_tensors="pt", padding=False)
        ids = out.input_ids[0].tolist()
        if (
            len(ids) >= 2
            and ids[0] == getattr(tokenizer, "bos_token_id", None)
            and ids[-1] == tokenizer.eos_token_id
        ):
            ids = ids[1:-1]
        return ids

    def _chunk_ids(ids: list[int], bos_id: int, eos_id: int) -> list[list[int]]:
        chunks: list[list[int]] = []
        i = 0
        while i < len(ids):
            block = ids[i : i + CONTENT_TOKENS_PER_CHUNK]
            i += len(block)
            pad_len = CONTENT_TOKENS_PER_CHUNK - len(block)
            chunk = [bos_id] + block + [eos_id] * (pad_len + 1)
            chunks.append(chunk)
        return chunks

    # Tokenizar sin truncar
    p1 = _tokenize_no_trunc(tok1, full_prompt)
    p2 = _tokenize_no_trunc(tok2, full_prompt)
    n1 = _tokenize_no_trunc(tok1, full_negative)
    n2 = _tokenize_no_trunc(tok2, full_negative)

    cp1 = _chunk_ids(p1, bos_id1, eos_id1)
    cp2 = _chunk_ids(p2, bos_id2, eos_id2)
    cn1 = _chunk_ids(n1, bos_id1, eos_id1)
    cn2 = _chunk_ids(n2, bos_id2, eos_id2)

    num_chunks = max(len(cp1), len(cp2), len(cn1), len(cn2))
    if num_chunks == 0:
        num_chunks = 1

    # Rellenar con chunks de padding si hace falta
    def _pad_chunks(chunks: list[list[int]], bos: int, eos: int) -> list[list[int]]:
        while len(chunks) < num_chunks:
            chunks.append([bos] + [eos] * (CONTENT_TOKENS_PER_CHUNK + 1))
        return chunks

    cp1 = _pad_chunks(cp1, bos_id1, eos_id1)
    cp2 = _pad_chunks(cp2, bos_id2, eos_id2)
    cn1 = _pad_chunks(cn1, bos_id1, eos_id1)
    cn2 = _pad_chunks(cn2, bos_id2, eos_id2)

    # Misma capa que Compel (penúltima) para consistencia
    layer_idx = -2  # PENULTIMATE_HIDDEN_STATES
    pooled_pos = None
    pooled_neg = None
    dtype = enc2.dtype
    embeds_pos: list[torch.Tensor] = []
    embeds_neg: list[torch.Tensor] = []

    for i in range(num_chunks):
        # Positive
        t1 = torch.tensor([cp1[i]], dtype=torch.long, device=device)
        t2 = torch.tensor([cp2[i]], dtype=torch.long, device=device)
        o1 = enc1(t1, output_hidden_states=True)
        o2 = enc2(t2, output_hidden_states=True)
        if i == 0:
            pooled_pos = o2[0]
        h1 = o1.hidden_states[layer_idx].to(dtype)
        h2 = o2.hidden_states[layer_idx].to(dtype)
        embeds_pos.append(torch.cat([h1, h2], dim=-1))

        # Negative
        t1n = torch.tensor([cn1[i]], dtype=torch.long, device=device)
        t2n = torch.tensor([cn2[i]], dtype=torch.long, device=device)
        o1n = enc1(t1n, output_hidden_states=True)
        o2n = enc2(t2n, output_hidden_states=True)
        if i == 0:
            pooled_neg = o2n[0]
        h1n = o1n.hidden_states[layer_idx].to(dtype)
        h2n = o2n.hidden_states[layer_idx].to(dtype)
        embeds_neg.append(torch.cat([h1n, h2n], dim=-1))

    prompt_embeds = torch.cat(embeds_pos, dim=1)
    negative_embeds = torch.cat(embeds_neg, dim=1)

    if pooled_pos is None:
        pooled_pos = enc2(torch.tensor([cp2[0]], dtype=torch.long, device=device))[0]
    if pooled_neg is None:
        pooled_neg = enc2(torch.tensor([cn2[0]], dtype=torch.long, device=device))[0]

    pooled_pos = pooled_pos.to(dtype)
    pooled_neg = pooled_neg.to(dtype)

    return prompt_embeds, negative_embeds, pooled_pos, pooled_neg
