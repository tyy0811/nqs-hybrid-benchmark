"""
Transformer-style autoregressive ansatz for 2D TFIM benchmarks.

This module is designed to be imported by 02_tfim_2d_ground_state.py so that the
benchmark/plotting harness stays unchanged while the autoregressive model can be
swapped independently.

Notes
-----
- This is a lightweight Flax/JAX implementation of a causal transformer over
  binary spin tokens.
- The model is written in a "best effort" NetKet-compatible style: it exposes a
  ``conditionals`` method and returns ``0.5 * log P(sigma)`` from ``__call__``
  so that ``|psi(sigma)|^2 = P(sigma)`` for a positive-real autoregressive
  wavefunction.
- Depending on the exact NetKet version in the runtime environment, you may
  need small interface adjustments for direct autoregressive sampling.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax.numpy as jnp


class _TransformerBlock(nn.Module):
    """Single pre-norm causal transformer block."""

    d_model: int
    n_heads: int
    ff_mult: int = 4
    dropout_rate: float = 0.0
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        seq_len = x.shape[-2]
        # Broadcastable causal mask: [1, 1, T, T]
        causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=bool))

        y = nn.LayerNorm(dtype=self.param_dtype)(x)
        y = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=False,
            deterministic=deterministic,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )(y, mask=causal_mask)
        x = x + y

        y = nn.LayerNorm(dtype=self.param_dtype)(x)
        y = nn.Dense(
            self.ff_mult * self.d_model,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )(y)
        y = nn.gelu(y)
        y = nn.Dense(
            self.d_model,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )(y)
        return x + y


class SpinTransformerAR(nn.Module):
    """
    Causal transformer for spin-1/2 product-basis autoregressive amplitudes.

    The model treats each spin as a binary token. It predicts conditional
    log-probabilities for each site given the prefix. The wavefunction is taken
    to be positive-real with

        log psi(sigma) = 0.5 * sum_i log p(sigma_i | sigma_{<i}).

    This makes it immediately suitable for TFIM ground-state VMC, where a real,
    nodeless ansatz is often a reasonable starting point.
    """

    n_sites: int
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ff_mult: int = 4
    dropout_rate: float = 0.0
    param_dtype: Any = jnp.float32
    machine_pow: int = 2

    @staticmethod
    def _tokens_from_spins(spins):
        spins = jnp.asarray(spins)
        # Accept {-1, +1} or {0, 1} encodings.
        is_pm_one = jnp.logical_or(spins == -1, spins == 1)
        tokens_pm = ((spins + 1) // 2).astype(jnp.int32)
        tokens_01 = spins.astype(jnp.int32)
        return jnp.where(is_pm_one, tokens_pm, tokens_01)

    @nn.compact
    def __call__(self, sigma, *, deterministic: bool = True):
        log_cond = self.conditionals(sigma, deterministic=deterministic)
        tokens = self._tokens_from_spins(sigma)
        chosen = jnp.take_along_axis(log_cond, tokens[..., None], axis=-1)[..., 0]
        return 0.5 * jnp.sum(chosen, axis=-1)

    @nn.compact
    def conditionals(self, sigma, *, deterministic: bool = True):
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            sigma = sigma[None, :]

        tokens = self._tokens_from_spins(sigma)
        bos = jnp.full((tokens.shape[0], 1), 2, dtype=jnp.int32)
        prefix = jnp.concatenate([bos, tokens[:, :-1]], axis=1)

        tok_emb = nn.Embed(
            num_embeddings=3,  # spin down, spin up, BOS
            features=self.d_model,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            name="token_embed",
        )(prefix)

        pos_emb = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (self.n_sites, self.d_model),
            self.param_dtype,
        )
        x = tok_emb + pos_emb[None, :, :]

        for layer_idx in range(self.n_layers):
            x = _TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ff_mult=self.ff_mult,
                dropout_rate=self.dropout_rate,
                param_dtype=self.param_dtype,
                name=f"block_{layer_idx}",
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(dtype=self.param_dtype)(x)
        logits = nn.Dense(
            2,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            name="logit_head",
        )(x)
        return nn.log_softmax(logits, axis=-1)

    def conditional(self, sigma, index: int, *, deterministic: bool = True):
        """Convenience helper for sampler APIs that query one site at a time."""
        log_cond = self.conditionals(sigma, deterministic=deterministic)
        return log_cond[..., index, :]


def build_transformer_model(
    hilbert,
    *,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    ff_mult: int = 4,
    dropout_rate: float = 0.0,
    param_dtype=float,
):
    """Factory returning a transformer-style autoregressive spin model."""
    n_sites = int(getattr(hilbert, "size", getattr(hilbert, "N", None)))
    if n_sites is None:
        raise ValueError("Could not infer the number of sites from the Hilbert space.")

    dtype = jnp.float32 if param_dtype in (float, None) else param_dtype
    return SpinTransformerAR(
        n_sites=n_sites,
        d_model=int(d_model),
        n_layers=int(n_layers),
        n_heads=int(n_heads),
        ff_mult=int(ff_mult),
        dropout_rate=float(dropout_rate),
        param_dtype=dtype,
    )


def estimate_transformer_n_parameters(
    hilbert,
    *,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    ff_mult: int = 4,
    dropout_rate: float = 0.0,
    n_samples: int = 16,
    param_dtype=float,
) -> int:
    """Instantiate a tiny VQS to estimate transformer parameter count."""
    import netket as nk

    model = build_transformer_model(
        hilbert,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_mult=ff_mult,
        dropout_rate=dropout_rate,
        param_dtype=param_dtype,
    )

    # Prefer direct autoregressive sampling; fall back if the local NetKet build
    # expects a stricter interface than this lightweight model provides.
    try:
        sampler = nk.sampler.ARDirectSampler(hilbert)
        vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)
    except Exception:
        sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=4)
        vstate = nk.vqs.MCState(sampler, model, n_samples=max(64, n_samples))
    return int(vstate.n_parameters)
