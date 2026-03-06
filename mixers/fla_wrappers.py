"""
fla_wrappers.py — Zoology-compatible wrappers for fla RetNet and GLA
RISE Lab, Purdue University  |  March 2026

════════════════════════════════════════════════════════════════════════════════
WHY THESE WRAPPERS EXIST
════════════════════════════════════════════════════════════════════════════════

Zoology's ModuleConfig instantiates sequence mixers as:

    Mixer(d_model, **kwargs)               ← Zoology calling convention

fla's layer classes use:

    MultiScaleRetention(hidden_size, ...)  ← fla calling convention
    GatedLinearAttention(hidden_size, ...) ← fla calling convention

The wrappers below translate between the two, so configs can reference
"fla_wrappers.RetNetWrapper" and "fla_wrappers.GLAWrapper" without depending
on Zoology's built-in zoology/mixers/fla.py (whose wrapper class names differ
across versions and may not exist in some installs).

════════════════════════════════════════════════════════════════════════════════
fla OUTPUT PROTOCOL
════════════════════════════════════════════════════════════════════════════════

fla layers return a tuple:  (output, *extras)
  output: (B, T, hidden_size)  — the tensor Zoology needs
  extras: recurrent state, attention weights, etc. — Zoology ignores these

Both wrappers unpack the tuple and return just the output tensor, which is
what Zoology's TransformerBlock expects.

════════════════════════════════════════════════════════════════════════════════
fla API REFERENCE (confirmed from fla-org/flash-linear-attention README)
════════════════════════════════════════════════════════════════════════════════

MultiScaleRetention(
    hidden_size : int,
    num_heads   : int,
    mode        : str   = 'chunk',   # 'chunk' | 'fused_recurrent' | 'parallel'
    ...
)
  forward(x: Tensor[B,T,H]) -> (Tensor[B,T,H], ...)

GatedLinearAttention(
    hidden_size       : int,
    num_heads         : int,
    mode              : str   = 'chunk',
    expand_k          : float = 0.5,
    expand_v          : float = 1.0,
    use_output_gate   : bool  = True,
    gate_fn           : str   = 'swish',
    ...
)
  forward(x: Tensor[B,T,H]) -> (Tensor[B,T,H], ...)
"""

import torch
import torch.nn as nn


class RetNetWrapper(nn.Module):
    """
    Zoology-compatible wrapper around fla.layers.MultiScaleRetention.

    Accepts:  RetNetWrapper(d_model, num_heads=2, mode='chunk')
    Returns:  Tensor[B, T, d_model]  (not a tuple — Zoology expects raw tensor)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        mode: str = "chunk",
        **kwargs,
    ):
        super().__init__()
        # Import here so the module is importable even if fla isn't installed yet
        # (setup.sh verifies the import; this defers the hard error to runtime)
        try:
            from fla.layers import MultiScaleRetention
        except ImportError as e:
            raise ImportError(
                "fla is required for RetNetWrapper. "
                "Install with: pip install flash-linear-attention\n"
                f"Original error: {e}"
            ) from e

        self.retnet = MultiScaleRetention(
            hidden_size=d_model,
            num_heads=num_heads,
            mode=mode,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        x: (B, T, d_model)
        returns: (B, T, d_model)

        fla returns (output, recurrent_state, ...) — we unpack and return only output.
        The *args/**kwargs absorb any extra arguments Zoology may pass
        (e.g. attention_mask, position_ids) without breaking.
        """
        output, *_ = self.retnet(x)
        return output


class GLAWrapper(nn.Module):
    """
    Zoology-compatible wrapper around fla.layers.GatedLinearAttention.

    Accepts:  GLAWrapper(d_model, num_heads=2, mode='chunk', ...)
    Returns:  Tensor[B, T, d_model]

    GLA defaults match the Based paper comparison settings:
      expand_k=0.5, expand_v=1.0, use_output_gate=True, mode='chunk'
    These are fla's own defaults and are kept as-is for reproducing
    published GLA MQAR results.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        mode: str = "chunk",
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        use_output_gate: bool = True,
        gate_fn: str = "swish",
        **kwargs,
    ):
        super().__init__()
        try:
            from fla.layers import GatedLinearAttention
        except ImportError as e:
            raise ImportError(
                "fla is required for GLAWrapper. "
                "Install with: pip install flash-linear-attention\n"
                f"Original error: {e}"
            ) from e

        self.gla = GatedLinearAttention(
            hidden_size=d_model,
            num_heads=num_heads,
            mode=mode,
            expand_k=expand_k,
            expand_v=expand_v,
            use_output_gate=use_output_gate,
            gate_fn=gate_fn,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        x: (B, T, d_model)
        returns: (B, T, d_model)
        """
        output, *_ = self.gla(x)
        return output
