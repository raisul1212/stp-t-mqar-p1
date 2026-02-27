"""
STP (Short-Term Plasticity) Mixers for the Zoology Framework
=============================================================

Biologically-inspired short-term synaptic plasticity applied to
transformer sequence mixing. Derived from first-principles MOSFET
device physics (EKV unified model).

Architecture versions:
  v3 (STPAttention):       Content-INDEPENDENT. V_gs is a fixed scalar per head.
                           λ, G, sign pre-computed once. Same scan class as RetNet.
  v4 (STPAttentionV4):     Content-ADAPTIVE. V_gs(t) = V_gs_base + W_vgs · x(t).
                           λ(t), G(t), sign(t) are input-dependent. Same scan class as GLA.

Also includes:
  HybridSTPAttention:      Learned gate between softmax and STP attention (Approach B).
  STPStateMixer:           STP applied to the feed-forward network (Approach C).

Physics equations (from MOSFET / FeFET device physics):
  V_eff = V_gs - V_T0 + W_LTM                        (effective overdrive voltage)
  g_ch  = β_τ · softplus(V_eff / U_T)                 (channel conductance)
  λ     = 1 - exp(-g_ch / C_ch)                       (decay rate, from τ = C_ch/g_ch)
  G     = β_gm · softplus(V_eff/U_T) · σ(V_eff/U_T)  (transconductance gate, ∝ ∂I_D/∂V_gs)
  sign  = tanh(α · (g_ch - IC_threshold))             (PPF/PPD regime switch)

Recurrence:
  F(t) = (1-λ) ⊙ F(t-1) + γ · sign ⊙ G ⊙ (v_t ⊗ k_t^T)
  y(t) = (W_LTM + F(t)) · q(t)

Reference: Wang 2026 (FeFET NQS); Enz et al. 1995 (EKV model).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────────────────────
# STP Physics Core (shared by all STP mixers)
# ─────────────────────────────────────────────────────────────

def compute_stp_physics(W_LTM, V_gs, V_T0, U_T, beta_tau, beta_gm, C_ch, 
                        alpha_ppd, IC_threshold, use_sign=True):
    """
    Compute STP physics quantities from device parameters.
    
    All outputs are (H, dk, dk) or (B, H, dk, dk) depending on V_gs shape.
    
    Returns: Lambda, G, sign_mask, retention
    """
    # ── Effective voltage (gate overdrive analog) ──
    # MOSFET: V_od = V_gs - V_T0   (gate overdrive voltage)
    # STP-T:  V_eff = V_gs - V_T0 + W_LTM
    # W_LTM acts as a per-element threshold voltage shift, analogous to
    # ferroelectric polarization shifting V_T in a FeFET.
    # Each element of W_LTM (d_k × d_k) gives a unique operating point.
    V_eff = V_gs - V_T0 + W_LTM  # broadcasts against W_LTM
    V_norm = V_eff / U_T
    
    # ── Channel conductance ──
    # MOSFET: I_D = I_spec · [softplus(V_od / 2nU_T)]²  (EKV unified model)
    # STP-T:  g_ch = β_τ · softplus(V_eff / U_T)         (1st-order approximation)
    # β_τ absorbs I_spec, n, and other process constants.
    # softplus provides smooth subthreshold → strong-inversion transition.
    g_ch = beta_tau * F.softplus(V_norm)
    
    # ── Decay rate ──
    # MOSFET: τ = C_ch / g_ch  (RC time constant of channel)
    # STP-T:  retention = exp(-Δt/τ) = exp(-g_ch/C_ch), then λ = 1 - retention
    # High g_ch (strong inversion, large V_eff) → small τ → large λ → fast forgetting
    # Low  g_ch (subthreshold, small V_eff)     → large τ → small λ → long memory
    Lambda = 1.0 - torch.exp(-g_ch / C_ch)
    retention = 1.0 - Lambda  # (1-λ): fraction of state retained per token
    
    # ── Transconductance gate ──
    # MOSFET: g_m = ∂I_D/∂V_gs ∝ softplus(V_norm) · sigmoid(V_norm)
    # STP-T:  G = β_gm · softplus(V_norm) · sigmoid(V_norm)
    # NOTE: We use softplus(V_norm) directly, NOT g_ch (= β_τ · softplus).
    # This keeps G independent of β_τ, matching the whitepaper (§2.2.2).
    # β_gm absorbs all process constants (I_spec, n, U_T) from MOSFET g_m.
    G = beta_gm * F.softplus(V_norm) * torch.sigmoid(V_norm)
    
    # ── Sign control (PPF/PPD regime switching) ──
    # No single-equation MOSFET analog. Captures the experimental observation
    # in FeFET synaptic devices: paired-pulse facilitation (PPF, sign > 0)
    # at low g_ch (weak inversion) and paired-pulse depression (PPD, sign < 0)
    # at high g_ch (strong inversion). tanh provides smooth differentiable switch.
    if use_sign:
        sign_mask = torch.tanh(alpha_ppd * (g_ch - IC_threshold))
    else:
        sign_mask = torch.ones_like(g_ch)
    
    return Lambda, G, sign_mask, retention


# ─────────────────────────────────────────────────────────────
# Approach A: STP Attention (v3 — content-independent)
# ─────────────────────────────────────────────────────────────

class STPAttention(nn.Module):
    """
    STP-T v3: Content-independent per-element decay attention.
    
    λ, G, sign are pre-computed once from W_LTM (constant across tokens).
    Same parallel scan class as RetNet, but with d_k² granularity instead
    of scalar decay.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads (default: derived from d_model)
        chunk_size: Chunk size for gradient checkpointing (default: 64)
        use_sign: Whether to use PPF/PPD sign switching (default: True)
        gamma_init: Initial Hebbian learning rate (default: 0.1)
        W_LTM_init_std: Std of W_LTM initialization (default: 0.5)
    """
    
    def __init__(self, d_model, layer_idx=None, num_heads=None, chunk_size=64, use_sign=True,
                 gamma_init=0.1, W_LTM_init_std=0.5, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads or max(1, d_model // 64)
        self.head_dim = d_model // self.num_heads
        self.chunk_size = chunk_size
        self.use_sign = use_sign
        
        H, dk = self.num_heads, self.head_dim
        
        # QKV projections
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # STP physics parameters
        self.W_LTM = nn.Parameter(torch.randn(H, dk, dk) * W_LTM_init_std)
        self.V_gs = nn.Parameter(torch.zeros(H))
        self.V_T0 = nn.Parameter(torch.zeros(H))
        self.U_T = 1.0  # thermal voltage (fixed)
        self.beta_tau = nn.Parameter(torch.ones(H))
        self.beta_gm = nn.Parameter(torch.ones(H) * 0.1)
        self.C_ch = nn.Parameter(torch.ones(H))
        self.gamma = nn.Parameter(torch.ones(H) * gamma_init)
        self.alpha_ppd = nn.Parameter(torch.ones(1))
        self.IC_threshold = nn.Parameter(torch.ones(1))
        
        # Key normalization for stable outer products
        self.k_norm = nn.LayerNorm(dk, elementwise_affine=False)
    
    def _forward_chunk(self, q_chunk, k_chunk, v_chunk, F_state,
                       retention, G, sign_mask, gamma, W_LTM):
        """Process one chunk of tokens sequentially."""
        B, C, H, dk = q_chunk.shape
        outputs = []
        
        for t in range(C):
            q_t = q_chunk[:, t]  # (B, H, dk)
            k_t = k_chunk[:, t]
            v_t = v_chunk[:, t]
            
            # Hebbian outer product: v_t ⊗ k_t^T
            hebbian = torch.einsum("bhd,bhe->bhde", v_t, k_t)  # (B, H, dk, dk)
            
            # Gated update: γ · sign ⊙ G ⊙ hebbian
            update = gamma * sign_mask * G * hebbian
            
            # State transition: F(t) = (1-λ) ⊙ F(t-1) + update
            F_state = retention * F_state + update
            
            # Output: y = (W_LTM + F(t)) · q(t)
            W_eff = W_LTM + F_state
            y_t = torch.einsum("bhde,bhe->bhd", W_eff, q_t)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1), F_state
    
    def forward(self, x, **kwargs):
        B, L, D = x.shape
        H, dk = self.num_heads, self.head_dim
        
        # QKV projection
        qkv = self.W_qkv(x).reshape(B, L, 3, H, dk)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Normalize keys
        k = self.k_norm(k)
        
        # Pre-compute physics (constant across tokens for v3)
        V_gs_expanded = self.V_gs.reshape(H, 1, 1)  # (H, 1, 1)
        V_T0_expanded = self.V_T0.reshape(H, 1, 1)
        beta_tau = self.beta_tau.reshape(H, 1, 1)
        beta_gm = self.beta_gm.reshape(H, 1, 1)
        C_ch = self.C_ch.reshape(H, 1, 1)
        
        Lambda, G, sign_mask, retention = compute_stp_physics(
            self.W_LTM, V_gs_expanded, V_T0_expanded, self.U_T,
            beta_tau, beta_gm, C_ch, self.alpha_ppd, self.IC_threshold,
            use_sign=self.use_sign
        )
        
        gamma = self.gamma.reshape(1, H, 1, 1)
        W_LTM = self.W_LTM.unsqueeze(0)  # (1, H, dk, dk)
        retention = retention.unsqueeze(0)  # (1, H, dk, dk)
        G = G.unsqueeze(0)
        sign_mask = sign_mask.unsqueeze(0)
        
        # Initialize state
        F_state = torch.zeros(B, H, dk, dk, device=x.device, dtype=x.dtype)
        
        # Process in chunks with gradient checkpointing
        all_outputs = []
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            q_c = q[:, start:end].contiguous()
            k_c = k[:, start:end].contiguous()
            v_c = v[:, start:end].contiguous()
            
            if self.training:
                chunk_out, F_state = checkpoint(
                    self._forward_chunk, q_c, k_c, v_c, F_state,
                    retention, G, sign_mask, gamma, W_LTM,
                    use_reentrant=False,
                )
            else:
                chunk_out, F_state = self._forward_chunk(
                    q_c, k_c, v_c, F_state, retention, G, sign_mask, gamma, W_LTM
                )
            all_outputs.append(chunk_out)
        
        # Reshape and project
        y = torch.cat(all_outputs, dim=1)  # (B, L, H, dk)
        y = y.reshape(B, L, D)
        y = self.W_o(y)
        
        return y


# ─────────────────────────────────────────────────────────────
# STP Attention v4: Content-Adaptive
# ─────────────────────────────────────────────────────────────

class STPAttentionV4(nn.Module):
    """
    STP-T v4: Content-adaptive per-element decay attention.
    
    V_gs(t) = V_gs_base + W_vgs · x(t) makes λ, G, sign input-dependent.
    One scalar per head per token → d_k² unique content-adaptive decay rates.
    Same scan class as GLA (diagonal), but 256× fewer params for same expressiveness.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        chunk_size: Chunk size for gradient checkpointing (default: 64)
        use_sign: Whether to use PPF/PPD sign switching (default: True)
        gamma_init: Initial Hebbian learning rate (default: 0.1)
        W_LTM_init_std: Std of W_LTM initialization (default: 0.5)
    """
    
    def __init__(self, d_model, layer_idx=None, num_heads=None, chunk_size=64, use_sign=True,
                 gamma_init=0.1, W_LTM_init_std=0.5, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads or max(1, d_model // 64)
        self.head_dim = d_model // self.num_heads
        self.chunk_size = chunk_size
        self.use_sign = use_sign
        
        H, dk = self.num_heads, self.head_dim
        
        # QKV projections
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # STP physics parameters
        self.W_LTM = nn.Parameter(torch.randn(H, dk, dk) * W_LTM_init_std)
        self.V_gs_base = nn.Parameter(torch.zeros(H))
        self.V_T0 = nn.Parameter(torch.zeros(H))
        self.U_T = 1.0
        self.beta_tau = nn.Parameter(torch.ones(H))
        self.beta_gm = nn.Parameter(torch.ones(H) * 0.1)
        self.C_ch = nn.Parameter(torch.ones(H))
        self.gamma = nn.Parameter(torch.ones(H) * gamma_init)
        self.alpha_ppd = nn.Parameter(torch.ones(1))
        self.IC_threshold = nn.Parameter(torch.ones(1))
        
        # v4: input-dependent V_gs projection
        # V_gs(t) = V_gs_base + W_vgs · x(t)    [whitepaper §2.4]
        # W_vgs: d_model → H projection (one scalar per head per token)
        # bias=False because V_gs_base already serves as the bias term
        self.W_vgs = nn.Linear(d_model, H, bias=False)
        nn.init.normal_(self.W_vgs.weight, std=0.01)
        
        # Key normalization
        self.k_norm = nn.LayerNorm(dk, elementwise_affine=False)
    
    def _forward_chunk(self, q_chunk, k_chunk, v_chunk, x_chunk, F_state):
        """Process one chunk with per-token physics recomputation."""
        B, C, H, dk = q_chunk.shape
        outputs = []
        
        for t in range(C):
            q_t = q_chunk[:, t]
            k_t = k_chunk[:, t]
            v_t = v_chunk[:, t]
            x_t = x_chunk[:, t]  # (B, d_model) for V_gs computation
            
            # v4: compute V_gs(t) from input
            V_gs_t = self.W_vgs(x_t) + self.V_gs_base  # (B, H)
            V_gs_t = V_gs_t.reshape(B, H, 1, 1)  # broadcast against dk×dk
            
            # Compute physics for this token
            V_T0 = self.V_T0.reshape(1, H, 1, 1)
            beta_tau = self.beta_tau.reshape(1, H, 1, 1)
            beta_gm = self.beta_gm.reshape(1, H, 1, 1)
            C_ch = self.C_ch.reshape(1, H, 1, 1)
            W_LTM = self.W_LTM.unsqueeze(0)  # (1, H, dk, dk)
            
            Lambda_t, G_t, sign_t, retention_t = compute_stp_physics(
                W_LTM, V_gs_t, V_T0, self.U_T,
                beta_tau, beta_gm, C_ch, self.alpha_ppd, self.IC_threshold,
                use_sign=self.use_sign
            )
            
            gamma = self.gamma.reshape(1, H, 1, 1)
            
            # Hebbian update
            hebbian = torch.einsum("bhd,bhe->bhde", v_t, k_t)
            update = gamma * sign_t * G_t * hebbian
            
            # State transition with content-adaptive retention
            F_state = retention_t * F_state + update
            
            # Output
            W_eff = W_LTM + F_state
            y_t = torch.einsum("bhde,bhe->bhd", W_eff, q_t)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1), F_state
    
    def forward(self, x, **kwargs):
        B, L, D = x.shape
        H, dk = self.num_heads, self.head_dim
        
        # QKV projection
        qkv = self.W_qkv(x).reshape(B, L, 3, H, dk)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        k = self.k_norm(k)
        
        # Initialize state
        F_state = torch.zeros(B, H, dk, dk, device=x.device, dtype=x.dtype)
        
        # Process in chunks
        all_outputs = []
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            q_c = q[:, start:end].contiguous()
            k_c = k[:, start:end].contiguous()
            v_c = v[:, start:end].contiguous()
            x_c = x[:, start:end].contiguous()
            
            if self.training:
                chunk_out, F_state = checkpoint(
                    self._forward_chunk, q_c, k_c, v_c, x_c, F_state,
                    use_reentrant=False,
                )
            else:
                chunk_out, F_state = self._forward_chunk(
                    q_c, k_c, v_c, x_c, F_state
                )
            all_outputs.append(chunk_out)
        
        y = torch.cat(all_outputs, dim=1).reshape(B, L, D)
        y = self.W_o(y)
        return y


# ─────────────────────────────────────────────────────────────
# Approach B: Hybrid Softmax + STP with Learned Gating
# ─────────────────────────────────────────────────────────────

class HybridSTPAttention(nn.Module):
    """
    Learned gate between standard softmax attention and STP fast-weight output.
    
    y(t) = σ(gate) · softmax_output + (1 - σ(gate)) · stp_output
    
    Gate is learned per-head, allowing the model to decide how much to rely
    on full attention vs STP memory per head.
    """
    
    def __init__(self, d_model, layer_idx=None, num_heads=None, chunk_size=64, use_sign=True,
                 gamma_init=0.1, W_LTM_init_std=0.5, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads or max(1, d_model // 64)
        self.head_dim = d_model // self.num_heads
        self.chunk_size = chunk_size
        self.use_sign = use_sign
        
        H, dk = self.num_heads, self.head_dim
        
        # QKV projections (shared)
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # STP parameters (same as v3)
        self.W_LTM = nn.Parameter(torch.randn(H, dk, dk) * W_LTM_init_std)
        self.V_gs = nn.Parameter(torch.zeros(H))
        self.V_T0 = nn.Parameter(torch.zeros(H))
        self.U_T = 1.0
        self.beta_tau = nn.Parameter(torch.ones(H))
        self.beta_gm = nn.Parameter(torch.ones(H) * 0.1)
        self.C_ch = nn.Parameter(torch.ones(H))
        self.gamma = nn.Parameter(torch.ones(H) * gamma_init)
        self.alpha_ppd = nn.Parameter(torch.ones(1))
        self.IC_threshold = nn.Parameter(torch.ones(1))
        
        # Learned gate: initialized to 0.5 (equal mix)
        self.gate = nn.Parameter(torch.zeros(H))
        
        self.k_norm = nn.LayerNorm(dk, elementwise_affine=False)
    
    def _forward_chunk(self, q_chunk, k_chunk, v_chunk, F_state,
                       retention, G, sign_mask, gamma, W_LTM):
        """Same as STPAttention._forward_chunk."""
        B, C, H, dk = q_chunk.shape
        outputs = []
        for t in range(C):
            q_t, k_t, v_t = q_chunk[:, t], k_chunk[:, t], v_chunk[:, t]
            hebbian = torch.einsum("bhd,bhe->bhde", v_t, k_t)
            update = gamma * sign_mask * G * hebbian
            F_state = retention * F_state + update
            W_eff = W_LTM + F_state
            y_t = torch.einsum("bhde,bhe->bhd", W_eff, q_t)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1), F_state
    
    def forward(self, x, **kwargs):
        B, L, D = x.shape
        H, dk = self.num_heads, self.head_dim
        
        # QKV
        qkv = self.W_qkv(x).reshape(B, L, 3, H, dk)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        k_normed = self.k_norm(k)
        
        # --- Softmax attention path ---
        q_s = q.permute(0, 2, 1, 3)  # (B, H, L, dk)
        k_s = k.permute(0, 2, 1, 3)
        v_s = v.permute(0, 2, 1, 3)
        
        scale = 1.0 / math.sqrt(dk)
        attn = torch.matmul(q_s, k_s.transpose(-2, -1)) * scale
        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        softmax_out = torch.matmul(attn, v_s).permute(0, 2, 1, 3)  # (B, L, H, dk)
        
        # --- STP path ---
        V_gs_exp = self.V_gs.reshape(H, 1, 1)
        V_T0_exp = self.V_T0.reshape(H, 1, 1)
        bt = self.beta_tau.reshape(H, 1, 1)
        bg = self.beta_gm.reshape(H, 1, 1)
        cc = self.C_ch.reshape(H, 1, 1)
        
        Lambda, G_phys, sign_mask, retention = compute_stp_physics(
            self.W_LTM, V_gs_exp, V_T0_exp, self.U_T,
            bt, bg, cc, self.alpha_ppd, self.IC_threshold, self.use_sign
        )
        
        gam = self.gamma.reshape(1, H, 1, 1)
        W_LTM = self.W_LTM.unsqueeze(0)
        retention = retention.unsqueeze(0)
        G_phys = G_phys.unsqueeze(0)
        sign_mask = sign_mask.unsqueeze(0)
        
        F_state = torch.zeros(B, H, dk, dk, device=x.device, dtype=x.dtype)
        all_stp = []
        
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            q_c = q[:, start:end].contiguous()
            k_c = k_normed[:, start:end].contiguous()
            v_c = v[:, start:end].contiguous()
            
            if self.training:
                chunk_out, F_state = checkpoint(
                    self._forward_chunk, q_c, k_c, v_c, F_state,
                    retention, G_phys, sign_mask, gam, W_LTM,
                    use_reentrant=False,
                )
            else:
                chunk_out, F_state = self._forward_chunk(
                    q_c, k_c, v_c, F_state, retention, G_phys, sign_mask, gam, W_LTM
                )
            all_stp.append(chunk_out)
        
        stp_out = torch.cat(all_stp, dim=1)  # (B, L, H, dk)
        
        # --- Gated combination ---
        gate = torch.sigmoid(self.gate).reshape(1, 1, H, 1)  # broadcast
        y = gate * softmax_out + (1 - gate) * stp_out
        
        y = y.reshape(B, L, D)
        y = self.W_o(y)
        return y


# ─────────────────────────────────────────────────────────────
# Approach C: STP State Mixer (applied to FFN/MLP)
# ─────────────────────────────────────────────────────────────

class STPStateMixer(nn.Module):
    """
    STP applied to the feed-forward network (state mixer in Zoology).
    
    Instead of y = W2 · GELU(W1 · x + b1) + b2,
    we have  y = (W2 + F_state) · GELU(W1 · x + b1) + b2
    where F_state evolves with Hebbian plasticity.
    
    This adds short-term plasticity to the MLP, allowing it to adapt
    its behavior based on recent tokens.
    """
    
    def __init__(self, d_model, layer_idx=None, d_inner=None, chunk_size=64,
                 gamma_init=0.01, **kwargs):
        super().__init__()
        d_inner = d_inner or d_model * 4
        self.d_model = d_model
        self.d_inner = d_inner
        self.chunk_size = chunk_size
        
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        self.activation = nn.GELU()
        
        # STP parameters for the MLP weight (simpler than attention STP)
        self.Lambda_F = nn.Parameter(torch.ones(d_inner, d_model) * 0.1)
        self.Gamma_F = nn.Parameter(torch.ones(d_inner, d_model) * gamma_init)
    
    def _forward_chunk(self, x_chunk, F_state):
        B, C, D = x_chunk.shape
        retention = 1.0 - torch.sigmoid(self.Lambda_F)
        gamma = torch.sigmoid(self.Gamma_F)
        
        outputs = []
        for t in range(C):
            x_t = x_chunk[:, t, :]
            
            # Apply STP-augmented fc1
            W1_eff = self.fc1.weight + F_state  # (B, d_inner, d_model)
            h = torch.einsum("bij,bj->bi", W1_eff, x_t)
            if self.fc1.bias is not None:
                h = h + self.fc1.bias
            
            h_pre = h
            h = self.activation(h)
            out = self.fc2(h)
            outputs.append(out)
            
            # Hebbian update to F_state
            hebbian = torch.einsum("bi,bj->bij", h_pre, x_t)
            F_state = retention.unsqueeze(0) * F_state + gamma.unsqueeze(0) * hebbian
        
        return torch.stack(outputs, dim=1), F_state
    
    def forward(self, x, **kwargs):
        B, L, D = x.shape
        F_state = torch.zeros(B, self.d_inner, self.d_model,
                              device=x.device, dtype=x.dtype)
        
        all_outputs = []
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            x_chunk = x[:, start:end, :]
            
            if self.training:
                chunk_out, F_state = checkpoint(
                    self._forward_chunk, x_chunk, F_state,
                    use_reentrant=False,
                )
            else:
                chunk_out, F_state = self._forward_chunk(x_chunk, F_state)
            all_outputs.append(chunk_out)
        
        return torch.cat(all_outputs, dim=1)


# ─────────────────────────────────────────────────────────────
# Simple baselines (self-contained, no external deps)
# ─────────────────────────────────────────────────────────────

class RetNetAttention(nn.Module):
    """
    Simplified RetNet-style attention with scalar per-head decay.
    S(t) = γ · S(t-1) + v_t ⊗ k_t^T;  y(t) = S(t) · q(t)
    """
    
    def __init__(self, d_model, layer_idx=None, num_heads=None, chunk_size=64, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads or max(1, d_model // 64)
        self.head_dim = d_model // self.num_heads
        self.chunk_size = chunk_size
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # One scalar decay per head (learned in log-space for stability)
        H = self.num_heads
        self.log_gamma = nn.Parameter(torch.zeros(H))
        
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
    
    def _forward_chunk(self, q_c, k_c, v_c, S_state, gamma):
        B, C, H, dk = q_c.shape
        outputs = []
        for t in range(C):
            hebbian = torch.einsum("bhd,bhe->bhde", v_c[:, t], k_c[:, t])
            S_state = gamma * S_state + hebbian
            y_t = torch.einsum("bhde,bhe->bhd", S_state, q_c[:, t])
            outputs.append(y_t)
        return torch.stack(outputs, dim=1), S_state
    
    def forward(self, x, **kwargs):
        B, L, D = x.shape
        H, dk = self.num_heads, self.head_dim
        
        qkv = self.W_qkv(x).reshape(B, L, 3, H, dk)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        k = self.k_norm(k)
        
        gamma = torch.sigmoid(self.log_gamma).reshape(1, H, 1, 1)
        S_state = torch.zeros(B, H, dk, dk, device=x.device, dtype=x.dtype)
        
        all_outputs = []
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            q_c = q[:, start:end].contiguous()
            k_c = k[:, start:end].contiguous()
            v_c = v[:, start:end].contiguous()
            
            if self.training:
                chunk_out, S_state = checkpoint(
                    self._forward_chunk, q_c, k_c, v_c, S_state, gamma,
                    use_reentrant=False,
                )
            else:
                chunk_out, S_state = self._forward_chunk(
                    q_c, k_c, v_c, S_state, gamma
                )
            all_outputs.append(chunk_out)
        
        y = torch.cat(all_outputs, dim=1).reshape(B, L, D)
        return self.W_o(y)


class LinearAttention(nn.Module):
    """
    Linear attention (no decay). S(t) = S(t-1) + v_t ⊗ k_t^T.
    No gating, no decay — pure accumulation.
    """
    
    def __init__(self, d_model, layer_idx=None, num_heads=None, chunk_size=64, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads or max(1, d_model // 64)
        self.head_dim = d_model // self.num_heads
        self.chunk_size = chunk_size
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
    
    def _forward_chunk(self, q_c, k_c, v_c, S_state):
        B, C, H, dk = q_c.shape
        outputs = []
        for t in range(C):
            hebbian = torch.einsum("bhd,bhe->bhde", v_c[:, t], k_c[:, t])
            S_state = S_state + hebbian
            y_t = torch.einsum("bhde,bhe->bhd", S_state, q_c[:, t])
            outputs.append(y_t)
        return torch.stack(outputs, dim=1), S_state
    
    def forward(self, x, **kwargs):
        B, L, D = x.shape
        H, dk = self.num_heads, self.head_dim
        
        qkv = self.W_qkv(x).reshape(B, L, 3, H, dk)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        k = self.k_norm(k)
        
        S_state = torch.zeros(B, H, dk, dk, device=x.device, dtype=x.dtype)
        all_outputs = []
        
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            q_c = q[:, start:end].contiguous()
            k_c = k[:, start:end].contiguous()
            v_c = v[:, start:end].contiguous()
            
            if self.training:
                chunk_out, S_state = checkpoint(
                    self._forward_chunk, q_c, k_c, v_c, S_state,
                    use_reentrant=False,
                )
            else:
                chunk_out, S_state = self._forward_chunk(q_c, k_c, v_c, S_state)
            all_outputs.append(chunk_out)
        
        y = torch.cat(all_outputs, dim=1).reshape(B, L, D)
        return self.W_o(y)


class SoftmaxAttention(nn.Module):
    """Standard causal softmax attention. The upper bound baseline."""
    
    def __init__(self, d_model, layer_idx=None, num_heads=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads or max(1, d_model // 64)
        self.head_dim = d_model // self.num_heads
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, **kwargs):
        B, L, D = x.shape
        H, dk = self.num_heads, self.head_dim
        
        qkv = self.W_qkv(x).reshape(B, L, 3, H, dk)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        scale = 1.0 / math.sqrt(dk)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        y = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, L, D)
        return self.W_o(y)
