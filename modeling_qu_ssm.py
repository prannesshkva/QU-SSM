import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers import AutoConfig, AutoModelForCausalLM

from configuration_qu_ssm import QUSSMConfig

try:
    import qu_ssm_cuda_ext as _cuda_ext
except Exception:  # noqa: BLE001 - optional dependency, must never break CPU-only usage
    _cuda_ext = None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class StaticTPUMoE(nn.Module):
    """
    Sparse Mixture-of-Experts with Top-K Token-Dispatch Routing.
    Evaluates each expert only on the subset of tokens routed to it,
    yielding true sub-linear compute scaling (Top-2 compute = 2x rather than 8x).
    """
    def __init__(self, d_model=512, d_ff=1024, num_experts=8, moe_top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.moe_top_k = min(moe_top_k, num_experts)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        num_tokens = x_flat.size(0)

        # 1. Router logits & normalized top-k probabilities
        router_logits = self.router(x_flat) * (1.0 / math.sqrt(self.d_model))
        probs = F.softmax(router_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(probs, self.moe_top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 2. Sparse Token Dispatch: execute only the tokens assigned to each expert
        out_flat = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i)  # [num_tokens, top_k]
            token_idx, k_slot = torch.where(mask)
            if token_idx.shape[0] > 0:
                expert_in = x_flat[token_idx]
                expert_out = expert(expert_in)
                weights = topk_weights[token_idx, k_slot].unsqueeze(-1)
                out_flat.index_add_(0, token_idx, (weights * expert_out).to(out_flat.dtype))

        out = out_flat.view(*orig_shape)

        # 3. Differentiable auxiliary load balancing loss
        aux_loss = None
        if self.training:
            tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
            tokens_per_expert.scatter_add_(
                0, 
                topk_indices.view(-1), 
                torch.ones_like(topk_indices.view(-1), dtype=torch.float)
            )
            f_e = tokens_per_expert / (num_tokens * self.moe_top_k)
            P_e = probs.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(f_e * P_e)

        return out, aux_loss


class ExactRealQUBlock(nn.Module):
    """
    Quasi-Unitary State Space Model (QU-SSM) Layer.
    Implements:
      1. Continuous SO(2) Lie Group Isometry (Theorem 1)
      2. Strictly Bounded Differentiable Log-Space Highway (Theorem 2):
         S(raw_S) = raw_S / (1 - raw_S / clamp_min)
         Guarantees S in [clamp_min, 0] (no overflow of exp(-S)) and dS/d(raw_S) > 0 everywhere!
      3. Pure-Real Dual Prefix Scan Parallelization (Theorem 3)
      4. Modulo 2*pi Phase Wrapping on S^1 to eliminate FP32 drift
      5. Multi-Chunk Prefix Scan with exact threaded boundary accumulators
      6. Optional torch.utils.checkpoint gradient checkpointing across chunks
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.clamp_min = getattr(config, 'clamp_min', -12.0)
        # S naturally satisfies S <= clamp_max (0.0) for all raw_S <= 0 without explicit clamping
        self.clamp_max = getattr(config, 'clamp_max', 0.0)
        self.chunk_size = getattr(config, 'chunk_size', 512)
        self.use_checkpoint = getattr(config, 'use_checkpoint', False)

        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_model, bias=True)

        self.gate_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            # NOTE: gate_bias_min/max name a *spectrum endpoint*, not a numeric
            # min/max -- gate_bias_min (-6.90) > gate_bias_max (-9.21) is
            # intentional (it just linearly spaces the per-channel decay
            # rates), but the naming is easy to misread as a bug. Left as-is
            # for config/checkpoint compatibility.
            b_spectrum = torch.linspace(config.gate_bias_min, config.gate_bias_max, config.d_model)
            self.gate_proj.bias.copy_(b_spectrum)
        # Skip these in _init_weights: they carry a deliberate, mathematically
        # motivated initialization (zero weight + fixed decay-rate spectrum),
        # not a generic one, and must survive post_init()/from_pretrained().
        self.gate_proj._is_custom_init = True

        self.theta_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.normal_(self.theta_proj.weight, std=0.01)
        self.theta_proj._is_custom_init = True
        tb = torch.linspace(config.theta_base_min, config.theta_base_max, config.d_state)
        self.theta_base = nn.Parameter(tb.view(1, 1, 1, config.d_state).repeat(1, 1, config.d_model, 1))

        self.C_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.D = nn.Parameter(torch.ones(config.d_model))

    def _smooth_log_decay(self, raw_S):
        """
        Algebraic Soft-Saturating Log-Decay Function:
          S(raw_S) = raw_S / (1.0 - raw_S / clamp_min)   for raw_S <= 0
        Properties:
          1. Strictly bounded in (clamp_min, 0] (e.g. (-12.0, 0]).
             exp(-S) <= exp(12) ≈ 162,754 (Zero float32/bfloat16 overflow forever).
          2. Smooth algebraic decay of gradient magnitude: dS/d(raw_S) = 1 / (1.0 - raw_S / clamp_min)^2 > 0.
             Eliminates the abrupt hard-zero cliff of torch.clamp, preserving gradient flow
             with smooth continuous decay as raw_S -> -inf.
        """
        return raw_S / (1.0 + raw_S / self.clamp_min)

    def _scan_chunk(self, u_c, theta_c, log_g_c, C_re_offset, C_im_offset, raw_S_offset, Phi_offset):
        """
        Threaded intra-chunk prefix scan with mathematically exact boundary propagation.
        Maintains complete mathematical equivalence to global full-sequence scan.
        """
        TWO_PI = 2.0 * math.pi

        raw_S_c = raw_S_offset + torch.cumsum(log_g_c, dim=1)
        S_c = self._smooth_log_decay(raw_S_c)
        Phi_c = torch.remainder(Phi_offset + torch.cumsum(theta_c, dim=1), TWO_PI)

        exp_S_c = torch.exp(S_c)
        exp_neg_S_c = torch.exp(-S_c)
        cos_P_c = torch.cos(Phi_c)
        sin_P_c = torch.sin(Phi_c)

        A_c = torch.cumsum(u_c * exp_neg_S_c * cos_P_c, dim=1)
        B_c = torch.cumsum(-u_c * exp_neg_S_c * sin_P_c, dim=1)

        total_C_re = C_re_offset + A_c
        total_C_im = C_im_offset + B_c

        chunk_re = exp_S_c * (cos_P_c * total_C_re - sin_P_c * total_C_im)

        # Offsets for next chunk
        next_C_re_off = total_C_re[:, -1:]
        next_C_im_off = total_C_im[:, -1:]
        next_raw_S_off = raw_S_c[:, -1:]
        next_Phi_off = Phi_c[:, -1:]

        return chunk_re, next_C_re_off, next_C_im_off, next_raw_S_off, next_Phi_off

    def forward(self, x, chunk_size=None):
        orig_dtype = x.dtype
        B, L, D = x.shape
        N = self.d_state
        effective_chunk = chunk_size or self.chunk_size

        x_norm = self.norm(x)
        u_val, gate = self.in_proj(x_norm).chunk(2, dim=-1)

        u = u_val.float().unsqueeze(-1).expand(B, L, D, N)
        th_p = self.theta_proj(x_norm).float().unsqueeze(-1)
        theta = th_p + self.theta_base.float()

        g_val = self.gate_proj(x_norm).float()

        # Optional fast inference path: only taken when (a) no gradients are
        # needed at all, (b) we're on CUDA, (c) d_state matches the kernel's
        # hardcoded head count, and (d) the extension actually built. Any
        # False here falls straight through to the verified pure-PyTorch
        # scan below -- this path is a pure speed optimization and must never
        # be able to change training behavior.
        use_fused = (
            _cuda_ext is not None
            and not self.training
            and not torch.is_grad_enabled()
            and x.is_cuda
            and _cuda_ext.fused_forward_available(self)
        )
        if use_fused:
            theta_base_flat = self.theta_base[0, 0, 0, :]
            h_sum = _cuda_ext.fused_forward(u_val.float(), g_val, th_p.squeeze(-1), theta_base_flat)
            h_sum = h_sum.to(orig_dtype)
            y_ssm = self.C_proj(h_sum) + x * self.D
            return y_ssm * F.silu(gate)

        log_g = -F.softplus(g_val).unsqueeze(-1).expand(B, L, D, N)

        # Threaded multi-chunk scan with exact boundary offsets.
        # Handles any sequence length L identically, eliminating single-pass/multi-chunk divergence.
        h_chunks = []
        C_re_off = torch.zeros(B, 1, D, N, device=x.device, dtype=torch.float32)
        C_im_off = torch.zeros(B, 1, D, N, device=x.device, dtype=torch.float32)
        raw_S_off = torch.zeros(B, 1, D, N, device=x.device, dtype=torch.float32)
        Phi_off = torch.zeros(B, 1, D, N, device=x.device, dtype=torch.float32)

        for start in range(0, L, effective_chunk):
            end = min(start + effective_chunk, L)
            u_slice = u[:, start:end]
            th_slice = theta[:, start:end]
            lg_slice = log_g[:, start:end]

            if self.training and self.use_checkpoint:
                c_re, C_re_off, C_im_off, raw_S_off, Phi_off = torch.utils.checkpoint.checkpoint(
                    self._scan_chunk,
                    u_slice, th_slice, lg_slice,
                    C_re_off, C_im_off, raw_S_off, Phi_off,
                    use_reentrant=False
                )
            else:
                c_re, C_re_off, C_im_off, raw_S_off, Phi_off = self._scan_chunk(
                    u_slice, th_slice, lg_slice,
                    C_re_off, C_im_off, raw_S_off, Phi_off
                )
            h_chunks.append(c_re)

        h_re = torch.cat(h_chunks, dim=1) if len(h_chunks) > 1 else h_chunks[0]

        h_sum = h_re.sum(dim=-1).to(orig_dtype)
        y_ssm = self.C_proj(h_sum) + x * self.D
        return y_ssm * F.silu(gate)

    def step(self, x_t, state=None):
        """
        Recurrent step for O(1) inference.
        Tracks cumulative raw log-decay to maintain exact parity with training forward pass.
        """
        orig_dtype = x_t.dtype
        B, D = x_t.shape
        N = self.d_state

        x_norm = self.norm(x_t)
        u_val, gate = self.in_proj(x_norm).chunk(2, dim=-1)

        u = u_val.float().unsqueeze(-1).expand(B, D, N)
        tb_sq = self.theta_base.squeeze(0).squeeze(0).float()
        th_p = self.theta_proj(x_norm).float().unsqueeze(-1)
        theta = th_p + tb_sq

        g_val = self.gate_proj(x_norm).float()
        log_g = -F.softplus(g_val).unsqueeze(-1).expand(B, D, N)

        if state is None:
            h_real = torch.zeros(B, D, N, device=x_t.device, dtype=torch.float32)
            h_imag = torch.zeros(B, D, N, device=x_t.device, dtype=torch.float32)
            cum_raw_S = torch.zeros(B, D, N, device=x_t.device, dtype=torch.float32)
        else:
            h_real, h_imag, cum_raw_S = state

        # Compute delta_S via exact algebraic soft-decay
        new_raw_S = cum_raw_S + log_g
        S_prev = self._smooth_log_decay(cum_raw_S)
        S_curr = self._smooth_log_decay(new_raw_S)
        delta_S = S_curr - S_prev
        gamma = torch.exp(delta_S)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        new_h_real = gamma * (cos_t * h_real - sin_t * h_imag) + u
        new_h_imag = gamma * (sin_t * h_real + cos_t * h_imag)

        h_sum = new_h_real.sum(dim=-1).to(orig_dtype)
        y_ssm = self.C_proj(h_sum) + x_t * self.D
        out = y_ssm * F.silu(gate)

        return out, (new_h_real, new_h_imag, new_raw_S)


class QUSSMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ssm = ExactRealQUBlock(config)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = StaticTPUMoE(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_experts=config.num_experts,
            moe_top_k=config.moe_top_k,
        )

    def forward(self, x):
        x = x + self.ssm(x)
        moe_out, aux_loss = self.moe(self.mlp_norm(x))
        x = x + moe_out
        return x, aux_loss

    def step(self, x_t, state=None):
        ssm_out, new_state = self.ssm.step(x_t, state)
        x_t = x_t + ssm_out
        moe_out, _ = self.moe(self.mlp_norm(x_t))
        x_t = x_t + moe_out
        return x_t, new_state


class QUSSMForCausalLM(PreTrainedModel):
    config_class = QUSSMConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([QUSSMBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        self.post_init()

    def _init_weights(self, module):
        """
        Required by HF's post_init()/from_pretrained() fast-init path: without
        an override, PreTrainedModel._init_weights() is a no-op, which is
        harmless for a plain `QUSSMForCausalLM(config)` construction but means
        weights loaded via `from_pretrained(..., low_cpu_mem_usage=True)` can
        be left uninitialized for any parameter not present in a checkpoint
        (e.g. newly added layers). gate_proj/theta_proj are tagged
        `_is_custom_init` in ExactRealQUBlock.__init__ and are skipped here so
        this can never overwrite their deliberate initialization.
        """
        if getattr(module, "_is_custom_init", False):
            return
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        elif input_ids is not None:
            x = self.embed(input_ids)
        elif inputs_embeds is not None:
            x = inputs_embeds
        else:
            raise ValueError("Input tokens must be provided.")

        total_aux_loss = torch.tensor(0.0, device=x.device)
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
            x, aux_loss = layer(x)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ce_loss + (self.config.aux_loss_coef * total_aux_loss)

        if not return_dict:
            output = (logits,) + ((all_hidden_states,) if output_hidden_states else ())
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
        )

    @torch.no_grad()
    def generate_recurrent(
        self,
        input_ids,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
    ):
        self.eval()
        B, L = input_ids.shape
        states = [None for _ in range(self.config.n_layers)]

        # Warmup prompt recurrence
        for t in range(L):
            tok_t = input_ids[:, t]
            x_t = self.embed(tok_t)
            for i, layer in enumerate(self.layers):
                x_t, states[i] = layer.step(x_t, states[i])

        generated_tokens = []
        curr_token = input_ids[:, -1]

        for _ in range(max_new_tokens):
            x_t = self.embed(curr_token)
            for i, layer in enumerate(self.layers):
                x_t, states[i] = layer.step(x_t, states[i])

            x_norm = self.final_norm(x_t)
            logits = self.lm_head(x_norm)

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float("Inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            generated_tokens.append(next_token.unsqueeze(1))
            curr_token = next_token

        return torch.cat([input_ids, torch.cat(generated_tokens, dim=1)], dim=1)


AutoConfig.register("qu_ssm_moe", QUSSMConfig)
AutoModelForCausalLM.register(QUSSMConfig, QUSSMForCausalLM)
