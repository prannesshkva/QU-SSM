from transformers.configuration_utils import PretrainedConfig


class QUSSMConfig(PretrainedConfig):
    model_type = "qu_ssm_moe"

    def __init__(
        self,
        vocab_size=50257,
        d_model=512,
        n_layers=8,
        d_state=8,  # 8 coordinate pairs (16 real state coordinates per channel)
        d_ff=1024,
        num_experts=8,
        moe_top_k=2,
        aux_loss_coef=0.01,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        # NOTE: these name a *spectrum endpoint*, not a numeric min/max --
        # gate_bias_min (-6.90) > gate_bias_max (-9.21) is intentional: it's
        # the start/end of a torch.linspace() used to spread per-channel
        # decay-rate init across d_model. Kept as-is for compatibility with
        # any existing config.json / checkpoints; just don't "fix" the
        # ordering, it isn't a bug.
        gate_bias_min=-6.90,
        gate_bias_max=-9.21,
        theta_base_min=0.01,
        theta_base_max=0.50,
        clamp_min=-12.0,
        clamp_max=0.0,
        chunk_size=512,
        use_checkpoint=False,
        initializer_range=0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.moe_top_k = min(moe_top_k, num_experts)
        self.aux_loss_coef = aux_loss_coef
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.gate_bias_min = gate_bias_min
        self.gate_bias_max = gate_bias_max
        self.theta_base_min = theta_base_min
        self.theta_base_max = theta_base_max
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint
        self.num_hidden_layers = n_layers
        self.hidden_size = d_model
        self.intermediate_size = d_ff
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
