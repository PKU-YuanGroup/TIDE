import torch
from torch import nn

from dllm.pipelines import a2d


class EditFlowQwen3Config(a2d.A2DQwen3Config):
    model_type = "editflow-qwen3"  # <- NEW model_type


class EditFlowQwen3Model(a2d.A2DQwen3LMHeadModel):
    config_class = EditFlowQwen3Config
    modules_to_save = {
        "rate_heads",
        "sub_logits",
        "ins_logits",
    }  # fully fintuned even using lora

    def __init__(self, config):
        # TODO: time embedding
        super().__init__(config)
        ff = self.lm_head
        in_f, out_f = ff.in_features, ff.out_features
        use_bias = ff.bias is not None
        # Create new, independent heads (no deepcopy)
        self.sub_logits = nn.Linear(in_f, out_f, bias=use_bias)
        self.ins_logits = nn.Linear(in_f, out_f, bias=use_bias)
        self.rate_heads = nn.Sequential(nn.Linear(config.hidden_size, 3), nn.Softplus())
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        **kwargs,
    ):
        # TODO: time embedding
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        h = output["hidden_states"][-1]  # final hidden states
        # Position heads
        sub_log = self.sub_logits(h)  # [B, L, V]
        ins_log = self.ins_logits(h)  # [B, L, V]

        rates = self.rate_heads(h)
        sub_rate_hat, del_rate_hat, ins_rate_hat = rates.unbind(
            -1
        )  # [B, L], [B, L], [B, L]
        return dict(
            sub_rate_hat=sub_rate_hat,  # [B,L]
            del_rate_hat=del_rate_hat,  # [B,L]
            ins_rate_hat=ins_rate_hat,  # [B,L]
            ins_logits=ins_log,  # [B,L,V]
            sub_logits=sub_log,  # [B,L,V]
        )


from transformers.models.auto import AutoModel, AutoConfig

# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoConfig.register("editflow-qwen3", EditFlowQwen3Config)
AutoModel.register(EditFlowQwen3Config, EditFlowQwen3Model)


if __name__ == "__main__":
    import dllm
    import torch
    from transformers import AutoConfig, AutoModel

    # Load a config from a local path (either a directory containing config.json, or the file itself)
    config_path = dllm.utils.resolve_with_base_env("Qwen/Qwen3-0.6B", "BASE_MODELS_DIR")
    config = EditFlowQwen3Config.from_pretrained(config_path)
    if hasattr(config, "auto_map"):
        delattr(config, "auto_map")
    if hasattr(config, "architectures"):
        delattr(config, "architectures")

    torch.set_default_device("cuda")
    model = EditFlowQwen3Model(config)
    model.save_pretrained("models-tmp/editflow-qwen3")
    auto_model = AutoModel.from_pretrained("models-tmp/editflow-qwen3")
