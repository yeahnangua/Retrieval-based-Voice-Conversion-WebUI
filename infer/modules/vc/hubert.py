import os
import logging
import pickle
import re
import types

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fairseq checkpoint loading utilities (bypass fairseq/omegaconf dependency)
# ---------------------------------------------------------------------------

class _DummyObj:
    """Placeholder for unpickling fairseq/omegaconf objects we don't need."""
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return _DummyObj()
    def __call__(self, *args, **kwargs):
        return _DummyObj()
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that replaces missing fairseq/omegaconf classes with dummies."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return _DummyObj


def _make_safe_pickle_module():
    mod = types.ModuleType("_safe_pickle")
    mod.Unpickler = _SafeUnpickler
    for attr in dir(pickle):
        if not hasattr(mod, attr):
            setattr(mod, attr, getattr(pickle, attr))
    return mod


_SAFE_PICKLE = _make_safe_pickle_module()

# ---------------------------------------------------------------------------
# State-dict key conversion: fairseq HuBERT → transformers HuBERT
# ---------------------------------------------------------------------------

def _convert_fairseq_key(key):
    """Map one fairseq state-dict key to its transformers equivalent.

    Returns the new key, or *None* if the key should be dropped.
    """
    # Keys we don't need for inference
    if key in ("mask_emb", "label_embs_concat"):
        return None
    # final_proj handled separately (RVC v1 projection head)
    if key.startswith("final_proj."):
        return None

    # --- feature extractor conv layers ---
    # conv weight: .{i}.0.weight → .{i}.conv.weight
    m = re.match(r"feature_extractor\.conv_layers\.(\d+)\.0\.weight$", key)
    if m:
        return f"feature_extractor.conv_layers.{m.group(1)}.conv.weight"

    # layer-0 group-norm: .0.2.{w|b} → .0.layer_norm.{w|b}
    m = re.match(r"feature_extractor\.conv_layers\.0\.2\.(weight|bias)$", key)
    if m:
        return f"feature_extractor.conv_layers.0.layer_norm.{m.group(1)}"

    # --- feature projection ---
    if key.startswith("post_extract_proj."):
        return key.replace("post_extract_proj.", "feature_projection.projection.", 1)
    # layer_norm right after projection
    m = re.match(r"layer_norm\.(weight|bias)$", key)
    if m:
        return f"feature_projection.layer_norm.{m.group(1)}"

    # --- positional conv embedding ---
    m = re.match(r"encoder\.pos_conv\.0\.(weight_g|weight_v|bias)$", key)
    if m:
        return f"encoder.pos_conv_embed.conv.{m.group(1)}"

    # --- transformer encoder layers ---
    # attention projections: self_attn.X → attention.X
    m = re.match(
        r"encoder\.layers\.(\d+)\.self_attn\."
        r"(k_proj|v_proj|q_proj|out_proj)\.(weight|bias)$",
        key,
    )
    if m:
        i, proj, wb = m.groups()
        return f"encoder.layers.{i}.attention.{proj}.{wb}"

    # attention layer norm: self_attn_layer_norm → layer_norm
    m = re.match(r"encoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$", key)
    if m:
        return f"encoder.layers.{m.group(1)}.layer_norm.{m.group(2)}"

    # FFN: fc1 → feed_forward.intermediate_dense
    m = re.match(r"encoder\.layers\.(\d+)\.fc1\.(weight|bias)$", key)
    if m:
        return f"encoder.layers.{m.group(1)}.feed_forward.intermediate_dense.{m.group(2)}"

    # FFN: fc2 → feed_forward.output_dense
    m = re.match(r"encoder\.layers\.(\d+)\.fc2\.(weight|bias)$", key)
    if m:
        return f"encoder.layers.{m.group(1)}.feed_forward.output_dense.{m.group(2)}"

    # final_layer_norm (same name in both)
    m = re.match(r"encoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)$", key)
    if m:
        return key

    # encoder-level layer norm (same name)
    if key in ("encoder.layer_norm.weight", "encoder.layer_norm.bias"):
        return key

    logger.debug("Skipping unknown fairseq key: %s", key)
    return None


def _convert_state_dict(fairseq_state):
    """Convert a full fairseq HuBERT state dict to transformers format.

    Returns (hf_state_dict, final_proj_state_dict).
    """
    hf_state = {}
    final_proj = {}
    for key, value in fairseq_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("final_proj."):
            final_proj[key.split(".", 1)[1]] = value  # "weight" / "bias"
            continue
        new_key = _convert_fairseq_key(key)
        if new_key is not None:
            hf_state[new_key] = value
    return hf_state, final_proj


# ---------------------------------------------------------------------------
# HuBERT wrapper with fairseq-compatible API
# ---------------------------------------------------------------------------

class HubertModelWrapper(nn.Module):
    """HuBERT model using *transformers* architecture with weights loaded from
    the local fairseq checkpoint that RVC models were trained against.

    Provides the ``extract_features()`` / ``final_proj`` interface expected by
    the rest of the RVC codebase.
    """

    FAIRSEQ_CKPT = "assets/hubert/hubert_base.pt"

    def __init__(self):
        super().__init__()
        from transformers import HubertModel, HubertConfig

        model, fp_state = self._load_model(HubertModel, HubertConfig)
        self.model = model
        self.num_layers = self.model.config.num_hidden_layers  # 12

        # final_proj: RVC v1 uses this to project 768 → 256
        self.final_proj = nn.Linear(self.model.config.hidden_size, 256)
        if "weight" in fp_state and "bias" in fp_state:
            self.final_proj.weight.data.copy_(fp_state["weight"])
            self.final_proj.bias.data.copy_(fp_state["bias"])
            logger.info("Loaded final_proj weights from checkpoint.")
        else:
            logger.warning(
                "final_proj weights not found in checkpoint; "
                "RVC v1 models may produce incorrect output."
            )

    # ---- model loading ---------------------------------------------------

    def _load_model(self, HubertModel, HubertConfig):
        if os.path.exists(self.FAIRSEQ_CKPT):
            return self._from_fairseq_ckpt(HubertModel, HubertConfig)
        logger.warning(
            "%s not found — falling back to HuggingFace download. "
            "Voice quality may differ from original RVC training.",
            self.FAIRSEQ_CKPT,
        )
        return self._from_huggingface(HubertModel), {}

    def _from_fairseq_ckpt(self, HubertModel, HubertConfig):
        logger.info("Loading HuBERT from local checkpoint: %s", self.FAIRSEQ_CKPT)

        ckpt = torch.load(
            self.FAIRSEQ_CKPT,
            map_location="cpu",
            pickle_module=_SAFE_PICKLE,
            weights_only=False,
        )
        fairseq_state = ckpt.get("model", ckpt)

        # Hard-coded config matching facebook/hubert-base-ls960
        config = HubertConfig(
            vocab_size=32,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout=0.1,
            activation_dropout=0.1,
            attention_dropout=0.1,
            feat_proj_layer_norm=True,
            feat_proj_dropout=0.0,
            final_dropout=0.1,
            layerdrop=0.0,  # disable for inference
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            feat_extract_norm="group",
            feat_extract_activation="gelu",
            conv_dim=[512, 512, 512, 512, 512, 512, 512],
            conv_stride=[5, 2, 2, 2, 2, 2, 2],
            conv_kernel=[10, 3, 3, 3, 3, 2, 2],
            conv_bias=False,
            num_conv_pos_embeddings=128,
            num_conv_pos_embedding_groups=16,
            do_stable_layer_norm=False,
            apply_spec_augment=False,
        )

        model = HubertModel(config)
        hf_state, fp_state = _convert_state_dict(fairseq_state)

        missing, unexpected = model.load_state_dict(hf_state, strict=False)
        # masked_spec_embed is training-only — safe to ignore
        missing = [k for k in missing if k != "masked_spec_embed"]
        if missing:
            logger.warning("Missing keys loading HuBERT: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys loading HuBERT: %s", unexpected)

        logger.info("HuBERT loaded from fairseq checkpoint successfully.")
        return model, fp_state

    def _from_huggingface(self, HubertModel):
        repo = "facebook/hubert-base-ls960"
        logger.info("Downloading HuBERT from HuggingFace: %s", repo)
        try:
            return HubertModel.from_pretrained(repo)
        except OSError as e:
            raise OSError(
                f"Failed to download HuBERT. Set HF_ENDPOINT env var "
                f"(e.g. export HF_ENDPOINT=https://hf-mirror.com) and retry. "
                f"Original error: {e}"
            ) from e

    # ---- inference --------------------------------------------------------

    def extract_features(self, source, padding_mask=None, output_layer=None, **kwargs):
        """Extract features — drop-in replacement for fairseq HuBERT API.

        Args:
            source: Raw waveform tensor (batch, seq_len).
            padding_mask: Bool tensor, True = padded (fairseq convention).
            output_layer: 1-indexed transformer layer (9 for v1, 12 for v2).
        Returns:
            (features, padding_mask)
        """
        if padding_mask is not None and padding_mask.any():
            attention_mask = (~padding_mask).long()
        else:
            attention_mask = None

        need_hidden = output_layer is not None and output_layer < self.num_layers
        outputs = self.model(
            input_values=source,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden,
        )

        if need_hidden:
            feats = outputs.hidden_states[output_layer]
        else:
            feats = outputs.last_hidden_state

        return (feats, padding_mask)


def load_hubert(config):
    """Load HuBERT model (drop-in replacement for the old fairseq loader)."""
    model = HubertModelWrapper()
    model = model.to(config.device)
    if config.is_half:
        model = model.half()
    else:
        model = model.float()
    return model.eval()
