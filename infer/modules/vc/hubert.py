import os
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HubertModelWrapper(nn.Module):
    """HuBERT model wrapper using HuggingFace transformers with fairseq-compatible API.

    Provides the same extract_features() and final_proj interface that the rest
    of the RVC codebase expects from the fairseq HuBERT model.
    """

    HUBERT_REPO = "facebook/hubert-base-ls960"
    FAIRSEQ_CKPT = "assets/hubert/hubert_base.pt"

    def __init__(self):
        super().__init__()
        from transformers import HubertModel

        logger.info("Loading HuBERT model via transformers (%s)...", self.HUBERT_REPO)
        self.model = HubertModel.from_pretrained(self.HUBERT_REPO)
        self.num_layers = self.model.config.num_hidden_layers  # 12

        # final_proj: needed by RVC v1 (projects 768 -> 256)
        self.final_proj = nn.Linear(self.model.config.hidden_size, 256)
        self._load_final_proj()

    def _load_final_proj(self):
        """Try to load final_proj weights from the existing fairseq checkpoint."""
        if not os.path.exists(self.FAIRSEQ_CKPT):
            logger.warning(
                "Fairseq checkpoint not found at %s. "
                "final_proj uses random init (RVC v1 models may sound wrong).",
                self.FAIRSEQ_CKPT,
            )
            return
        try:
            ckpt = torch.load(self.FAIRSEQ_CKPT, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt)
            if "final_proj.weight" in state and "final_proj.bias" in state:
                self.final_proj.weight.data.copy_(state["final_proj.weight"])
                self.final_proj.bias.data.copy_(state["final_proj.bias"])
                logger.info("Loaded final_proj weights from fairseq checkpoint.")
            else:
                logger.warning("final_proj weights not found in checkpoint.")
        except Exception as e:
            logger.warning("Could not load final_proj from checkpoint: %s", e)

    def extract_features(self, source, padding_mask=None, output_layer=None, **kwargs):
        """Extract features matching the fairseq HuBERT API.

        Args:
            source: Raw waveform (batch, seq_len).
            padding_mask: Bool tensor, True = padded (fairseq convention).
            output_layer: 1-indexed transformer layer to extract from.
                          9 for RVC v1, 12 for RVC v2.

        Returns:
            Tuple (features, padding_mask) where features is (batch, feat_len, hidden).
        """
        # fairseq: padding_mask True = ignore; transformers: attention_mask 1 = attend
        if padding_mask is not None and padding_mask.any():
            attention_mask = (~padding_mask).long()
        else:
            attention_mask = None

        # Only request all hidden states when we need an intermediate layer
        need_hidden = output_layer is not None and output_layer < self.num_layers
        outputs = self.model(
            input_values=source,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden,
        )

        if need_hidden:
            # fairseq output_layer is 1-indexed and maps directly to
            # hidden_states[output_layer] in transformers
            feats = outputs.hidden_states[output_layer]
        else:
            feats = outputs.last_hidden_state

        return (feats, padding_mask)


def load_hubert(config):
    """Load HuBERT model using transformers (drop-in replacement for fairseq version)."""
    model = HubertModelWrapper()
    model = model.to(config.device)
    if config.is_half:
        model = model.half()
    else:
        model = model.float()
    return model.eval()
