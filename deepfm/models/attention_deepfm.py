import torch.nn as nn

from deepfm.models.base import BaseCTRModel
from deepfm.models.layers.attention import FieldAttention
from deepfm.models.layers.dnn import DNN
from deepfm.models.layers.fm import FMInteraction


class AttentionDeepFM(BaseCTRModel):
    """DeepFM with attention-enhanced feature interactions.

    Inspired by AutoInt (Song et al., 2019).

    Architecture:
        y = sigmoid(y_FM + y_DNN_attn)
        y_FM       = first_order + FM_second_order(field_embeddings)
        y_DNN_attn = W_out * DNN(flatten(Attention(field_embeddings)))

    Field embeddings pass through stacked self-attention layers before
    being flattened and fed to the DNN.
    """

    def _build_components(self):
        cfg = self.config

        # FM
        self.fm = FMInteraction(reduce_sum=True)

        # Stacked attention layers
        embedding_dim = cfg.feature.fm_embedding_dim
        self.attention_layers = nn.ModuleList(
            [
                FieldAttention(
                    embedding_dim=embedding_dim,
                    num_heads=cfg.attention.num_heads,
                    attention_dim=cfg.attention.attention_dim,
                    dropout=cfg.attention.dropout,
                    use_residual=cfg.attention.use_residual,
                )
                for _ in range(cfg.attention.num_layers)
            ]
        )

        # DNN on attention-refined embeddings
        num_fields = self.schema.num_fields
        attn_output_dim = num_fields * embedding_dim

        self.dnn = DNN(
            input_dim=attn_output_dim,
            hidden_units=cfg.dnn.hidden_units,
            activation=cfg.dnn.activation,
            dropout=cfg.dnn.dropout,
            use_batch_norm=cfg.dnn.use_batch_norm,
        )
        self.dnn_output = nn.Linear(cfg.dnn.hidden_units[-1], 1, bias=False)

    def _forward_components(self, first_order, field_embeddings, flat_embeddings):
        # FM path (on raw embeddings)
        fm_output = first_order + self.fm(field_embeddings)  # (B, 1)

        # Attention-refined embeddings
        attn_embeddings = field_embeddings
        for attn_layer in self.attention_layers:
            attn_embeddings = attn_layer(attn_embeddings)  # (B, F, D)

        # Flatten and pass through DNN
        B = attn_embeddings.size(0)
        attn_flat = attn_embeddings.reshape(B, -1)  # (B, F*D)

        dnn_out = self.dnn(attn_flat)  # (B, last_hidden)
        dnn_logit = self.dnn_output(dnn_out)  # (B, 1)

        return fm_output + dnn_logit  # (B, 1)
