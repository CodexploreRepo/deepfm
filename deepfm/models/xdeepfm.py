import torch.nn as nn

from deepfm.models.base import BaseCTRModel
from deepfm.models.layers.cin import CIN
from deepfm.models.layers.dnn import DNN


class xDeepFM(BaseCTRModel):
    """xDeepFM: Combining Explicit and Implicit Feature Interactions.

    (Lian et al., 2018)

    Architecture:
        y = sigmoid(y_linear + y_CIN + y_DNN)
        y_linear = first_order
        y_CIN    = W_cin * CIN(field_embeddings)   — explicit interactions
        y_DNN    = W_dnn * DNN(flat_embeddings)     — implicit interactions

    Key difference from DeepFM: CIN replaces FM's 2nd-order interaction,
    capturing arbitrary-order explicit interactions at vector-wise level.
    """

    def _build_components(self):
        cfg = self.config

        # CIN for explicit interactions
        num_fields = self.schema.num_fields
        embedding_dim = cfg.feature.fm_embedding_dim

        self.cin = CIN(
            num_fields=num_fields,
            embedding_dim=embedding_dim,
            layer_sizes=cfg.cin.layer_sizes,
            activation=cfg.cin.activation,
            split_half=cfg.cin.split_half,
        )
        self.cin_output = nn.Linear(self.cin.output_dim, 1, bias=False)

        # DNN for implicit interactions
        dnn_input_dim = self.schema.total_embedding_dim
        self.dnn = DNN(
            input_dim=dnn_input_dim,
            hidden_units=cfg.dnn.hidden_units,
            activation=cfg.dnn.activation,
            dropout=cfg.dnn.dropout,
            use_batch_norm=cfg.dnn.use_batch_norm,
        )
        self.dnn_output = nn.Linear(cfg.dnn.hidden_units[-1], 1, bias=False)

    def _forward_components(self, first_order, field_embeddings, flat_embeddings):
        # Linear (first order)
        linear_out = first_order  # (B, 1)

        # CIN path (explicit)
        cin_out = self.cin(field_embeddings)  # (B, cin_output_dim)
        cin_logit = self.cin_output(cin_out)  # (B, 1)

        # DNN path (implicit)
        dnn_out = self.dnn(flat_embeddings)  # (B, last_hidden)
        dnn_logit = self.dnn_output(dnn_out)  # (B, 1)

        return linear_out + cin_logit + dnn_logit  # (B, 1)
