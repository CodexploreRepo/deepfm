import torch
import torch.nn as nn

from deepfm.models.base import BaseCTRModel
from deepfm.models.layers.dnn import DNN
from deepfm.models.layers.fm import FMInteraction


class DeepFM(BaseCTRModel):
    """DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.

    (Guo et al., 2017)

    Architecture:
        y = sigmoid(y_FM + y_DNN)
        y_FM  = first_order + FM_second_order(field_embeddings)
        y_DNN = W_out * DNN(flat_embeddings)

    Key: FM and DNN share the same embedding vectors.
    """

    def _build_components(self):
        cfg = self.config

        # FM second-order interaction
        self.fm = FMInteraction(reduce_sum=True)

        # DNN component
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
        # FM path: first-order + second-order
        fm_output = first_order + self.fm(field_embeddings)  # (B, 1)

        # DNN path
        dnn_out = self.dnn(flat_embeddings)  # (B, last_hidden)
        dnn_output = self.dnn_output(dnn_out)  # (B, 1)

        return fm_output + dnn_output  # (B, 1)
