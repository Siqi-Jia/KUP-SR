# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel


class AutoCF(SequentialModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=100,
                            help='Size of hidden vectors in AutoCF.')
        parser.add_argument('--L', type=int, default=3,
                            help='Number of convolutional layers.')
        parser.add_argument('--n_filters', type=int, default=100,
                            help='Number of filters in each convolutional layer.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.L = args.L
        self.n_filters = args.n_filters
        super().__init__(args, corpus)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.n_filters, kernel_size=(h, self.emb_size))
            for h in range(1, self.L + 1)
        ])
        self.fc = nn.Linear(self.n_filters * self.L, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.emb_size, bias=False)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]

        i_vectors = self.i_embeddings(i_ids)
        his_vectors = self.i_embeddings(history)  # [batch_size, history_max, emb_size]

        # Reshape for convolution
        his_vectors = his_vectors.unsqueeze(1)  # [batch_size, 1, history_max, emb_size]

        # Convolutional layers
        conv_outputs = []
        for conv in self.conv_layers:
            output = F.relu(conv(his_vectors))  # [batch_size, n_filters, history_max - kernel_size + 1, 1]
            output = output.squeeze(-1)  # [batch_size, n_filters, history_max - kernel_size + 1]
            output = F.max_pool1d(output, kernel_size=output.size(2))  # [batch_size, n_filters, 1]
            output = output.squeeze(-1)  # [batch_size, n_filters]
            conv_outputs.append(output)

        # Concatenate outputs from all convolutional layers
        conv_output = torch.cat(conv_outputs, dim=1)  # [batch_size, n_filters * L]

        # Fully connected layer
        hidden = F.relu(self.fc(conv_output))  # [batch_size, hidden_size]

        # Output layer
        rnn_vector = self.out(hidden)  # [batch_size, emb_size]

        # Predicts
        prediction = (rnn_vector[:, None, :] * i_vectors).sum(-1)  # [batch_size, -1]
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}