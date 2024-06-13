import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Define the flags using argparse
parser = argparse.ArgumentParser(description='MoeModel parameters.')
parser.add_argument('--moe_num_mixtures', type=int, default=2,
                    help='The number of mixtures (excluding the dummy "expert") used for MoeModel.')
parser.add_argument('--moe_l2', type=float, default=1e-8, help='L2 penalty for MoeModel.')
parser.add_argument('--moe_low_rank_gating', type=int, default=-1, help='Low rank gating for MoeModel.')
parser.add_argument('--moe_prob_gating', action='store_true', help='Prob gating for MoeModel.')
parser.add_argument('--moe_prob_gating_input', type=str, default='prob', help='Input Prob gating for MoeModel.')
parser.add_argument('--gating_remove_diag', action='store_true', help='Remove diagonal elements for gating.')


FLAGS = parser.parse_args()


class MoeModel(nn.Module):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def __init__(self, input_size, vocab_size, num_mixtures=None, l2_penalty=None, low_rank_gating=None,
                 gating_probabilities=None, gating_input=None, remove_diag=None):
        super(MoeModel, self).__init__()
        self.vocab_size = vocab_size
        self.num_mixtures = num_mixtures if num_mixtures is not None else FLAGS.moe_num_mixtures
        self.l2_penalty = l2_penalty if l2_penalty is not None else FLAGS.moe_l2
        self.low_rank_gating = low_rank_gating if low_rank_gating is not None else FLAGS.moe_low_rank_gating
        self.gating_probabilities = gating_probabilities if gating_probabilities is not None else FLAGS.moe_prob_gating
        self.gating_input = gating_input if gating_input is not None else FLAGS.moe_prob_gating_input
        self.remove_diag = remove_diag if remove_diag is not None else FLAGS.gating_remove_diag

        self.g = nn.Parameter(torch.rand(num_mixtures, vocab_size))
        nn.init.xavier_uniform_(self.g)

        if self.low_rank_gating == -1:
            self.fc1 = nn.Linear(input_size, self.vocab_size * (self.num_mixtures + 1), bias=False)
            nn.init.xavier_uniform_(self.fc1.weight)
        else:
            self.fc2 = nn.Linear(input_size, self.low_rank_gating, bias=False)
            nn.init.xavier_uniform_(self.fc2.weight)
            self.fc3 = nn.Linear(self.low_rank_gating, self.vocab_size * (self.num_mixtures + 1), bias=False)
            nn.init.xavier_uniform_(self.fc3.weight)

        self.expert_activations = nn.Linear(input_size, self.vocab_size * self.num_mixtures, bias=False)
        nn.init.xavier_uniform_(self.expert_activations.weight)

        if self.gating_probabilities:
            if self.gating_input == 'prob':
                self.gating_weights = nn.Parameter(
                    torch.randn(self.vocab_size, self.vocab_size) / math.sqrt(self.vocab_size))
            else:
                self.gating_weights = nn.Parameter(torch.randn(input_size, self.vocab_size) / math.sqrt(self.vocab_size))

    def forward(self, V):
        # Apply softmax to the weights along the first dimension
        g_softmax = F.softmax(self.g, dim=0)  # shape: (num_mix, c)

        multiplies = [V[:, i, :].squeeze(1) * g_softmax[i] for i in range(self.num_mixtures)]

        # Concatenate tensors along a new dimension (default is dim=0)
        stacked_tensors = torch.stack(multiplies)

        # Sum along the newly created dimension (dim=0)
        model_input = torch.sum(stacked_tensors, dim=0)

        if self.low_rank_gating == -1:
            gate_output = self.fc1(model_input)
        else:
            gate_output = self.fc3(self.fc2(model_input))

        gating_distribution = F.softmax(gate_output.view(-1, self.num_mixtures + 1), dim=-1)
        expert_distribution = torch.sigmoid(self.expert_activations(model_input).view(-1, self.num_mixtures))

        probabilities_by_class_and_batch = torch.sum(gating_distribution[:, :self.num_mixtures] * expert_distribution,
                                                     dim=1)
        probabilities = probabilities_by_class_and_batch.view(-1, self.vocab_size)

        if self.gating_probabilities:
            if self.gating_input == 'prob':
                gates = torch.matmul(probabilities, self.gating_weights)
            else:
                gates = torch.matmul(model_input, self.gating_weights)

            if self.remove_diag:
                diagonals = torch.diag(self.gating_weights)
                gates = gates - torch.mul(diagonals, probabilities)

            gates = nn.BatchNorm1d(self.vocab_size)(gates)
            gates = torch.sigmoid(gates)
            probabilities = torch.mul(probabilities, gates)

        return probabilities


# Example usage
# Initialize the model
# vocab_size = 50  # Example vocab size
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MoeModel(50, vocab_size, num_mixtures=3).to(device)
#
# # Create some dummy input data
# V = torch.randn(32, 3, 50).to(device)  # Batch size of 32 and 100 features
#
# # Forward pass
# output = model(V)  # Batch size of 32 and 100 features)
# print(output["predictions"])
