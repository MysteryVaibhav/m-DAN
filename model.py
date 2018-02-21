import torch
import numpy as np
import torch.nn as nn
from properties import *
from util import to_variable, to_tensor


class mDAN(torch.nn.Module):
    def __init__(self, pre_trained_embeddings):
        super(mDAN, self).__init__()
        # Create a biLSTM object
        bi_lstm = biLSTM(pre_trained_embeddings)
        self.text_encoder = bi_lstm
        t_attn = T_Att()
        self.t_attn = t_attn

    def forward(self, input):
        h = self.text_encoder(input)
        self.u_0 = h.sum(1)*(1/MAX_CAPTION_LEN)     # Take care of masking here
        u_1 = self.t_attn(h, self.u_0)    # Since m_0 = u_0
        return h


class biLSTM(torch.nn.Module):
    def __init__(self, pre_trained_embeddings):
        super(biLSTM, self).__init__()
        self.batch_size = BATCH_SIZE
        self.hidden_dim = HIDDEN_DIMENSION
        self.word_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION)
        # Assigning pre-trained embeddings as initial weights
        self.word_embeddings.weight = nn.Parameter(to_tensor(pre_trained_embeddings))

        self.lstm_forward = nn.LSTM(EMBEDDING_DIMENSION, HIDDEN_DIMENSION)
        self.hidden_forward = self.init_hidden_forward()
        self.lstm_backward = nn.LSTM(EMBEDDING_DIMENSION, HIDDEN_DIMENSION)
        self.hidden_backward = self.init_hidden_forward()

    def init_hidden_forward(self):
        # Assigning initial hidden and cell state
        return (to_variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                to_variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        sentence_backward = np.flip(sentence.data.cpu().numpy(), 1).copy()
        embeds_forward = self.word_embeddings(sentence)
        embeds_backward = self.word_embeddings(to_variable(torch.LongTensor(sentence_backward)))

        lstm_out_forward, self.hidden_forward = self.lstm_forward(
            embeds_forward.view(MAX_CAPTION_LEN, self.batch_size, -1), self.hidden_forward)
        # lstm_out: MAX_LEN * BATCH_SIZE * EMBEDDING_DIMENSION
        lstm_out_backward, self.hidden_backward = self.lstm_backward(
            embeds_backward.view(MAX_CAPTION_LEN, self.batch_size, -1), self.hidden_backward)

        # clear out the hidden state of the LSTM,
        self.hidden_forward = self.init_hidden_forward()
        self.hidden_backward = self.init_hidden_forward()

        # Adding the forward and backward embedding as per the paper
        return (lstm_out_forward + lstm_out_backward).permute(1, 0, 2)


class T_Att(torch.nn.Module):
    def __init__(self):
        super(T_Att, self).__init__()
        self.layer1 = nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=HIDDEN_DIMENSION)
        self.activation = nn.Tanh()
        self.layer2 = nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=HIDDEN_DIMENSION)
        self.linear = nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=1)
        self.softmax = nn.Softmax()

    def forward(self, u, m_u):
        h_u = self.activation(self.layer1(u)) * torch.unsqueeze(self.activation(self.layer2(m_u)), 1)
        alpha_u = self.linear(h_u)  # BATCH_SIZE * MAX_LEN * 1
        # TODO: do masking before taking softmax to nullify padding
        alpha_u = self.softmax(alpha_u)
        return (alpha_u * u).sum(1)     # Context vector: BATCH_SIZE * HIDDEN_DIMENSION


class V_Att(torch.nn.Module):
    def __init__(self):
        super(V_Att, self).__init__()
        self.layer1 = nn.Linear(in_features=VISUAL_FEATURE_DIMENSION, out_features=HIDDEN_DIMENSION)
        self.activation = nn.Tanh()
        self.layer2 = nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=HIDDEN_DIMENSION)
        self.linear = nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=1)
        self.softmax = nn.Softmax()

    def forward(self, u, m_u):
        h_u = self.activation(self.layer1(u)) * torch.unsqueeze(self.activation(self.layer2(m_u)), 1)
        alpha_u = self.linear(h_u)  # BATCH_SIZE * MAX_LEN * 1
        # TODO: do masking before taking softmax to nullify padding
        alpha_u = self.softmax(alpha_u)
        return (alpha_u * u).sum(1)     # Context vector: BATCH_SIZE * HIDDEN_DIMENSION

