import torch
import numpy as np
import torch.nn as nn
from properties import *
from util import to_variable, to_tensor


class mDAN(torch.nn.Module):
    def __init__(self, embeddings):
        super(mDAN, self).__init__()
        # Create a biLSTM object
        bi_lstm = biLSTM(embeddings)
        self.text_encoder = bi_lstm
        t_attn = T_Att()
        self.t_attn = t_attn
        v_attn = V_Att()
        self.v_attn = v_attn

    def forward(self, input_caption, mask, input_image):
        h = self.text_encoder(input_caption)
        i = input_image
        # Textual Attention
        self.u_0 = h.sum(1)/torch.unsqueeze(torch.sum(mask, dim=1), 1)     # Take care of masking here
        self.t_attn.m_u = self.u_0      # Since m_0 = u_0
        u_1 = self.t_attn(h, mask)

        # Visual Attention
        avg_v = to_variable((i.sum(1)*(1/NO_OF_REGIONS_IN_IMAGE)).data, requires_grad=True)
        self.v_0 = self.v_attn.activation(self.v_attn.linear_transform(avg_v))
        self.v_attn.m_v = self.v_0      # Since m_0 = v_0
        v_1 = self.v_attn(i)

        # Similarity, will be used to compute loss and do backprop
        S = torch.sum(self.u_0 * self.v_0, 1)
        S = S + torch.sum(u_1 * v_1, 1)

        # Repeating the above process for NO_OF_STEPS
        for steps in range(NO_OF_STEPS-1):
            self.t_attn.m_u = self.t_attn.m_u + u_1
            self.v_attn.m_v = self.v_attn.m_v + v_1
            u_1 = self.t_attn(h, mask)
            v_1 = self.v_attn(i)
            S = S + torch.sum(u_1 * v_1, 1)
        return S


class biLSTM(torch.nn.Module):
    def __init__(self, embeddings):
        super(biLSTM, self).__init__()
        self.batch_size = BATCH_SIZE
        self.hidden_dim = HIDDEN_DIMENSION
        self.word_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION)
        # Assigning pre-trained embeddings as initial weights
        self.word_embeddings.weight.data.copy_(to_tensor(embeddings))
        self.lstm = nn.LSTM(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, bidirectional=True)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Assigning initial hidden and cell state
        # 2, since single layered LSTM
        return (to_variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                to_variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(MAX_CAPTION_LEN, self.batch_size, -1), self.hidden)
        # lstm_out: MAX_LEN * BATCH_SIZE * (2*EMBEDDING_DIMENSION)
        out_forward, out_backward = np.hsplit(lstm_out.data.permute(0, 2, 1).cpu().numpy(), 2)
        # clear out the hidden state of the LSTM
        self.hidden = self.init_hidden()
        # Adding the forward and backward embedding as per the paper
        return to_variable(to_tensor(out_forward + out_backward), requires_grad=True).permute(2, 0, 1)  # BATCH_SIZE * MAX_LEN * HIDDEN_DIMENSION


class T_Att(torch.nn.Module):
    def __init__(self):
        super(T_Att, self).__init__()
        self.layer1 = nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=HIDDEN_DIMENSION)
        self.activation = nn.Tanh()
        self.layer2 = nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=HIDDEN_DIMENSION)
        self.linear = nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=1)
        self.softmax = nn.Softmax()

    def forward(self, u, mask):
        h_u = self.activation(self.layer1(u)) * torch.unsqueeze(self.activation(self.layer2(self.m_u)), 1)
        alpha_u = self.linear(h_u) * torch.unsqueeze(mask, 2)  # BATCH_SIZE * MAX_LEN * 1
        # masking before taking softmax to nullify padding
        alpha_u = self.softmax(alpha_u)
        return (alpha_u * u).sum(1)     # Context vector: BATCH_SIZE * HIDDEN_DIMENSION


class V_Att(torch.nn.Module):
    def __init__(self):
        super(V_Att, self).__init__()
        self.layer1 = nn.Linear(in_features=VISUAL_FEATURE_DIMENSION, out_features=HIDDEN_DIMENSION)
        self.activation = nn.Tanh()
        self.layer2 = nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=HIDDEN_DIMENSION)
        self.linear = nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=1)
        self.linear_transform = nn.Linear(in_features=VISUAL_FEATURE_DIMENSION, out_features=EMBEDDING_DIMENSION)
        self.softmax = nn.Softmax()

    def forward(self, v):
        h_v = self.activation(self.layer1(v)) * torch.unsqueeze(self.activation(self.layer2(self.m_v)), 1)
        alpha_v = self.linear(h_v)  # BATCH_SIZE * NO_OF_REGIONS_IN_IMAGE * 1
        alpha_v = self.softmax(alpha_v)
        return self.activation(self.linear_transform((alpha_v * v).sum(1)))    # Context vector: BATCH_SIZE * EMBEDDING_DIMENSION

