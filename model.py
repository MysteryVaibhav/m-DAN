import torch
import numpy as np
import torch.nn as nn
from properties import *
from util import to_variable, to_tensor
import torch.nn.functional as F


class mDAN(torch.nn.Module):
    def __init__(self):
        super(mDAN, self).__init__()
        # Create a biLSTM object
        # Get pre-trained embeddings
        #embeddings = np.random.uniform(-1, 1, (VOCAB_SIZE, EMBEDDING_DIMENSION))
        self.bi_lstm = biLSTM()
        self.text_encoder = self.bi_lstm
        t_attn = T_Att()
        self.t_attn = t_attn
        v_attn = V_Att()
        self.v_attn = v_attn

    def forward(self, input_caption, mask, input_image, is_inference):
        h = self.text_encoder(input_caption)
        i = input_image

        # Textual Attention
        self.u_0 = (h * mask.unsqueeze(2)).sum(1)/torch.unsqueeze(torch.clamp(torch.sum(mask, dim=1), min=1), 1)     # Take care of masking here
        self.t_attn.m_u = self.u_0      # Since m_0 = u_0

        # Visual Attention
        avg_v = to_variable((i.sum(1)*(1/NO_OF_REGIONS_IN_IMAGE)).data, requires_grad=True)
        self.v_0 = self.v_attn.activation(self.v_attn.P[0](avg_v))
        self.v_attn.m_v = self.v_0      # Since m_0 = v_0

        # Creating similarity vectors for inference
        if is_inference:
            z_u = to_variable(self.u_0.data, requires_grad=False)
            z_v = to_variable(self.v_0.data, requires_grad=False)

        # Similarity, will be used to compute loss and do backprop
        S = torch.sum(self.u_0 * self.v_0, 1)

        # Repeating the above process for NO_OF_STEPS
        for steps in range(NO_OF_STEPS):
            if steps > 0:
                self.t_attn.m_u = self.t_attn.m_u + u_1
                self.v_attn.m_v = self.v_attn.m_v + v_1
            u_1 = self.t_attn(h, mask, steps)
            v_1 = self.v_attn(i, steps)
            if is_inference:
                z_u = z_u.cat((z_u, u_1), 1)
                z_v = z_v.cat((z_v, v_1), 1)
            S = S + torch.sum(u_1 * v_1, 1)
        if is_inference:
            return S, z_u, z_v
        return S


class biLSTM(torch.nn.Module):
    def __init__(self):
        super(biLSTM, self).__init__()
        self.batch_size = BATCH_SIZE
        self.hidden_dim = HIDDEN_DIMENSION
        self.word_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION)
        # Assigning pre-trained embeddings as initial weights
        #self.word_embeddings.weight.data.copy_(to_tensor(embeddings))
        nn.init.xavier_uniform(self.word_embeddings.weight)
        self.lstm = nn.LSTM(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, bidirectional=False)

    def init_hidden(self):
        # Assigning initial hidden and cell state
        # 2, since single layered LSTM
        return (to_variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                to_variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence.t())
        self.batch_size = embeds.size()[1]
        # clear out the hidden state of the LSTM
        self.hidden = self.init_hidden()

        outputs, self.hidden = self.lstm(embeds, self.hidden)

        #out_forward = outputs[:MAX_CAPTION_LEN, :self.batch_size, :self.hidden_dim]
        #out_backward = outputs[:MAX_CAPTION_LEN, :self.batch_size, self.hidden_dim:]
        # Adding the forward and backward embedding as per the paper
        return outputs.permute(1, 0, 2)  # BATCH_SIZE * MAX_LEN * HIDDEN_DIMENSION


class T_Att(torch.nn.Module):
    def __init__(self):
        super(T_Att, self).__init__()
        self.activation = nn.Tanh()
        self.layer1 = nn.ModuleList([nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=HIDDEN_DIMENSION)
                                    for _ in range(NO_OF_STEPS)])
        self.layer2 = nn.ModuleList([nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=HIDDEN_DIMENSION)
                                    for _ in range(NO_OF_STEPS)])
        self.linear_sm = nn.ModuleList([nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=1)
                                        for _ in range(NO_OF_STEPS)])

    def forward(self, u, mask, k):
        h_u = self.activation(self.layer1[k](u)) * torch.unsqueeze(self.activation(self.layer2[k](self.m_u)), 1)
        alpha_u = self.linear_sm[k](h_u)
        alpha_u.data.masked_fill_((1-mask).data.unsqueeze(2).byte(), -float('inf'))
        alpha_u = F.softmax(alpha_u, dim=1)
        # masking to nullify padding
        return (alpha_u * u).sum(1)  # Context vector: BATCH_SIZE * HIDDEN_DIMENSION


class V_Att(torch.nn.Module):
    def __init__(self):
        super(V_Att, self).__init__()
        self.activation = nn.Tanh()
        self.layer1 = nn.ModuleList([nn.Linear(in_features=VISUAL_FEATURE_DIMENSION, out_features=HIDDEN_DIMENSION)
                                    for _ in range(NO_OF_STEPS)])
        self.layer2 = nn.ModuleList([nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=HIDDEN_DIMENSION)
                                     for _ in range(NO_OF_STEPS)])
        self.linear_sm = nn.ModuleList([nn.Linear(in_features=EMBEDDING_DIMENSION, out_features=1)
                                        for _ in range(NO_OF_STEPS)])
        self.P = nn.ModuleList([nn.Linear(in_features=VISUAL_FEATURE_DIMENSION, out_features=EMBEDDING_DIMENSION)
                                for _ in range(NO_OF_STEPS + 1)])

    def forward(self, v, k):
        h_v = self.activation(self.layer1[k](v)) * torch.unsqueeze(self.activation(self.layer2[k](self.m_v)), 1)
        alpha_v = self.linear_sm[k](h_v)  # BATCH_SIZE * NO_OF_REGIONS_IN_IMAGE * 1
        alpha_v = F.softmax(alpha_v, dim=1)
        return self.activation(
            self.P[k + 1]((alpha_v * v).sum(1)))  # Context vector: BATCH_SIZE * EMBEDDING_DIMENSION

