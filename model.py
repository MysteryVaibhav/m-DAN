import torch
import torch.nn as nn
from util import to_variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class mDAN(torch.nn.Module):
    def __init__(self, params):
        super(mDAN, self).__init__()
        self.regions_in_image = params.regions_in_image
        self.number_of_steps = params.number_of_steps
        # Initialize embeddings
        # embeddings = np.random.uniform(-1, 1, (VOCAB_SIZE, EMBEDDING_DIMENSION))

        self.text_encoder = Encoder(params.mini_batch_size, params.hidden_dimension, params.embedding_dimension,
                                    params.vocab_size, params.max_caption_len)
        t_attn = T_Att(params.hidden_dimension, params.embedding_dimension, params.number_of_steps)
        self.t_attn = t_attn
        v_attn = V_Att(params.hidden_dimension, params.embedding_dimension, params.visual_feature_dimension,
                       params.number_of_steps)
        self.v_attn = v_attn

    def forward(self, input_caption, mask, input_image, is_inference):
        # Sorting sequences by their lengths for packing
        seq_lens = mask.sum(dim=1).long()
        seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        input_caption = input_caption[perm_idx]                     # bs * max_seq_len * 512

        h = self.text_encoder(input_caption, seq_lens)              # bs * seq_len * 512

        # Unsort everything back
        h = h[perm_idx]
        i = input_image

        # Modify mask, according to max-seq-len in the batch
        max_seq_len_in_batch = h.size(1)
        mask = mask[:, :max_seq_len_in_batch]

        # Textual Attention
        self.t_attn.u_0 = (h * mask.unsqueeze(2)).sum(1) / torch.unsqueeze(torch.sum(mask, dim=1), dim=1)
        self.t_attn.m_u = self.t_attn.u_0  # Since m_0 = u_0

        # Visual Attention
        avg_v = torch.autograd.Variable((i.sum(1) * (1 / self.regions_in_image)).data, requires_grad=True)
        self.v_attn.v_0 = self.v_attn.activation(self.v_attn.P[0](avg_v))
        self.v_attn.m_v = self.v_attn.v_0  # Since m_0 = v_0

        # Creating similarity vectors for inference
        if is_inference:
            z_u = torch.autograd.Variable(self.t_attn.u_0.data, requires_grad=False)
            z_v = torch.autograd.Variable(self.v_attn.v_0.data, requires_grad=False)

        # Similarity, will be used to compute loss and do backprop
        S = torch.sum(self.t_attn.u_0 * self.v_attn.v_0, 1)

        # Repeating the above process for NO_OF_STEPS
        for k in range(self.number_of_steps):
            if k > 0:
                self.t_attn.m_u = self.t_attn.m_u + u_1
                self.v_attn.m_v = self.v_attn.m_v + v_1
            u_1 = self.t_attn(h, mask, k)
            v_1 = self.v_attn(i, k)
            if is_inference:
                z_u = z_u.cat((z_u, u_1), 1)
                z_v = z_v.cat((z_v, v_1), 1)
            S = S + torch.sum(u_1 * v_1, 1)
        if is_inference:
            return S, z_u, z_v
        return S


class Encoder(torch.nn.Module):
    def __init__(self, mini_batch_size, hidden_dimension, embedding_dimension, vocab_size, max_caption_len):
        super(Encoder, self).__init__()
        self.batch_size = mini_batch_size
        self.hidden_dim = hidden_dimension
        self.max_caption_len = max_caption_len
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dimension)

        # Assigning pre-trained embeddings as initial weights
        # self.word_embeddings.weight.data.copy_(to_tensor(embeddings))

        nn.init.xavier_uniform(self.word_embeddings.weight)
        self.lstm = nn.LSTM(embedding_dimension, hidden_dimension, bidirectional=True)

    def init_hidden(self):
        # Assigning initial hidden and cell state
        # 2, since single layered LSTM
        return (to_variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                to_variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, seq, seq_lens):
        seq = seq.transpose(0, 1)  # seq_len * batch_size * embedding_dimension
        embeds = self.word_embeddings(seq)
        self.batch_size = embeds.size()[1]
        # clear out the hidden state of the LSTM
        self.hidden = self.init_hidden()

        packed_input = pack_padded_sequence(embeds, seq_lens.data.cpu().numpy())
        packed_outputs, self.hidden = self.lstm(packed_input, self.hidden)
        outputs, _ = pad_packed_sequence(packed_outputs)

        outputs = outputs.view(outputs.size(0), outputs.size(1), 2, -1).sum(2) / 2
        # Adding the forward and backward embedding as per the paper, and unsorting sequences for modules ahead
        return outputs.permute(1, 0, 2)  # BATCH_SIZE * MAX_LEN * HIDDEN_DIMENSION


class T_Att(torch.nn.Module):
    def __init__(self, hidden_dimension, embedding_dimension, number_of_steps):
        super(T_Att, self).__init__()
        self.u_0 = None
        self.activation = nn.Tanh()
        self.layer1 = nn.ModuleList([nn.Linear(in_features=embedding_dimension, out_features=hidden_dimension)
                                    for _ in range(number_of_steps)])
        self.layer2 = nn.ModuleList([nn.Linear(in_features=embedding_dimension, out_features=hidden_dimension)
                                    for _ in range(number_of_steps)])
        self.linear_sm = nn.ModuleList([nn.Linear(in_features=embedding_dimension, out_features=1)
                                        for _ in range(number_of_steps)])

    def forward(self, u, mask, k):
        h_u = self.activation(self.layer1[k](u)) * torch.unsqueeze(self.activation(self.layer2[k](self.m_u)), 1)
        alpha_u = self.linear_sm[k](h_u)
        alpha_u = F.softmax(alpha_u, dim=1)
        # masking to nullify padding
        alpha_u = alpha_u * mask.unsqueeze(2)
        alpha_u = alpha_u / alpha_u.sum(1).unsqueeze(1)
        return (alpha_u * u).sum(1)  # Context vector: BATCH_SIZE * HIDDEN_DIMENSION


class V_Att(torch.nn.Module):
    def __init__(self, hidden_dimension, embedding_dimension, visual_feature_dimension, number_of_steps):
        super(V_Att, self).__init__()
        self.v_0 = None
        self.activation = nn.Tanh()
        self.layer1 = nn.ModuleList([nn.Linear(in_features=visual_feature_dimension, out_features=hidden_dimension)
                                    for _ in range(number_of_steps)])
        self.layer2 = nn.ModuleList([nn.Linear(in_features=embedding_dimension, out_features=hidden_dimension)
                                     for _ in range(number_of_steps)])
        self.linear_sm = nn.ModuleList([nn.Linear(in_features=embedding_dimension, out_features=1)
                                        for _ in range(number_of_steps)])
        self.P = nn.ModuleList([nn.Linear(in_features=visual_feature_dimension, out_features=embedding_dimension)
                                for _ in range(number_of_steps + 1)])

    def forward(self, v, k):
        h_v = self.activation(self.layer1[k](v)) * torch.unsqueeze(self.activation(self.layer2[k](self.m_v)), 1)
        alpha_v = self.linear_sm[k](h_v)  # BATCH_SIZE * NO_OF_REGIONS_IN_IMAGE * 1
        alpha_v = F.softmax(alpha_v, dim=1)
        return self.activation(
            self.P[k + 1]((alpha_v * v).sum(1)))  # Context vector: BATCH_SIZE * EMBEDDING_DIMENSION
