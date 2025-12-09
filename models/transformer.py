import copy
import math
import torch
import torch.nn as nn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class AITextDetectionModel(nn.Module):
    def __init__(self, encoder, classifier_head):
        super(AITextDetectionModel, self).__init__()
        self.encoder = encoder
        self.classifier_head = classifier_head

    def forward(self, source, source_mask):
        encoded_output = self.encoder(source, source_mask)
        output_embedding = encoded_output[:, 0, :]  # [CLS] token
        logits = self.classifier_head(output_embedding)
        return logits


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attention, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer1 = ResidualConnection(size, dropout)
        self.sublayer2 = ResidualConnection(size, dropout)
        self._size = size

    @property
    def size(self):
        return self._size

    def forward(self, x, mask):
        x = self.sublayer1(x, lambda x: self.self_attention(x, x, x, mask))
        return self.sublayer2(x, self.feed_forward)


class EncoderStack(nn.Module):
    def __init__(self, encoding_layer, N):
        super(EncoderStack, self).__init__()
        self.encoding_layers = clones(encoding_layer, N)
        self.norm = LayerNorm(encoding_layer.size)

    def forward(self, x, mask):
        for layer in self.encoding_layers:
            x = layer(x, mask)
        return self.norm(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class ClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.linear2(x)
        return x


class Encoder(nn.Module):
    """Encoder-only model for pretraining (no classification head)."""
    def __init__(self, embedding, encoder_stack):
        super().__init__()
        self.embedding = embedding
        self.encoder_stack = encoder_stack

    def forward(self, x, mask):
        return self.encoder_stack(self.embedding(x), mask)


def make_encoder(vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Creates encoder-only model for pretraining."""
    c = copy.deepcopy
    attention_head = MultiHeadedAttention(h, d_model)
    feed_forward_network = PositionwiseFeedForward(d_model, d_ff, dropout)
    position_encoder = PositionalEncoding(d_model, dropout)

    model = Encoder(
        nn.Sequential(Embeddings(d_model, vocab), c(position_encoder)),
        EncoderStack(EncoderLayer(d_model, c(attention_head), c(feed_forward_network), dropout), N),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def make_classifier(vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, pretrained_encoder=None):
    """
    Creates classification model for AI text detection (fine-tuning).
    
    Args:
        vocab: Vocabulary size.
        N: Number of encoder layers.
        d_model: Model dimension.
        d_ff: Feed-forward dimension.
        h: Number of attention heads.
        dropout: Dropout rate.
        pretrained_encoder: Optional pretrained Encoder to use (for fine-tuning).
    """
    if pretrained_encoder is not None:
        encoder = pretrained_encoder
    else:
        encoder = make_encoder(vocab, N, d_model, d_ff, h, dropout)

    classifier_head = ClassifierHead(d_model, d_model // 2)
    
    model = AITextDetectionModel(encoder, classifier_head)

    # Only init classifier head if using pretrained encoder
    if pretrained_encoder is not None:
        for p in classifier_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    return model

