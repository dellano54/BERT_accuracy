import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Embeddings(nn.Module):
    def __init__(self, vocab_size=5000, max_length=1024, embed_dim=768, dropout=0.2):
        super().__init__()
        self.vocab_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, s = x.shape
        x = self.vocab_embedding(x)
        pos = torch.arange(s).repeat(b, 1).to(x.device)
        pos = self.pos_embedding(pos)
        
        x += pos

        x = self.drop(x)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim = 768, n_heads=8, dropout=0.2, enc = False):
        super().__init__()

        self.enc = enc

        self.embed_dim = embed_dim
        self.h_dim = embed_dim // n_heads
        self.n_heads = n_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.drop = nn.Dropout(dropout)

        self.out = nn.Linear(embed_dim, embed_dim)


    @staticmethod
    def apply_padding_mask(attention_scores, padding_mask):
        padding_mask = padding_mask.unsqueeze(1)  # Shape: [batch_size, 1, seq_length]
        padding_mask = padding_mask.expand(-1, attention_scores.size(2), -1)  # Shape: [batch_size, seq_length, seq_length]
        
        # Replace 1s in padding mask with 0 and 0s with -inf
        padding_mask = padding_mask.masked_fill(padding_mask == 0, float('-1e+9'))  # Mask padded tokens with -inf
        padding_mask = padding_mask.masked_fill(padding_mask == 1, 0) 
        # Add the padding mask to attention scores
        masked_attention_scores = attention_scores + padding_mask
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(masked_attention_scores, dim=-1)
        
        return attention_weights
    

    @staticmethod
    def apply_look_ahead_mask(attention_scores, seq_length):
        # Create the look-ahead mask
        look_ahead_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)  # Upper triangular matrix
        look_ahead_mask = look_ahead_mask.masked_fill(look_ahead_mask == 1, float('-inf'))  # Fill upper triangle with -inf
        
        # Add the look-ahead mask to attention scores
        masked_attention_scores = attention_scores + look_ahead_mask
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(masked_attention_scores, dim=-1)
        
        return attention_weights



    def forward(self, q, k, v, mask=None):
        q_b, q_s, q_e = q.shape
        k_b, k_s, k_e = k.shape
        v_b, v_s, v_e = v.shape


        q, k, v = self.q(q), self.k(k), self.v(v)

        q = q.view(q_b, q_s, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(k_b, k_s, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(v_b, v_s, self.n_heads, self.h_dim).transpose(1, 2)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.h_dim)

        if (self.enc == True) and (mask is not None):
            attention_scores = self.apply_padding_mask(attention_scores, mask)

        elif (self.enc == False) and (mask is not None):
            attention_scores = self.apply_look_ahead_mask(attention_scores, v_s)


        attention_scores = self.drop(attention_scores)
        x = (attention_scores @ v).transpose(1, 2).contiguous().view(v_b, -1, self.embed_dim)

        x = self.out(x)

        return x


class ResAdd(nn.Module):
    def __init__(self, embed_dim=768, forward_expansion=4, dropout=0.2):
        super().__init__()

        self.w1 = nn.Linear(embed_dim, embed_dim * forward_expansion)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.w2 = nn.Linear(embed_dim * forward_expansion, embed_dim)

    def forward(self, x):
        x = self.w1(x)
        x = self.drop(x)
        x = self.act(x)
        x = self.w2(x)

        return x



class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, n_heads=8, forward_expansion=4, dropout=0.2):
        super().__init__()

        self.AttnMech = MultiHeadAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            enc=True
        )

        self.Norm1 = nn.LayerNorm(embed_dim)

        self.ResAdd = ResAdd(
            embed_dim=embed_dim,
            forward_expansion=forward_expansion,
            dropout=dropout
        )

        self.Norm2 = nn.LayerNorm(embed_dim)


    def forward(self, x, mask):
        res = x
        x = self.AttnMech(x, x, x, mask)
        x += res
        x = self.Norm1(x)

        res = x
        x = self.ResAdd(x)
        x += res
        x = self.Norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers=12, embed_dim=768, n_heads=8,
                  forward_expansion=4, dropout=0.2):
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim=embed_dim,
                                     n_heads=n_heads,
                                     forward_expansion=forward_expansion,
                                     dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x
    


class Transformer(nn.Module):
    def __init__(self, vocab_size=5000, n_layers=12, embed_dim=768, n_heads=8,
                  forward_expansion=4, dropout=0.2, max_length=1000):
        super().__init__()

        self.emb = Embeddings(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim,
            dropout=dropout
        )

        self.Encoder = TransformerEncoder(
            n_layers=n_layers,
            embed_dim=embed_dim,
            n_heads=n_heads,
            forward_expansion=forward_expansion,
            dropout=dropout
        )

        self.act = nn.GELU()
        self.decoder = nn.Linear(embed_dim, 1)

    def forward(self, x, mask=None):
        x = self.emb(x)
        x = self.Encoder(x, mask)
        x = self.act(x)
        x = self.decoder(x)

        return x
    