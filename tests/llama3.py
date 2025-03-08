import torch
import torch.nn as nn
import torch.nn.functional as F


"""
32 Decoder Layers:
    input: (tokens, embed_dim)

    - pre attention layer norm
    - grouped multi query attention layer
    - skip connection (add)
    - post attention layer norm
    - feedforward: (mul(up, F.silu(gate))), down

Post Decoder Layers:
    - post decoder norm
    - lm_head (output: logits)
    - softmax() -> probabilities
"""


class Llama3_Attention(nn.Module):
    def __init__(self, nheads, embed_dim):
        super().__init__()

        assert embed_dim % nheads == 0, "embed dim must be equally divisible by heads"

        self.nheads = nheads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // nheads

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim // 4)
        self.Wv = nn.Linear(embed_dim, embed_dim // 4)

        self.Wo = nn.Linear(embed_dim, embed_dim)

    def custom_expand(self, X, num):
        return X.repeat(1, num, 1, 1)

    def create_mask(self, tokens, device):
        mask = torch.triu(torch.ones(tokens, tokens), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf")).unsqueeze(0).unsqueeze(1)
        return mask.to(device)

    def forward(self, X):
        # X: (b, tokens, embed_dim)

        batch, tokens, embed_dim = X.shape

        # Q: (b, tokens, 4096) -> (b, nheads, tokens, head_dim)
        Q = self.Wq(X).reshape(batch, tokens, self.nheads, self.head_dim).transpose(1, 2)

        # K: (b, tokens, 1024) -> (b, nheads // 4, tokens, head_dim)
        K = self.Wk(X).reshape(batch, tokens, self.nheads // 4, self.head_dim).transpose(1, 2)

        # V: (b, tokens, 1024) -> (b, nheads // 4, tokens, head_dim)
        V = self.Wv(X).reshape(batch, tokens, self.nheads // 4, self.head_dim).transpose(1, 2)

        K = self.custom_expand(K, 4)
        V = self.custom_expand(V, 4)

        # attention_scores: (b, tokens, tokens)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.nheads**0.5)
        attention_scores += self.create_mask(tokens, X.device)
        attention_scores = F.softmax(attention_scores, dim=-1)

        # X: (b, tokens, nheads, head_dim)
        X = torch.matmul(attention_scores, V).transpose(1, 2).reshape(batch, tokens, embed_dim)
        X = self.Wo(X)

        return X


class Llama3_FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()

        self.W_Up = nn.Linear(embed_dim, hidden_dim)
        self.W_Gate = nn.Linear(embed_dim, hidden_dim)
        self.W_Down = nn.Linear(hidden_dim, embed_dim)

    def forward(self, X):
        # Up: (b, tokens, hidden_dim)
        Up = self.W_Up(X)
        # Gate: (b, tokens, hidden_dim)
        Gate = self.W_Gate(X)

        # X: (b, tokens, embed_dim)
        X = self.W_Down(F.silu(Gate) * Up)

        return X


class RMSLayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)  # rms calculation
        return x / rms * self.weight


class Llama3_Decoder(nn.Module):
    def __init__(self, nheads, embed_dim, ffn_hidden_dim):
        super().__init__()

        self.pre_attention_norm = RMSLayerNorm(embed_dim)

        self.attention = Llama3_Attention(nheads, embed_dim)

        self.feedforward = Llama3_FFN(embed_dim, ffn_hidden_dim)

        self.post_attention_norm = RMSLayerNorm(embed_dim)

    def forward(self, X):
        residual = X
        X = self.pre_attention_norm(X)
        X = self.attention(X)

        # skip connection
        X = residual + X

        residual = X
        X = self.post_attention_norm(X)
        X = self.feedforward(X)

        # skip connection
        X = residual + X

        return X


class Llama3(nn.Module):
    def __init__(self, nheads=32, embed_dim=4096, ffn_hidden_dim=14336, decoder_layers=32, vocab_size=128256):
        super().__init__()

        self.layer_embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoders = nn.ModuleList([Llama3_Decoder(nheads, embed_dim, ffn_hidden_dim) for i in range(decoder_layers)])
        self.post_decoder_norm = RMSLayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens, inference=False):
        X = self.layer_embedding(tokens)

        for decoder_layer in self.decoders:
            X = decoder_layer(X)

        X = self.post_decoder_norm(X)

        X = self.lm_head(X[:, -1, :])
        if inference:
            X = F.softmax(X, dim=-1)

        return X


if __name__ == "__main__":
    vocab_size = 128
    batch_size = 2
    seq_len = 5
    X = torch.randint(0, vocab_size, (batch_size, seq_len))

    device = "cpu"
    model = Llama3(nheads=4, embed_dim=64, ffn_hidden_dim=256, decoder_layers=4, vocab_size=vocab_size).to(device)

    print(f"\n{model}\n")

    Y = model(X, True).argmax(dim=-1)

    print(f"X ({X.shape}):\n{X}")
    print(f"\nY ({Y.shape}):\n{Y}")
