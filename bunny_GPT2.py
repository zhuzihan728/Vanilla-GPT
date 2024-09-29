import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.utils import get_network_paras_amount
from bunny_BPE import BPETokenizer

class CausalSelfAttention(nn.Module):
    def __init__(self, block_size, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        self.att_w_dropout = nn.Dropout(dropout)
        self.res_dropout = nn.Dropout(dropout)
        
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        
    def forward(self, x):
        b, t, e = x.size()
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, t.size(1), self.num_heads, self.head_dim).transpose(1, 2), (q, k, v))
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.att_w_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).contiguous().view(b, t, e)
        x = self.out(x)
        return self.res_dropout(x)
    
    def forward_w_att(self, x):
        b, t, e = x.size()
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, t.size(1), self.num_heads, self.head_dim).transpose(1, 2), (q, k, v))
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.att_w_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).contiguous().view(b, t, e)
        x = self.out(x)
        return self.res_dropout(x), attn

class DecoderBlock(nn.Module):
    def __init__(self, block_size, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(block_size, embed_dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
    def forward_w_att(self, x):
        x_ = x.clone()
        x, att = self.attn.forward_w_att(self.ln1(x))
        print(att.shape)
        x = x + x_
        x = x + self.mlp(self.ln2(x))
        return x, att
        
    
class GPT2(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, num_blocks, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        
        self.blocks = nn.ModuleList([DecoderBlock(block_size, embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_blocks)])
        self.embed_dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, x):
        _, t = x.size()
        device = x.device
        pos = torch.arange(0, t, device=device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.embed_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.head(x)
    
    def forward_w_att(self, x):
        _, t = x.size()
        device = x.device
        pos = torch.arange(0, t, device=device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.embed_dropout(x)
        atts = []
        for block in self.blocks:
            x, att = block.forward_w_att(x)
            atts.append(att)
        x = self.ln(x)
        return self.head(x), atts
    
    def compute_loss(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss
    
    
    def generate(self, input_ids, max_new_tokens):
        """

        Returns:
            _type_: _description_
        """
        
        output = {'input_ids': [], 'probs': []}
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        prefix_len = input_ids.size(1)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(input_ids)
                logits = logits[:, -1, :]
                logits = F.softmax(logits, dim=-1)
                output['probs'].append(logits.squeeze(1).cpu().numpy())
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        output['input_ids'] = input_ids[:, prefix_len:].cpu().numpy()
        return output
if __name__ == "__main__":
    
    tokenizer = BPETokenizer()
    vocab_size = tokenizer.vocab_size
    block_size = 10
    embed_dim = 512
    num_heads = 8
    num_blocks = 8
    batch_size = 2
    
    model = GPT2(vocab_size, block_size, embed_dim, num_heads, num_blocks)
    print("=== GPT2 ===")
    print(f"===> Number of trainable parameters: {get_network_paras_amount({'GPT2': model})}")
    
    text = "Do I have real friends? I often thought of this question after I had entered junior high."
    
    
    token_ids = tokenizer.encode(text)
    
    if len(token_ids) % block_size != 0:
        token_ids += [50256] * (block_size - len(token_ids) % block_size)
    
    token_ids = torch.tensor(token_ids).reshape(batch_size, -1)
    output = model(token_ids)
 
    print(output.shape)