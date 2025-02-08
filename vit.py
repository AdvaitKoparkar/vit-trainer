import math
import torch

class PatchEmbedding(torch.nn.Module):
    def __init__(self, config : dict ) :
        '''
            split image into patches and encode each patch into d-dim space
        '''
        super().__init__()
        self.img_dim = config['img_dim']
        self.dim = config['dim']
        self.patch_size = config['patch_size']

        C, H, W = self.img_dim
        self.proj = torch.nn.Conv2d(C, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.num_patches = (H//self.patch_size)*(W//self.patch_size)
    
    def forward(self, x : torch.Tensor ) -> torch.Tensor :
        N, _, _, _ = x.shape
        x = self.proj(x)
        x = x.reshape(N, self.dim, self.num_patches).transpose(1,2)
        return x # N, #patches, D

class Embedding(torch.nn.Module):
    def __init__(self, config : dict ) :
        '''
            get patch embedding, create class label and add positional encoding
        '''
        super().__init__()
        self.patch_embedding = PatchEmbedding(config)
        self.cls = torch.nn.Parameter((torch.rand(1, 1, config['dim'])))
        self.pos_enc = torch.nn.Parameter(self._generate_pos_enc(config['dim'], self.patch_embedding.num_patches+1, config['wavelength']))
    
    def forward(self, x : torch.Tensor ) -> torch.Tensor :
        N, _, _, _ = x.shape
        x = self.patch_embedding(x)
        c = self.cls.expand(N, -1, -1)
        x = torch.cat([c, x], axis=1)
        x = self.pos_enc + x
        return x

    def _generate_pos_enc(self, dim : int , seq_len : int , k : int ) :
        pos = torch.arange(seq_len)[:, None]
        pe = torch.zeros((1, seq_len, dim))
        pe[0, :, 0::2] = torch.cos(pos/(k**(2*torch.arange(dim//2)[None, :]/dim)))
        pe[0, :, 1::2] = torch.sin(pos/(k**(2*torch.arange(dim-dim//2)[None, :]/dim)))
        return pe

class SSAttn(torch.nn.Module):
    def __init__(self, config : dict ):
        super().__init__()
        self.dim = config['dim']//config['n_head']
        self.hidden_size = config['hidden_size']
        self.query = torch.nn.Linear(self.dim, self.hidden_size)
        self.key = torch.nn.Linear(self.dim, self.hidden_size)
        self.val = torch.nn.Linear(self.dim, self.hidden_size)
        self.proj = torch.nn.Linear(self.hidden_size, self.dim)

    def forward(self, x : torch.Tensor ) -> torch.Tensor :
        q = self.query(x) # N, H, S, h
        k = self.key(x)
        v = self.val(x)

        attn = q @ k.transpose(-1, -2) # N, H, S, S
        attn = attn / math.sqrt(self.dim)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = attn @ v

        out = self.proj(out)
        return out

class MultiHeadAttn(torch.nn.Module):
    def __init__(self, config : dict ) :
        super().__init__()
        self.dim = config['dim']
        self.n_head = config['n_head']
        # assert self.dim%self.n_head == 0

        self.attn = SSAttn(config)
        self.dropout = torch.nn.Dropout(p=config['dropout'])
        self.norm = torch.nn.LayerNorm(self.dim)
        self.ff = torch.nn.Linear(self.dim, self.dim)
    
    def forward(self, x : torch.Tensor ) -> torch.Tensor :
        N, S, d = x.shape

        # split d into n_head chunks of size d//n_head each
        # permute to size N, n_head, S, d//n_head
        x = torch.split(x[:, :, None, :], d//self.n_head, dim=-1)
        x = torch.cat(x, dim=2)
        x = x.permute(0, 2, 1, 3)

        # calculate self-attention for each N, n_head
        x = self.attn(x)

        # permute to N, S, (d//n_head, n_head)
        # reshape to N, S, d
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(N, S, d)

        # FF proj
        x = self.dropout(x)
        x = x + self.ff(self.norm(x))
        
        return x
    
class ViT(torch.nn.Module):
    def __init__(self, config : dict ):
        super().__init__()
        self.n_layers = config['n_layers']
        self.emb = Embedding(config)
        self.layers = torch.nn.ModuleList([MultiHeadAttn(config) for _ in range(self.n_layers)])
        self.mlp = torch.nn.Linear(config['dim'], config['n_classes'])
    
    def forward(self, x : torch.Tensor ) -> torch.Tensor :
        # N, C, H, W = x.shape
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x =  self.mlp(x[:, 0, :])
        return x
