import sys
sys.path.append('..')

import torch
import unittest
import matplotlib.pyplot as plt

import vit

config = {
    'img_dim': (3, 224, 224),
    'dim': 512,
    'patch_size': 16,
    'wavelength': 1000,
    'n_head': 8,
    'hidden_size': 64,
    'dropout': 0.5,
    'n_layers': 2,
    'n_classes': 10,
}

class TestViT(unittest.TestCase):    
    def test_patch_embedding(self, ):
        img = torch.rand(16, *config['img_dim'])
        pe = vit.PatchEmbedding(config)
        P = pe(img)
        d = config['dim']
        num_patches = (config['img_dim'][1] // config['patch_size']) * (config['img_dim'][2] // config['patch_size'])
        assert P.shape[0] == img.shape[0] and P.shape[1] == num_patches and P.shape[2] == d

    def test_pos_encoding(self, ):
        emb = vit.Embedding(config)
        plt.imshow(emb.pos_enc.detach().numpy()[0, :, :])
        plt.title('viz pos encoding')
        plt.show()

    def test_embedding(self, ):
        img = torch.rand(16, *config['img_dim'])
        emb = vit.Embedding(config)
        E = emb(img)
        d = config['dim']
        num_patches = (config['img_dim'][1] // config['patch_size']) * (config['img_dim'][2] // config['patch_size'])
        assert E.shape[0] == img.shape[0] and E.shape[1] == num_patches+1 and E.shape[2] == d
    
    def test_ssattn(self, ):
        d = config['dim']
        nh = config['n_head']
        emb = torch.rand(16, nh, 197, d//nh)
        A = vit.SSAttn(config)(emb)
        assert A.shape[0] == emb.shape[0] and A.shape[1] == emb.shape[1] and A.shape[2] == emb.shape[2] and A.shape[3] == emb.shape[3]

    def test_mhattn(self, ):
        d = config['dim']
        nh = config['n_head']
        emb = torch.rand(16, 197, d)
        A = vit.MultiHeadAttn(config)(emb)
        assert A.shape[0] == emb.shape[0] and A.shape[1] == emb.shape[1] and A.shape[2] == emb.shape[2]

    def test_vit(self, ):
        model = vit.ViT(config)
        img = torch.rand(16, *config['img_dim'])
        O = model(img)
        assert O.shape[0] == img.shape[0] and O.shape[1] == config['n_classes']

if __name__ == '__main__':
    unittest.main()    

