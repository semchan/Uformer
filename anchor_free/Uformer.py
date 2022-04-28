import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import nn
from anchor_free import anchor_free_helper
from anchor_based import anchor_helper
from helpers import bbox_helper


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return x + position_embeddings


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))





class FeedForwardNet(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim_out),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)




class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim_in,#1024
        dim_out,#1024
        heads,#16
        mlp_dim,#1024
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
                    Residual(
                        PreNorm(
                            dim_in,
                            # dropout_rate,
                            SelfAttention(
                                dim_in, heads=heads, dropout_rate=attn_dropout_rate
                                ),
                            )
                    ),
                    PreNorm(dim_in, FeedForwardNet(dim_in,dim_out, mlp_dim, dropout_rate))
        )

    def forward(self, x):
        x = self.net(x)#1,1296,512 ->1,1296,24



        return x

class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs



class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = IntermediateSequential(*layers)

    def forward(self, x):
        return self.net(x)

def _upsample_like(src,tar):
    src = src.permute(0, 2, 1).contiguous()
    src = F.upsample(src,size=tar.shape[1],mode='linear')
    src = src.permute(0, 2, 1).contiguous()
    return src


### UTransHeader ###
class UTransHeader(nn.Module):

    def __init__(self, num_feature, num_hidden):
        super(UTransHeader,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),

            nn.Tanh(),
            nn.Dropout(0.1),
            nn.LayerNorm(num_hidden),
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)
    def forward(self,x):
        _, seq_len, _ = x.shape#1, 531, 1024

        out = self.fc1(x)    
        pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        pred_loc = self.fc_loc(out).exp().view(seq_len, 2)
        pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)

        output_dict = {
            'pred_cls': pred_cls,     
            'pred_loc': pred_loc,    
            'pred_ctr': pred_ctr,
        }

        return output_dict

### UTransformer ###
class UTransformer(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, dim_in, dim_out, heads,mlp_dim,dropout_rate=0.1,attn_dropout_rate=0.1):
        super(UTransformer,self).__init__()
        dim_mid = 64
        heads=8
        mlp_dim=64

        self.linear_encoding1 = nn.Sequential(
            nn.Linear(dim_in, dim_mid),
            nn.ReLU6(inplace=False),
            nn.Dropout(0.1),
            nn.LayerNorm(dim_mid)
        )
        self.linear_encoding2 = nn.Sequential(
            nn.Linear(dim_in, dim_mid),
            nn.ReLU6(inplace=False),
            nn.Dropout(0.1),
            nn.LayerNorm(dim_mid)
        )
        self.layer_fusing = nn.LayerNorm(dim_mid*2)
        self.linear_fusing = nn.Sequential(
            nn.Linear(dim_mid*2, dim_mid*2),
            nn.ReLU6(inplace=False),
            nn.Dropout(0.1),
            nn.LayerNorm(dim_mid*2)
        )
        self.trans1 = TransformerLayer(dim_mid*2, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)
        self.pool1 = nn.MaxPool1d(2,stride=2,ceil_mode=True)
        self.trans2 = TransformerLayer(dim_mid, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)

        self.pool2 = nn.MaxPool1d(2,stride=2,ceil_mode=True)
        self.trans3 = TransformerLayer(dim_mid, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)

        self.layer_norm3d = nn.LayerNorm(dim_mid*2)
        self.trans3d = TransformerLayer(dim_mid*2, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)
        self.layer_norm2d = nn.LayerNorm(dim_mid*2)
        self.trans2d = TransformerLayer(dim_mid*2, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)
        self.layer_norm1d = nn.LayerNorm(dim_mid*2)
        self.trans1d = TransformerLayer(dim_mid*2, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)
        self.layer_header0 = nn.LayerNorm(dim_mid*5)
        self.header0 = UTransHeader(dim_mid*5, dim_out)
    def forward(self,x,xdiff):

        hx = x #1,1296,1024
        
        x = self.linear_encoding1(x)
        xdiff = self.linear_encoding2(xdiff)
        hx = torch.cat((x,xdiff),2)
        hx = self.linear_fusing(hx)
        hx1 = self.trans1(hx)
        phx1 = self.pool1(hx1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        hx2 = self.trans2(phx1)
        phx2 = self.pool2(hx2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        hx3 = self.trans3(phx2)
        hx = torch.cat((hx3,phx2),2)
        hx3d = self.trans3d(hx)
        hx3d = hx3d+hx3
        hx3dup = _upsample_like(hx3d,hx1)

        hx3dup_or = _upsample_like(hx3d,hx2)
        hx = torch.cat((hx2,hx3dup_or),2)
        hx2d = self.trans2d(hx)
        hx2d = hx2d+hx2
        hx2dup = _upsample_like(hx2d,hx1)

        hx = torch.cat((hx2dup,hx1),2)
        hx1d = self.trans1d(hx)

        hx1d = hx1d+hx1
        headin = torch.cat((x,xdiff,hx1d,hx2dup,hx3dup),2)
        headin = self.layer_header0(headin)
        d0 = self.header0(headin)


        output_dict = {
            'd0': d0, 
        }


        return output_dict

    def predict(self, seq,seqdiff):
        out =  self(seq,seqdiff)
        output_dict = out['d0']
        pred_cls = output_dict['pred_cls']
        pred_loc = output_dict['pred_loc']
        pred_ctr = output_dict['pred_ctr']

        pred_cls *= pred_ctr
        pred_cls /= pred_cls.max() + 1e-8

        pred_cls = pred_cls.cpu().numpy()
        pred_loc = pred_loc.cpu().numpy()

        pred_bboxes = anchor_free_helper.offset2bbox(pred_loc)

        return pred_cls, pred_bboxes


### UTransformer ###
class UTransHeaderAB(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, anchor_scales,num_feature, num_hidden):
        super(UTransHeaderAB,self).__init__()

        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]
        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.LayerNorm(num_hidden),
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
    def forward(self,x):
        _, seq_len, _ = x.shape#1, 531, 1024
        out = x#1, 149, 48
        out = out.transpose(2, 1)#1, 48, 149
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]#[1, 1024, 762];[1, 1024, 762];[1, 1024, 762];[1, 1024, 762]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]#761, 4, 1024

        out = self.fc1(out)#761, 4, 128

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)#761, 4
        pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2)#761, 4, 2

        output_dict = {
            'pred_cls': pred_cls,     
            'pred_loc': pred_loc,    
        }

        return output_dict
### UTransformer ###
class UTransformerAB(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, anchor_scales,dim_in, dim_out, heads,mlp_dim,dropout_rate=0.1,attn_dropout_rate=0.1):
        super(UTransformerAB,self).__init__()
        dim_mid = 64
        heads=8
        mlp_dim=64

        self.anchor_scales = anchor_scales
        self.linear_encoding1 = nn.Sequential(
            nn.Linear(dim_in, dim_mid),
            nn.ReLU6(inplace=False),
            nn.Dropout(0.1),
            nn.LayerNorm(dim_mid)
        )
        self.linear_encoding2 = nn.Sequential(
            nn.Linear(dim_in, dim_mid),
            nn.ReLU6(inplace=False),
            nn.Dropout(0.1),
            nn.LayerNorm(dim_mid)
        )
        self.layer_fusing = nn.LayerNorm(dim_mid*2)
        self.linear_fusing = nn.Sequential(
            nn.Linear(dim_mid*2, dim_mid*2),
            nn.ReLU6(inplace=False),
            nn.Dropout(0.1),
            nn.LayerNorm(dim_mid*2)
        )

        self.trans1 = TransformerLayer(dim_mid*2, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)
        self.pool1 = nn.MaxPool1d(2,stride=2,ceil_mode=True)
        self.trans2 = TransformerLayer(dim_mid, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)
        self.pool2 = nn.MaxPool1d(2,stride=2,ceil_mode=True)
        self.trans3 = TransformerLayer(dim_mid, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)
        self.layer_norm3d = nn.LayerNorm(dim_mid*2)
        self.trans3d = TransformerLayer(dim_mid*2, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)
        self.layer_norm2d = nn.LayerNorm(dim_mid*2)
        self.trans2d = TransformerLayer(dim_mid*2, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)
        self.layer_norm1d = nn.LayerNorm(dim_mid*2)
        self.trans1d = TransformerLayer(dim_mid*2, dim_mid, heads, mlp_dim, dropout_rate, attn_dropout_rate)
        self.layer_header0 = nn.LayerNorm(dim_mid*5)
        self.header0 = UTransHeaderAB(anchor_scales,dim_mid*5, dim_out)
    def forward(self,x,xdiff):

        hx = x #1,1296,1024
        
        x = self.linear_encoding1(x)
        xdiff = self.linear_encoding2(xdiff)
        hx = torch.cat((x,xdiff),2)
        hx = self.linear_fusing(hx)
        hx1 = self.trans1(hx)
        phx1 = self.pool1(hx1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        hx2 = self.trans2(phx1)
        phx2 = self.pool2(hx2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        hx3 = self.trans3(phx2)
        hx = torch.cat((hx3,phx2),2)
        hx3d = self.trans3d(hx)
        hx3d = hx3d+hx3
        hx3dup = _upsample_like(hx3d,hx1)
        hx3dup_or = _upsample_like(hx3d,hx2)
        hx = torch.cat((hx2,hx3dup_or),2)
        hx2d = self.trans2d(hx)
        hx2d = hx2d+hx2
        hx2dup = _upsample_like(hx2d,hx1)        
        hx = torch.cat((hx2dup,hx1),2)
        hx1d = self.trans1d(hx)

        hx1d = hx1d+hx1
        headin = torch.cat((x,xdiff,hx1d,hx2dup,hx3dup),2)
        headin = self.layer_header0(headin)
        d0 = self.header0(headin)


        output_dict = {
            'd0': d0, 
        }


        return output_dict

    def predict(self, seq,seqdiff):
        seq_len = seq.shape[1]
        out =  self(seq,seqdiff)
        output_dict = out['d0']
        pred_cls = output_dict['pred_cls']
        pred_loc = output_dict['pred_loc']
        pred_cls = pred_cls.cpu().numpy().reshape(-1)
        pred_loc = pred_loc.cpu().numpy().reshape((-1, 2))
        anchors = anchor_helper.get_anchors(seq_len, self.anchor_scales)
        anchors = anchors.reshape((-1, 2))
        pred_bboxes = anchor_helper.offset2bbox(pred_loc, anchors)
        pred_bboxes = bbox_helper.cw2lr(pred_bboxes)

        return pred_cls, pred_bboxes

