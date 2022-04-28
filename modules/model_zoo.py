
from anchor_free.Uformer import UTransformerAB
from anchor_free.Uformer import UTransformer
def get_anchor_based(base_model, num_feature, num_hidden, anchor_scales,
                     num_head, **kwargs):
    return UTransformerAB(anchor_scales=anchor_scales,
                         dim_in=num_feature,
                         dim_out=num_feature,
                         heads=8,
                         mlp_dim=16,
                         dropout_rate=0.3,
                         attn_dropout_rate=0.3)

def get_anchor_free(base_model, num_feature, num_hidden, num_head, **kwargs):
    return UTransformer( dim_in=num_feature,
                         dim_out=num_feature,
                         heads=8,
                         mlp_dim=16,
                         dropout_rate=0.3,
                         attn_dropout_rate=0.3)

def get_model(model_type, **kwargs):
    if model_type == 'anchor-based':
        return get_anchor_based(**kwargs)
    elif model_type == 'anchor-free':
        return get_anchor_free(**kwargs)
    else:
        raise ValueError(f'Invalid model type {model_type}')
