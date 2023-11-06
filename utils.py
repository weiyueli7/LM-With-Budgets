def estimate_parameters(s_vocab, n_ctx, n_layers, d_model, n_heads, d_head):
    """
    Calculate the number of parameters (in Millions) in the GPT-3 models.
    Args:

        s_vocab (int): Size of the vocabulary.
        n_ctx (int): Context size.
        n_layers (int): Number of layers.
        d_model (int): Embedding size.
        n_heads (int): Number of heads.
        d_head (int): Size of each head.
    Returns:
        float: Number of parameters (in Millions).

    """
    word_embed = s_vocab * d_model
    pos_embed = n_ctx * d_model
    qkv = (3 * d_model * d_head) * n_heads * n_layers
    proj = (n_heads * d_head * d_model) * n_layers
    w1, b1 = (d_model * (4 * d_model)) * n_layers, 4 * d_model * n_layers
    w2, b2 = ((4 * d_model) * d_model) * n_layers, d_model * n_layers
    l_norm = 2 * (2 * d_model) * n_layers
    
    total = word_embed+pos_embed+qkv+proj+w1+b1+w2+b2+l_norm
    return total / 1e6