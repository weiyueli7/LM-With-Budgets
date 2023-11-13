def estimate_parameters(s_vocab, n_ctx, n_layers, d_model, n_heads, d_head, hidden_ratio):
    """
    Calculate the number of parameters (in Millions) in the GPT-3 models.
    Args:

        s_vocab (int): Size of the vocabulary.
        n_ctx (int): Context size.
        n_layers (int): Number of layers.
        d_model (int): Embedding size.
        n_heads (int): Number of heads.
        d_head (int): Size of each head.
        hidden_ratio (float): Ratio of hidden size to embedding size.
    Returns:
        float: Number of parameters (in Millions).

    """
    word_embed = s_vocab * d_model
    pos_embed = n_ctx * d_model
    qkv = (3 * d_model * d_head) * n_heads * n_layers
    proj = (n_heads * d_head * d_model) * n_layers
    w1, b1 = (d_model * (hidden_ratio * d_model)) * n_layers, hidden_ratio * d_model * n_layers
    w2, b2 = ((hidden_ratio * d_model) * d_model) * n_layers, d_model * n_layers
    l_norm = 2 * (2 * d_model) * n_layers
    
    total = word_embed+pos_embed+qkv+proj+w1+b1+w2+b2+l_norm
    return total / 1e6


def estimate_forward_flops(s_vocab, n_ctx, n_layers, d_model, n_heads, d_head):
    """
    Calculate the number of forward FLOPs (in Millions) in the GPT-3 models.
    Args:

        s_vocab (int): Size of the vocabulary.
        n_ctx (int): Context size.
        n_layers (int): Number of layers.
        d_model (int): Embedding size.
        n_heads (int): Number of heads.
        d_head (int): Size of each head.
    Returns:
        float: Number of forward FLOPs (in Millions).

    """
    word_embed = 2 * n_ctx * s_vocab * d_model
    pos_embed = n_ctx * d_model
    qkv = 3 * 2 * n_ctx * d_model * d_head * n_heads * n_layers
    attention = 2 * n_ctx * n_ctx * d_head * n_heads * n_layers
    proj = n_ctx * n_heads * n_heads * d_model * n_layers
    ff = (16 * n_ctx * d_model * d_model + 26 * n_ctx * d_model) * n_layers
    output = 2 * n_ctx * d_model * s_vocab + 3 * n_ctx * s_vocab

    total = word_embed+pos_embed+qkv+attention+proj+ff+output
    return total


def estimate_backward_flops(s_vocab, n_ctx, n_layers, d_model, n_heads, d_head):
    """
    Calculate the number of backward FLOPs (in Millions) in the GPT-3 models.
    Args:

        s_vocab (int): Size of the vocabulary.
        n_ctx (int): Context size.
        n_layers (int): Number of layers.
        d_model (int): Embedding size.
        n_heads (int): Number of heads.
        d_head (int): Size of each head.
    Returns:
        float: Number of backward FLOPs (in Millions).

    """
    total = d_model * n_ctx + \
        (n_ctx * d_model + n_ctx * d_model * d_model 
         + d_model * n_ctx * d_head * n_heads) * \
        n_layers + (n_ctx * d_head * n_heads + 
                        n_ctx * n_ctx * d_model + 
                        n_ctx * d_model + d_head 
                        * n_ctx * d_model) * \
        n_layers * n_heads + d_model * n_ctx + \
            d_model * n_ctx * s_vocab
    return total