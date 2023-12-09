def estimate_parameters(s_vocab, n_ctx, n_layers, d_model, n_heads, d_head, hidden_ratio=4):
    """
    Calculate the number of parameters (in Millions) in the GPT-3 models.
    Args:

        s_vocab (int): Size of the vocabulary.
        n_ctx (int): Context size.
        n_layers (int): Number of layers.
        d_model (int): Embedding size.
        n_heads (int): Number of heads.
        d_head (int): Size of each head.
        hidden_ratio (float): Ratio of hidden size to embedding size. GPT-3 uses 4 and LLaMA uses 2.3
    Returns:
        float: Number of parameters (in Millions).

    """
    # The usage of magic numbers are specified in the report
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
    # The usage of magic numbers are specified in the report
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
    # The usage of 3 is specified in the report
    total = estimate_forward_flops(s_vocab, n_ctx, n_layers, d_model, n_heads, d_head) * 3
    return total

def estimate_memory(s_vocab, n_ctx, n_layers, d_model, n_heads, d_head, hidden_ratio=4):
    """
    Calculate the memory (in GB) in the GPT-3 models.
    Args:

        s_vocab (int): Size of the vocabulary.
        n_ctx (int): Context size.
        n_layers (int): Number of layers.
        d_model (int): Embedding size.
        n_heads (int): Number of heads.
        d_head (int): Size of each head.
        hidden_ratio (float): Ratio of hidden size to embedding size. GPT-3 uses 4 and LLaMA uses 2.3
    Returns:
        float: Memory (in GB).
    """
    # The usage of 20 is specified in the report
    return 20 * estimate_parameters(s_vocab, n_ctx, n_layers, d_model, n_heads, d_head, hidden_ratio)
