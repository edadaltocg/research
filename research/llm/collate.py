from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def get_causal_mask_from_padding_mask(
    padding_mask: torch.Tensor, target_seq_len: Optional[int] = None
) -> torch.Tensor:
    """
    Converts a padding mask of shape ``[bsz, seq_len]`` to a ``[bsz, seq_len, seq_len]`` causal attention mask suitable for
    consumption by :func:`~torch.nn.functional.scaled_dot_product_attention`. If ``target_seq_len``
    is provided, this will return a mask of shape ``[bsz, seq_len, target_seq_len]``. This is useful
    when generating masks for static KV caches where the maximum length the caches have been setup with
    are longer than the current sequence.

    Args:
        padding_mask (torch.Tensor): Boolean tensor where False indicates the corresponding token in the sequence
            is a padding token and should be masked out in attention, with shape [bsz x seq_length]
        target_seq_len (Optional[int]): target sequence length to create attention mask with. Default None.

    Returns:
        torch.Tensor: Boolean causal mask with shape
            - [bsz, seq_length, seq_length] or
            - [bsz, seq_length, target_seq_len] if ``target_seq_len`` was specified.

    Raises:
        AssertionError: if ``target_seq_len > seq_len``, the sequence length of the padding mask.

    Example:
        >>> padding_mask = torch.tensor([[False, True, True, True]])
        >>> get_causal_mask_from_padding_mask(padding_mask, target_seq_len=5)
        tensor([[[ True, False, False, False, False],
                  [False,  True, False, False, False],
                  [False,  True,  True, False, False],
                  [False,  True,  True,  True, False]]])
        ])
    """
    bsz, seq_len = padding_mask.shape
    target_seq_len = seq_len if target_seq_len is None else target_seq_len

    if target_seq_len < seq_len:
        raise AssertionError(
            "target_seq_len cannot be shorter than the sequence length of the padding mask."
        )

    mask = torch.tril(
        torch.ones(seq_len, target_seq_len, device=padding_mask.device, dtype=bool),
        diagonal=0,
    ).repeat(bsz, 1, 1)
    mask.narrow(2, 0, seq_len).mul_(padding_mask[:, None, :].expand(-1, seq_len, -1))
    mask.diagonal(dim1=1, dim2=2).copy_(torch.Tensor([True]))
    return mask


def get_position_ids_from_padding_mask(
    padding_mask: torch.Tensor,
):
    """
    Calculates position ids given a padding mask which right-shifts position ids to start
    from the first valid token.

    Args:
        padding_mask (torch.Tensor): Boolean tensor where False indicates the corresponding token in the sequence
            is a padding token and should be masked out in attention. Shape [bsz, seq_len]

    Returns:
        torch.Tensor: position ids which are appropriately shifted according to any padding values.

    Example:
        >>> padding_mask = torch.tensor([False, False, False, True, True, True, True, True])
        >>> get_position_ids_from_padding_mask(padding_mask)
        torch.Tensor([0, 0, 0, 0, 1, 2, 3, 4])
    """
    return ((padding_mask.cumsum(-1) - 1) * padding_mask).to(torch.int)


def left_pad_sequence(
    sequences: List[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0,
) -> torch.Tensor:
    """
    This function is identical to :func:`torch.nn.utils.rnn.pad_sequence`, but
    instead pads a list of variable length Tensors from the left to the length
    of the longest sequence.

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (List[torch.Tensor]): list of variable length sequences.
        batch_first (bool): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise. Default False.
        padding_value (float): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise

    Example:
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5, 6, 7])
        >>> c = torch.tensor([8, 9, 10, 11, 12])
        >>> left_pad_sequence([a, b, c], batch_first=True, padding_value=0)
        tensor([[ 0,  0,  1,  2,  3],
                [ 0,  4,  5,  6,  7],
                [ 8,  9, 10, 11, 12]])
    """
    return pad_sequence(
        map(lambda x: torch.flip(x, dims=[0]), sequences),
        batch_first=batch_first,
        padding_value=padding_value,
    ).flip(dims=[int(batch_first)])


def collate_self_consistency(
    batch,
    n_paths: int,
) -> Dict[str, Union[Tensor, str]]:
    collated = {}
    for k, v in batch[0].items():
        if isinstance(v, Tensor):
            collated[k] = torch.stack([x[k] for x in batch]).repeat(n_paths, 1)
        else:
            collated[k] = [x[k] for x in batch]
    return collated


def create_block_causal_mask(seq_lens: List[torch.Tensor]) -> torch.Tensor:
    """
    Given a batch tensor of seq lens defining the lengths of samples in each pack,
    Construct a 2D block causal mask for each pack in the batch. For example, if
    a single sample's seq_lens is [3, 2, 1], the mask would be::

        mask = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.


    Returns:
        Tensor: Block causal mask of shape (batch_size, max_seq_len, max_seq_len).
    """
    batch_block_attn_masks = []
    batch_size = len(seq_lens)
    for sample_idx in range(batch_size):
        block_attn_masks = [
            torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=seq_len.device))
            for i, seq_len in enumerate(seq_lens[sample_idx])
        ]

        batch_block_attn_masks.append(torch.block_diag(*block_attn_masks))
    return torch.stack(batch_block_attn_masks)


def padded_collate_packed(batch) -> Dict[str, torch.Tensor]:
    """Collate packed sequences into a batch. Only convert the seq lens into
    a block mask for use with attention. Tokens, labels, and input_pos are
    already padded to the same length within :class:`~torchtune.datasets.PackedDataset`.

    Args:
        batch (List[PACK_TYPE]): A list of pack dictionaries containing the following keys:
            - tokens: input token ids
            - labels: label token ids
            - input_pos: relative position ids for each sequence in pack
            - seq_lens: lengths of each sample within the pack

    Returns:
        Dict[str, torch.Tensor]: Collated input, label, input_pos, mask tensors.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3, 4, 5, 6], "labels": [7, 8, 9, 10, 11, 12],
        >>>     "input_pos": [0, 1, 2, 0, 1, 0], "seq_lens": [3, 2, 1]},
        >>>    {"tokens": [13, 14, 15, 16, 17, 18], "labels": [19, 20, 21, 22, 23, 24],
        >>>     "input_pos": [0, 1, 0, 1, 0, 1], "seq_lens": [2, 2, 2]},
        >>> ]
        >>> collated = padded_collate_packed(
        >>>    batch=token_pairs,
        >>> )
        >>> collated["mask"]
        >>> tensor([
        >>> [[1, 0, 0, 0, 0, 0],
        >>>  [1, 1, 0, 0, 0, 0],
        >>>  [1, 1, 1, 0, 0, 0],
        >>>  [0, 0, 0, 1, 0, 0],
        >>>  [0, 0, 0, 1, 1, 0],
        >>>  [0, 0, 0, 0, 0, 1]],
        >>> [[1, 0, 0, 0, 0, 0],
        >>>  [1, 1, 0, 0, 0, 0],
        >>>  [0, 0, 1, 0, 0, 0],
        >>>  [0, 0, 1, 1, 0, 0],
        >>>  [0, 0, 0, 0, 1, 0],
        >>>  [0, 0, 0, 0, 1, 1]])
    """

    tokens = torch.stack([x["tokens"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    input_pos = torch.stack([x["input_pos"] for x in batch])
    seq_lens = [x["seq_lens"] for x in batch]

    block_mask = create_block_causal_mask(seq_lens=seq_lens)

    return {
        "tokens": tokens,
        "labels": labels,
        "input_pos": input_pos,
        "mask": block_mask,
    }
