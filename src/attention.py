#########################
# Tokenizer Definition
# Author: Koureas Stavros
#########################

from src.attentions.mha import MHA
from src.attentions.moh import MOH
from src.attentions.gqa import GQA
from src.attentions.swh import SWH
from src.attentions.aft import AFT

class Attention:
    def __new__(cls, mp, c_attention, n_emb, n_ctx, p_dropout, head_size, n_heads):
        match c_attention:
            case "mha":
                return MHA(mp, n_emb, n_ctx, p_dropout, head_size, n_heads)
            case "moh":
                return MOH(mp, n_emb, n_ctx, p_dropout, head_size, n_heads)
            case "gqa":
                return GQA(mp, n_emb, n_ctx, p_dropout, head_size, n_heads)
            case "swh":
                return SWH(mp, n_emb, n_ctx, p_dropout, head_size, n_heads)
            case "aft":
                return AFT(mp, n_emb, n_ctx, p_dropout)
            case _:
                raise ValueError(f"Unknown attention type: {c_attention}")