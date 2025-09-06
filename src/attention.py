#########################
# Tokenizer Definition
# Author: Koureas Stavros
#########################

from src.attentions.mha import MHA
from src.attentions.moh import MOH
from src.attentions.gqa import GQA
from src.attentions.swh import SWH
from src.attentions.aft import AFT
from src.attentions.lda import LDA
from src.attentions.rfa import RFA
from src.attentions.sca import SCA

class Attention:
    def __new__(cls, mp, c_attention, n_ctx, n_emb, r_dropout, head_size, n_heads):
        match c_attention:
            case "mha":
                return MHA(mp, n_ctx, n_emb, r_dropout, head_size, n_heads)
            case "moh":
                return MOH(mp, n_ctx, n_emb, r_dropout, head_size, n_heads)
            case "gqa":
                return GQA(mp, n_ctx, n_emb, r_dropout, head_size, n_heads)
            case "swh":
                return SWH(mp, n_ctx, n_emb, r_dropout, head_size, n_heads)
            case "aft":
                return AFT(mp, n_ctx, n_emb, r_dropout, clip=20.0)
            case "lda":
                return LDA(mp, n_ctx, n_emb, r_dropout, kernel_size=8)
            case "rfa":
                return RFA(mp, n_ctx, n_emb, r_dropout, head_size, n_heads, window_size=8)
            case "sca":
                return SCA(mp, n_ctx, n_emb, r_dropout, head_size, window_size=8)
            case _:
                raise ValueError(f"Unknown attention type: {c_attention}")