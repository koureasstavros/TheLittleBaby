#########################
# Attention Definition
# Author: Koureas Stavros
#########################

from src.attentions.mha import MHA
from src.attentions.moh import MOH
from src.attentions.gqa import GQA
from src.attentions.swh import SWH
from src.attentions.aft import AFT
from src.attentions.lda import LDA
from src.attentions.rfa import RFA

class Attention:
    def __new__(cls, mp, c_attention, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads):
        match c_attention:
            case "mha":
                return MHA(mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads)
            case "moh":
                return MOH(mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads)
            case "gqa":
                return GQA(mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads, n_kv_heads=None)
            case "swh":
                return SWH(mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads)
            case "aft":
                return AFT(mp, d_type, n_ctx, n_emb, r_dropout, r_temp, r_clip=20.0)
            case "lda":
                return LDA(mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_kernel=8)
            case "rfa":
                return RFA(mp, d_type, n_ctx, n_emb, r_dropout, r_temp, s_head, n_heads, s_window=8)
            case _:
                raise ValueError(f"Unknown attention type: {c_attention}")