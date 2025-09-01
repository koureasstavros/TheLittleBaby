#########################
# Tokenizer Definition
# Author: Koureas Stavros
#########################

from src.networks.mlp import MLP
from src.networks.moe import MOE
from src.networks.swi import SWI
from src.networks.lor import LOR
from src.networks.nft import NFT
from src.networks.lin import LIN
from src.networks.ggl import GGL

class Network:
    def __new__(cls, mp, c_network, n_ctx, n_emb, p_dropout):
        match c_network:
            case "mlp":
                return MLP(mp, n_ctx, n_emb, p_dropout, n_expansion=4)
            case "moe":
                return MOE(mp, n_ctx, n_emb, p_dropout, n_expansion=4, n_experts=4)
            case "swi":
                return SWI(mp, n_ctx, n_emb, n_emb, n_expansion=4)
            case "lor":
                return LOR(mp, n_ctx, n_emb, n_emb, p_dropout, rank=4, alpha=1.0)
            case "nft":
                return NFT(mp, n_ctx, n_emb, p_dropout, use_gate=True, clip=20.0)
            case "lin":
                return LIN(mp, n_ctx, n_emb, p_dropout, use_gate=True)
            case "ggl":
                return GGL(mp, n_ctx, n_emb, p_dropout, n_groups=4)
            case _:
                raise ValueError(f"Unknown network type: {c_network}")