#########################
# Tokenizer Definition
# Author: Koureas Stavros
#########################

from src.networks.mlp import MLP
from src.networks.moe import MOE
from src.networks.swi import SWI
from src.networks.lor import LOR
from src.networks.nft import NFT

class Network:
    def __new__(cls, mp, c_network, n_emb, n_ctx, p_dropout):
        match c_network:
            case "mlp":
                return MLP(mp, n_emb, p_dropout, n_expansion=4)
            case "moe":
                return MOE(mp, n_emb, p_dropout, n_expansion=4, n_experts=4)
            case "swi":
                return SWI(mp, n_emb, n_emb, n_expansion=4)
            case "lor":
                return LOR(mp, n_emb, n_emb, p_dropout, rank=4, alpha=1.0)
            case "nft":
                return NFT(mp, n_emb, n_ctx, p_dropout, use_gate=True, clip=20.0)
            case _:
                raise ValueError(f"Unknown network type: {c_network}")