#########################
# Tokenizer Definition
# Author: Koureas Stavros
#########################

from src.networks.mlp import MLP
from src.networks.moe import MOE
from src.networks.nft import NFT

class Network:
    def __new__(cls, mp, c_network, n_emb, n_ctx, p_dropout):
        match c_network:
            case "mlp":
                return MLP(mp, n_emb, p_dropout)
            case "moe":
                return MOE(mp, n_emb, p_dropout, n_experts=4, expansion=4)
            case "nft":
                return NFT(mp, n_emb, n_ctx, p_dropout)
            case _:
                raise ValueError(f"Unknown network type: {c_network}")