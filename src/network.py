#########################
# Network Definition
# Author: Koureas Stavros
#########################

from src.networks.mlp import MLP
from src.networks.moe import MOE
from src.networks.swi import SWI
from src.networks.lor import LOR
from src.networks.gln import GLN
from src.networks.ggl import GGL
from src.networks.nft import NFT
from src.networks.sms import SMS  

class Network:
    def __new__(cls, mp, n_layers, i_layer, c_network, d_type, n_ctx, n_emb, r_dropout):
        match c_network:
            case "mlp":
                return MLP(mp, d_type, n_ctx, n_emb, r_dropout, n_expansion=4)
            case "moe":
                return MOE(mp, d_type, n_ctx, n_emb, r_dropout, n_expansion=4, n_experts=4)
            case "lor":
                return LOR(mp, d_type, n_ctx, n_emb, n_emb, r_dropout, rank=4, alpha=1.0)
            case "swi":
                return SWI(mp, d_type, n_ctx, n_emb, n_emb, n_expansion=4)
            case "gln":
                return GLN(mp, d_type, n_ctx, n_emb, r_dropout)
            case "ggl":
                return GGL(mp, d_type, n_ctx, n_emb, r_dropout, n_groups=4)
            case "nft":
                return NFT(mp, d_type, n_ctx, n_emb, r_dropout, clip=20.0)
            case "sms":
                return SMS(mp, d_type, n_ctx, n_emb, r_dropout)
            case _:
                raise ValueError(f"Unknown network type: {c_network}")