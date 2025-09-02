#########################
# Tokenizer Definition
# Author: Koureas Stavros
#########################

from src.optimizers.adam import AdamW

class Optimizer:
    def __new__(cls, mp, c_optimizer, parameters, r_learn):
        match c_optimizer:
            case "adamw":
                return AdamW(mp, parameters, r_learn, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)           
            case _:
                raise ValueError(f"Unknown optimizer type: {c_optimizer}")