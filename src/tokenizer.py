#########################
# Tokenizer Definition
# Author: Koureas Stavros
#########################

from src.tokenizers.char import CharTokenizer

class Tokenizer:
    def __new__(cls, mp, c_tokenizer):
        match c_tokenizer:
            case "char":
                return CharTokenizer(mp)
            case _:
                raise ValueError(f"Unknown tokenizer type: {c_tokenizer}")