## 📦 Changelog
In this changelog, you can find a concise summary of all the updates made to the software—ranging from new features and performance improvements to bug fixes. It serves as a transparent record that helps users track the evolution of the software across different versions. Whether you're looking for what's new, what's changed, or what’s been resolved, the changelog is your go-to snapshot of progress.

- Version v0.1.20
  - 

- Version v0.1.19
  - Removed conditional architectural flow change on network
  
- Version v0.1.18
  - Renamed attention and network due to misleading title and description

- Version v0.1.17
  - Fixed Comments

- Version v0.1.16
  - Added expirements
  - Added skeleton for dynamic tokenizer

- Version v0.1.15
  - Added expirements
  - Fixed attention lda process issues

- Version v0.1.14
  - Added expirements
  - Added visual matrixes

- Version v0.1.13
  - Added expirements
  - Added d_type for data types of floats
  - Added attention r_temp for temperature

- Version v0.1.12
  - Aligned expirement

- Version v0.1.11
  - Added expirements
  - Added library information
  - Enhanched expirements visual management
  - Enhanched expirements visual evaluation

- Version v0.1.10
  - Added expirements
  - Aligned naming conventions
  - Added c_shuffle option into tokenizer for shuffling input data before split
  - Added r_split option into tokenizer for spliting input data into training and validation

- Version v0.1.9
  - Added expirements
  - Minor notebook fix

- Version v0.1.8
  - Added youtube video into readme file
  - Centralized head management for attentions

- Version v0.1.7
  - Fixed config files information cpu / gpu
  - Fixed attentions and networks comments order and description
  - Fixed attentions and networks prime functions for backward steps

- Version v0.1.6
  - Fixed blank charts

- Version v0.1.5
  - Added optimizer warmup
  - Added warmup options
  - Fixed naming conventions

- Version v0.1.4
  - Added expirements
  - Added attention and network comment step matching

- Version v0.1.3
  - Added expirements
  - Added gpu / cuda version handling
  - Added estimation for process flops
  - Fixed models biases / nullabilities

- Version v0.1.2
  - Added expirements
  - Added visual material
  - Added charts with loss
  - Added additional settings desciptions
  - Added c_tokenizer config to choose tokenizer
  - Added Word Tokenizer
  - Added Linear Diagonal Attention (LDA)
  - Added Linear Instant Network (LIN)

- Version v0.1.1
  - Added expirement
  - Fixed Naming Conventions
  - Added c_optimizer config to choose optimizer
  - Added Low-Rank Adaptation (lor)
  - Added Shifted Window Interaction (swi)
  - Added Unicode Support for inputs and outputs
  - Abstracted Import / Export for attentions and networks

- Version v0.1.0
  - Refactored code from notebook (ipynb) to python (py)
  - Added c_device config to choose between cpu and gpu
  - Added GPU Support
  - Added Grouped Query Attention (gqa)
  - Added Switch Head Attention (swh)
  - Added Attention Free Transformer (aft)
  - Added Network Free Transformer (nft)

- Version v0.0.10
  - Added expirement

- Version v0.0.9
  - Moved config files into Hugging Face Repo

- Version v0.0.8
  - Added debug option
  - Added architecture diagram
  - Added inference cache (kv)
  - Added c_sequence config to choose between pre and post
  - Added Post Norm sequence into the Block
  - Added c_attention config to choose between mha and moh
  - Added Multi Head Mixture Attention (moh)
  - Added model load / save for Multi Head Mixture Attention (moh)
  - Added c_network config to choose between mlp and moe
  - Added Mixture of Experts (moe)
  - Added model load / save for Mixture of Experts (moe)
  - Aligned naming conventions for common attributes
  
- Version v0.0.7
  - Fixed epoch and batch report prints

- Version v0.0.6
  - Switched model file extension from weights to json

- Version v0.0.5
  - Refactored Little Baby's code
  - Implemented save and load of tokenizer
  - Added finetune option into the workflow
  - Added cache into inference process (kv_cache)
  - Removed some python loops to increase cores used