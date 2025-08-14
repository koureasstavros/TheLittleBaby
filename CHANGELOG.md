## 📦 Changelog

- Version v0.0.5
  - Refactored Little Baby's code
  - Implemented save and load of tokenizer
  - Added finetune option into the workflow
  - Added cache into inference process (kv_cache)
  - Removed some python loops to increase cores used

- Version v0.0.6
  - Switched model file extension from weights to json

- Version v0.0.7
  - Fixed epoch and batch report prints

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

- Version v0.0.9
  - Moved config files into Hugging Face Repo