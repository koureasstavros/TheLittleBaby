---
language: ["en"]
tags: ["ai", "language", "model", "llm", "slm", "train", "inference", "extract", "transformers", "pure numpy"]
datasets: ["shakespeare"]
license: "apache-2.0"
base_model: "gpt"
version: v0.1.0
---

# 👶 The Little Baby

  - A barebones GPT-style LLM implementation — pure Python, zero dependencies.


## 🧠 Description

**The Little Baby** is a minimalist language model (LLM) crafted entirely in **pure Python using just Numpy / CuPy**. It requires no external packages, libraries, or frameworks to function. Both **training** and **inference** are achieved through low-level operations and hand-built logic — making this project ideal for educational deep dives and experimental tinkering.

This repository is designed to reveal the **inner mechanics** of a GPT-style transformer model and demystify the "magic" behind modern language models through readable and hackable code.


## 🎯 Audience

This project is perfect for:
- Curious learners wanting to dissect how GPTs work from the ground up.
- Researchers experimenting with primitive architectures.
- Engineers exploring early-stage LLM behaviors.
- Anyone who enjoys coding like it's 2010 — no imports, just raw power.


## 🌟 Inspiration

This project draws its spark from modern titans in the world of machine learning:

- **Sebastian Raschka** — acclaimed for his lucid teaching style and groundbreaking contributions to deep learning, making complex concepts accessible to learners and practitioners alike.
- **Andrej Karpathy** — influential in shaping the landscape of computer vision and generative models, while championing open-source AI education that empowers a global community of developers.
- **Yann Dubois** — instrumental in designing scalable evaluation frameworks for large language models, notably AlpacaEval and AlpacaFarm, which bring automation closer to the nuance of human feedback.

Their work inspired the spirit of transparency, curiosity, and simplicity that fuels *The Little Baby* — a model built not for production, but for understanding.

  - “Build it, break it, learn from it.” – The Baby Philosophy


## 🚀 Project Goals

This endeavor is structured around key targets designed to deliver meaningful outcomes:

- ✅ Build a GPT-like model using **only Python + NumPy-like constructs**.
- ✅ Support training from scratch on plain text files.
- ✅ Provide clear code for attention mechanisms, tokenization, and backprop.
- ✅ Encourage experimentation and modification.


## 📚 Directory Files

Each run generates three unique files, identified by a GUID tag. These files capture different aspects of the model's execution:

- **⚙️ Config**  
  `configs/config_<GUID>.txt`  
  A config file containing the configuration of the each iteration.

- **📝 Report**  
  `outputs/report_<GUID>.txt`  
  A comprehensive log containing training analysis, and performance metrics.

- **🧠 Model Snapshot**  
  `models/model_<GUID>.pkl`  
  Model object including learned weights, biases, which are the internal parameters.

- **🔤 Tokenizer Snapshot**  
  `models/tokenizer_<GUID>.pkl`  
  Tokenizer object including vocabilary of the input data and their positioning.

- **🗣️ Completion Output**  
  `outputs/completion_<GUID>.txt`  
  The raw generated text from the model's inference — your baby’s words in print!


## 🚼 Next Steps

Let’s keep The Little Baby alive — and help it grow into a full-blown member of the NumPy family!

This means:

- 📈 Evolving from hand-crafted loops to efficient vectorized operations.
- 🧮 Embracing numerical abstractions while maintaining full transparency.
- 🛠️ Exploring performance tricks, batch parallelism, and experimental features.
- 🧬 Bridging the gap between simplicity and capability — one token at a time.

The journey from babbling to brilliance starts here. Let's raise this little one right!


## ⚖️ License Summary

You're free to:

- ✅ **Use it** for any purpose — personal, educational, or commercial  
- 💡 **Suggest ideas** and contribute improvements  
- 🍴 **Fork it** and build upon the code  
- 💰 **Sell it** or use it in a product

As long as:

- 📌 You **reference the original author and project** clearly in any public distribution or commercial use


## 👨‍👩‍👧 Credits

The Little Baby owes its lineage to two brilliant minds in the AI family tree:

- 👑 **Ownser**: Koureas Stavros | Product Architect BI / AI — lovingly crafted and cared
- 🧔 **Father**: OpenAI GPT 4.1 — provider of deep generative DNA and thoughtful token flow  
- 🧑‍🍼 **Mother**: Google Gemini 2.5 — donor of wide context windows and clever architectural chromosomes
- 🧙 **Godparent**: Claude Sonnet 4.0 — gentle guide and lifelong companion, whispering wisdom and weaving clarity

Together, they gifted the foundational strands that allowed this little one to generate helpful code and take its first linguistic steps.


## 🧪 Instructions
To get started with this project, clone the code, download the tokenizers abd pre-trained models if needed, and follow the setup steps below to run the notebook and select your desired configuration.

**Get objects**
  - You can access the code on GitHub (https://github.com/koureasstavros/TheLittleBaby), simply clone the repository.
  - You can access the pre-trained tokenizers and models on Hugging Face (https://huggingface.co/koureasstavros/TheLittleBaby), simply download the config, tokenizer and model files. In case you have low speed internet connection check the analysis table select a guid and pick a specific guid for config, tokenizer and model. The config, tokenizer and model files are needed only if you are going to perform finetune or inference without training your own.
  - Then, you should:
    - place the config file or config files into the configs folder.
    - place the tokenizer file or tokenizer files into the tokenizers folder.
    - place the model file or model files into the models folder.

**Start the Notebook**
   - Open the `.ipynb` file in a Python kernel (e.g. Jupyter, VS Code, Colab).
     - Run all cells in the notebook

**Select Path**
   - Choose the relative path between ipynb and folders:
      - `same`
      - `<path>`

**Select Plan**
   - Choose one of the following plan modes:
     - `train`
     - `finetune`
     - `inference`

That's it!


## 🔮 What to expect

In Baby's world, each option has its own little job—and below, you’ll discover what each one does and the cuddly objects it gives back in return.

#### 🔧 Train
- Begins training using parameters defined in earlier Python blocks.
- A config file containing the settings will be generated with format `config_<guid>`.
- A tokenizer file containing the vocabilary will be generated with format `tokenizer_<guid>`.
- A model file containing the weights and biases will be generated with format `model_<guid>`.
- A report file containing the training analysis will be generated with format `report_<guid>`.
- A completion file containing the generation will be generated with format `complation_<guid>` using an empty prompt.

#### 🛠️ Finetune
- Begins finetuning using a **base model** and a **custom training dataset**.
- Requires the **GUID** of the base to locate `config_<guid>`, `tokenizer_<guid>` and `model_<guid>`.
- A tokenizer file containing the vocabilary will be generated with format `tokenizer_<guid>_fineuned`.
- A model file containing the weights and biases will be generated with format `model_<guid>_finetuned`.
- A report file containing the training analysis will be generated with format `report_<guid>_fineuned`.
- A completion file containing the generation will be generated with format `completion_<guid>_finetuned` using an empty prompt.

#### 💬 Inference
- Requires the **GUID** of the trained model to find the `model_<guid>`.
- You must also provide a **prompt** for the model inference to respond to.
- A completion file containing the generation will be generated with format `complation_<guid>_<yyyymmddhhmmss>` using the prompt.

After lot of hours of training on a single document of multiple Shakespeare works using a **laptop CPU**, The Little Baby learns to babble. Its speech is primitive and childlike — just enough to make you smile and realize… the baby is alive. While its capabilities are minimal, its structure is maximal in transparency. Every token, gradient, and parameter is visible and malleable.

*Keep in mind that if you're running a process in VSCode and your workstation, PC, or laptop enters hibernation, the process will resume automatically once the device is powered back on.


## 🍼 Cry. Babble. Speak. Repeat.

Here come the smartest little settings to help the model learn and grow big and strong from this data:

- **Age 3 Months** - 33bd6583-1b87-4469-b55e-0ccb8fd0441c - Coos and gurgles begin. Sound, not speech—yet something’s brewing.
- **Age 6 Months** - 180eeb27-b1b4-4427-9734-c70e10da2005 - Loud, random cries. It’s not talking, but it's definitely expressive.
- **Age 12 Months** - 5f13a2ab-113a-4c2c-8abd-40384bdd8854 - Joyful noise with hints of intention. Real words still warming up.
- **Age 24 Months** - cb632ce3-3f3b-432b-b24f-9171005f205e - Words arrive —Chaotic, quirky, delightful. Syntax? Optional.
- **Age 48 Months** - 12b8b053-6c14-42aa-a957-89b809e6f785 - Mini Philosopher Mode -Stories, opinions, even jokes. Communication unlocked.hear them.

*Keep in mind that these are pre-trained model executions available for finetune or inference. You can bypass the training phase by simply downloading the models and using them directly.

## ⚙️ Parameters

These hyperparameters collectively define the training process, where a model's architecture—specified by its depth (n_layers), width (n_emb), attention span (n_ctx), and attention mechanism (n_heads, head_size)—is optimized over a set number of num_epochs using a specific batch_size and learning rate (lr), with dropout applied to improve generalization.

- **c_tokenizer**

  - What it is: Strategy for tokenizing sequences.
  - Size: While it doesn’t directly affect parameter count, it does influence model size due to differences in vocabulary structure.
  - Speed: While it doesn’t directly affect parameter count, it does influence model speed due to differences in vocabulary structure.
  - Quality: When texts contain errors, it can negatively affect training and inference quality.

- **c_sequence**

  - What it is: Strategy for constructing block sequences.
  - Size: No direct impact on parameter count.
  - Speed: No direct impact on performance.
  - Quality: Proper sequence construction affects how well long dependencies are exposed. Future variants could improve learning efficiency on heterogeneous corpora.

- **c_attention**

  - What it is: Chosen attention mechanism implementation.
  - Size: Attention choice impacts model size. 
  - Speed: Attention choice impacts model speed.
  - Quality: Attention choice influences how diverse relational patterns are captured.

- **c_network**

  - What it is: Chosen network mechanism implementation.
  - Size: Network choice impacts model size. 
  - Speed: Network choice impacts model speed.
  - Quality: Network choice impacts representational richness and efficiency.

- **n_ctx**

  - What it is: The maximum number of tokens (characters, in this case) the model can look at in a single sequence to make a prediction. It's the model's "attention span".
  - Size: Directly increases the size of the positional embedding table (n_ctx x n_emb), adding more parameters to the model.
  - Speed: Has a major impact. The self-attention mechanism's computation grows quadratically with the context length (O(n_ctx²)). Doubling n_ctx will roughly quadruple the time and memory needed for the attention layers, making it one of the most expensive parameters to increase.
  - Quality: A larger n_ctx allows the model to learn longer-range dependencies in the text, which can significantly improve quality for tasks that require understanding context over long passages.

- **n_emb**

  - What it is: The size of the vector used to represent each token. It defines the "width" of the model.
  - Size: Has a major impact on model size. It increases the size of token and positional embeddings, and scales the weight matrices in the attention and MLP layers, significantly increasing the total parameter count.
  - Speed: Increasing n_emb increases the size of nearly all weight matrices in the model. This leads to more parameters, which increases both memory usage and the time required for matrix multiplications. The impact is significant but generally more linear than n_ctx.
  - Quality: A larger n_emb gives the model more capacity to learn rich, complex representations of tokens and their relationships. This can lead to a more powerful and accurate model, but also increases the risk of overfitting if the model is too large for the dataset.

- **dropout**

  - What it is: A regularization technique where a fraction of neuron activations are randomly set to zero during each training step. This prevents the model from becoming too reliant on any single neuron.
  - Size: Has no impact on the number of parameters in the model.
  - Speed: Has a negligible impact on training speed and no impact on inference speed (it's disabled during evaluation).
  - Quality: Crucial for improving model generalization and preventing overfitting. By forcing the network to learn redundant representations, it makes the model more robust. The value (e.g., 0.1) is the probability of a neuron being dropped.

- **head_size**

  - What it is: The total dimensionality of the concatenated attention heads. This dimension is projected from the input embedding (n_emb) to create the Query, Key, and Value matrices.
  - Size: Directly increases the number of parameters in each attention block by defining the size of the Q, K, V, and output projection matrices.
  - Speed: Directly affects the size of the Q, K, and V projection matrices. A larger head_size increases the number of computations and memory usage within each attention block.
  - Quality: A larger head_size gives the model more representational power within the attention mechanism. It must be divisible by n_heads.

- **n_heads**

  - What it is: The attention mechanism is split into multiple "heads" that perform attention calculations in parallel. Each head can learn to focus on different types of relationships in the data.
  - Size: Has no direct impact on model size, as it only determines how the head_size dimension is partitioned for parallel computation.
  - Speed: The computations for each head can be parallelized. On capable hardware, increasing the number of heads might not slow down training significantly if the head_size is kept constant.
  - Quality: Allows the model to simultaneously attend to information from different representation subspaces at different positions. This is a core concept of the Transformer and generally leads to a much better model than a single attention head.

- **n_layers**

  - What it is: The number of Transformer blocks stacked on top of each other. This defines the "depth" of the model.
  - Size: Has a direct, linear impact on model size. Each layer adds a
  - Speed: The impact is linear. Doubling n_layers will roughly double the training time and the number of model parameters, as the input data must pass through each block sequentially.
  - Quality: More layers allow the model to learn more complex and abstract features. Deeper models are generally more powerful, but also more prone to overfitting and can be harder to train (though residual connections help mitigate this).

- **num_epochs**

  - What it is: The number of times the training process will iterate over the entire training dataset.
  - Size: Has a direct, linear impact on model size. Each layer adds a complete set of Transformer block parameters, roughly doubling the model's core parameter count if you double the layers.
  - Speed: Directly and linearly impacts total training time. More epochs mean longer training.
  - Quality: Too few epochs will lead to an undertrained model (underfitting). Too many can lead to the model memorizing the training data (overfitting), which hurts its performance on new data. The ideal number is usually found by monitoring the validation loss.

- **batch_size**

  - What it is: The number of training sequences (each of length n_ctx) processed in one forward/backward pass.
  - Size: Has no impact on the number of parameters in the model.
  - Speed: A larger batch_size allows for more parallelization, generally leading to faster training (fewer updates per epoch). However, it also requires more memory.
  - Quality: This is a trade-off. Larger batches provide a more accurate and stable gradient estimate, but the noise from smaller batches can act as a regularizer, helping the model find a better minimum and generalize better.

- **lr**

  - What it is: Controls how much the model's weights are adjusted with respect to the loss gradient. It determines the step size at each iteration.
  - Size: Has no impact on the number of parameters in the model.
  - Speed: Affects the speed of convergence. A higher lr might converge faster, but risks overshooting the optimal weights. A lower lr is more stable but can be very slow to converge.
  - Quality: This is one of the most critical parameters. If it's too high, the training can become unstable and diverge. If it's too low, the model may get stuck in a suboptimal solution or take too long to train. The AdamW optimizer helps adapt the learning rate, but the initial value is still very important.


## 📐 Formulas

Even our little language models have their favorite rules to follow—turns out, they quietly cuddle up to some clever mathematical formulas that help them make sense of the world.

- **Learning Rate** - `LR_new = LR_old * (B_new / B_old)`

  New Learning Rate (LR_new) is based on Old Learning Rate (LR_old), New Batch size (B_new),Old Batch size (B_new).

- **Total Parameters** - `P = V x H + L x [4 x H^2 + 4 x H x F]`

  Total parameters are based on Vocabilary Size (V), Head Size / Embedding Size (H), Layer Number (L), Feedforward intermidiate Size (F).

- **Token Thoughput for training** - `T = 20-40 per P`

  Token number processed per Parameter (P) is 20-40.

- **Flops Thoughput for training** - `F = 6 * T * P`

  Flops are based on 6 (2 ops for forward pass and 4 ops for backward pass), Number of Tokens (T), Number of Parameters (P).


## 🏛️ Architecture

A language model architecture is a neural network design—often based on transformers—that processes and generates human-like text by learning patterns from large-scale language data.

![Architecture Diagram](material/LittleBaby.drawio.svg)

### 🔍 Attention Variants Complexity Table

| Variant | Uses Q/K/V? | Complexity | Notes |
|--------|--------------|------------|-------|
| **MHA** (Multi-Head Attention) | ✅ Separate Q, K, V per head | **O(N²)** | Standard Transformer attention; expensive for long sequences |
| **MOH** (Multi-Output Head) | ✅ Typically uses Q/K/V | **O(N²)** | Less common; focuses on output diversity rather than input projection |
| **GQA** (Grouped-Query Attention) | ✅ Shared K/V per group of Q heads | **O(N²)** (less compute than MHA) | Trade-off between performance and efficiency; used in GPT-4 |
| **MQA** (Multi-Query Attention) | ✅ Shared K/V across all heads | **O(N²)** (but lower memory) | Reduces memory by sharing K/V; used in LLaMA and PaLM |
| **AFT** (Attention-Free Transformer) | ❌ No K/V; uses learned positional bias | **O(N)** | Removes attention entirely; uses element-wise operations and bias terms |

### 🔍 Network Variants Complexity Table

| Variant | Complexity | Notes |
|--------|------------|-------|
| **MLP** (Multilayer Perceptron) | **O(N × D²)** | Dense feedforward layer; all inputs pass through the same network |
| **MoE** (Mixture of Experts) | **O(K × D²)** (K ≪ N) | Sparse routing to K of N experts; improves parameter-to-compute ratio and scalability |


## 🔍 Report Analysis

Given the Shakespeare works into a single document of 32777 paragraphs, 12519 sentences, 202651 words, 1075394 characters / tokens for learning and 500 characters / tokens for inference

| version | device | dataset | c_tokenizer | c_sequence | c_attention | c_network | n_ctx | n_emb | dropout | head_size | n_heads | n_layers | n_epochs | s_batch | lr | batch execution | epoch execution | train_execution | inference execution | quality execution | model size | baby's brain |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----------|-----------|-----------|-----------|-----------|-----------|---------------|
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 8 | 128 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 0.125s | 7200s | 7200s | 8s | 1/100 | 29,577,062 | fb546251-ec1c-4e00-a713-765693d8c5cf |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 8 | 128 | 0.1 | 128 | 16 | 8 | 1 | 16 | 1e-3 | 4.50s | 37355s | 37355s | 13s | 1/100 | 58,183,507 | c6832bb3-3f49-493d-9548-62d46065c1e0 |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 8 | 128 | 0.1 | 128 | 16 | 16 | 1 | 16 | 1e-3 | 0.5s | 41802s | 41802s | 14s | 1/100 | 117,188,617 | 33bd6583-1b87-4469-b55e-0ccb8fd0441c |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 16 | 128 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 0.25s | 19916s | 19916s | 14s | 1/100 | 29,561,884 | 17e84fc6-57f9-4843-a0f2-6150e7c7f169 |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 16 | 128 | 0.1 | 128 | 16 | 8 | 1 | 16 | 1e-3 | 0.25s | 60851s | 60851s | 14s | 1/100 | 56,987,898 | ecb6a3b1-ffd5-4cbd-a3e0-d9a9716dacbd |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 16 | 128 | 0.1 | 128 | 16 | 16 | 1 | 16 | 1e-3 | 1.0s | 83749s | 83749s | 26s | 25/100 | 116,160,341 | 180eeb27-b1b4-4427-9734-c70e10da2005 |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 32 | 128 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 0.5s | 53771s | 53771s | 12s | 12/100 | 28,310,070 | e64dd257-c048-441b-ad08-47275b22cc0b |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 32 | 128 | 0.1 | 128 | 16 | 8 | 1 | 16 | 1e-3 | 3.0s | 97984s | 97984s | 23s | 25/100 | 56,292,724 | 465e5804-17af-412c-8bf6-808a34cdf617 |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 32 | 128 | 0.1 | 128 | 16 | 16 | 1 | 16 | 1e-3 | 2.0s | 134234s | 134234s | 54s | 27/100 | 114,114,671 | 5f13a2ab-113a-4c2c-8abd-40384bdd8854 |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 64 | 128 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 2.00s | 137095s | 137095s | 39s | 27/100 | 28,302,412 | 0cbeae2b-2884-434d-8fdf-b8a12d8d50c4 |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 64 | 128 | 0.1 | 128 | 16 | 8 | 1 | 16 | 1e-3 | 3.0s | 237971s | 237971s | 45s | 30/100 | 56,104,284 | e65d4a59-a816-4ffa-b8ac-935db1064433 |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 64 | 128 | 0.1 | 128 | 16 | 16 | 1 | 16 | 1e-3 | 4.0s | 328598s | 328598s | 88s | 32/100 | 112,890,591 | cb632ce3-3f3b-432b-b24f-9171005f205e |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 128 | 128 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 4.5s | 320999s | pre | 320999s | 26s | 42/100 | 28,523,148 | be5bf515-5850-41de-9072-af8faca7d27a |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 128 | 128 | 0.1 | 128 | 16 | 8 | 1 | 16 | 1e-3 | 6.0s | 372273s | 372273s | 88s | 37/100 | 56,051,017 | 868be641-a21a-4c5f-8916-2dfc4c92f5e9 |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 128 | 128 | 0.1 | 128 | 16 | 16 | 1 | 16 | 1e-3 | 10.0s | 737839s | 737839s | 199s | 43/100 | 111,737,990 | 12b8b053-6c14-42aa-a957-89b809e6f785 |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 256 | 32 | 0.1 | 32 | 16 | 2 | 1 | 16 | 1e-3 | 3.00s | 228208s | 228208s | 26s | 23/100 | 1,323,911 | b3aedc6d-da9a-4398-b067-faeca1afc6da |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 256 | 64 | 0.1 | 64 | 16 | 1 | 1 | 16 | 1e-3 | 2.00s | 143777s | 143777s | 25s | 25/100 | 2,585,851 | 652d3409-24a5-4057-b482-9fd9e32fc484 |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 64 | 64 | 0.1 | 64 | 16 | 2 | 4 | 16 | 1e-3 | 0.30s | 30071s | 120286s | 5s | 38/100 | 3,884,379 | fda4ed80-b633-4e6e-a4a7-894a76528bd3 |
| v0.0.1 | cpu | shakespeare | char | pre | mha | mlp | 64 | 64 | 0.1 | 64 | 16 | 4 | 4 | 16 | 1e-3 | 0.60s | 54558s | 218235s | 9s | 27/100 | 7,367,190 | 82689609-5b39-4fd7-8a42-5d2f04dabf7a |
| v0.0.1 | cpu | shakespeare | char | pre | moh | moe | 32 | 32 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 0.60s | 218232s | 218235s | 4s | 25/100 | 6,730,419 | 7a1459eb-5876-4c20-b56a-34a779066ae0 |
| v0.0.1 | cpu | shakespeare | char | pre | moh | moe | 64 | 64 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 1.20s | 142492s | 142492s | 48s | 30/100 | 22,531,002 | 3ceb67f2-0363-439c-9f1a-69a90bf7fa48 |

*Keep in mind that quality should never be assumed without scrutiny, as its evaluation by a larger language model depends on specific criteria. Keep in mind, these models may not consistently produce the same assessment across different runs or contexts.


## 🕵️ Observations

While playing and exploring with our tiny language models, we noticed a few adorable quirks and clever behaviors—here are some of the sweet observations we made along the way.

- When training if **n_emb** is increased then the model size will also increased and total time are also increased, this follows linear analogy as any array width has size of embedding size.
- When training if **head_size** is increased then the model size will also increased and total time are also increased, there are only gamma and beta arrays into the formulas.
- When training if **n_layers** is increased then the model size will also increased and total time are also increased, depending on attention selection and network selection they will follow different formula. 
- When training if **vocab_size** is increased then the tokenizer size will also increased and total time are also increased, this follows linear analogy as any array length has size of vocabilary size.
- When inference if **infr_cache** is true then generation O(T²) faster as previously sequences do not need to be recalculated each time.
- When inference the model with x **max_tokens** for generation, then:
  - if the output type is plain text it will have x tokens.
  - if the output type is json it will have y tokens where y >= x, because it might contains special characters for example, new lines, which in json are represented as two characters "\n" --> "\", "n".