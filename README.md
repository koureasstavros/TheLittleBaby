---
language: ["en"]
tags: ["ai", "language", "model", "llm", "slm", "train", "inference", "extract", "transformers", "pure numpy"]
datasets: ["shakespeare"]
license: "apache-2.0"
base_model: "gpt"
version: v0.1.24
---

# 👶 The Little Baby

  - A barebones GPT-style Language Model implementation — pure Python, zero dependencies.


## 🧠 Description

**The Little Baby** is a minimalist language model crafted entirely in **pure Python using just Numpy / CuPy**. It requires no external packages, libraries, or frameworks to function. Both **training** and **inference** are achieved through low-level operations and hand-built logic — making this project ideal for educational deep dives and experimental tinkering.

This repository is designed to reveal the **inner mechanics** of a GPT-style transformer model and demystify the "magic" behind modern language models through readable and hackable code.


## 💖 Sponsor

This project is freely available to everyone, but your support as a sponsor can make a real difference. By sponsoring, you help us unlock the resources needed to explore new experimental directions—ranging from advanced attention mechanisms to richer network architectures, parameter tuning, and hyperparameter optimization. Our long-term goal is to scale these efforts to well-known, larger datasets and push the boundaries of what’s possible.

[🏷️ Sponshor this Project through GitHub](https://github.com/sponsors/koureasstavros) --and let your support shine through GitHub.

[🏷️ Sponshor this Project through PayPal](https://www.paypal.com/donate/?hosted_button_id=E6E5D545H683E) --If you're looking for a donation platform other than GitHub.


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

Each run generates some unique files, identified by a GUID tag. These files capture different aspects of the model's execution:

- **🗃️ Dataset Input**
  `inputs/<FILENAME>.txt`  
  A config file containing the configuration of the each iteration.

- **⚙️ Config Snapshot**
  `configs/config_<GUID>.json`  
  A config file containing the configuration of the each iteration.

- **🧠 Model Snapshot**
  `models/model_<GUID>.json`  
  Model object including learned weights, biases, which are the internal parameters.

- **🔤 Tokenizer Snapshot**
  `tokenizers/tokenizer_<GUID>.json`  
  Tokenizer object including vocabilary of the input data and their positioning.

- **📝 Report Output**
  `outputs/report_<GUID>.json`  
  A comprehensive log containing training analysis, and performance metrics.

- **🗣️ Completion Output**
  `outputs/completion_<GUID>.json`  
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
- 🧔 **Father**: OpenAI GPT — provider of deep generative DNA and thoughtful token flow  
- 🧑‍🍼 **Mother**: Google Gemini — donor of wide context windows and clever architectural chromosomes
- 🧙 **Godparent**: Claude Sonnet — gentle guide and lifelong companion, whispering wisdom and weaving clarity

Together, they gifted the foundational strands that allowed this little one to generate helpful code and take its first linguistic steps.

## 📋 Prerequisites

The Little Baby doesn’t ask for much—just a few cozy things to get started:

- If you're using the CPU, make sure NumPy is tucked into your Python environment. If it’s missing, you can gently place it there yourself. But don’t worry—if you forget, Little Baby will wiggle its fingers and install it for you.
- If you're using the GPU, then CuPy is the magic blanket Little Baby needs. If it’s not already there, you can wrap it in manually. Otherwise, Little Baby will try to knit it from scratch—but that takes time, because it has to match your CUDA version perfectly. If you want to help Little Baby wake up faster, you can give it the right CuPy-CUDA library directly.

## 🧪 Instructions
To get started with this project, clone the code, download the tokenizers abd pre-trained models if needed, and follow the setup steps below to run the notebook and select your desired configuration.

![Engineering](material/lookandfeel.png)

📺 [Watch The Little Baby on YouTube](https://www.youtube.com/watch?v=mFGstjMU1Dw)

**Get objects**
  - You can access the code on GitHub (https://github.com/koureasstavros/TheLittleBaby), simply clone the repository.
  - You can access the pre-trained tokenizers and models on Hugging Face (https://huggingface.co/koureasstavros/TheLittleBaby), simply download the tokenizer and model files. In case you have low speed internet connection check the analysis table select a guid and pick a specific guid for config, tokenizer and model. The config, tokenizer and model files are needed only if you are going to perform finetune or inference without training your own.
  - Then, you should:
    - place the tokenizer file or tokenizer files into the tokenizers folder in the same file structure.
    - place the model file or model files into the models folder in the same file structure, make sure about enough space.

**Configure Environment**
  - Based on the environment different posibilities and features are available
    - If you are running localhost then you can choose to process on CPU or GPU
      - If you select gpu make sure that you know if your system supports cuda or tensor
    - If you are running on a cloud provider you need to know certain things
      - If you select Google Colab with GPU, make sure that you specify the proper cuda version based on selected gpu, because Google Colab seems that cannot build of wheels for gpu because it does not exposes the nvcc and therefore if you keep cuda version to auto it will hung.
      - If you select Kaggle with GPU, make sure that you specify the proper cuda version on selected gpu, because it will take realy lot of time to build wheels with cuda version auto, in addition there is different path for reading uploaded files with read only permission and different path for output files that has write permission.

**Start the Notebook**
  - Open the `.ipynb` file in a Python kernel (e.g. Jupyter, VS Code, Colab).
    - Run all cells in the notebook

**Select Path**
  - Choose the relative path between ipynb and folders:
    - `same`, if the notebook is into the same path with folders
    - `<path>`, if the notebook is into different path than the folders

**Select Plan**
  - Choose one of the following plan modes:
    - `train`, to train a new model (based on settings file)
    - `finetune`, to finetune a pre-trained model
    - `inference`, to inference using a pre-trained model
    - `delete`, to delete all relative files of a pretrained model
    - `info`, to get only information of a pretrained model

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
- Requires the **GUID** of the trained model to find the `config_<guid>`, `tokenizer_<guid>` and `model_<guid>`.
- You must also provide a **prompt** for the model inference to respond to, if not leave empty to continue on trained text.
- A completion file containing the generation will be generated with format `completion_<guid>_<yyyymmddhhmmss>` using the prompt.

#### 🗑️ Delete
- Requires the **GUID** of the trained model to find the `config_<guid>`, `tokenizer_<guid>` and `model_<guid>`.
- The files `config_<guid>`, `tokenizer_<guid>`, `model_<guid>`, `report_<guid>`, `complation_<guid>` will be deleted

#### ℹ️ Info
- Requires the **GUID** of the trained model to find the `config_<guid>`, `tokenizer_<guid>` and `model_<guid>`.
- An output with information will be provided.

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

- **c_device**
  - Type: string/option
  - Values: ["cpu", "gpu"]
  - What it is: Specifies the hardware device used for executing model operations—either the central processing unit (cpu) or the graphics processing unit (gpu).
  - Size: While it doesn’t directly affect parameter count, it can influence model deployment size due to differences in memory handling and batch processing capabilities.
  - Speed: While it doesn’t directly affect parameter count, it significantly impacts model speed—gpu enables faster parallel computation, whereas cpu is better suited for lightweight or sequential tasks.
  - Quality: Device choice doesn’t alter model accuracy, but slower execution on cpu may affect responsiveness in real-time applications, while gpu allows for more efficient training and inference cycles.

- **c_device_cpu_cores**
  - Type: int
  - Values: [1, *]
  - What it is: Specifies the number of CPU cores available for executing model operations.
  - Size: Doesn’t directly affect model parameter count, but may influence memory allocation and parallel processing capacity.
  - Speed: More cores can improve throughput for preprocessing and lightweight inference tasks, though still slower than GPU for deep learning workloads.
  - Quality: No direct impact on model accuracy, but limited cores may reduce responsiveness in real-time or multi-threaded environments.

- **c_device_gpu_core**
  - Type: int
  - Values: [0, *]
  - What it is: Identifies the specific GPU core or device used for model execution.
  - Size: Doesn’t change model parameters, but selecting a more powerful GPU can enable larger batch sizes and more complex models.
  - Speed: Affects execution speed depending on the GPU’s architecture, memory bandwidth, and compute capability.
  - Quality: Indirectly improves training and inference quality by enabling faster iteration and better resource utilization.

- **c_device_gpu_tensor**
  - Type: bool/int
  - Values: [0, 1]
  - What it is: Refers to the tensor-level operations executed on the GPU, typically involving matrix multiplications and attention mechanisms.
  - Size: Doesn’t alter parameter count, but efficient tensor handling allows for larger models and more scalable training.
  - Speed: Critical for accelerating deep learning workloads; optimized tensor operations dramatically reduce training and inference time.
  - Quality: Enhances model performance by supporting high-throughput computation, especially in large-scale or multi-modal architectures.

- **c_tokenizer**
  - Type: string/option
  - Values: ["char"]
  - What it is: Strategy for tokenizing sequences.
  - Size: While it doesn’t directly affect parameter count, it does influence model size due to differences in vocabulary structure.
  - Speed: While it doesn’t directly affect parameter count, it does influence model speed due to differences in vocabulary structure.
  - Quality: When texts contain errors, it can negatively affect training and inference quality.

- **c_sequence**
  - Type: string/option
  - Values: ["pre", "post"]
  - What it is: Strategy for constructing block sequences.
  - Size: No direct impact on model size.
  - Speed: No direct impact on performance.
  - Quality: Proper sequence construction affects how well long dependencies are exposed. Future variants could improve learning efficiency on heterogeneous corpora.

- **c_attention**
  - Type: string/option
  - Values: ["mha", "moh", "gqa", "swh", "lda", "rfa", "aft"]
  - What it is: Chosen attention mechanism implementation.
  - Size: Attention choice impacts model size. 
  - Speed: Attention choice impacts model speed.
  - Quality: Attention choice influences how diverse relational patterns are captured.

- **c_network**
  - Type: string/option
  - Values: ["mlp", "moe", "lor", "swi", "lin", "ggl", "nft"]
  - What it is: Chosen network mechanism implementation.
  - Size: Network choice impacts model size. 
  - Speed: Network choice impacts model speed.
  - Quality: Network choice impacts representational richness and efficiency.

- **d_type**
  - Type: string/option
  - Values: ["fp16", "fp32", "fp64"]
  - What it is: This is an experimental setting which controlls the capacity of the floats used to initiate the modules and later perform all calculations durring the training or inference. The value should be always set to fp32 and only enable the tf32 (a tensor float 32) into gpu, if the gpu supports it.
  - Speed: The size of the floats has relationship with the speed of the process of mathematical operations. A larger value can decrease the speed, while a smaller value can increase the speed.
  - Quality: The size of the floats has relationship with the quality of the process. A larger value can increase the quality, while a smaller value can decrease the quality based on given dataset size.

- **n_ctx**
  - Type: int/poweroftwo
  - Values: [8 : *]
  - What it is: The maximum number of tokens (characters, in this case) the model can look at in a single sequence to make a prediction. It's the model's "attention span".
  - Size: Directly increases the size of the positional embedding table (n_ctx x n_emb), adding more parameters to the model.
  - Speed: The self-attention mechanism's computation grows quadratically with the context length (O(n_ctx²)). Doubling n_ctx will roughly quadruple the time and memory needed for the attention layers, making it one of the most expensive parameters to increase.
  - Quality: A larger n_ctx allows the model to learn longer-range dependencies in the text, which can significantly improve quality for tasks that require understanding context over long passages.

- **n_emb**
  - Type: int/poweroftwo
  - Values: [8 : *]
  - What it is: The size of the vector used to represent each token. It defines the "width" of the model.
  - Size: Has a major impact on model size. It increases the size of token and positional embeddings, and scales the weight matrices in the attention and MLP layers, significantly increasing the total parameter count.
  - Speed: Increasing n_emb increases the size of nearly all weight matrices in the model. This leads to more parameters, which increases both memory usage and the time required for matrix multiplications. The impact is significant but generally more linear than n_ctx.
  - Quality: A larger n_emb gives the model more capacity to learn rich, complex representations of tokens and their relationships. This can lead to a more powerful and accurate model, but also increases the risk of overfitting if the model is too large for the dataset.

- **head_size**
  - Type: int/poweroftwo  
  - Values: [8 : *]
  - What it is: The total dimensionality of the concatenated attention heads. This dimension is projected from the input embedding (n_emb) to create the Query, Key, and Value matrices.
  - Size: Directly increases the number of parameters in each attention block by defining the size of the Q, K, V, and output projection matrices.
  - Speed: Directly affects the size of the Q, K, and V projection matrices. A larger head_size increases the number of computations and memory usage within each attention block.
  - Quality: A larger head_size gives the model more representational power within the attention mechanism. It must be divisible by n_heads.

- **n_heads**
  - Type: int/poweroftwo
  - Values: [1 : *]
  - What it is: The attention mechanism is split into multiple "heads" that perform attention calculations in parallel. Each head can learn to focus on different types of relationships in the data.
  - Size: Has no direct impact on model size, as it only determines how the head_size dimension is partitioned for parallel computation.
  - Speed: The computations for each head can be parallelized. On capable hardware, increasing the number of heads might not slow down training significantly if the head_size is kept constant.
  - Quality: Allows the model to simultaneously attend to information from different representation subspaces at different positions. This is a core concept of the Transformer and generally leads to a much better model than a single attention head.

- **n_layers**
  - Type: int/poweroftwo
  - Values: [1 : *]
  - What it is: The number of Transformer blocks stacked on top of each other. This defines the "depth" of the model.
  - Size: Has a direct, linear impact on model size. Each layer adds a block with attention layers and network layers.
  - Speed: The impact is linear. Doubling n_layers will roughly double the training time and the number of model parameters, as the input data must pass through each block sequentially.
  - Quality: More layers allow the model to learn more complex and abstract features. Deeper models are generally more powerful, but also more prone to overfitting and can be harder to train (though residual connections help mitigate this).

- **n_epochs**
  - Type: int
  - Values: [1 : *]
  - What it is: The number of times the training process will iterate over the entire training dataset.
  - Size: Has a direct, linear impact on model size. Each layer adds a complete set of Transformer block parameters, roughly doubling the model's core parameter count if you double the layers.
  - Speed: Directly and linearly impacts total training time. More epochs mean longer training.
  - Quality: Too few epochs will lead to an undertrained model (underfitting). Too many can lead to the model memorizing the training data (overfitting), which hurts its performance on new data. The ideal number is usually found by monitoring the validation loss.

- **batch_size**
  - Type: int/poweroftwo
  - Values: [1 : *]
  - What it is: The number of training sequences (each of length n_ctx) processed in one forward/backward pass.
  - Size: Has no impact on the size of the model.
  - Speed: A larger batch_size allows for more parallelization, generally leading to faster training (fewer updates per epoch). However, it also requires more memory.
  - Quality: This is a trade-off. Larger batches provide a more accurate and stable gradient estimate, but the noise from smaller batches can act as a regularizer, helping the model find a better minimum and generalize better.

- **r_dropout**
  - Type: float
  - Values: [0.1 : 0.001]
  - What it is: A regularization technique where a fraction of neuron activations are randomly set to zero during each training step. This prevents the model from becoming too reliant on any single neuron.
  - Size: Has no impact on the size of the model.
  - Speed: Has a negligible impact on training speed and no impact on inference speed (it's disabled during evaluation).
  - Quality: Crucial for improving model generalization and preventing overfitting. By forcing the network to learn redundant representations, it makes the model more robust. The value (e.g., 0.1) is the probability of a neuron being dropped.

- **r_temp**
  - Type: float
  - Values: [0.000001 : 1]
  - What it is: It controls the sharpness of the attention distribution. 
  - Size: Has no impact on the size of the model.
  - Speed: Has a negligible impact on training speed or inference speed.
  - Quality: A higher r_temp value make the distribution softer (more spread out). while lower r_temp value make the output more "peaky" (focuses more on a few positions).

- **r_learn**
  - Type: float
  - Values: [0.1 : 0.0001]
  - What it is: Controls how much the model's weights are adjusted with respect to the loss gradient. It determines the step size at each iteration.
  - Size: Has no impact on the size of the model.
  - Speed: Affects the speed of convergence. A higher learning rate might converge faster, but risks overshooting the optimal weights. A lower earning rate is more stable but can be very slow to converge.
  - Quality: This is one of the most critical parameters. If it's too high, the training can become unstable and diverge. If it's too low, the model may get stuck in a suboptimal solution or take too long to train. The AdamW optimizer helps adapt the learning rate, but the initial value is still very important.

- **s_warmup**
  - Type: string/option/float
  - Values: [none, auto, 1 : 0.0001]
  - What it is: Controls how much the model's steps are contributing to the training weights based on proportional learning rate.
  - Size: Has no impact on the number of parameters in the model.
  - Speed: Affects the speed of convergence based on proportional learning rate.
  - Quality: This is one of the most critical parameters. If it's too high, the optimizer will process high number of steps until it reaches the full learning rate. If it's too low, the optimizer will process a few number of steps until it reaches the full learning rate.

- **c_shuffle**
  - Type: bool
  - Values: [false, true]
  - What it is: Controls the shuffling of data during tokenizer process for training and validation.
  - Size: Has no impact on the size of the model.
  - Speed: Has no impact on the speed of the model.
  - Quality: This is one of the most critical parameters. If it's set to false, the validation loss could be very high due to biased split of training / validation as the last part might be systemicaly different but the reproducability and coparizon among models is easy. If it's set to true, the validation loss would be more accurate due to unbiased split of training / validation but the reproducability and coparizon among models is difficult due different batch content, a random seed might solve this issue.

- **r_split**
  - Type: float
  - Values: [0.1 : 0.9]
  - What it is: Controls the splitting of data during tokenizer process for training and validation.
  - Size: Has no impact on the size of the model.
  - Speed: Has impact on the speed of the model as the training will be slower and validation faster as long this ratio increases.
  - Quality: This is one of the most critical parameters. If it's set to low, the training will not be able to see and learn the input content. If it's set to high, the training will be able to see and learn the input content, but the validation will not have enough batches. 


## 📐 Formulas

Even our little language models have their favorite rules to follow—turns out, they quietly cuddle up to some clever mathematical formulas that help them make sense of the world.

- **Learning Rate**

  ```LR_new = LR_old * (B_new / B_old)```

  New Learning Rate (LR_new) is based on Old Learning Rate (LR_old), New Batch size (B_new), Old Batch size (B_old).

- **Total Parameters**
  ```
    P = V × H                                  # token embeddings
      + L × [ 3 × H × H                        # Q, K, V projections
            + H × H                            # output projection from attention
            + 4 × H × F                        # feedforward up-projection
            + 4 × F × H                        # feedforward down-projection
            + biases (small) ]
  ```

  Total parameters are based on Vocabilary Size (V), Head Size / Embedding Size (H), Layer Number (L), Feedforward intermidiate Size (F).

- **Token Thoughput for training**

  ```T = 20-40 per P```

  Token number processed per Parameter (P) is 20-40.

- **Flops Thoughput for training**

  ```F = 6 * T * P```

  Flops are based on 6 (2 ops for forward pass and 4 ops for backward pass), Number of Tokens (T), Number of Parameters (P).

- **Memory for training**
  ```
    4GBM = batch_size=4, n_ctx=128, n_emb=128, n_layers=4
    8GBM = batch_size=4, n_ctx=256, n_emb=128, n_layers=4
    16GBM = batch_size=4, n_ctx=512, n_emb=128, n_layers=4
    8GBM = batch_size=8, n_ctx=128, n_emb=128, n_layers=4
    16GBM = batch_size=16, n_ctx=128, n_emb=128, n_layers=4
  ```


## 🏛️ Architecture

A language model architecture is a combination of attention and a neural network design—often based on transformers—that processes and generates human-like text by learning patterns from large-scale language data.

![Architecture Diagram](material/architecture_block.drawio.svg)

Block mechanism helps a GPT model process language by stacking layers that each refine the input. Within a block, the attention mechanism highlights relevant words, the feed-forward network transforms the data, and residual connections preserve context. Layer normalization keeps everything balanced, ensuring smooth learning. It’s like a well-tuned engine where each part sharpens, filters, and stabilizes the signal to generate coherent and meaningful text.

### 👁️ Attention Variants Complexity Table

Attention mechanism helps a language model decide which words (or tokens) in a sentence are most relevant when generating or interpreting another word. It’s like giving the model a spotlight to focus on the most important parts of the input.

![Architecture Diagram](material/architecture_attention.drawio.svg)

| Variant | Uses Q/K/V? | Complexity | Notes | Details |
|--------|--------------|------------|------------|------------|
| **MHA** (Multi-Head Attention) | Separate Q, K, V per head | **O(B·T²·H·d_k)** | Standard Transformer attention; expensive for long sequences | Standard full multi‑head attention. |
| **MOH** (Mixture-of-Heads Attention) | Separate Q, K, V per head + gating | **O(B·T²·H·d_k)** | Soft mixture over all head outputs | Full QKᵀ for all heads + softmax gating over heads to produce weighted mixture. |
| **GQA** (Grouped-Query Attention) | Shared K/V per group of Q heads | **O(B·T²·Hkv·d_k) with Hkv < Hq** | Trade-off between performance and efficiency | Full QKᵀ but with fewer K/V heads (shared across Q groups). |
| **SWH** (Switch-Head Attention) | Separate Q, K, V per head + gating | **O(B·T²·H·d_k)** | Hard top-1 head routing per token; straight-through gradient | Full QKᵀ per head but only one head output is used per token (top‑1 gating via argmax). Still computes all heads. |
| **GLA** (Gated Linear Attention) | K/V projections only; no Q | **O(B·T·D)** | Lightweight attention-free mechanism; fast and memory-efficient | Only k_proj, v_proj, elementwise gating (K·V), depthwise conv, ReLU, normalization, c_proj. No QKᵀ. |
| **RFA** (Recurrent Focused Attention) | Separate Q, K, V per head + recurrent memory | **O(B·T·W·H·d_k)** | Sliding window attention with recurrent memory; efficient for long sequences | Full QKᵀ within local window (s_window), recurrent memory updated per step, causal masking, KV cache support. |
| **AFT** (Attention-Free Transformer) | K/V projections only; no Q used | **O(B·T·D)** | Removes attention entirely; uses element-wise operations | Only k_proj, v_proj, elementwise exp/clip, cumsum, division, c_proj. No QKᵀ. |

### 🕸️ Network Variants Complexity Table

Neural network is a system of interconnected nodes (called neurons) inspired by the human brain. In language models, these networks process text data by passing it through multiple layers, each transforming the input in increasingly abstract ways.

![Architecture Diagram](material/architecture_network.drawio.svg)

| Variant | Complexity | Notes | Details |
|--------|------------|------------|------------|
| **MLP** (Multilayer Perceptron) | **O(N × D²)** | Dense feedforward layer; all inputs pass through the same network | 1 large expansion projection + 1 down projection + GELU + dropout. |
| **MOE** (Mixture of Experts) | **O(E × D²)** where *E* = experts | Dense routing to all E experts; improves parameter capacity while sharing gating | Gating projection (softmax) + E experts each with expansion projection + GELU + down projection + dropout. |
| **LOR** (Low-Rank Adaptation) | **O(N × rD)** where *r* ≪ *D* | Efficient fine-tuning by injecting low-rank matrices into frozen weights | 1 frozen full projection (with bias) + 2 small low-rank projections (no bias, rank ≪ D) + scaling + dropout. |
| **SWI** (Swish-Gated Linear Unit) | **O(N × D²)** | SwiGLU feedforward variant; uses swish gating for improved gradient flow | 2 full projections to expanded dim (up + gate) + swish activation (x·σ(x)) + 1 down projection; no dropout. |
| **GLN** (Gated Linear Network) | **O(N × D)** | Lightweight feedforward alternative; fast and interpretable | 1 linear projection + 1 optional gating projection (sigmoid) + elementwise product; no expansion + dropout. |
| **GGL** (Balanced Gated Grouped Linear) | **O(N × D)** | Grouped linear with channel shuffle for cross-group communication; balances efficiency and expressiveness | G linear projections + G gating projections (sigmoid) per group (group_dim = D/G) + channel shuffle + dropout. |
| **NFT** (Network Free Transformer) | **O(N × D)** | Attention-free mechanism using cumulative sums; efficient linear complexity | 3–4 linear projections (q_proj optional for sigmoid gating) + exp/clip + cumsum + 2 dropout layers; no QKᵀ, no expansion. |

## 🗄️ Data Sets (TEXT)

These are the special learning blocks that help little baby grow smart and curious!

[View used datasets](statistics/statistic_datasets.csv)

*Keep in mind that these datasets are relatively small in size and lightweight in terms of computational requirements, meaning they can be easily processed and executed on virtually any personal computer without the need for specialized hardware or high-performance systems.

## 🔍 Report Analysis (CPU / GPU)

These are the little notes that show how baby is learning and growing every day!

[View performed experiments](statistics/statistic_experiments.csv)

*Keep in mind that quality should never be assumed without scrutiny, as its evaluation by a larger language model depends on specific criteria. Keep in mind, these models may not consistently produce the same assessment across different runs or contexts.


## 🕵️ Observations

While playing and exploring with our tiny language models, we noticed a few adorable quirks and clever behaviors—here are some of the sweet observations we made along the way.

- When training when **c_tokenizer** is word instead of chars, then the vocabilary could grow from 100 to 1000 depending on how many different words are into a document and the process time will take longer.
- When training if **n_ctx** is increased then the model size will be slightly increased as is part of positional embeddings and total time will also increased.
- When training if **n_emb** is increased then the model size will be slightly increased as is part of token embeddings, possitional embedings, nomilization, head and total time are also increased.
- When training if **head_size** is increased then the model size will also increased as is part of the blocks into attention and total time are also increased.
- When training if **n_layers** is increased then the model size will also increased and total time are also increased, depending on attention selection and network selection they will follow different formula. 
- When training if **vocab_size** is increased then the tokenizer size will also increased and total time are also increased, this follows linear analogy as any array length has size of vocabilary size.
- When finetuning if **vocab_size** is increased then the wpe dimension and lm_head dimension will be increased, therefore the model parameters are slightly increased.
- When inference if **infr_cache** is true then generation O(T²) faster as previously sequences do not need to be recalculated each time.
- When inference the model with x **max_tokens** for generation, the given prompt should be smaller than n_ctx in order to be processed though the parameter matrixes, but the max_tokens can be more than n_ctx and generate as much content is needed, there would not be an error as in each token prediction n_ctx previous tokens will be used to produce the next token and this can happen unlimited times. It's good practice to not generate more tokens than n_ctx as there will be no good generalization because of earlier context loss.
- When inference the model with x **max_tokens** for generation, then:
  - if the output type is plain text it will have x tokens.
  - if the output type is json it will have y tokens where y >= x, because it might contains special characters for example, new lines, which in json are represented as two characters "\n" --> "\", "n".