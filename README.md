---
language: ["en"]
tags: ["ai", "language", "model", "llm", "slm", "train", "inference", "extract", "transformers", "pure numpy"]
datasets: ["shakespeare"]
license: "apache-2.0"
base_model: "gpt"
new_version: v0.0.3
---

# 👶 The Little Baby

> A barebones GPT-style LLM implementation — pure Python, zero dependencies.


## 🧠 Description

**The Little Baby** is a minimalist language model (LLM) crafted entirely in **pure Python using just Numpy**. It requires no external packages, libraries, or frameworks to function. Both **training** and **inference** are achieved through low-level operations and hand-built logic — making this project ideal for educational deep dives and experimental tinkering.

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

> “Build it, break it, learn from it.” – The Baby Philosophy


## 🚀 Project Goals

This endeavor is structured around key targets designed to deliver meaningful outcomes:

- ✅ Build a GPT-like model using **only Python + NumPy-like constructs**.
- ✅ Support training from scratch on plain text files.
- ✅ Provide clear code for attention mechanisms, tokenization, and backprop.
- ✅ Encourage experimentation and modification.


## 📚 Directory Files

Each run generates three unique files, identified by a GUID tag. These files capture different aspects of the model's execution:

- **📝 Report**  
  `outputs/report_<GUID>.txt`  
  A comprehensive log containing configuration settings, training analysis, and performance metrics.

- **🗣️ Completion Output**  
  `outputs/completion_<GUID>.txt`  
  The raw generated text from the model's inference — your baby’s words in print!

- **🧠 Model Snapshot**  
  `models/model_<GUID>.pkl`  
  Serialized model object including class definitions, learned weights, biases, and internal parameters.


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


## 🔮 What to expect

After ~24 hours of training on a single document of multiple Shakespeare works using a **laptop CPU**, The Little Baby learns to babble. Its speech is primitive and childlike — just enough to make you smile and realize… the baby is alive. 👼

While its capabilities are minimal, its structure is maximal in transparency. Every token, gradient, and parameter is visible and malleable. 🧩


## 🧪 Instructions

You can access the code on GitHub (https://github.com/koureasstavros/TheLittleBaby), simply clone the repository.
You can access the pre-trained models on Hugging Face (https://huggingface.co/koureasstavros/TheLittleBaby), simply download the model files, or a specific guid model file and place them / it into the models folder.

1. **Start the Notebook**
   - Open the `.ipynb` file in a Python kernel (e.g. Jupyter, VS Code, Colab).

2. **Select Path**
   - Choose the relative path between ipynb and folders:
      - `same`
      - `<path>`

3. **Select Plan**
   - Choose one of the following plan modes:
     - `train`
     - `inference`
     - `extract`

#### 🔧 Train
- Begins training using parameters defined in earlier Python blocks.
- A model file and the individual files will will be generated with format model_<guid>.
- After training completes, performs **inference with an empty prompt** using the trained model.
- A completion file will be generated with format complation_<guid>

#### 💬 Inference
- Requires the **GUID** of the trained model to find the model_<guid>.
- You must also provide a **prompt** for the model inference to respond to.
- A completion file will be generated with format complation_<guid>_<yyyymmddhhmmss>

#### 📦 Extract
- Requires the **GUID** of the trained model  to find the model_<guid>.
- Extracts components from the `model_<guid>s` object file into `model_<guid>` weight file.
- A model file and the individual files will be generated with format model_<guid>_<type>

*Keep in mind that if you're running a process in VSCode and your workstation, PC, or laptop enters hibernation, the process will resume automatically once the device is powered back on.


## 🍼 Cry. Babble. Speak. Repeat.

Here come the smartest little settings to help the model learn and grow big and strong from this data:

- **Age 3 Months** - 33bd6583-1b87-4469-b55e-0ccb8fd0441c - Coos and gurgles begin. Sound, not speech—yet something’s brewing.
- **Age 6 Months** - 180eeb27-b1b4-4427-9734-c70e10da2005 - Loud, random cries. It’s not talking, but it's definitely expressive.
- **Age 12 Months** - 5f13a2ab-113a-4c2c-8abd-40384bdd8854 - Joyful noise with hints of intention. Real words still warming up.
- **Age 24 Months** - cb632ce3-3f3b-432b-b24f-9171005f205e - Words arrive —Chaotic, quirky, delightful. Syntax? Optional.
- **Age 48 Months** - 12b8b053-6c14-42aa-a957-89b809e6f785 - Mini Philosopher Mode -Stories, opinions, even jokes. Communication unlocked.hear them.

*Keep in mind that these are pre-trained model executions available for inference. You can bypass the training phase by simply downloading the models and using them directly.


## 🔍 Report Analysis
Given the Shakespeare works into a single document of 32777 paragraphs, 12519 sentences, 202651 words, 1075394 characters / tokens for learning and 500 characters / tokens for inference

| n_ctx | n_emb | dropout | head_size | n_heads | n_layers | n_epochs | s_batch | lr | batch execution | epoch execution | inference execution | quality execution | baby's brain |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----------|-----------|-----------|-----------|---------------|
| 8 | 128 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 0.125s | 7200s | 8s | 1/100 | fb546251-ec1c-4e00-a713-765693d8c5cf |
| 8 | 128 | 0.1 | 128 | 16 | 8 | 1 | 16 | 1e-3 | 4.5.0s | 37355s | 13s | 1/100 | c6832bb3-3f49-493d-9548-62d46065c1e0 |
| 8 | 128 | 0.1 | 128 | 16 | 16 | 1 | 16 | 1e-3 | 0.5s | 41802s | 14s | 1/100 | 33bd6583-1b87-4469-b55e-0ccb8fd0441c |
| 16 | 128 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 0.25s | 19916s | 14s | 1/100 | 17e84fc6-57f9-4843-a0f2-6150e7c7f169 |
| 16 | 128 | 0.1 | 128 | 16 | 8 | 1 | 16 | 1e-3 | 0.25s | 60851s | 14s | 1/100 | ecb6a3b1-ffd5-4cbd-a3e0-d9a9716dacbd |
| 16 | 128 | 0.1 | 128 | 16 | 16 | 1 | 16 | 1e-3 | 1.0s | 83749s | 26s | 25/100 | 180eeb27-b1b4-4427-9734-c70e10da2005 |
| 32 | 128 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 0.5s | 53771s | 12s | 12/100 | e64dd257-c048-441b-ad08-47275b22cc0b |
| 32 | 128 | 0.1 | 128 | 16 | 8 | 1 | 16 | 1e-3 | 3.0s | 97984s | 23s | 25/100 | 465e5804-17af-412c-8bf6-808a34cdf617 |
| 32 | 128 | 0.1 | 128 | 16 | 16 | 1 | 16 | 1e-3 | 2.0s | 134234s | 54s | 27/100 | 5f13a2ab-113a-4c2c-8abd-40384bdd8854 |
| 64 | 128 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 2.00s | 137095s | 39s | 27/100 | 0cbeae2b-2884-434d-8fdf-b8a12d8d50c4 |
| 64 | 128 | 0.1 | 128 | 16 | 8 | 1 | 16 | 1e-3 | s | s | s | |  |
| 64 | 128 | 0.1 | 128 | 16 | 16 | 1 | 16 | 1e-3 | 4.0s | 328598s | 88s | 32/100 | cb632ce3-3f3b-432b-b24f-9171005f205e |
| 128 | 128 | 0.1 | 128 | 16 | 4 | 1 | 16 | 1e-3 | 4.5s | 320999s | 26s | 42/100 | be5bf515-5850-41de-9072-af8faca7d27a |
| 128 | 128 | 0.1 | 128 | 16 | 8 | 1 | 16 | 1e-3 | s | s | s | |  |
| 128 | 128 | 0.1 | 128 | 16 | 16 | 1 | 16 | 1e-3 | 10.0s | 763757s | 199s | 43/100 | 12b8b053-6c14-42aa-a957-89b809e6f785 |
| 256 | 32 | 0.1 | 32 | 16 | 2 | 1 | 16 | 1e-3 | 3.00s | 228208s | 26s | 23/100 | b3aedc6d-da9a-4398-b067-faeca1afc6da |
| 256 | 64 | 0.1 | 64 | 16 | 2 | 1 | 16 | 1e-3 | 2.00s | 143777s | 25s | 25/100 | 652d3409-24a5-4057-b482-9fd9e32fc484 |

*Keep in mind that quality should never be assumed without scrutiny, as its evaluation by a larger language model depends on specific criteria. Keep in mind, these models may not consistently produce the same assessment across different runs or contexts.


## 📐 Mathematic Formulas

**Learning Rate** - `LR_new = LR_old * (B_new / B_old)`

New Learning Rate (LR_new) is based on Old Learning Rate (LR_old), New Batch size (B_new),Old Batch size (B_new).

**Total Parameters** - `P = V x H + L x [4 x H^2 + 4 x H x F]`

Total parameters are based on Vocabilary Size (V), Head Size / Embedding Size (H), Layer Number (L), Feedforward intermidiate Size (F).

**Token Thoughput for training** - `T = 20-40 per P`

Token number processed per Parameter (P) is 20-40.

**Flops Thoughput for training** - `F = 6 * T * P`

Flops are based on 6 (2 ops for forward pass and 4 ops for backward pass), Number of Tokens (T), Number of Parameters (P).


## 🕵️ Observations

When inference the model with x max tokens for generation, then
 - if the output type is plain text it will have x tokens
 - if the output type is json it will have y tokens where y >= x, because it might contains special characters for example, new lines, which in json are represented as two characters "\n" --> "\", "n"


## Further Thoughts

🧠 "Let’s imagine what shiny new toys and big upgrades the little model needs to turn into a grown-up LLM who knows all about the big wide world!

 **Known DataSets**

| DataSet Type | DataSet Type | DataSet Name | DataSet Tokens |
|-----|-----|-----|-----|
| open | train | SlimPajama | 627B |
| open | train | RedPajama v1 | 1T |
| open | train | RedPajama v2 | 30T |
| open | eval | HellaSwag | 30T |


**Known Models**

| Model | Type | Parameters | Training Model Tokens | Training Model Flops | Training Environment | Training Environment Flops /s | Training Content | Training Duration |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| GPT2 | s | 117M | 3.3B | 2.3e18F | 1-2 x A100 | 100P | WebText (Reddit outbound links with ≥3 karma; ~40GB of filtered internet text) | 60D |
| GPT2 | m | 335M | 3.3B | 7e18F | 4-8 × A100 | 200P | Same as Small; byte-level BPE tokenization, 50,257 vocab size | 60Days |
| GPT2 | l | 774B | 3.3B | 15e18F | 8-16 × V100 | 400P | Same as Small; trained with causal LM objective | 60Days |
| GPT3 | s | 125M | 300B | 2.25e21F | 1-2 × A100 | 100P | Common Crawl (filtered), WebText2, Books1/2, Wikipedia (~570GB filtered) | 180D |
| GPT3 | m | 350M | 300B | 6.3e21F | 8-16 × A100 | 200P | Same as Small; scaled architecture with 24 layers and 16 attention heads | 180D |
| GPT3 | l | 760M | 300B | 3.7e21F | 100-200 × A100 | 400P | Same as Small; deeper model with wider layers and more attention heads | 180D |
| GPT4 | s | 1B | 6B | 1.8e21F | 100-200 × A100 | 1OOP | Filtered Common Crawl, Books, Wikipedia, WebText2, code, academic papers | 160D |
| GPT4 | m | 13B | 1.7T | 9.4e23F | 400-600 × A100 | 400P | Same as Small; with broader multilingual and multimodal data | 160D |
| GPT4 | l | 65B | 13T | 3e25F | 2k-4K × A100 | 1E | Massive curated dataset: text, code, images, audio (for GPT-4o), RLHF tuning | 90D  |
| LLAMA2 | s | 7B | 2T | 1.5e24F | 32-64 × A100 | 400P | Publicly available web data (filtered), books, code, academic papers | 180D |
| LLAMA2 | m | 13B | 2T | 2.6e24F | 128-256 × A100 | 400P | Same as Small; with additional curated datasets for scaling | 180D |
| LLAMA2 | l | 70B | 2T | 14e24F | 1024K+ x A100 | 800P | Same as Small; plus enhanced filtering, grouped-query attention optimization | 180D |
| LLAMA3 | s | 8B | 15T | 7.2e24F | 64-128 x A100 | 700P | Books, Wikipedia, GitHub, StackExchange | 70D |
| LLAMA3 | m | 70B | 15T | 63e24F | 512-1024 x A100 | 800P | Books, Wikipedia, GitHub, StackExchange | 70D |
| LLAMA3 | l | 405B | 15T | 365e24F | 1024+ x A100 | 1E | Books, Wikipedia, GitHub, StackExchange | 70D |
| LLAMA4 | s | 109B total / 17B active | ~30T | ~8e25F | 32-64 x H100 |	~400T |	Text, image, video (multimodal) |	Unknown |
| LLAMA4 | m | 400B total / 17B active | ~30T | ~38e25F | 128-256 × H100 | ~3200T | Text, image, code, multilingual data | Unknown |
| LLAMA4 | l | ~2T total / 288B active | ~30T | ~100e25F | 32K+ x H100 | Unknown | STEM-heavy, multimodal, synthetic distill. | Still training |

*Do not try to relate Training Model Flops, Training Environment Training Environment Flops, Training Duration as there are other factors which are playing role, like: number of epochs, number of precision parallel efficiency, memory bandwidth, thermal limitations, etc.


## 📖 Terminology

🧠 **Core Concepts**

**Transformer** – The backbone of most LLMs. It processes input all at once (not word-by-word) using a technique called self-attention, which helps the model understand relationships between words.

**Parameters** – The internal settings (weights) that the model learns during training. More parameters equaks more learning capacity.

**Embedding** – A way to turn words into numbers. These numbers (vectors) capture meaning, so similar words have similar embeddings.

🧮 **Model Architecture**

**Layer** – A building block of the model which transforms the input data and passes it to the next. LLMs have many layers stacked together.

**Embedding Layer** – Converts tokens into vectors.

**Attention Layer** – Applies self-attention to understand relationships.

**Feed-Forward Layer** – Adds complexity and depth to the model’s understanding.

**Head** – A sub-unit inside an attention layer. Each head focuses on different aspects of the input (e.g., grammar, relationships, facts).

**Multi-Head Attention** – Uses multiple heads in parallel to capture diverse patterns in the data4.

🔁 **Training Process**

**Training** – The process of teaching the model by showing it lots of text and adjusting its parameters to reduce errors. It involves feeding data, calculating predictions, comparing them to actual results, and updating weights.

**Epoch** – One full pass through the training data. Usually repeated many times to help the model learn better.

**Batch** – A small group of training examples processed together. This makes training faster and more efficient.

**Iteration** – One update to the model’s parameters. If you have 10,000 samples and a batch size of 100, you’ll do 100 iterations per epoch.

**Gradient Descent** – The method used to adjust parameters during training. It helps the model get better by reducing errors step-by-step.

**Loss Function** – A mathematical formula that measures how far off the model’s predictions are from the correct answers. The goal is to minimize this loss during training.

🧪 **Inference Process**

**Inference** – When the model uses what it learned to generate answers. This is what happens when you chat with it.

**Zero-shot Learning** – The model solves tasks it hasn’t seen before, using general knowledge.

**Few-shot Learning** – The model is given a few examples before solving a task.

**Hallucination** – When the model makes up facts or gives incorrect information confidently.

📊 **Evaluation**

**MMLU** (Massive Multitask Language Understanding) – A benchmark that tests how well a model performs across 57 subjects (like math, law, and history). Scores range from 0 to 100.

**GLUE** (General Language Understanding Evaluation) – A set of tasks used to measure how well a model understands language. Includes things like sentiment analysis and question answering.

📈 **Performance**

**FLOPs** (Floating Point Operations) – A measure of how much computing power is needed. More FLOPs = more expensive and slower processing. GPT-3 uses ~350 billion FLOPs per token.

**Latency** – How long it takes for the model to respond. Lower latency = faster answers.



## 🧾 References

**Yann Dubois**

https://www.youtube.com/watch?v=9vM4p9NN0Ts / Stanford CS229 I Machine Learning I Building Large Language Models (LLMs)

**Sebastian Raschka**

https://www.youtube.com/watch?v=79F32D9aM8U / Build LLMs From Scratch with Sebastian Raschka #52

https://www.youtube.com/watch?v=Zar2TJv-sE0 / Build an LLM from Scratch 5: Pretraining on Unlabeled Data

**Andrej Karpathy**

https://www.youtube.com/watch?v=l8pRSuU81PU / Let's reproduce GPT-2 (124M)

https://www.youtube.com/watch?v=EWvNQjAaOHw / How I use LLMs