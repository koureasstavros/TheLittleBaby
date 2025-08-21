## 📖 Terminology

This section defines key terms used throughout the content, ensuring clarity and consistency. Whether you're new to the topic or just need a quick refresher, these definitions will help you navigate the material with confidence.

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

**Multi Head Attention (MHA)** – is a core component of Transformer architectures which allows the model to attend to different parts of the input sequence in parallel, using multiple attention "heads."

**Grouped Query Attention (GQA)** – it groups multiple heads to share the same key and value projections.

**Multi-Head Latent Attention (MLA)** – it compresses the key and value tensors into a lower-dimensional space before storing them in the KV cache.

**Mixture-of-Experts (MoE)** – is a modular architecture where different "expert" subnetworks are selectively activated per input token, often used to scale models efficiently.

**Mixture Head Attention (MoH)** – is reimagined as an MoE system, where heads = experts while replaces the standard summation of heads with a weighted, token-specific selection.

🧩 **Components**

**Tensor** - A tensor is just a multi-dimensional array of data—like a matrix, but more general. In deep learning frameworks like PyTorch or TensorFlow, tensors are the building blocks for storing inputs, outputs, weights, and more.

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


## 💡 Explanations

This section breaks down complex ideas into clear, digestible insights. It offers thoughtful interpretations and contextual background to help deepen your understanding of the concepts discussed. Whether you're exploring unfamiliar territory or refining your expertise, these explanations aim to illuminate the “why” behind the “what.”

**Grad**

The .grad attribute of a tensor is where the gradient is stored after backpropagation. It tells you how much a change in that tensor would affect the final output (usually the loss). Think of it as the sensitivity of the loss to that tensor.During training, we want to minimize the loss. To do that, we need to know:

- Which direction to adjust each parameter (positive or negative).

- How much to adjust it (the magnitude of the gradient).

#### Define a simple function
`y = x**2 + 3*x + 1`

Here, the gradient of y with respect to x is dy/dx = 2x + 3, so at x = 2, the gradient is 7. That’s what gets stored in x.grad.


## 🧾 References

Here you can find references that support the key concepts discussed, offering deeper insights and sources for further exploration.

**Yann Dubois**

https://www.youtube.com/watch?v=9vM4p9NN0Ts / Stanford CS229 I Machine Learning I Building Large Language Models (LLMs)

**Sebastian Raschka**

https://www.youtube.com/watch?v=79F32D9aM8U / Build LLMs From Scratch with Sebastian Raschka #52

https://www.youtube.com/watch?v=Zar2TJv-sE0 / Build an LLM from Scratch 5: Pretraining on Unlabeled Data

**Andrej Karpathy**

https://www.youtube.com/watch?v=l8pRSuU81PU / Let's reproduce GPT-2 (124M)

https://www.youtube.com/watch?v=EWvNQjAaOHw / How I use LLMs