## 🔨 Implementation

This section defines key terms used throughout the content, ensuring clarity and consistency. Whether you're new to the topic or just need a quick refresher, these definitions will help you navigate the material with confidence.

### GPT

The GPT module is the main model definition that orchestrates the entire architecture. It initializes the computation environment (supporting both CPU via NumPy and GPU via CuPy), manages token and positional embeddings, stacks transformer blocks, and handles the final layer normalization and language modeling head. The module provides complete training and inference pipelines, including forward and backward passes, text generation with optional KV-caching, and serialization methods for saving and loading model weights.

### Block

The Block module represents a single Transformer block that serves as the fundamental building unit of the model. Each block contains layer normalization, an attention mechanism, and a feed-forward network with residual connections. It supports both pre-normalization and post-normalization sequences, allowing flexibility in the architectural design while maintaining gradient flow through skip connections.

### Optimizer

The Optimizer module acts as a factory for creating optimization algorithms used during training. It currently supports variant optimizers that incorporates different capabilities. The factory pattern allows easy switching between optimizers through configuration.

### Attention

The Attention module serves as a factory for instantiating different attention mechanisms. It supports a variety of implementations. This modular design enables experimentation with different attention paradigms.

### Network

The Network module is a factory for creating feed-forward network components within transformer blocks. It offers multiple architectures including standard. Each variant provides different trade-offs between computational efficiency, expressiveness, and memory characteristics.

### Training Flow

During training, multiple text sequences are grouped together into a batch. Each sequence in the batch passes through the forward pass independently, producing its own loss value. When computing gradients during the backward pass, these individual gradients are averaged across all sequences in the batch. The optimizer then uses this averaged gradient to update the model parameters. This batching approach provides two key benefits: it stabilizes training by reducing gradient variance through averaging, and it improves computational efficiency by leveraging parallel processing capabilities of modern hardware.

### Inference Flow

During inference, the model generates text autoregressively, producing one token at a time. Starting from an initial prompt, the model performs a forward pass to compute logits for the next token position. These logits are converted to probabilities via softmax, and a token is sampled from this distribution. The newly generated token is appended to the sequence, and the process repeats until the desired number of tokens is reached. To improve efficiency, the model supports KV-caching, which stores the key and value projections from previous positions so they don't need to be recomputed at each step. Unlike training, inference runs in evaluation mode with dropout disabled and requires no backward pass or gradient computation.