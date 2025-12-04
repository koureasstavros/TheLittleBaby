#########################
# GPT Model Definition
# Author: Koureas Stavros
#########################

import os
import gc
import time as tm
from src.block import Block
from src.module import Module
from src.tokenizer import Tokenizer
from src.optimizer import Optimizer
from src.layers.linear import Linear
from src.layers.embedding import Embedding
from src.layers.normalization import Normalization
from src.functions.runtime import is_debug, pt_debug
from src.functions.helper import load_package, get_cpu_properties, get_gpu_properties
from src.functions.process import softmax, value_and_grad, value_and_nograd

class GPT(Module):
    """
    A minimal GPT (Generative Pre-trained tokenizer) model.
    """
    def __init__(self, config):
        super().__init__()

        gc.collect()

        device = config["c_device"]
        if device == "cpu":
            # Use NumPy
            print(f"{'-'*10} {'Using CPU (with NumPy)'} {'-'*10}" )
            load_package(["numpy"])
            system_cores = os.cpu_count()
            print(f"Detected CPU cores: {system_cores}")
            selected_cores = config["c_device_cpu_cores"]
            print(f"Configured CPU cores: {selected_cores}")
            cpu_cores = min(int(selected_cores), int(system_cores))
            print(f"Chosen CPU cores: {cpu_cores}")
            if "OMP_NUM_THREADS" not in os.environ:
                os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
            if "MKL_NUM_THREADS" not in os.environ:
                os.environ["MKL_NUM_THREADS"] = str(cpu_cores)
            if "OPENBLAS_NUM_THREADS" not in os.environ:
                os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_cores)
            if "NUMEXPR_NUM_THREADS" not in os.environ:
                os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_cores)
            # Load package after environment variables are set
            import numpy as mp
            device_name = get_cpu_properties(mp)
            device_cores = str(cpu_cores)
            library_version = str(mp.__version__)
            library_driver = "default"
            print(f"Processor: {device_name}")

        elif device == "gpu":
            # Use CuPy
            print(f"{'-'*10} {'Using GPU (with CuPy)'} {'-'*10}" )
            if config["c_device_gpu_cuda"] == "auto":
                load_package(["cupy"])
            elif config["c_device_gpu_cuda"] == "cuda11":
                load_package(["cupy-cuda11x"])
            elif config["c_device_gpu_cuda"] == "cuda12":
                load_package(["cupy-cuda12x"])
            elif config["c_device_gpu_cuda"] == "cuda13":
                load_package(["cupy-cuda13x"])
            os.environ["CUPY_TF32"] = config["c_device_gpu_tensor"] #[0=FP32, 1=TF32]
            os.environ["CUPY_ACCELERATORS"] = "cub,cutensor,cutensornet"
            # Load package after environment variables are set
            import cupy as mp
            gpu_cores = mp.cuda.runtime.getDeviceCount()
            print(f"Detected GPU cores: {gpu_cores}")
            for i in range(mp.cuda.runtime.getDeviceCount()):
                props = mp.cuda.runtime.getDeviceProperties(i)
                print(f"Device {i}: {props['name'].decode('utf-8')}")
            selected_core = config["c_device_gpu_core"]
            print(f"Configured GPU core id: {selected_core}")
            device_name = get_gpu_properties(mp, selected_core)
            device_cores = "all"
            library_version = str(mp.__version__)
            library_driver = str(mp.cuda.runtime.runtimeGetVersion())
            mp.cuda.Device(int(selected_core)).use()
            mp.get_default_memory_pool().free_all_blocks()
            mp.cuda.Device().synchronize()

        else:
            raise ValueError(f"Unsupported device type: {device}. Supported types are 'cpu' and 'gpu'.")
    
        print(f"{'-'*10} {'Using Configuration'} {'-'*10}")
        for key, value in config.items():
            if key != "runtime":
                print(f"{key}: {value}")

        # Set computation library
        self.mp = mp

        # Store device info
        self.device_name = device_name
        self.device_cores = device_cores

        # Store library info
        self.library_version = library_version
        self.library_driver = library_driver

        # Initialize attributes
        self.config_dict = None
        self.report_dict = None
        self.tokenizer_dict = None
        self.completion_dict = None

        # Configuration parameters
        self.c_tokenizer = config["c_tokenizer"]
        self.c_sequence = config["c_sequence"]
        self.c_attention = config["c_attention"]
        self.c_network = config["c_network"]
        self.c_optimizer = config["c_optimizer"]

        # Model architecture parameters
        self.config_dict = config

        match config["d_type"]:
            case "fp64":
                self.d_type = mp.float64
            case "fp32":
                self.d_type = mp.float32
            case "fp16":
                self.d_type = mp.float16

        self.n_ctx = config["n_ctx"]
        self.n_emb = config["n_emb"]
        self.s_head = config["s_head"]
        self.n_heads = config["n_heads"]
        self.r_dropout = config["r_dropout"]
        self.r_temp = max(1e-6, float(config["r_temp"]))
        self.n_layers = config["n_layers"]

        # Model components
        self.vocab_size = 1 # Placeholder for vocabulary size, should updated
        self.tokenizer = Tokenizer(self.mp, self.c_tokenizer)     # Tokenizer
        self.wte = Embedding(self.mp, self.d_type, self.vocab_size, self.n_emb)                              # Token embeddings
        self.wpe = Embedding(self.mp, self.d_type, self.n_ctx, self.n_emb)                                   # Positional embeddings
        self.blocks = [Block(self.mp, self.c_sequence, self.c_attention, self.c_network, self.d_type, self.n_emb, self.n_ctx, self.r_dropout, self.r_temp, self.s_head, self.n_heads)
                    for _ in range(self.n_layers)]                                              # Stack of tokenizer blocks
        self.ln_f = Normalization(self.mp, self.d_type, self.n_emb)                                          # Final Layer Normalization
        self.lm_head = Linear(self.mp, self.d_type, self.n_emb, self.vocab_size, bias=True)                  # Language modeling head (output logits)
        
    def parameters(self):
        """Returns all parameters of the GPT model."""
        params = []
        params += self.wte.parameters()
        params += self.wpe.parameters()
        for block in self.blocks:
            params += block.parameters()
        params += self.ln_f.parameters()
        params += self.lm_head.parameters()
        return params

    def flops(self, batch_size, training):
        """
        Estimate total FLOPs for the GPT model forward (and backward if training=True).
        """
        flops = 0

        # Embedding lookups (approximate as 0 FLOPs or 1 multiply-add per element)
        flops += batch_size * self.n_ctx * self.n_emb  # token embeddings
        flops += batch_size * self.n_ctx * self.n_emb  # positional embeddings

        # Transformer blocks
        for block in self.blocks:
            flops += block.flops(batch_size, training)

        # Final normalization
        norm_flops = 4 * batch_size * self.n_ctx * self.n_emb
        flops += norm_flops

        # LM head projection
        flops += 2 * batch_size * self.n_ctx * self.n_emb * self.vocab_size

        return flops
    
    def set(self, mode=True):
        """Sets the GPT model and all its sub-modules to training/eval mode."""
        super().set(mode) # Call base Module train to set self.set
        self.wte.set(mode)
        self.wpe.set(mode)
        for block in self.blocks:
            block.set(mode)
        self.ln_f.set(mode)
        self.lm_head.set(mode)

    def clear_cache(self):
        """Clear all KV caches in the model."""
        for block in self.blocks:
            block.att.clear_cache()
        if hasattr(self, '_position_offset'):
            del self._position_offset

    def forward(self, x, use_cache):
        """
        Forward pass for the GPT model.
        x: input token IDs, shape (B, T)
        Returns: logits, shape (B, T, vocab_size)
        """
        B, T = x.shape
        pt_debug("Start Forward pass through wte")
        tok_emb = self.wte.forward(x)   # (B, T, n_emb)
        pt_debug("Stop Forward pass through wte")
        
        # For KV cache, we need to offset position indices
        if use_cache and hasattr(self, '_position_offset'):
            # Calculate starting position with modulo wrap
            pos_start = self._position_offset % self.n_ctx
            pos_indices = self.mp.arange(T) + pos_start
            pos_indices = pos_indices % self.n_ctx  # Wrap around if needed
        else:
            pos_indices = self.mp.arange(T) % self.n_ctx # Wrap around if needed

        pt_debug("Start Forward pass through wpe")
        pos_emb = self.wpe.forward(pos_indices) # (T, n_emb)
        pt_debug("Stop Forward pass through wpe")
        
        # Combine token and position embeddings (positional embeddings are broadcasted)
        x_combined_emb = tok_emb + pos_emb

        # Pass through tokenizer blocks
        current_x = x_combined_emb

        # We need to store the output of each block to correctly backpropagate through the sequential blocks.
        # However, the Block's backward method only needs its *own* input gradient, not the full history.
        # The chain rule handles this sequentially.
        for block in self.blocks:
            pt_debug("Start Forward pass through block")
            current_x = block.forward(current_x, use_cache)
            pt_debug("Stop Forward pass through block")

        pt_debug("Start Forward pass through final layer norm")
        ln_f_out = self.ln_f.forward(current_x)
        pt_debug("Stop Forward pass through final layer norm")
        pt_debug("Start Forward pass through head")
        logits = self.lm_head.forward(ln_f_out)  # (B, T, vocab_size)
        pt_debug("Stop Forward pass through head")
        
        # Update position offset for next iteration
        if use_cache:
            if not hasattr(self, '_position_offset'):
                self._position_offset = T
            else:
                self._position_offset += T

        # Store intermediate values for backward pass
        if is_debug():
            self._cache = (x_combined_emb, current_x, ln_f_out)

        return logits

    def backward(self, grad_output):
        """
        Backward pass for the GPT model.
        grad_output: gradient from the loss function, shape (B, T, vocab_size)
        Returns: (None, list_of_param_grads) - no grad_input for the whole model.
        """
        # Load cached values for backward pass
        if is_debug():
            (x_combined_emb, current_x_before_lnf, ln_f_out) = self._cache

        # Initialize an empty list to store gradients in the correct order
        # The order must match self.parameters()
        ordered_param_grads = []

        # 1. Backward through lm_head
        pt_debug("Start Backward pass through head")
        grad_ln_f_out, lm_head_grads = self.lm_head.backward(grad_output)
        pt_debug("Stop Backward pass through head")

        # 2. Backward through ln_f (final Normalization)
        pt_debug("Start Backward pass through final layer norm")
        grad_current_x_before_lnf, ln_f_grads = self.ln_f.backward(grad_ln_f_out)
        pt_debug("Stop Backward pass through final layer norm")

        # 3. Backward through blocks in reverse order
        # Need to store block gradients temporarily in correct order (forward pass order)
        # to match how self.blocks are added in parameters()
        block_grads_temp = [None] * len(self.blocks) # Temporary storage for block gradients
        grad_for_prev_block = grad_current_x_before_lnf
        for i in reversed(range(len(self.blocks))):
            pt_debug("Start Backward pass through block")
            block = self.blocks[i]
            grad_for_prev_block, current_block_grads = block.backward(grad_for_prev_block)
            block_grads_temp[i] = current_block_grads # Store in forward order index
            pt_debug("Stop Backward pass through block")

        # 4. Backward through token + position embeddings addition
        # grad_for_prev_block is now the gradient for x_combined_emb (tok_emb + pos_emb)
        grad_tok_emb = grad_for_prev_block # Gradient for token embeddings
        # For position embeddings, sum gradients over the batch dimension
        grad_pos_emb = self.mp.sum(grad_for_prev_block, axis=0)

        # 5. Backward through wte (token embeddings)
        # Embedding.backward returns (None, [grad_weight])
        pt_debug("Start Backward pass through wte")
        _, wte_grads = self.wte.backward(grad_tok_emb)
        pt_debug("Stop Backward pass through wte")

        # 6. Backward through wpe (position embeddings)
        # Embedding.backward returns (None, [grad_weight])
        pt_debug("Start Backward pass through wpe")
        _, wpe_grads = self.wpe.backward(grad_pos_emb)
        pt_debug("Stop Backward pass through wpe")

        # Now, assemble the ordered_param_grads list in the same order as self.parameters()
        ordered_param_grads.extend(wte_grads)
        ordered_param_grads.extend(wpe_grads)
        for grads_list_for_block in block_grads_temp: # These are already in forward order
            ordered_param_grads.extend(grads_list_for_block)
        ordered_param_grads.extend(ln_f_grads)
        ordered_param_grads.extend(lm_head_grads)

        return None, ordered_param_grads # No grad_input for the entire model

    def generate(self, stoi, prompt, use_cache, max_new_tokens):
        """
        Generates new tokens based on the model's learned probabilities.
        prompt_ids: input sequence of token IDs to start generation.
        max_new_tokens: maximum number of tokens to generate.
        Returns: generated sequence of token IDs.
        """
        # Set model to evaluation mode for generation (disables dropout)
        self.eval()

        # Clear any existing cache
        self.clear_cache()
        
        # Start with a starting token (here we use index 0, assuming it's a valid token)
        # This could be a special <SOS> token in a more robust implementation.
        if prompt is None:
            ctx = self.mp.zeros((1, 1), dtype=self.mp.int32) # Initial context: a single token
        else:
            def encode_(s):
                return self.mp.array([stoi[c] for c in s]).reshape(1, -1)
            prompt_ids = encode_(prompt)  # Your tokenizer should return shape (1, prompt_length)
            ctx = prompt_ids # Initial context: sequence of tokens

        if max_new_tokens is None:
            max_new_tokens = 500
            
        # Process initial prompt (if any) without cache to establish context
        if ctx.shape[1] > 0:
            input_seq = ctx[:, -self.n_ctx:]
            _ = self.forward(input_seq, use_cache)

        for _ in range(max_new_tokens):
            if use_cache and ctx.shape[1] > 1:
                # For cached generation, only process the last token
                input_seq = ctx[:, -1:]
            else:
                # Use the last self.n_ctx tokens as input (or all if shorter)
                input_seq = ctx[:, -self.n_ctx:]

            logits = self.forward(input_seq, use_cache)

            # Get logits for the last token in the sequence (the one to predict)
            logits = logits[:, -1, :] # Shape: (1, vocab_size)

            # Convert logits to probabilities
            probs = softmax(self.mp, logits, axis=-1).flatten() # Flatten to (vocab_size,)

            # Sample the next token based on probabilities
            # device = os.getenv("DEVICE", "").lower()
            # if device == "cpu":
            #     next_tok = self.mp.random.choice(self.mp.arange(probs.shape[0]), p=probs)
            # elif device == "gpu":
            #     next_tok = int(self.mp.random.choice(self.mp.arange(probs.shape[0]), size=1, replace=True, p=probs).item())

            r = self.mp.random.rand()
            next_tok = self.mp.searchsorted(self.mp.cumsum(probs), r)

            # Append the new token to the context
            ctx = self.mp.concatenate([ctx, self.mp.array([[next_tok]], dtype=self.mp.int32)], axis=1)

        return ctx

    def expand_embeddings(self, new_vocab_size):
        """ Expands the token embeddings and output layer to accommodate a new vocabulary size. """

        # old_vocab_size is set to 1 for initial training
        # old_vocab_size could be smaller than new_vocab_size for fine tuning
        old_vocab_size = self.vocab_size
        
        if new_vocab_size > old_vocab_size:
            print(f"Expanding model embeddings from {old_vocab_size} to {new_vocab_size}.")
            # Expand token embeddings
            self.vocab_size = new_vocab_size
            new_wte_weight = self.mp.random.randn(new_vocab_size, self.wte.weight.shape[1]) * 0.02
            new_wte_weight[:old_vocab_size] = self.wte.weight
            self.wte.weight = new_wte_weight
            
            # Expand output layer
            new_lm_head_weight = self.mp.random.randn(self.lm_head.weight.shape[0], new_vocab_size) * 0.02
            new_lm_head_weight[:, :old_vocab_size] = self.lm_head.weight
            self.lm_head.weight = new_lm_head_weight
            
            if self.lm_head.bias is not None:
                new_bias = self.mp.zeros(new_vocab_size)
                new_bias[:old_vocab_size] = self.lm_head.bias
                self.lm_head.bias = new_bias

            self.wte._parameters = [self.wte.weight]
            self.lm_head._parameters = [self.lm_head.weight]
            if self.lm_head.bias is not None:
                self.lm_head._parameters.append(self.lm_head.bias)
    
    def params_from_dict(self, json_weights_dict):
        """ Model parameters from a JSON file. """

        # Convert lists back to numpy arrays
        weights_dict = {}
        for key, value in json_weights_dict.items():
            if value is not None and isinstance(value, list):
                weights_dict[key] = self.mp.array(value, dtype=self.d_type)
            else:
                weights_dict[key] = value

        # Update vocab_size from loaded weights BEFORE restoring weights
        self.vocab_size = weights_dict['wte_weight'].shape[0]

        # Restore weights to the model
        self.wte.weight = weights_dict['wte_weight']
        self.wpe.weight = weights_dict['wpe_weight']
        self.ln_f.gamma = weights_dict['ln_f_gamma']
        self.ln_f.beta = weights_dict['ln_f_beta']
        self.lm_head.weight = weights_dict['lm_head_weight']
        self.lm_head.bias = weights_dict['lm_head_bias']

        # Update the main model's top-level _parameters
        self.wte.synchronize()
        self.wpe.synchronize()
        self.ln_f.synchronize()
        self.lm_head.synchronize()
        
        # Restore block weights
        for i, block in enumerate(self.blocks):
            # Normalization
            block.ln_1.gamma = weights_dict[f'block_{i}_ln1_gamma']
            block.ln_1.beta = weights_dict[f'block_{i}_ln1_beta']
            block.ln_2.gamma = weights_dict[f'block_{i}_ln2_gamma']
            block.ln_2.beta = weights_dict[f'block_{i}_ln2_beta']

            # Update Normalization _parameters
            block.ln_1._parameters = [block.ln_1.gamma, block.ln_1.beta]
            block.ln_2._parameters = [block.ln_2.gamma, block.ln_2.beta]
            
            # Attention
            block.att.from_dict(weights_dict, i)

            # Network
            block.net.from_dict(weights_dict, i)

    def params_towa_dict(self):     
        """ Model parameters to a JSON file. """

        # Extract all weight arrays from the model
        weights_dict = { }
        
        # Token and position embeddings
        weights_dict['wte_weight'] = self.wte.weight
        weights_dict['wpe_weight'] = self.wpe.weight

        # Final layer norm
        weights_dict['ln_f_gamma'] = self.ln_f.gamma
        weights_dict['ln_f_beta'] = self.ln_f.beta

        # Language model head
        weights_dict['lm_head_weight'] = self.lm_head.weight
        weights_dict['lm_head_bias'] = self.lm_head.bias
        
        # Add block parameters
        for i, block in enumerate(self.blocks):
            # Layer norms
            weights_dict[f'block_{i}_ln1_gamma'] = block.ln_1.gamma
            weights_dict[f'block_{i}_ln1_beta'] = block.ln_1.beta
            weights_dict[f'block_{i}_ln2_gamma'] = block.ln_2.gamma
            weights_dict[f'block_{i}_ln2_beta'] = block.ln_2.beta
            
            # Attention
            block.att.towa_dict(weights_dict, i)

            # Network
            block.net.towa_dict(weights_dict, i)
        
        # Convert numpy arrays to lists for JSON serialization
        json_weights_dict = {}
        for key, value in weights_dict.items():
            if value is not None and hasattr(value, 'tolist'):
                json_weights_dict[key] = value.tolist()
            else:
                json_weights_dict[key] = value

        # Save weights only
        return json_weights_dict
    
    def tokenizer_from_dict(self, json_tokenizer_dict):     
        """ Tokenizer to a JSON file. """
        self.tokenizer_dict = self.tokenizer.from_dict(json_tokenizer_dict)

    def tokenizer_towa_dict(self):     
        """ Tokenizer to a JSON file. """
        return self.tokenizer_dict
    
    def config_towa_dict(self):     
        """ Config to a JSON file. """
        self.config_dict["runtime"] = {
            "model_version": "v0.1.0",
            "model_params": self.count_parameters(),
            "model_size": self.count_totalsize(),
            "device_name": self.device_name,
            "device_cores": self.device_cores,
            "library_version": self.library_version,
            "library_driver": self.library_driver
            }
        return self.config_dict

    def report_towa_dict(self):     
        """ Report to a JSON file. """
        return self.report_dict
    
    def completion_towa_dict(self):     
        """ Completion to a JSON file. """
        return self.completion_dict
    
    def count_parameters(self):
        """Return the total number of parameters in the model."""
        total_params = sum((p.size for p in self.parameters()))
        return total_params
    
    def count_totalsize(self):
        """Return the total size of parameters in the model."""
        total_size = sum(p.size for p in self.parameters()) * 4
        return total_size

    def backup(model):
        """ Backup the model weights and configuration. """
        model_ = model
        model.wte = model_.wte
        model.wpe = model_.wpe
        model.blocks = model_.blocks
        model.ln_f = model_.ln_f
        model.lm_head = model_.lm_head
        model.vocab_size = model_.vocab_size
        model._parameters = model_.parameters()
        return model

    def train(self, input_name, input_text, train_cache, n_epochs, s_batch, r_learn, s_warmup, c_shuffle, r_split):
        """ Train the GPT model on the provided input data. """
        print(f"{'-'*10} {'Training in progress'} {'-'*10}" )

        # File name from path
        file_name = os.path.basename(input_name)

        # Tokenize the input data
        train_data, val_data = self.tokenizer.tokenize(input_text, c_shuffle, r_split)        

        # Create the tokenizer object
        tokenizer_dict = self.tokenizer.towa_dict()

        # Save the tokenizer to JSON
        self.tokenizer_dict = tokenizer_dict
        
        X_train, y_train = self.tokenizer.prepare_data(train_data, self.n_ctx)
        X_val, y_val = self.tokenizer.prepare_data(val_data, self.n_ctx)

        # If vocabulary expanded, resize model embeddings
        vocab_size = self.tokenizer.vocab_size  # Get the size of the vocabulary from the tokenizer
        if vocab_size > self.vocab_size: # If the vocabulary size has changed, expand embeddings
            self.expand_embeddings(vocab_size)

        # Get parameters from the model
        params = self.parameters() 
    
        # Initialize the optimizer
        optimizer = Optimizer(self.mp, self.c_optimizer, params, r_learn)

        batch_logs = []
        epoch_logs = []
        train_batch_total_cnt = 0
        train_batch_total_time = 0
        val_batch_total_cnt = 0
        val_batch_total_time = 0
        epoch_total_time = 0
        total_time = 0

        for epoch in range(n_epochs):
            # Record start time
            epoch_start_time = tm.time()

            # Training phase
            self.set(True) # Set model to training mode (enables dropout)

            # Train Epoch Level
            running_train_loss = 0
            train_batch_cnt = 0
            train_batch_all = 0

            train_batches = list(self.tokenizer.get_batches(X_train, y_train, s_batch, shuffle=True))
            train_batch_all = len(train_batches)
            if not train_batches:
                print(f"Epoch {epoch+1}/{n_epochs} | No training data batches available. Skipping training for this epoch.")
                avg_train_loss = float('nan')
            else:
                for X_batch, y_batch in train_batches:
                    train_batch_cnt += 1
                    train_batch_start_time = tm.time() # Record start time
                    # Adjust learning rate during warm-up
                    optimizer.set_r_learn(r_learn, train_batch_cnt, train_batch_all, s_warmup)
                    # Compute loss and gradients
                    loss, grads = value_and_grad(self, X_batch, y_batch, train_cache)
                    # Update model parameters using the optimizer
                    optimizer.step(grads)
                    running_train_loss += loss
                    train_batch_stop_time = tm.time() # Record stop time
                    train_batch_elapsed_time = train_batch_stop_time - train_batch_start_time
                    train_batch_total_time += train_batch_elapsed_time
                    # Print loss for the current batch
                    batch_log = f"Epoch {epoch+1}/{n_epochs} | Batch {train_batch_cnt}/{train_batch_all} | train_loss = {loss:.4f} | execution_time = {train_batch_elapsed_time:.4f}"
                    batch_logs.append(batch_log)
                    print(batch_log)
                avg_train_loss = running_train_loss / train_batch_cnt
                train_batch_total_cnt += train_batch_cnt

            # Validation phase
            self.set(False) # Set model to evaluation mode (disables dropout)

            # Val Epoch Level
            running_val_loss = 0
            val_batch_cnt = 0
            val_batch_all = 0

            val_batches = list(self.tokenizer.get_batches(X_val, y_val, s_batch, shuffle=False))
            val_batch_all = len(val_batches)
            if not val_batches:
                print(f"Epoch {(epoch+1)}/{n_epochs} | No validation data batches available. Skipping validation for this epoch.")
                avg_val_loss = float('nan') # Indicate no validation was performed
            else:
                for X_batch, y_batch in val_batches:
                    val_batch_cnt += 1
                    val_batch_start_time = tm.time() # Record start time
                    # In validation, only forward pass and loss computation are needed
                    loss, _ = value_and_nograd(self, X_batch, y_batch, train_cache)
                    running_val_loss += loss
                    val_batch_stop_time = tm.time() # Record stop time
                    val_batch_elapsed_time = val_batch_stop_time - val_batch_start_time 
                    val_batch_total_time += val_batch_elapsed_time
                    # Print loss for the current batch
                    batch_log = f"Epoch {epoch+1}/{n_epochs} | Batch {val_batch_cnt}/{val_batch_all} | val_loss = {loss:.4f} | execution_time = {val_batch_elapsed_time:.4f}"
                    batch_logs.append(batch_log)
                    print(batch_log)
                avg_val_loss = running_val_loss / val_batch_cnt
                val_batch_total_cnt += val_batch_cnt

            # Record stop time
            epoch_stop_time = tm.time()

            # Epoch Level
            epoch_elapsed_time = epoch_stop_time - epoch_start_time  
            epoch_total_time += epoch_elapsed_time

            # Print average losses for the epoch
            epoch_log = f"Epoch {(epoch+1)}/{n_epochs} | train_loss = {avg_train_loss:.4f} | val_loss = {avg_val_loss:.4f} | execution_time = {epoch_elapsed_time:.4f}"
            epoch_logs.append(epoch_log)
            print(epoch_log)

        #Total Level
        avg_train_time_per_batch = train_batch_total_time / train_batch_total_cnt
        avg_val_time_per_batch = val_batch_total_time / val_batch_total_cnt
        epoch_total_time_avg = epoch_total_time / n_epochs
        total_time += epoch_total_time
        print(f"Average epoch time: {epoch_total_time_avg:.4f}")        

        # Create the report object
        report_dict = {
            "n_epochs": n_epochs,
            "dataset": file_name,
            "avg_train_time_per_batch": float(avg_train_time_per_batch),
            "train_batches_per_epoch": int(train_batch_all),
            "avg_val_time_per_batch": float(avg_val_time_per_batch),
            "val_batches_per_epoch": int(val_batch_all),
            "average_train_loss_per_epoch": float(avg_train_loss),
            "average_val_loss_per_epoch": float(avg_val_loss),
            "average_time_per_epoch": float(epoch_total_time_avg),
            "total_time": float(total_time),
            "train_cache": train_cache,
            "batch_logs": batch_logs,
            "epoch_logs": epoch_logs
        }

        # Save the report to JSON
        self.report_dict = report_dict

    def inference(self, prompt, infer_cache, max_tokens):
        """ Perform inference with the trained model using a given prompt. """
        print(f"{'-'*10} {'Infrerence in progress'} {'-'*10}" )

        # Update vocab size
        self.vocab_size = self.tokenizer.vocab_size  # Update model vocabulary size from tokenizer
        
        # If vocabulary expanded, resize model embeddings / Is not needed as load is loaded expanded matixes
        # vocab_size = tokenizer.vocab_size  # Get the size of the vocabulary from the tokenizer
        # if vocab_size > self.vocab_size: # If the vocabulary size has changed, expand embeddings
        #     self.expand_embeddings(vocab_size)

        # Load the tokenizer's string-to-index mapping
        stoi = self.tokenizer.stoi
        
        # Initialize the total inference time
        inference_total_time = 0

        # Generate new tokens
        inference_max_tokens = max_tokens

        # Record start time
        inference_start_time = tm.time()

        # Generate text based on given token ids
        generation_ids = self.generate(stoi, prompt, infer_cache, inference_max_tokens)
        
        # Decode the generated token IDs back to text
        generation = self.tokenizer.decode(generation_ids[0].tolist())

        # Record stop time
        inference_stop_time = tm.time() 

        # Calculate elapsed time for inference
        inference_elapsed_time = inference_stop_time - inference_start_time  
        inference_total_time += inference_elapsed_time

        # Create a completion object with the prompt, generated text, and inference time
        completion_dict = { "prompt": prompt, "generation": generation, "inference_max_tokens": inference_max_tokens, "inference_total_time": inference_total_time, "inference_cache": infer_cache }

        print(completion_dict)

        self.completion_dict = completion_dict