#########################
# Character Tokenizer
# Author: Koureas Stavros
#########################

class CharTokenizer:
    """
    Character-level tokenizer for text processing.
    Separates encoding/decoding logic for better modularity.
    """
    def __init__(self, mp, text=None, vocab=None, stoi=None, itos=None):
        self.mp = mp
        self.vocab = None
        self.vocab_size = None
        self.itos = None
        self.stoi = None

    def init(self, text):
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.itos = {i: c for i, c in enumerate(self.vocab)}
        self.stoi = {c: i for i, c in enumerate(self.vocab)}

    def expand_vocabilary(self, text):
        """ Expand the tokenizer vocabulary with new characters from the text. """
        new_chars = set(text) - set(self.vocab)
        if new_chars:
            print(f"Found {len(new_chars)} new characters: {new_chars}")
            # Add new characters to existing vocabulary and sort everything
            all_chars = set(self.vocab) | new_chars  # Union of sets
            self.vocab = sorted(list(all_chars))
            self.vocab_size = len(self.vocab)
            self.itos = {i: c for i, c in enumerate(self.vocab)}
            self.stoi = {c: i for i, c in enumerate(self.vocab)}

    def fit(self, text):
        """ Fit the tokenizer to the provided text. """
        if self.vocab is None:
            self.init(text)
        else:
            self.expand_vocabilary(text)

    def encode(self, text):
        """ Convert string to list of token IDs. """        
        return [self.stoi[c] for c in text]
    
    def decode(self, token_ids):
        """ Convert list of token IDs to string. """
        return ''.join([self.itos[i] for i in token_ids])
    
    def from_dict(self, tokenizer_dict):
        """ Load tokenizer from dictionary. """
        self.vocab=tokenizer_dict['vocab']
        self.vocab_size=len(self.vocab)
        self.itos={int(k): v for k, v in enumerate(self.vocab)} # Convert keys back to integers
        self.stoi={str(v): k for k, v in enumerate(self.vocab)} # Convert keys back to strings
    
    def towa_dict(self):
        """ Export tokenizer state as dictionary for JSON serialization. """
        return {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size
        }
    
    def tokenize(self, text, c_shuffle, r_split):
        """ Tokenize input text and return data. """        
        
        # Fit the tokenizer to the text
        self.fit(text)

        # Encode the text
        data = self.encode(text)

        # Shuffle the data
        if c_shuffle:
            indices = self.mp.arange(len(data))
            self.mp.random.shuffle(indices)
            data = [data[i] for i in indices]

        # Split into training and validation sets
        split = int(r_split * len(data))
        train_data = data[:split]
        val_data = data[split:]

        return train_data, val_data
    
    def prepare_data(self, data, n_ctx):
        """ Prepares data for training by creating input-output pairs. """

        X, y = [], []
        # Ensure there's enough data for at least one full context length + 1 for target
        if len(data) < n_ctx + 1:
            # Return empty arrays with correct shapes for concatenation later
            return self.mp.array([], dtype=self.mp.int32).reshape(0, n_ctx), self.mp.array([], dtype=self.mp.int32).reshape(0, n_ctx)

        for i in range(0, len(data) - n_ctx):
            X.append(data[i:i+n_ctx])
            y.append(data[i+1:i+n_ctx+1])
        return self.mp.array(X, dtype=self.mp.int32), self.mp.array(y, dtype=self.mp.int32)
    
    def get_batches(self, X, y, b_size, shuffle=True):
        """ Generates batches of data for training. """
        
        N = X.shape[0]
        if N == 0: # Handle empty data case
            return # Yield nothing if no data
        indices = self.mp.arange(N)
        if shuffle:
            self.mp.random.shuffle(indices)
        for i in range(0, N, b_size):
            batch_idx = indices[i:i+b_size]
            yield X[batch_idx], y[batch_idx]