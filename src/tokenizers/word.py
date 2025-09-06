#########################
# Word Tokenizer
# Author: Koureas Stavros
#########################

class WordTokenizer:
    """
    Word-level tokenizer for text processing.
    Similar to CharTokenizer but splits on whitespace.
    """
    def __init__(self, mp, text=None, vocab=None, stoi=None, itos=None):
        self.mp = mp
        self.vocab = None
        self.vocab_size = None
        self.itos = None
        self.stoi = None

    def init(self, text):
        words = text.split()
        self.vocab = sorted(list(set(words)))
        self.vocab_size = len(self.vocab)
        self.itos = {i: w for i, w in enumerate(self.vocab)}
        self.stoi = {w: i for i, w in enumerate(self.vocab)}

    def expand_vocabulary(self, text):
        """ Expand the tokenizer vocabulary with new words from the text. """
        new_words = set(text.split()) - set(self.vocab)
        if new_words:
            print(f"Found {len(new_words)} new words: {new_words}")
            all_words = set(self.vocab) | new_words
            self.vocab = sorted(list(all_words))
            self.vocab_size = len(self.vocab)
            self.itos = {i: w for i, w in enumerate(self.vocab)}
            self.stoi = {w: i for i, w in enumerate(self.vocab)}

    def fit(self, text):
        """ Fit the tokenizer to the provided text. """
        if self.vocab is None:
            self.init(text)
        else:
            self.expand_vocabulary(text)

    def encode(self, text):
        """ Convert string to list of token IDs (word-based). """
        return [self.stoi[w] for w in text.split()]
    
    def decode(self, token_ids):
        """ Convert list of token IDs to string. """
        return ' '.join([self.itos[i] for i in token_ids])
    
    def from_dict(self, tokenizer_dict):
        """ Load tokenizer from dictionary. """
        self.vocab = tokenizer_dict['vocab']
        self.vocab_size = len(self.vocab)
        self.itos = {int(k): v for k, v in enumerate(self.vocab)}
        self.stoi = {str(v): k for k, v in enumerate(self.vocab)}
    
    def towa_dict(self):
        """ Export tokenizer state as dictionary for JSON serialization. """
        return {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size
        }

    def tokenize(self, text):
        """ Tokenize input text and return train/val data. """
        self.fit(text)
        data = self.encode(text)
        split = int(0.9 * len(data))
        train_data = data[:split]
        val_data = data[split:]
        return train_data, val_data
    
    def prepare_data(self, data, n_ctx):
        """ Prepares data for training by creating input-output pairs. """
        X, y = [], []
        if len(data) < n_ctx + 1:
            return self.mp.array([], dtype=self.mp.int32).reshape(0, n_ctx), self.mp.array([], dtype=self.mp.int32).reshape(0, n_ctx)
        for i in range(0, len(data) - n_ctx):
            X.append(data[i:i+n_ctx])
            y.append(data[i+1:i+n_ctx+1])
        return self.mp.array(X, dtype=self.mp.int32), self.mp.array(y, dtype=self.mp.int32)
    
    def get_batches(self, X, y, b_size, shuffle=True):
        """ Generates batches of data for training. """
        N = X.shape[0]
        if N == 0:
            return
        indices = self.mp.arange(N)
        if shuffle:
            self.mp.random.shuffle(indices)
        for i in range(0, N, b_size):
            batch_idx = indices[i:i+b_size]
            yield X[batch_idx], y[batch_idx]