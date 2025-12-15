### Tokenizer Size
Given a dataset like sophocles works which has 283822 characters, 293679 total characters (including special characters), in UTF-8 Greek characters take 2 bytes, therefore the total size would be 587358 bytes as a file. If the n_ctx is 128 it means that there would be in total 283568 sequences of 128 size each, this is because each sequence is a sliding window of one character into the dataset while the sequences are 254 less than the actual characters because coming to the end of the dataset the last 127 x 2 (because train and validation) characters will not be slided as there would not be other characters to populate the sequence. Comming to the end, the total size of a such table if calculated at once would be  283568 x 128 = 36296704 x 2 bytes = 72MB

So the formula is (characters - ((n_ctx - 1) x 2)) x 2 bytes
For a dataset like simple_wiki of 8411633 characters with n_ctx of 256 there would be 8411123 batches with 256 length which is 2153247488 x 2 bytes = 4GB
For a dataset like simple_wiki of 8411633 characters with n_ctx of 512 there would be 8410611 batches with 512 length which is 4306232832 x 2 bytes = 8GB
For a dataset like simple_wiki of 8411633 characters with n_ctx of 1024 there would be 8409587 batches with 1024 length which is 8611417088 x 2 bytes = 16GB

Instead of this code:
train_batches = list(self.tokenizer.get_batches(X_train, y_train, s_batch, shuffle=True))
train_batch_all = len(train_batches)
val_batches = list(self.tokenizer.get_batches(X_val, y_val, s_batch, shuffle=False))
val_batch_all = len(val_batches)

Use this code:
train_batch_all = int(self.mp.ceil(X_train.shape[0] / s_batch)) if X_train.shape[0] > 0 else 0
train_batches = self.tokenizer.get_batches(X_train, y_train, s_batch, shuffle=True)
val_batch_all = int(self.mp.ceil(X_val.shape[0] / s_batch)) if X_val.shape[0] > 0 else 0
val_batches = self.tokenizer.get_batches(X_val, y_val, s_batch, shuffle=False)