# Seq2Seq-with-Attention

## Model Description (Training Procedure)

1. **Embedding**: 
   - Convert source sentence words/characters into embeddings.
   - Feed embeddings into a convolutional layer.

2. **Bidirectional LSTM Encoder**: 
   - Process embeddings through a bidirectional LSTM.
   - Generate and concatenate forward and backward hidden states.

3. **Decoder Initialization**: 
   - Initialize the decoder’s hidden and cell states using the encoder’s final states.

4. **Attention Mechanism**: 
   - Apply attention to the encoder's hidden states.
   - Combine attention output with the decoder's hidden states.

5. **Output Generation**: 
   - Generate a probability distribution over the target vocabulary.
   - Train the model using cross-entropy loss.
