# Task 1 — Basic Sequence-to-Sequence Translation

## Overview

This task implements a basic **Sequence-to-Sequence (Seq2Seq)** neural machine translation model for translating English sentences into French.

The purpose of this task is to establish a baseline encoder-decoder architecture before extending it with attention mechanisms and more advanced decoding strategies in the next task.

---

## Objective

Build and evaluate a basic Seq2Seq translation model using:

- An **Encoder GRU**
- A **Decoder GRU**
- Learned word embeddings
- Teacher forcing during training
- Greedy decoding during inference
- BLEU score for evaluation

The model was trained from scratch using an English-French parallel dataset.

---

## Dataset

The dataset contains parallel English and French sentences.

| Column | Description |
|---|---|
| `en` | English sentence |
| `fr` | Corresponding French translation |

### Dataset Split

| Split | Number of Examples |
|---|---:|
| Training | 232,825 |
| Validation | 890 |
| Test | 8,597 |

For computational efficiency, a subset of **20,000 training examples** was used for model training.

---

## Preprocessing

The text was tokenized and converted into integer IDs.

Four special tokens were used:

| Token | ID | Purpose |
|---|---:|---|
| `<PAD>` | 0 | Padding sequences |
| `<SOS>` | 1 | Start of sequence |
| `<EOS>` | 2 | End of sequence |
| `<UNK>` | 3 | Unknown words |

The vocabulary was constructed from the training data with a minimum frequency threshold of 2.

### Vocabulary Sizes

- English vocabulary: **40,776**
- French vocabulary: **53,548**

---

## Model Architecture

The model follows a standard encoder-decoder Seq2Seq architecture.

```text
English Sentence
       │
       ▼
   Tokenization
       │
       ▼
   Word Embedding
       │
       ▼
   GRU Encoder
       │
       │ Hidden State
       ▼
   GRU Decoder
       │
       ▼
 French Sentence
```

### Encoder
The encoder processes the complete English input sequence and produces a final hidden state representing the input.

**Configuration:**
- **Embedding size:** 128
- **Hidden size:** 256
- **Recurrent unit:** GRU

### Decoder
The decoder generates the French translation one token at a time. At each step, it receives:
- The previous target token
- The decoder hidden state

**Configuration:**
- **Embedding size:** 128
- **Hidden size:** 256
- **Recurrent unit:** GRU

### Teacher Forcing
Teacher forcing was used during training with a ratio of 0.5. This means that during training, the decoder uses the correct previous target token as its next input with probability 0.5. Otherwise, it uses its own previous prediction.

Teacher forcing helps the decoder learn the target sequence more effectively during training.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Training subset | 20,000 |
| Batch size | 8 |
| Embedding size | 128 |
| Hidden size | 256 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Teacher forcing ratio | 0.5 |
| Gradient clipping | 1 |
| Epochs | 1 |
| Hardware | NVIDIA RTX 4050 Laptop GPU |

*Note: Only one epoch was used to keep the experiment computationally manageable.*

### Loss Function
Cross-entropy loss was used for token prediction. Padding tokens were ignored while calculating the loss. The objective was to minimize the difference between the predicted French token and the actual target token at each decoding step.

---

## Results

### Training and Validation Loss

| Metric | Result |
|---|---:|
| Training Loss | 6.0935 |
| Validation Loss | 6.3143 |

### Test Evaluation

- **BLEU Score:** 0.0159

The relatively low BLEU score is expected given the limited training setup, especially the use of only 20,000 training examples and one training epoch.

### Qualitative Results
The model was able to generate French-like output, but the translations were often:
- Incomplete
- Repetitive
- Incorrect
- Affected by `<UNK>` tokens

This indicates that the basic Seq2Seq architecture was able to learn some translation patterns but struggled to retain and use all the information from longer input sequences.

---

## Limitations

1. **Limited Training:** Only one epoch was used, so the model had limited opportunity to learn the translation task.
2. **Reduced Training Dataset:** Only 20,000 examples were used instead of the complete training set.
3. **No Attention Mechanism:** The basic architecture relies primarily on the encoder's final hidden representation. This creates a bottleneck when representing longer input sequences.
4. **Unknown Tokens:** Words outside the learned vocabulary are represented using `<UNK>`, which can reduce translation quality.
5. **Greedy Decoding:** The decoder selects the highest-probability token at each step without considering alternative sequences.

These limitations motivate the next task, where an attention mechanism and different decoding strategies are introduced.

---

## Conclusion

A basic GRU-based Seq2Seq model was successfully created for English-to-French translation. The model achieved a test BLEU score of 0.0159 after one epoch of training.

Although the translation quality was limited, this experiment establishes a baseline for comparing more advanced architectures. The next task extends this model with Bahdanau attention and compares different decoding strategies.