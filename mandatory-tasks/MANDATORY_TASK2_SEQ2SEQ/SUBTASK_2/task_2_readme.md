# Task 2 — Attention Mechanism and Decoding Strategies

## Overview

This task extends the basic Sequence-to-Sequence (Seq2Seq) translation model from Task 1 by introducing an **attention mechanism** and comparing different decoding strategies.

The objective is to determine whether allowing the decoder to dynamically focus on different parts of the input sequence improves translation performance.

The same English-to-French translation dataset and preprocessing pipeline from Task 1 were used.

---

## Objectives

The main objectives of this task are:

1. Extend the basic Seq2Seq model with **Bahdanau additive attention**.
2. Allow the decoder to access all encoder hidden states instead of relying only on the final encoder state.
3. Compare the attention-based model with the basic Seq2Seq baseline.
4. Compare different decoding strategies.
5. Evaluate the models using **BLEU score**.
6. Analyze the effect of attention and decoding on translation quality.

---

## Dataset

The same English-French parallel dataset used in Task 1 was used.

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

For computational efficiency, a subset of **20,000 training examples** was used during model training.

No new dataset or additional training data was introduced in this task.

---

## Preprocessing

The preprocessing pipeline from Task 1 was retained.

The following special tokens were used:

| Token | ID | Purpose |
|---|---:|---|
| `<PAD>` | 0 | Padding sequences |
| `<SOS>` | 1 | Start of sequence |
| `<EOS>` | 2 | End of sequence |
| `<UNK>` | 3 | Unknown words |

Vocabulary construction used a minimum frequency of 2.

### Vocabulary Sizes

- English vocabulary: **40,776**
- French vocabulary: **53,548**

---

# Model Architecture

## Basic Seq2Seq Baseline

The baseline model from Task 1 uses:

```text
English Input
     │
     ▼
Word Embedding
     │
     ▼
GRU Encoder
     │
     │ Final Hidden State
     ▼
GRU Decoder
     │
     ▼
French Output
```

The main limitation of this architecture is that the decoder primarily depends on the encoder's final hidden state to represent the entire input sequence.

## Attention-Based Seq2Seq

The baseline was extended using Bahdanau additive attention.

```text
                    ┌──────────────────────┐
                    │ Encoder Hidden States│
                    └──────────┬───────────┘
                               │
                               ▼
                         Bahdanau
                         Attention
                               │
                               ▼
English Input → GRU Encoder → Context Vector
                               │
                               ▼
                         GRU Decoder
                               │
                               ▼
                        French Output
```

Instead of using only the final encoder state, the attention mechanism calculates a weighted combination of all encoder hidden states. At each decoding step, the decoder determines which parts of the input sequence are most relevant.

### Bahdanau Attention

Bahdanau attention is an additive attention mechanism. The decoder hidden state and encoder hidden states are combined to calculate attention scores.

Conceptually:

```text
Encoder States + Decoder State
             │
             ▼
       Attention Scores
             │
             ▼
       Softmax Weights
             │
             ▼
       Context Vector
```

The context vector is then provided to the decoder along with the current input token. Padding positions are masked so that the attention mechanism does not assign meaningful attention to padded tokens.

---

## Model Configuration

The attention model uses the same main dimensions as the baseline to make the comparison fair.

| Parameter | Value |
|---|---|
| Training subset | 20,000 |
| Batch size | 8 |
| Embedding size | 128 |
| Hidden size | 256 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Teacher forcing ratio | 0.5 |
| Gradient clipping | 1.0 |
| Epochs | 1 |
| Attention | Bahdanau |
| Hardware | NVIDIA RTX 4050 Laptop GPU |

The attention model was trained for one epoch under the same computational constraints as Task 1.

---

## Training

The attention model was trained using cross-entropy loss. Padding tokens were ignored during loss calculation. Gradient clipping was applied with a maximum norm of 1 to improve training stability. The same teacher forcing ratio of 0.5 was used.

### Attention Model Training Results

| Metric | Result |
|---|---:|
| **Training Loss** | 5.7056 |
| **Validation Loss** | 6.0270 |

Compared with the basic Seq2Seq model, the attention model achieved a lower training and validation loss.

---

## Evaluation

The models were evaluated on the test set using BLEU score. BLEU was used to compare generated French translations with the reference translations.

### BLEU Results

| Model | Test BLEU |
|---|---:|
| Basic Seq2Seq | 0.0159 |
| **Seq2Seq + Bahdanau Attention** | **0.0596** |

The attention-based model achieved approximately a **3.75× improvement** in BLEU compared with the basic Seq2Seq model:

$$\text{Improvement} = \frac{\text{Attention BLEU}}{\text{Basic Seq2Seq BLEU}} = \frac{0.0596}{0.0159} \approx 3.75$$

This demonstrates a substantial improvement from introducing attention under the same limited training setup.

---

## Decoding Strategies

Two decoding strategies were compared.

### 1. Greedy Decoding

At every decoding step, the token with the highest predicted probability is selected.

```text
Choose the highest-probability token
            ↓
Use it as the next input
            ↓
Repeat
```

* **Advantages:** Simple, fast, low computational cost.
* **Limitation:** It makes locally optimal decisions and does not consider alternative multi-step sequence probabilities.

### 2. Beam Search

Beam search maintains multiple candidate sequences during decoding. A beam width of 3 was used. Instead of keeping only the single best token at each step, the decoder maintains the three best partial sequences and expands them as generation continues.

Conceptually:

```text
                 Start
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
        Beam 1   Beam 2   Beam 3
          │        │        │
          └────────┼────────┘
                   │
              Best sequence
```

Beam search produced different outputs from greedy decoding, but it did not substantially resolve the repetition and unknown-token problems.

---

## Qualitative Analysis

The generated translations showed several limitations. Common issues included:

* Repetition of words or phrases
* Incomplete translations
* `<UNK>` tokens
* Grammatically incorrect outputs
* Poor handling of longer sentences

Although beam search produced somewhat different outputs, the qualitative improvement was limited. This suggests that the main limitation was not only the decoding strategy but also the very limited amount of model training.

---

## Comparison Summary

| Aspect | Basic Seq2Seq | Seq2Seq + Attention |
|---|---|---|
| **Encoder** | GRU | GRU |
| **Decoder** | GRU | GRU |
| **Attention** | No | Bahdanau |
| **Embedding Size** | 128 | 128 |
| **Hidden Size** | 256 | 256 |
| **Training Subset** | 20,000 | 20,000 |
| **Epochs** | 1 | 1 |
| **Teacher Forcing** | 0.5 | 0.5 |
| **Test BLEU** | **0.0159** | **0.0596** |

---

## Key Findings

1. **Attention improved translation performance:** The attention-based model achieved a BLEU score of 0.0596, compared with 0.0159 for the basic Seq2Seq model (~3.75× improvement).
2. **Attention reduces the information bottleneck:** The basic Seq2Seq model relies primarily on the final encoder representation. Attention allows the decoder to dynamically access different encoder states during generation.
3. **Beam search did not solve the main generation problems:** Beam search generated somewhat different sequences, but repetition and `<UNK>` tokens remained.
4. **Training budget was a major limitation:** Both models were trained for only one epoch using a 20,000-example subset. Therefore, the results demonstrate the architectural comparison rather than fully optimized translation performance.

---

## Limitations

* **Limited training time:** Only one epoch was used.
* **Reduced training dataset:** Only 20,000 training examples were used.
* **Large vocabulary:** The vocabulary contains tens of thousands of tokens, making learning difficult under a small training budget.
* **Unknown tokens:** Words outside the learned vocabulary are represented using `<UNK>`.
* **Limited decoding comparison:** Only greedy decoding and beam search with beam width 3 were evaluated.
* **Low absolute BLEU:** Although attention significantly improved BLEU, the absolute score remains low because of the limited training setup.

---

## Conclusion

This task successfully extended the basic Seq2Seq translation model with a Bahdanau attention mechanism.

The attention-based model achieved:
* **BLEU = 0.0596**

compared with:
* **BLEU = 0.0159** for the basic Seq2Seq model.

Thus, attention produced an approximately **3.75× improvement** in BLEU, demonstrating the benefit of allowing the decoder to dynamically focus on different encoder states. Beam search produced different outputs compared with greedy decoding but did not substantially eliminate repetition or `<UNK>` errors.

Overall, the experiment shows that attention was more impactful than changing the decoding strategy under the limited training budget used in this project.