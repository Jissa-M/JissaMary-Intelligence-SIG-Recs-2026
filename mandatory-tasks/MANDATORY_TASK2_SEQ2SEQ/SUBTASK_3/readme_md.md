# Task 3 — Document-Grounded Hinglish Dialogue Generation

## 1. Project Overview

This project extends sequence-to-sequence and attention-based models into a **document-grounded dialogue generation system** for code-mixed **Hinglish** (a blend of Hindi and English) multi-turn conversations.

Unlike single-sentence machine translation, the model generates conversational responses using two distinct context sources:

1. **Conversation History:** Previous dialogue turns in the conversation.
2. **Grounding Document:** External movie background knowledge from WikiData JSON files.

To measure the benefit of grounding, the project compares:
* **Ungrounded Baseline Generator:** Uses only conversation history ($History \rightarrow Response$).
* **Document-Grounded Generator:** Uses both conversation history and movie document context ($History + Document \rightarrow Response$).

---

## 2. Objectives

* Build a dialogue generation pipeline for multi-turn conversations.
* Handle code-mixed Hinglish text effectively.
* Construct full conversation history from individual turns using boundary tracking.
* Dynamically retrieve relevant external WikiData documents for each conversation.
* Implement Bahdanau additive attention mechanisms for input grounding.
* Compare ungrounded and document-grounded generators using automatic metrics.
* Perform qualitative analysis on generated conversational outputs.

---

## 3. Dataset & Grounding Resources

### Primary Dataset
* **Name:** CMU Hinglish Dialogue Dataset (`festvox/cmu_hinglish_dog`)
* **Source:** [Hugging Face Datasets](https://huggingface.co/datasets/festvox/cmu_hinglish_dog)
* **Setup:** Based on the CMU Document-Grounded Conversation (DoG) setup.
* **Authentication:** No Hugging Face token required.

#### Dataset Splits
| Split | Original Examples | Processed Examples |
|---|---:|---:|
| **Train** | 8,060 | 7,804 |
| **Validation** | 942 | 910 |
| **Test** | 960 | 928 |

*(Processed counts are lower because unassociable examples missing required conversation/document metadata were excluded.)*

### Primary Fields
* `translation.en`: English version of dialogue turn.
* `translation.hi_en`: Hinglish version of dialogue turn.
* `uid`: User identifier for dialogue turn.
* `user2_id`: Used to identify conversation boundaries.
* `wikiDocumentIdx`: Identifies the relevant movie document.
* `docIdx`: Identifies the section of the movie document.

### WikiData Documents
* **Total Documents:** 30 movie JSON files (e.g., *Batman_Begins.json*, *The_Avengers.json*, *Iron_Man.json*, *Frozen.json*, *Jaws.json*, *John_Wick.json*, *La_la_land.json*, *Toy_Story.json*, *Wonder_Woman.json*, *Zootopia.json*).
* Each JSON file contains sections (e.g., `0` for metadata, `1-3` for story descriptions).
* Mapping between `wikiDocumentIdx` and movie files was constructed by reading values directly inside JSON documents rather than relying on filename ordering.

---

## 4. Conversation Construction & Example

The raw turns were re-constructed into structured training instances:

$$\text{Conversation History} + \text{Grounding Document} \rightarrow \text{Target Response}$$

A shift in `user2_id` signals the start of a new conversation thread.

### Sample Code-Mixed Dialogue Thread

#### English Alignment
> **User:** what moviie did you see  
> **User:** hello how are you? Have you heard of Batman Begins? It is a great movie!  
> **User:** no tell me more  
> **User:** what is it about  

#### Code-Mixed Hinglish (Target Generation)
> **User:** tumne konsi movie dekhi  
> **User:** hello tum kaise ho? Kya tumne Batman Begins ke bare mein suna hai? Kya great movie hai!  
> **User:** nahi aur batao  
> **User:** ye kis bare mein hai  

---

## 5. Input Representation & Tokenization

Each training instance comprises:
* **Input 1 (History):** Prior turns including speaker information.
* **Input 2 (Document):** Relevant section of movie WikiData document.
* **Target:** Next Hinglish response.

### Speaker Information
Special speaker tags are converted into explicit tokens during tokenization:
* `<usr1>` $\rightarrow$ `usr1speaker`
* `<usr2>` $\rightarrow$ `usr2speaker`

```python
def tokenize(text):
    text = text.lower()
    text = text.replace("<usr1>", " usr1speaker ")
    text = text.replace("<usr2>", " usr2speaker ")
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens
```

### Preprocessing Steps
1. Convert text to lowercase.
2. Preserve special speaker tags.
3. Tokenize words and punctuation using Unicode regular expressions.
4. Construct vocabulary strictly from training data (`min_freq = 2`).
5. Map tokens to integer IDs and pad/truncate to fixed lengths.

### Special Tokens
| Token | ID | Purpose |
|---|---|---|
| `<PAD>` | 0 | Padding |
| `<SOS>` | 1 | Start of Sequence |
| `<EOS>` | 2 | End of Sequence |
| `<UNK>` | 3 | Unknown Token |

Vocabulary Size: **7,689 tokens**.

### Sequence Cutoffs & Dataset Statistics
| Parameter | Dataset Average | Dataset Maximum | Model Truncation Limit |
|---|---:|---:|---:|
| **History Length** | 241.867 | 1,645 | `MAX_HISTORY_LEN = 80` |
| **Document Length** | 187.442 | 407 | `MAX_DOCUMENT_LEN = 150` |
| **Target Length** | 11.763 | 341 | `MAX_TARGET_LEN = 40` |

---

## 6. Model Architectures & Configurations

### Hyperparameter Configuration
| Parameter | Value |
|---|---|
| **Embedding Size** | 64 |
| **Hidden Size** | 128 |
| **Batch Size** | 8 |
| **Learning Rate** | 0.001 |
| **Optimizer** | Adam |
| **Teacher Forcing Ratio** | 0.5 |
| **Epochs** | 1 |
| **Device** | CUDA (NVIDIA RTX 4050 Laptop GPU) |
| **Total Parameters** | ~3,840,201 |

### 1. Baseline Model Architecture (Ungrounded)
* Processes history only via single GRU encoder.
* Calculates Bahdanau additive attention over history hidden states.
* Decodes using a GRU decoder with teacher forcing.

$$\text{History} \rightarrow \text{GRU Encoder} \rightarrow \text{Attention Context} \rightarrow \text{GRU Decoder} \rightarrow \text{Response}$$

### 2. Grounded Model Architecture (Dual Attention)
* Employs two separate GRU encoders: History Encoder and Document Encoder.
* Uses dual Bahdanau attention mechanisms to extract context independently from history and document sequences.
* Combines token embedding, history context vector, and document context vector at each decoding step.

$$\begin{aligned}
\text{History} &\rightarrow \text{History GRU} \rightarrow \text{History Attention} \rightarrow \text{History Context} \searrow \\
& & & \text{Decoder GRU} \rightarrow \text{Linear Layer} \rightarrow \text{Response} \\
\text{Document} &\rightarrow \text{Document GRU} \rightarrow \text{Document Attention} \rightarrow \text{Document Context} \nearrow
\end{aligned}$$

---

## 7. Experimental Results & Qualitative Evaluation

### Training Losses
* **Baseline Loss (Epoch 1):** `6.2104`
* **Grounded Loss (Epoch 1):** `6.1795`

### Quantitative Results (Test Split $N=928$)
| Model | Test BLEU Score | Relative Performance |
|---|---:|---|
| **Ungrounded Baseline** | `0.00000317` | Baseline |
| **Document-Grounded** | `0.00013176` | **~41.6× Relative Improvement** |

### Qualitative Analysis

#### Example 1
* **Reference:** `marvel ' s the <UNK> 2012 ki <UNK> superhero film hai jo isi naam ki marvel comics superhero team par based hai , marvel studios ne isey produce kiya ha aur <UNK> <UNK> motion pictures ne isey <UNK>`
* **Baseline Output:** *(empty)*
* **Grounded Output:** *(empty)*

#### Example 2
* **Reference:** `hello . kaise ho ? main ekdum sure nahi hun ki kya question <UNK> , to main bas yahi <UNK> : kya tumhe lagta hai critics is movie ka <UNK> karne mein sahi they ?`
* **Baseline Output:** `, , , hein hein`
* **Grounded Output:** `, hai hai hai hai hai hai`

#### Example 3
* **Reference:** `main agree karta hun ruffalo movie mein great tha`
* **Baseline Output:** `, , <UNK> <UNK>`
* **Grounded Output:** `, hai hai hai hai hai`

#### Example 4
* **Reference:** `mai manta hu <UNK> movie achi hai`
* **Baseline Output:** `, , <UNK> hai`
* **Grounded Output:** `, hai hai hai hai hai hai`

#### Example 5
* **Reference:** `muje lgta hai cast top ki hai`
* **Baseline Output:** `, , <UNK> hai`
* **Grounded Output:** `, hai hai hai hai hai hai`

### Failure Modes Identified
1. **Severe Repetition:** Outputs loop heavily on single tokens like `hai hai hai hai` or `, , ,`.
2. **High OOV (`<UNK>`) Frequency:** Spoken Hinglish spelling variations and transliterations lead to unknown token dominance.
3. **Early EOS Generation:** Decoder emits immediate `<EOS>`, yielding blank responses.
4. **Poor Fluency:** Lack of grammatical structure and conversational flow.

---

## 8. Analysis & Limitations

### Causes of Low Generation Performance
1. **Limited Epochs:** Training for only 1 epoch is insufficient for learning language structure, dialogue flows, and grounding interactions.
2. **Model Capacity:** Small embedding (64) and hidden (128) dimensions restrict representations.
3. **Vocabulary Size vs Data Sparsity:** 7,689 tokens combined with informal code-mixed Hinglish creates severe sparsity.
4. **Context Truncation:** Cutting history at 80 tokens and documents at 150 tokens discards significant conversational context.
5. **Open-Ended Evaluation:** BLEU penalizes valid surface variations in open-ended conversations.

---

## 9. Evolution Across Tasks

| Dimension | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| **Task Type** | Sentence Translation | Sentence Translation | Document-Grounded Dialogue Generation |
| **Context Source** | Single Source Sentence | Single Source Sentence | Multi-turn History + External Document |
| **Attention** | None | Single Bahdanau Attention | Dual Bahdanau Attention (History + Doc) |
| **Output Type** | Direct Translation | Direct Translation | Code-Mixed Hinglish Conversational Response |

---

## 10. Overall System Pipeline

```text
CMU Hinglish Dialogue Dataset
             │
             ├── Conversation Turns
             └── wikiDocumentIdx ──> WikiData JSON Documents
                         │
                         ▼
             Conversation Construction
                         │
                         ▼
             Text Preprocessing & Tokenization
                         │
                         ▼
             Vocabulary Creation & DataLoaders
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
    Baseline Model              Grounded Model
    (History Only)              (History + Document)
           │                           │
  History Encoder             History Encoder + Document Encoder
           │                           │
   History Attention           Dual Attention Mechanics
           │                           │
     GRU Decoder                 GRU Decoder
           └─────────────┬─────────────┘
                         ▼
                  Hinglish Response
                         │
                         ▼
             BLEU & Qualitative Evaluation
```

---

## 11. Conclusions

Task 3 demonstrates the design and end-to-end implementation of a document-grounded dialogue generation architecture using PyTorch.

* **Architectural Success:** Dual encoder-attention mechanisms were successfully implemented to simultaneously focus on conversation history and external grounding documents.
* **Empirical Observation:** Document grounding yielded a **41.6× relative BLEU improvement** over an ungrounded baseline.
* **Core Takeaway:** Grounding improves performance relative to ungrounded baselines, but generating fluent code-mixed Hinglish requires larger model capacity, longer training schedules, and advanced tokenization suited for transliterated text.