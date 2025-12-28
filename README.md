# KGGuard: Semantic Fidelity Graph-Enhanced Hallucination Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HackLLM'25 ESYA**

Detects hallucinations in LLM outputs by converting text into knowledge graphs and analyzing them with Graph Neural Networks.

## What It Does

KGGuard catches when language models make stuff up. Instead of just checking surface-level text similarity, it extracts concepts from both the LLM output and reference text, builds a knowledge graph connecting them, then uses a GNN to spot inconsistencies.

Think of it like fact-checking with a visual map: if the LLM says "Paris is in Germany," our graph connects Paris→Germany, but also Paris→France from the reference. The GNN notices this conflict.

**Core approach:**
1. Extract entities and relations from text (spaCy)
2. Build semantic fidelity graphs linking hypothesis to reference
3. Run GATv2 neural network on the graph structure
4. Output hallucination probability

Works with any LLM. No model modification needed.

## Installation

```bash
git clone https://github.com/pros/kgguard.git
cd kgguard

# Core dependencies 
pip install "numpy<2.0"
pip install torch==2.1.0 torch_geometric==2.4.0 --extra-index-url https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install transformers==4.31.0 datasets==2.15.0 peft==0.5.0 accelerate==0.25.0
pip install sentence-transformers==2.3.1 spacy==3.7.2 PyDictionary==2.0.1 tqdm

# Language model
python -m spacy download en_core_web_sm
```

**Requirements:** Python 3.8+, GPU recommended, 8GB+ RAM

## Quick Start

```python
from kgguard import KGGuard

detector = KGGuard()

result = detector.predict(
    hypothesis="The Eiffel Tower is in Rome.",
    reference="The Eiffel Tower is in Paris.",
    task="PG"
)

print(f"Score: {result['score']:.4f}")  
```

### Supported Tasks

- `PG`: Paraphrase Generation
- `MT`: Machine Translation  
- `DM`: Definition Modeling

### Input Format

```json
{
    "hyp": "Generated text from LLM",
    "src": "Reference or source text", 
    "task": "PG"
}
```

## Architecture

**Pipeline:** Text → Concept Extraction → Graph Building → GNN Analysis → Score

**Graph Structure:**
- Concept nodes (entities, noun phrases)
- Text nodes (full hypothesis/reference)
- Knowledge nodes (external definitions for DM tasks)

**GNN Model:**
```python
# Two-layer GATv2 (Graph Attention Network)
Layer 1: GATv2Conv(encoder_dim → hidden_dim, 2 heads)
Layer 2: GATv2Conv(hidden_dim*2 → output_dim, 1 head)
Predictor: Linear layers → hallucination score
```

**Training:** Contrastive learning with positive/negative pairs, AdamW optimizer, 2e-4 learning rate

## Performance

| Approach | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Token confidence | 0.65 | 0.58 | 0.61 |
| Consistency checks | 0.71 | 0.63 | 0.67 |
| Basic fact-check | 0.69 | 0.61 | 0.65 |
| **KGGuard** | **0.78** | **0.73** | **0.75** |

## Configuration

Edit `config.py` for custom settings:

```python
ENCODER_MODEL_NAME = "microsoft/deberta-v3-small"
SIMILARITY_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
GNN_HIDDEN_CHANNELS = 128
GNN_OUT_CHANNELS = 64
```

## Project Structure

```
kgguard/
├── config.py                # Settings
├── kgguard/
│   ├── models/
│   │   ├── sfg_gnn.py       # GNN model
│   │   └── graph_builder.py # Graph construction
│   ├── utils/               # Data and text processing
│   └── evaluation/          # Metrics
├── notebooks/               # Kaggle implementation
└── tests/                   # Unit tests
```

## Advanced Usage

**Batch processing:**
```python
samples = [
    {"hyp": "Madrid is Spain's capital", "src": "Spain's capital is Madrid", "task": "PG"},
    {"hyp": "Moon is cheese", "src": "Moon is rocky", "task": "PG"}
]
results = detector.predict_batch(samples)
```

**Fine-tuning:**
```python
detector.fine_tune(
    train_data=your_dataset,
    validation_data=val_dataset,
    epochs=5
)
```

**Graph visualization:**
```python
result = detector.predict_with_explanation(
    hypothesis="Your text",
    reference="Reference",
    visualize_graph=True
)
result['graph_viz'].show()
```

## Known Limitations

- Large graphs eat GPU memory
- Complex text takes time to process
- Accuracy depends on external knowledge availability
- English only (for now)

## Roadmap

- Multi-language support
- Streaming/real-time mode
- Domain-specific fine-tuning
- Time-aware fact checking
- REST API deployment

## Citation

```bibtex
@inproceedings{kgguard2025,
    title={KGGuard: Knowledge Graph-Enhanced Hallucination Detection for Large Language Models},
    author={Bhalla, Mehardeep Singh and Agrawal, Kushagra and Agrawal, Nikhil and Basistha, Anurag},
    booktitle={SHROOM-Guard Challenge 2025},
    year={2025},
    organization={IIIT Delhi}
}
```

## License

MIT License - see LICENSE file

## Built With

PyTorch Geometric • Hugging Face Transformers • spaCy • Sentence Transformers

---

**Team:** Mehardeep Singh Bhalla, Kushagra Agrawal, Nikhil Agrawal, Anurag Basistha  
**Affiliation:** IIIT Delhi
