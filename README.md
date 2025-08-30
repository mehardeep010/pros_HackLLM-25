# pros_HackLLM-25
# KGGuard: Semantic Fidelity Graph-Enhanced Hallucination Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Solution for SHROOM-Guard Challenge 2025** 🏆  
> A novel model-agnostic hallucination detection system using Semantic Fidelity Graphs and Graph Neural Networks.

## 📖 Overview

KGGuard is an innovative hallucination detection system that transforms unstructured LLM outputs into structured knowledge representations for comprehensive verification. Unlike traditional surface-level approaches, our system performs deep structural analysis using Graph Neural Networks to identify factual, contextual, and logical inconsistencies in generated text.

### Key Features

- 🧠 **Semantic Fidelity Graphs**: Novel graph construction linking hypotheses with reference knowledge
- 🔗 **Multi-source Integration**: Combines internal references with external knowledge sources  
- 📊 **GNN Architecture**: Advanced GATv2-based learning on semantic structures
- 🎯 **Contrastive Learning**: Robust positive-negative sample discrimination
- 🔍 **Model-Agnostic**: Works with any LLM without modification
- 💡 **Explainable AI**: Graph visualizations show reasoning process

## 🏗️ Architecture

```
LLM Output → Concept Extraction → Graph Construction → SFG-GNN Model → Hallucination Score
```

Our system implements a four-stage pipeline:

1. **Concept Extraction**: Uses spaCy NLP for entity and relation extraction
2. **Graph Construction**: Builds semantic fidelity graphs with multi-source knowledge
3. **GNN Processing**: Employs Graph Attention Networks for structural analysis  
4. **Scoring**: Generates confidence scores via contrastive learning

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-team/kgguard.git
cd kgguard
```

2. **Install dependencies**
```bash
# Install core dependencies
pip install "numpy<2.0"
pip install torch==2.1.0 torch_geometric==2.4.0 --extra-index-url https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install transformers==4.31.0 datasets==2.15.0 peft==0.5.0 accelerate==0.25.0
pip install sentence-transformers==2.3.1 spacy==3.7.2 PyDictionary==2.0.1
pip install tqdm

# Download spaCy model
python -m spacy download en_core_web_sm
```

3. **Run the example**
```bash
python kgguard_demo.py
```

### 📊 Dataset Format

KGGuard supports multiple task types with the following JSON format:

```json
{
    "hyp": "Paris is the capital of France.",
    "src": "France's capital city is Paris.", 
    "task": "PG"
}
```

**Supported Tasks:**
- `PG`: Paraphrase Generation
- `MT`: Machine Translation  
- `DM`: Definition Modeling

### 🎯 Usage Examples

#### Basic Hallucination Detection

```python
from kgguard import KGGuard

# Initialize the model
detector = KGGuard()

# Single prediction
result = detector.predict(
    hypothesis="The Eiffel Tower is in Rome.",
    reference="The Eiffel Tower is in Paris.",
    task="PG"
)

print(f"Hallucination Score: {result['score']:.4f}")
print(f"Classification: {'Hallucination' if result['score'] < 0.5 else 'Factual'}")
```

#### Batch Processing

```python
# Batch predictions
samples = [
    {"hyp": "Madrid is Spain's capital", "src": "Spain's capital is Madrid", "task": "PG"},
    {"hyp": "The moon is made of cheese", "src": "The moon is rocky", "task": "PG"}
]

results = detector.predict_batch(samples)
for i, result in enumerate(results):
    print(f"Sample {i+1}: {result['score']:.4f}")
```

## 🔬 Model Architecture

### Semantic Fidelity Graph Construction

Our graphs contain three types of nodes:
- **Concept Nodes**: Extracted entities and noun phrases
- **Text Nodes**: Complete hypothesis and reference texts
- **Knowledge Nodes**: External definitions (for DM tasks)

### GNN Architecture

```python
# Graph Attention Network v2 layers
conv1 = GATv2Conv(encoder_dim, hidden_dim, heads=2, concat=True)
conv2 = GATv2Conv(hidden_dim * 2, output_dim, heads=1, concat=False)

# Final prediction layer
predictor = Sequential(
    Linear(output_dim * 2, output_dim),
    ReLU(),
    Linear(output_dim, 1)
)
```

### Training Process

- **Optimizer**: AdamW with learning rate 2e-4
- **Loss Function**: Margin-based contrastive loss
- **Training Strategy**: Positive-negative sample pairs
- **Regularization**: LoRA adaptation for efficient fine-tuning

## 📈 Performance

| Method | Precision | Recall | F1-Score |
|--------|-----------|---------|----------|
| Token Confidence | 0.65 | 0.58 | 0.61 |
| Consistency-Based | 0.71 | 0.63 | 0.67 |
| Simple Fact-Check | 0.69 | 0.61 | 0.65 |
| **KGGuard (Ours)** | **0.78** | **0.73** | **0.75** |

## 🔧 Configuration

Key hyperparameters can be adjusted in `config.py`:

```python
class Config:
    ENCODER_MODEL_NAME = "microsoft/deberta-v3-small"
    SIMILARITY_MODEL_NAME = "all-MiniLM-L6-v2"
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    GNN_HIDDEN_CHANNELS = 128
    GNN_OUT_CHANNELS = 64
    LORA_R = 8
    LORA_ALPHA = 16
```

## 📁 Project Structure

```
kgguard/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── config.py                # Configuration settings
├── kgguard/
│   ├── __init__.py
│   ├── models/
│   │   ├── sfg_gnn.py       # Main model architecture
│   │   └── graph_builder.py # Graph construction
│   ├── utils/
│   │   ├── data_utils.py    # Data processing
│   │   └── text_utils.py    # Text normalization
│   └── evaluation/
│       └── metrics.py       # Evaluation metrics
├── notebooks/
│   └── kaggle_notebook.ipynb # Complete implementation
├── examples/
│   └── demo.py              # Usage examples
└── tests/
    └── test_kgguard.py      # Unit tests
```

## 🚀 Advanced Usage

### Custom Knowledge Integration

```python
# Add custom knowledge sources
detector = KGGuard(
    knowledge_sources=['wikidata', 'custom_kb'],
    external_api_key='your_api_key'
)
```

### Model Fine-tuning

```python
# Fine-tune on domain-specific data
detector.fine_tune(
    train_data=your_dataset,
    validation_data=val_dataset,
    epochs=5
)
```

### Explainability

```python
# Get detailed explanations
result = detector.predict_with_explanation(
    hypothesis="Your text here",
    reference="Reference text",
    visualize_graph=True
)

# Visualize the semantic fidelity graph
result['graph_viz'].show()
```

## 🔬 Research & Citations

If you use KGGuard in your research, please cite:

```bibtex
@inproceedings{kgguard2025,
    title={KGGuard: Knowledge Graph-Enhanced Hallucination Detection for Large Language Models},
    author={Bhalla, Mehardeep Singh and Agrawal, Kushagra and Agrawal, Nikhil and Basistha, Anurag},
    booktitle={SHROOM-Guard Challenge 2025},
    year={2025},
    organization={IIIT Delhi}
}
```

### Development Setup

```bash
# Clone for development
git clone https://github.com/your-team/kgguard.git
cd kgguard

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## 📋 TODO & Roadmap

- [ ] **Multi-language Support**: Extend beyond English
- [ ] **Real-time Processing**: Optimize for streaming applications  
- [ ] **Domain Adaptation**: Fine-tune for specific domains
- [ ] **Temporal Validation**: Add time-aware fact checking
- [ ] **API Deployment**: REST API for production use
- [ ] **Web Interface**: User-friendly web dashboard

## ⚠️ Known Issues

1. **Memory Usage**: Large graphs may require significant GPU memory
2. **Processing Time**: Complex texts may take longer to process
3. **Knowledge Coverage**: Limited by external knowledge source availability

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SHROOM-Guard Challenge**: For providing the evaluation framework
- **Hugging Face**: For transformer models and datasets
- **PyTorch Geometric**: For graph neural network implementations
- **spaCy**: For natural language processing capabilities
- **Open Source Community**: For the tools that made this work possible

---

⭐ **Star this repository if you find it helpful!** ⭐
