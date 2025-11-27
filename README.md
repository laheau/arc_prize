# ARC Prize - Abstraction and Reasoning Corpus

This repository contains implementations for solving ARC (Abstraction and Reasoning Corpus) challenges.

## 🆕 Paper Reproduction Implementation

**NEW**: A complete implementation of techniques from the research paper is now available in [`paper_reproduction/`](paper_reproduction/).

### Quick Start

```bash
# View examples and documentation
cd paper_reproduction
cat INDEX.md              # Documentation map
python examples.py        # See usage examples

# Run tests
python test.py

# Start training
python main.py
```

### Key Features
- ✅ Deep recursion with gradient detachment
- ✅ Memory-efficient training (50-70% VRAM reduction)
- ✅ Task + output vector summing
- ✅ Multiple preset configurations
- ✅ Comprehensive documentation
- ✅ Complete test suite

See [`paper_reproduction/INDEX.md`](paper_reproduction/INDEX.md) for full documentation.

## Repository Structure

```
arc_prize/
├── paper_reproduction/       # NEW: Paper techniques implementation
│   ├── INDEX.md             # Start here!
│   ├── model.py             # Deep recursive model
│   ├── train.py             # Memory-efficient training
│   ├── main.py              # Training script
│   ├── configs.py           # Preset configurations
│   └── ... (see INDEX.md)
│
├── src/                     # Original implementations
│   ├── idea.py              # Sudoku tree model
│   ├── models.py            # Various model architectures
│   ├── arc_model.py         # ARC baseline model
│   └── datasets/            # Dataset loaders
│
└── data/                    # Dataset files (symlink)
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/laheau/arc_prize.git
cd arc_prize

# Install dependencies
pip install torch pandas jupyter kaggle
```

### Using the Paper Reproduction Implementation

See [`paper_reproduction/`](paper_reproduction/) folder for:
- Complete implementation of paper techniques
- Memory-efficient deep recursion model
- Multiple configurations (quick test, standard, high performance)
- Comprehensive documentation and examples
- Test suite

### Using Original Implementations

See `src/` folder for:
- Various model architectures
- Dataset loaders for ARC tasks
- Training scripts

## Paper Reference

This repository implements techniques from:
- **Paper**: arXiv:2511.14761
- **Techniques**: Deep recursion, gradient detachment, memory-efficient training

## Resources

- **ARC Challenge**: https://github.com/fchollet/ARC-AGI
- **Paper Reproduction**: [`paper_reproduction/INDEX.md`](paper_reproduction/INDEX.md)
- **Quick Reference**: [`paper_reproduction/QUICKREF.md`](paper_reproduction/QUICKREF.md)

## License

This repository is provided for research and educational purposes.
