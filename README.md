# Text-to-SQL Generation with Language Models

This repository contains the implementation and evaluation of different approaches for natural language to SQL query conversion using both traditional fine-tuning and prompting-based methods.

## Project Overview

The project explores two main approaches to text-to-SQL generation:

1. **Fine-tuning encoder-decoder models (T5)** - Training a T5 model specifically for the text-to-SQL task
2. **In-context learning with large language models (LLMs)** - Using prompt engineering with models like Gemma 1.1 2B IT

The system is evaluated on a flight database query task with natural language queries such as "Show all flights from New York to Boston on January 1st" being translated to SQL queries.

## Dataset

The dataset consists of:

- Natural language queries (`.nl` files)
- Corresponding SQL queries (`.sql` files)
- A flight database schema (`flight_database.schema`)
- Train/dev/test splits for evaluation

## Project Structure

```
.
├── data/                       # Dataset files
│   ├── train.nl                # Training natural language queries
│   ├── train.sql               # Training SQL queries
│   ├── dev.nl                  # Development natural language queries
│   ├── dev.sql                 # Development SQL queries
│   ├── test.nl                 # Test natural language queries
│   ├── flight_database.db      # SQLite database
│   └── flight_database.schema  # Database schema
├── records/                    # Evaluation records
├── results/                    # Generated SQL files
├── evaluate.py                 # Evaluation script
├── load_data.py                # Data loading utilities
├── prompting.py                # In-context learning implementation
├── prompting_utils.py          # Utilities for prompting
├── run_experiments.sh          # Script to run all experiments
├── t5_utils.py                 # Utilities for T5 model
├── train_t5.py                 # T5 training script
└── utils.py                    # General utility functions
```

## Implemented Approaches

### 1. T5 Fine-tuning

The project implements fine-tuning of a T5 model for the text-to-SQL task. The implementation includes:

- Custom dataset and dataloader for T5 model
- Training loop with teacher forcing
- Evaluation utilities for SQL generation

### 2. In-context Learning (Prompting)

The project explores various prompting strategies:

- **Shot settings**:

  - Zero-shot: No examples provided
  - One-shot: One example provided
  - Three-shot: Three examples provided

- **Prompt types**:

  - Basic: Simple natural language to SQL conversion instruction
  - Detailed: Instructions with emphasis on SQL features (sorting, filtering, etc.)
  - Chain-of-thought: Step-by-step reasoning for SQL generation

- **Example selection strategies**:
  - Random: Randomly selected examples
  - Similar: Examples with highest word overlap with the query
  - Diverse: Examples that are diverse from each other

## Running the Experiments

### Installation

Ensure you have the required packages installed:

```bash
pip install torch transformers tqdm nltk
```

### Running Fine-tuning Experiments

To train the T5 model:

```bash
python train_t5.py
```

### Running Prompting Experiments

To run all the prompting experiments, execute:

```bash
bash run_experiments.sh
```

This will run various combinations of shot settings, prompt types, and example selection strategies.

For individual experiments, you can use:

```bash
python prompting.py --shot 0 --ptype 0 --model google/gemma-1.1-2b-it --force_cpu --template_only --use_auth_token --experiment_name "gemma_basic"
```

Parameters:

- `--shot`: Number of examples (0, 1, 3)
- `--ptype`: Prompt type (0=basic, 1=detailed, 2=chain-of-thought)
- `--model`: Model to use (default: google/gemma-1.1-2b-it)
- `--example_selection`: Example selection strategy (random, similar, diverse)

### Evaluation

To evaluate the results:

```bash
python evaluate.py --predicted_sql results/model_output.sql --predicted_records records/model_output.pkl --development_sql data/dev.sql --development_records records/dev_gt_records.pkl
```

## Experimental Results

The evaluation metrics include:

- SQL Exact Match: Percentage of generated SQL queries that exactly match the reference
- Record Exact Match: Percentage of database records returned by the generated queries that exactly match those returned by reference queries
- Record F1: F1 score between the records returned by generated and reference queries

### Performance Summary

1. **T5 Fine-tuned Model**:

   - SQL EM: ~45-50%
   - Record EM: ~65-70%
   - Record F1: ~75-80%

2. **Gemma 1.1 2B IT (Zero-shot, Basic Prompt)**:

   - SQL EM: ~35-40%
   - Record EM: ~50-55%
   - Record F1: ~65-70%

3. **Gemma 1.1 2B IT (Few-shot, Detailed Prompt)**:
   - SQL EM: ~40-45%
   - Record EM: ~60-65%
   - Record F1: ~70-75%

The exact results may vary slightly across different runs.

## Conclusion

The project demonstrates that while fine-tuned T5 models generally perform better in terms of exact SQL match, prompting-based approaches with LLMs like Gemma can achieve competitive performance with appropriate prompt engineering, especially when using few-shot learning with carefully selected examples.

For detailed analysis and results, please refer to the Report.pdf document.
