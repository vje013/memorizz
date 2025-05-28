# LongMemEval Evaluation for Memorizz

This directory contains the implementation for evaluating Memorizz using the LongMemEval benchmark.

## About LongMemEval

LongMemEval is a comprehensive benchmark designed to evaluate the long-term memory capabilities of chat assistants. It consists of 500 human-curated questions that test five core memory abilities across scalable chat histories.

### Evaluation Categories

1. **Information Extraction (IE)**: Ability to recall specific information from extensive interactive histories
2. **Multi-Session Reasoning (MR)**: Ability to synthesize information across multiple history sessions  
3. **Knowledge Updates (KU)**: Ability to recognize changes in user information and update knowledge dynamically
4. **Temporal Reasoning (TR)**: Awareness of temporal aspects of user information
5. **Abstention (ABS)**: Ability to refrain from answering questions with unknown information

### Dataset Variants

- **LongMemEval-S**: ~115k tokens per problem (suitable for most models)
- **LongMemEval-M**: ~1.5M tokens per problem (extreme long-context evaluation)
- **LongMemEval-Oracle**: Only evidence sessions included (best-case baseline)

## Usage

### Prerequisites

```bash
pip install datasets transformers openai
```

### Environment Setup

```bash
export OPENAI_API_KEY="your_openai_api_key"
export MONGODB_URI="your_mongodb_uri"
```

### Running the Evaluation

```bash
python evaluate_memorizz.py [options]
```

### Options

- `--dataset_variant`: Choose from `oracle`, `s`, or `m` (default: `oracle`)
- `--num_samples`: Number of samples to evaluate (default: 50)
- `--memory_mode`: Memorizz memory mode to use (default: `general`)
- `--output_dir`: Directory to save results (default: `./results`)
- `--verbose`: Enable verbose logging

### Example

```bash
# Quick evaluation on oracle dataset
python evaluate_memorizz.py --dataset_variant oracle --num_samples 10

# Full evaluation on LongMemEval-S
python evaluate_memorizz.py --dataset_variant s --num_samples 100

# Memory mode comparison
python evaluate_memorizz.py --memory_mode conversation --num_samples 25
```

## Results

Results are saved as JSON files with the following structure:

```json
{
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "dataset_variant": "oracle",
    "num_samples": 50,
    "memory_mode": "general"
  },
  "overall_accuracy": 0.76,
  "category_results": {
    "information_extraction": 0.85,
    "multi_session_reasoning": 0.72,
    "knowledge_updates": 0.68,
    "temporal_reasoning": 0.71,
    "abstention": 0.84
  },
  "detailed_results": [...]
}
```

## Implementation Details

The evaluation script:

1. Loads the LongMemEval dataset from HuggingFace
2. Creates a fresh Memorizz agent for each evaluation run
3. Processes chat histories session by session to simulate real interactions
4. Tests the agent's ability to answer questions based on accumulated memory
5. Evaluates responses using GPT-4 as a judge
6. Generates comprehensive performance metrics

## Performance Tips

- Use `oracle` dataset for quick testing and development
- Start with smaller `num_samples` to validate the setup
- Consider using `s` variant for comprehensive evaluation
- Monitor memory usage as chat histories can be very long
- Results include response times to help optimize performance

## Interpreting Results

- **Information Extraction**: Basic memory recall capability
- **Multi-Session Reasoning**: Advanced memory synthesis across time
- **Knowledge Updates**: Dynamic memory management
- **Temporal Reasoning**: Time-aware memory retrieval
- **Abstention**: Confidence and uncertainty handling

Higher scores indicate better memory performance in each category. 