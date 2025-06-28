# LongMemEval Multi-Architecture Evaluation

This directory contains three evaluation scripts for testing Memorizz's long-term memory capabilities using the LongMemEval benchmark across different agentic architectures.

## Available Evaluation Scripts

### 1. Single Agent Evaluation (`evaluate_memorizz.py`)
**Architecture**: Single Agent  
**Description**: Evaluates a single MemAgent's memory capabilities using traditional single-agent architecture.

**Key Features**:
- Single agent handles all memory tasks
- Direct conversation processing
- Baseline performance measurement
- Simple architecture for comparison

**Usage**:
```bash
python evaluate_memorizz.py --variant oracle --samples 50 --verbose
```

### 2. Delegate Pattern Evaluation (`evaluate_delegate_pattern.py`)
**Architecture**: Multi-Agent Delegate Pattern  
**Description**: Evaluates multi-agent architecture where a root agent delegates tasks to specialized agents working in parallel.

**Key Features**:
- **Root Agent**: Coordinates and delegates tasks
- **Memory Specialist**: Focuses on memory retrieval and organization
- **Temporal Specialist**: Handles time-based queries and sequencing
- **Context Integrator**: Manages cross-session analysis and patterns
- Parallel task execution
- Flat delegation structure

**Agent Structure**:
```
Root Agent (Coordinator)
├── Memory Specialist
├── Temporal Specialist
└── Context Integrator
```

**Usage**:
```bash
python evaluate_delegate_pattern.py --variant oracle --samples 50 --verbose
```

### 3. Hierarchical Pattern Evaluation (`evaluate_hierarchical_pattern.py`)
**Architecture**: Multi-Agent Hierarchical Pattern  
**Description**: Evaluates hierarchical multi-agent architecture with multiple organizational levels and specialized branches.

**Key Features**:
- **Executive Agent**: Top-level strategic coordination
- **Branch Coordinators**: Middle management for specific domains
- **Specialist Agents**: Bottom-level task execution
- Hierarchical task distribution
- Structured command chain

**Agent Hierarchy**:
```
Executive Coordinator (Top Level)
├── Memory Branch
│   ├── Memory Coordinator (Middle Level)
│   └── Memory Retrieval Specialist (Bottom Level)
└── Analysis Branch
    ├── Analysis Coordinator (Middle Level)
    ├── Temporal Analysis Specialist (Bottom Level)
    └── Context Extraction Specialist (Bottom Level)
```

**Usage**:
```bash
python evaluate_hierarchical_pattern.py --variant oracle --samples 50 --verbose
```

## Architecture Comparison

| Feature | Single Agent | Delegate Pattern | Hierarchical Pattern |
|---------|-------------|------------------|---------------------|
| **Complexity** | Low | Medium | High |
| **Specialization** | None | High | Very High |
| **Coordination** | N/A | Flat | Multi-level |
| **Scalability** | Limited | Good | Excellent |
| **Task Distribution** | None | Parallel | Hierarchical |
| **Command Structure** | Direct | Delegate | Chain of Command |

## Evaluation Metrics

All evaluation scripts measure:
- Response accuracy against ground truth
- Response time performance
- Memory utilization effectiveness
- Architecture-specific metrics

## Expected Use Cases

### Single Agent
- Baseline performance measurement
- Simple memory tasks
- Resource-constrained environments

### Delegate Pattern
- Parallel processing requirements
- Specialized task domains
- Medium complexity scenarios

### Hierarchical Pattern
- Complex organizational tasks
- Large-scale coordination
- Enterprise-level scenarios

## Running Comparative Analysis

To compare all three architectures:

```bash
# Run all evaluations
python evaluate_memorizz.py --variant oracle --samples 50 --output-dir ./results/single
python evaluate_delegate_pattern.py --variant oracle --samples 50 --output-dir ./results/delegate  
python evaluate_hierarchical_pattern.py --variant oracle --samples 50 --output-dir ./results/hierarchical

# Results will be saved with architecture identifiers for comparison
```

## Dataset Variants

All scripts support three LongMemEval variants:
- `oracle`: Full dataset with ground truth
- `s`: Short conversation variant
- `m`: Medium conversation variant

## Output Format

Each evaluation produces JSON results with:
- Architecture identification
- Detailed sample results
- Aggregate performance metrics
- Timestamp and configuration info

Results are saved in the format: `longmemeval_{architecture}_results_{variant}_{timestamp}.json` 