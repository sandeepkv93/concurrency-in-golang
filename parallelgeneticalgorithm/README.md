# Parallel Genetic Algorithm

A high-performance, parallel implementation of genetic algorithms in Go, designed to solve optimization problems using evolutionary computation techniques with concurrent processing.

## Features

### Core Genetic Algorithm
- **Population-based Evolution**: Maintains and evolves populations of candidate solutions
- **Parallel Fitness Evaluation**: Concurrent evaluation of individuals using worker pools
- **Multiple Selection Methods**: Tournament selection with configurable tournament size
- **Crossover Operations**: Single-point crossover and uniform crossover implementations
- **Mutation Strategies**: Standard mutation and Gaussian mutation with adaptive parameters
- **Elitism**: Preserves best individuals across generations

### Advanced Parallel Features
- **Worker Pool Architecture**: Configurable number of workers for parallel fitness evaluation
- **Island Model**: Multiple isolated populations with periodic migration
- **Concurrent Evolution**: Parallel processing of genetic operations
- **Context Support**: Cancellation and timeout management for long-running optimizations
- **Dynamic Load Balancing**: Efficient distribution of fitness evaluation tasks

### Optimization Functions
- **Sphere Function**: Simple quadratic optimization benchmark
- **Rastrigin Function**: Highly multimodal function with many local optima
- **Rosenbrock Function**: Non-convex function with global optimum in narrow valley
- **Ackley Function**: Multimodal function with exponential and cosine components
- **Schwefel Function**: Deceptive function with global optimum far from local optima
- **Griewank Function**: Multimodal function with product and sum components

### Performance Optimizations
- **Stagnation Detection**: Automatic termination when evolution plateaus
- **Target Fitness**: Early termination when desired fitness is reached
- **Memory Efficiency**: Optimized data structures for large populations
- **Statistics Tracking**: Real-time monitoring of evolution progress
- **History Management**: Complete evolution history for analysis

## Usage Examples

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "time"
    
    "github.com/yourusername/concurrency-in-golang/parallelgeneticalgorithm"
)

func main() {
    config := parallelgeneticalgorithm.GAConfig{
        PopulationSize:   100,
        ChromosomeLength: 10,
        MutationRate:     0.1,
        CrossoverRate:    0.8,
        ElitismRate:      0.1,
        MaxGenerations:   200,
        NumWorkers:       8,
        MinGene:          -5.0,
        MaxGene:          5.0,
        TargetFitness:    -0.01,
        StagnationLimit:  50,
        RandomSeed:       42,
    }
    
    ga := parallelgeneticalgorithm.NewGeneticAlgorithm(config, parallelgeneticalgorithm.SphereFunction)
    
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
    defer cancel()
    
    best, err := ga.Run(ctx)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Printf("Best solution: %v\n", best.Genes)
    fmt.Printf("Best fitness: %f\n", best.Fitness)
}
```

### Island Model Evolution

```go
config := parallelgeneticalgorithm.GAConfig{
    PopulationSize:   50,
    ChromosomeLength: 20,
    MutationRate:     0.15,
    CrossoverRate:    0.85,
    ElitismRate:      0.05,
    MaxGenerations:   500,
    NumWorkers:       4,
    MinGene:          -10.0,
    MaxGene:          10.0,
    RandomSeed:       time.Now().UnixNano(),
}

ga := parallelgeneticalgorithm.NewGeneticAlgorithm(config, parallelgeneticalgorithm.RastriginFunction)

ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()

numIslands := 4
migrationRate := 0.1
migrationInterval := 10

best, err := ga.RunIslandModel(ctx, numIslands, migrationRate, migrationInterval)
if err != nil {
    fmt.Printf("Error: %v\n", err)
    return
}

fmt.Printf("Island model best solution: %v\n", best.Genes)
fmt.Printf("Island model best fitness: %f\n", best.Fitness)
```

### Custom Fitness Function

```go
func customFitness(genes []float64) float64 {
    // Example: Maximize sum of squares with penalty for values outside [-1, 1]
    sum := 0.0
    penalty := 0.0
    
    for _, gene := range genes {
        sum += gene * gene
        if gene < -1.0 || gene > 1.0 {
            penalty += 10.0 * math.Abs(gene)
        }
    }
    
    return sum - penalty
}

config := parallelgeneticalgorithm.GAConfig{
    PopulationSize:   80,
    ChromosomeLength: 5,
    MutationRate:     0.2,
    CrossoverRate:    0.7,
    ElitismRate:      0.15,
    MaxGenerations:   150,
    NumWorkers:       6,
    MinGene:          -2.0,
    MaxGene:          2.0,
    TargetFitness:    4.5,
    StagnationLimit:  30,
}

ga := parallelgeneticalgorithm.NewGeneticAlgorithm(config, customFitness)
```

### Evolution Monitoring

```go
ga := parallelgeneticalgorithm.NewGeneticAlgorithm(config, parallelgeneticalgorithm.AckleyFunction)

ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
defer cancel()

best, err := ga.Run(ctx)
if err != nil {
    fmt.Printf("Error: %v\n", err)
}

// Get evolution statistics
stats := ga.GetStats()
fmt.Printf("Evolution Statistics:\n")
fmt.Printf("Generations: %v\n", stats["generations"])
fmt.Printf("Best Fitness: %v\n", stats["best_fitness"])
fmt.Printf("Average Fitness: %v\n", stats["avg_fitness"])
fmt.Printf("Population Size: %v\n", stats["population_size"])
fmt.Printf("Workers Used: %v\n", stats["num_workers"])

// Get complete evolution history
history := ga.GetHistory()
for i, generation := range history {
    fmt.Printf("Generation %d: Best=%.6f, Avg=%.6f\n", 
        i, generation.BestFitness, generation.AvgFitness)
}
```

### Benchmark Functions Comparison

```go
functions := map[string]parallelgeneticalgorithm.FitnessFunction{
    "Sphere":     parallelgeneticalgorithm.SphereFunction,
    "Rastrigin":  parallelgeneticalgorithm.RastriginFunction,
    "Rosenbrock": parallelgeneticalgorithm.RosenbrockFunction,
    "Ackley":     parallelgeneticalgorithm.AckleyFunction,
    "Schwefel":   parallelgeneticalgorithm.SchwefelFunction,
    "Griewank":   parallelgeneticalgorithm.GriewankFunction,
}

for name, fn := range functions {
    fmt.Printf("Testing %s function:\n", name)
    
    config := parallelgeneticalgorithm.GAConfig{
        PopulationSize:   100,
        ChromosomeLength: 10,
        MutationRate:     0.1,
        CrossoverRate:    0.8,
        ElitismRate:      0.1,
        MaxGenerations:   200,
        NumWorkers:       8,
        MinGene:          -5.0,
        MaxGene:          5.0,
        StagnationLimit:  50,
    }
    
    ga := parallelgeneticalgorithm.NewGeneticAlgorithm(config, fn)
    
    ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
    best, err := ga.Run(ctx)
    cancel()
    
    if err != nil {
        fmt.Printf("  Error: %v\n", err)
    } else {
        fmt.Printf("  Best fitness: %f\n", best.Fitness)
    }
}
```

## Architecture

### Core Components

1. **GeneticAlgorithm**: Main algorithm controller with configuration and state
2. **Individual**: Represents a candidate solution with genes and fitness
3. **Population**: Collection of individuals with generation metadata
4. **GAConfig**: Configuration structure for algorithm parameters
5. **FitnessFunction**: Interface for objective function evaluation

### Parallel Processing

- **Worker Pool**: Distributes fitness evaluation across multiple goroutines
- **Job Queue**: Channel-based task distribution for parallel operations
- **Synchronization**: Mutex-based protection for shared state
- **Context Management**: Cancellation and timeout handling throughout

### Evolution Strategies

#### Selection Methods
- **Tournament Selection**: Selects best individual from random tournament
- **Configurable Tournament Size**: Adjustable selection pressure

#### Crossover Operations
- **Single-Point Crossover**: Exchanges genes at random crossover point
- **Uniform Crossover**: Randomly selects genes from either parent
- **Crossover Rate**: Probability of applying crossover vs. copying parent

#### Mutation Strategies
- **Standard Mutation**: Replaces genes with random values
- **Gaussian Mutation**: Applies normally distributed perturbations
- **Adaptive Mutation**: Configurable mutation rates and strategies

#### Population Management
- **Elitism**: Preserves best individuals across generations
- **Generation Replacement**: Replaces population with evolved offspring
- **Population Size**: Configurable number of individuals per generation

### Island Model Architecture

- **Multiple Populations**: Independent evolution on separate islands
- **Periodic Migration**: Exchange of best individuals between islands
- **Parallel Island Processing**: Concurrent evolution across all islands
- **Migration Parameters**: Configurable migration rate and interval

## Configuration Parameters

```go
type GAConfig struct {
    PopulationSize    int     // Number of individuals in population
    ChromosomeLength  int     // Number of genes per individual
    MutationRate      float64 // Probability of gene mutation
    CrossoverRate     float64 // Probability of crossover operation
    ElitismRate       float64 // Fraction of best individuals to preserve
    MaxGenerations    int     // Maximum number of generations
    NumWorkers        int     // Number of parallel workers
    MinGene           float64 // Minimum gene value
    MaxGene           float64 // Maximum gene value
    TargetFitness     float64 // Target fitness for early termination
    StagnationLimit   int     // Generations without improvement before stopping
    RandomSeed        int64   // Random seed for reproducible results
}
```

## Testing

Run the comprehensive test suite:

```bash
go test -v ./parallelgeneticalgorithm/
```

Run benchmarks:

```bash
go test -bench=. ./parallelgeneticalgorithm/
```

### Test Coverage

- Basic genetic algorithm operations
- Parallel fitness evaluation
- Crossover and mutation operations
- Selection mechanisms
- Island model evolution
- Context cancellation and timeout handling
- Stagnation detection
- Statistics and history tracking
- All benchmark functions
- Performance benchmarks

## Performance Characteristics

### Scalability
- **Population Size**: 10-10000 individuals (memory dependent)
- **Chromosome Length**: 1-1000 genes (problem dependent)
- **Worker Count**: 1-CPU cores (optimal at CPU count)
- **Generations**: 10-10000 (convergence dependent)

### Memory Usage
- **Per Individual**: ~8 bytes * chromosome length
- **Per Population**: ~(population size * chromosome length * 8) bytes
- **History Storage**: ~(generations * population size * chromosome length * 8) bytes

### Convergence Rates
- **Simple Functions**: 10-100 generations
- **Complex Functions**: 100-1000 generations
- **Multimodal Functions**: 500-5000 generations

## Use Cases

1. **Function Optimization**: Finding global optima of complex mathematical functions
2. **Parameter Tuning**: Optimizing hyperparameters for machine learning models
3. **Neural Network Evolution**: Evolving neural network architectures and weights
4. **Scheduling Problems**: Optimizing resource allocation and task scheduling
5. **Feature Selection**: Selecting optimal feature subsets for classification
6. **Game AI**: Evolving strategies and behaviors for game agents
7. **Engineering Design**: Optimizing engineering parameters and configurations
8. **Portfolio Optimization**: Finding optimal asset allocation strategies

## Advanced Features

### Adaptive Parameters
- **Dynamic Mutation Rates**: Adjust mutation based on convergence progress
- **Adaptive Crossover**: Modify crossover strategy based on population diversity
- **Population Sizing**: Automatic population size adjustment

### Diversity Management
- **Diversity Metrics**: Measure population genetic diversity
- **Diversity Preservation**: Maintain population diversity to avoid premature convergence
- **Niching Techniques**: Support for multiple species within populations

### Hybrid Approaches
- **Local Search Integration**: Combine with hill-climbing or simulated annealing
- **Multi-Objective Optimization**: Support for Pareto-optimal solutions
- **Constraint Handling**: Penalty functions and constraint satisfaction methods