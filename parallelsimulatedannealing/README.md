# Parallel Simulated Annealing

A high-performance, parallel implementation of the Simulated Annealing optimization algorithm in Go, featuring multiple cooling schedules, perturbation strategies, parallelization approaches, and advanced optimization techniques for solving complex continuous and discrete optimization problems.

## Features

### Core Optimization Algorithm
- **Simulated Annealing**: Classic metaheuristic inspired by metallurgical annealing process
- **Multiple Cooling Schedules**: Linear, exponential, logarithmic, adaptive, geometric, quadratic, cosine annealing
- **Advanced Acceptance Criteria**: Temperature-dependent probabilistic acceptance of worse solutions
- **Constraint Handling**: Automatic constraint enforcement for bounded optimization problems
- **Adaptive Parameters**: Self-adjusting step sizes and cooling rates based on search progress
- **Memory Management**: Elite solution preservation and search history tracking

### Perturbation Strategies
- **Gaussian Perturbation**: Normal distribution-based neighbor generation
- **Uniform Perturbation**: Uniform random perturbations within specified bounds
- **Cauchy Perturbation**: Heavy-tailed Cauchy distribution for global exploration
- **Levy Flight Perturbation**: Levy flight random walk for enhanced exploration
- **Adaptive Perturbation**: Dynamic adjustment of perturbation parameters
- **Hybrid Perturbation**: Combination of multiple perturbation strategies

### Parallel Processing Strategies
- **Independent Chains**: Multiple independent simulated annealing runs
- **Temperature Parallel**: Parallel chains with different temperature schedules
- **Multiple Restart**: Periodic restart of chains with best solutions preserved
- **Island Model**: Population-based approach with solution migration between islands
- **Hybrid Parallel**: Combination of independent and cooperative strategies
- **Cooperative Chains**: Information sharing between parallel search processes

### Advanced Features
- **Real-time Statistics**: Comprehensive performance monitoring and convergence tracking
- **Early Termination**: Configurable stopping criteria based on convergence or stagnation
- **Solution Exchange**: Inter-chain communication and best solution sharing
- **Diversification Control**: Mechanisms to maintain population diversity
- **Elitism**: Preservation of best solutions across generations and restarts
- **Context Support**: Cancellation and timeout handling for long-running optimizations

## Architecture Overview

### Algorithm Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Parallel Simulated Annealing            │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Chain 1   │  │   Chain 2   │  │   Chain N   │        │
│  │             │  │             │  │             │        │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │        │
│  │ │Current  │ │  │ │Current  │ │  │ │Current  │ │        │
│  │ │Solution │ │  │ │Solution │ │  │ │Solution │ │        │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │        │
│  │             │  │             │  │             │        │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │        │
│  │ │  Best   │ │  │ │  Best   │ │  │ │  Best   │ │        │
│  │ │Solution │ │  │ │Solution │ │  │ │Solution │ │        │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │        │
│  │             │  │             │  │             │        │
│  │ Temperature │  │ Temperature │  │ Temperature │        │
│  │   Control   │  │   Control   │  │   Control   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                 │                 │             │
│         └─────────────────┼─────────────────┘             │
│                           │                               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Solution Exchange Manager              │  │
│  │                                                     │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │  │
│  │  │   Island    │ │   Restart   │ │  Statistics │   │  │
│  │  │ Migration   │ │  Manager    │ │  Collector  │   │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘   │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                    Global Best Solution                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │        Convergence & Performance Statistics         │   │
│  │                                                     │   │
│  │ • Fitness History      • Temperature Evolution     │   │
│  │ • Acceptance Rates     • Diversity Measures        │   │
│  │ • Chain Performance    • Exchange Statistics       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Parallel Strategy Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| Independent Chains | Simple, no overhead | No information sharing | Embarrassingly parallel problems |
| Temperature Parallel | Different exploration patterns | Limited diversity | Multi-modal landscapes |
| Multiple Restart | Avoids local optima | Computational overhead | Rugged fitness landscapes |
| Island Model | Balance exploration/exploitation | Complex implementation | Large-scale optimization |
| Hybrid Parallel | Best of multiple strategies | Most complex | General-purpose optimization |
| Cooperative Chains | Guided search | Synchronization overhead | Problems requiring coordination |

## Usage Examples

### Basic Optimization

```go
package main

import (
    "fmt"
    "log"
    "math"
    
    "github.com/yourusername/concurrency-in-golang/parallelsimulatedannealing"
)

// Define objective function (minimize this function)
func sphereFunction(x []float64) float64 {
    sum := 0.0
    for _, xi := range x {
        sum += xi * xi
    }
    return sum
}

func main() {
    // Create configuration
    config := parallelsimulatedannealing.DefaultSAConfig()
    config.Dimensions = 5
    config.MaxIterations = 10000
    config.InitialTemperature = 100.0
    config.FinalTemperature = 0.01
    config.LowerBound = []float64{-10, -10, -10, -10, -10}
    config.UpperBound = []float64{10, 10, 10, 10, 10}
    
    // Create optimizer
    psa, err := parallelsimulatedannealing.NewParallelSimulatedAnnealing(config, sphereFunction)
    if err != nil {
        log.Fatalf("Failed to create optimizer: %v", err)
    }
    
    // Run optimization
    solution, err := psa.Optimize()
    if err != nil {
        log.Fatalf("Optimization failed: %v", err)
    }
    
    fmt.Printf("Best solution found:\n")
    fmt.Printf("  Variables: %v\n", solution.Variables)
    fmt.Printf("  Fitness: %f\n", solution.Fitness)
    fmt.Printf("  Iteration: %d\n", solution.Iteration)
}
```

### Advanced Configuration

```go
func advancedOptimization() {
    // Advanced configuration for complex optimization
    config := parallelsimulatedannealing.SAConfig{
        Dimensions:          20,
        LowerBound:         make([]float64, 20), // Will be set to -100
        UpperBound:         make([]float64, 20), // Will be set to 100
        InitialTemperature: 1000.0,
        FinalTemperature:   0.001,
        MaxIterations:      50000,
        CoolingSchedule:    parallelsimulatedannealing.AdaptiveCooling,
        CoolingRate:        0.95,
        PerturbationStrat:  parallelsimulatedannealing.HybridPerturbation,
        PerturbationSize:   2.0,
        ParallelStrategy:   parallelsimulatedannealing.IslandModel,
        NumWorkers:         8,
        NumChains:          16,
        RestartInterval:    5000,
        IslandExchangeRate: 1000,
        AdaptiveParameters: true,
        EnableMemory:       true,
        EnableStatistics:   true,
        EnableLogging:      true,
        Tolerance:          1e-8,
        StagnationLimit:    3000,
        ElitismRate:        0.2,
        DiversityThreshold: 0.05,
        RandomSeed:         12345,
    }
    
    // Initialize bounds
    for i := range config.LowerBound {
        config.LowerBound[i] = -100.0
        config.UpperBound[i] = 100.0
    }
    
    psa, err := parallelsimulatedannealing.NewParallelSimulatedAnnealing(config, rosenbrockFunction)
    if err != nil {
        log.Fatalf("Failed to create advanced optimizer: %v", err)
    }
    
    solution, err := psa.Optimize()
    if err != nil {
        log.Fatalf("Advanced optimization failed: %v", err)
    }
    
    fmt.Printf("Advanced optimization result: fitness=%f\n", solution.Fitness)
}

// Rosenbrock function - classic optimization benchmark
func rosenbrockFunction(x []float64) float64 {
    sum := 0.0
    for i := 0; i < len(x)-1; i++ {
        term1 := 100 * (x[i+1] - x[i]*x[i]) * (x[i+1] - x[i]*x[i])
        term2 := (1 - x[i]) * (1 - x[i])
        sum += term1 + term2
    }
    return sum
}
```

### Multiple Strategy Comparison

```go
func compareStrategies() {
    strategies := []struct {
        name     string
        strategy parallelsimulatedannealing.ParallelStrategy
    }{
        {"Independent Chains", parallelsimulatedannealing.IndependentChains},
        {"Temperature Parallel", parallelsimulatedannealing.TemperatureParallel},
        {"Island Model", parallelsimulatedannealing.IslandModel},
        {"Hybrid Parallel", parallelsimulatedannealing.HybridParallel},
        {"Cooperative Chains", parallelsimulatedannealing.CooperativeChains},
    }
    
    // Test function: Rastrigin (highly multimodal)
    rastrigin := func(x []float64) float64 {
        A := 10.0
        n := float64(len(x))
        sum := A * n
        for _, xi := range x {
            sum += xi*xi - A*math.Cos(2*math.Pi*xi)
        }
        return sum
    }
    
    for _, s := range strategies {
        config := parallelsimulatedannealing.DefaultSAConfig()
        config.Dimensions = 10
        config.MaxIterations = 20000
        config.ParallelStrategy = s.strategy
        config.NumChains = 8
        config.LowerBound = make([]float64, 10)
        config.UpperBound = make([]float64, 10)
        
        for i := range config.LowerBound {
            config.LowerBound[i] = -5.12
            config.UpperBound[i] = 5.12
        }
        
        psa, err := parallelsimulatedannealing.NewParallelSimulatedAnnealing(config, rastrigin)
        if err != nil {
            log.Printf("Failed to create optimizer for %s: %v", s.name, err)
            continue
        }
        
        start := time.Now()
        solution, err := psa.Optimize()
        duration := time.Since(start)
        
        if err != nil {
            log.Printf("Optimization failed for %s: %v", s.name, err)
            continue
        }
        
        fmt.Printf("%s Strategy:\n", s.name)
        fmt.Printf("  Best Fitness: %f\n", solution.Fitness)
        fmt.Printf("  Duration: %v\n", duration)
        fmt.Printf("  Iterations: %d\n", solution.Iteration)
        
        // Get detailed statistics
        stats := psa.GetStatistics()
        acceptanceRate := float64(stats.TotalAcceptances) / 
            float64(stats.TotalAcceptances+stats.TotalRejections+1) * 100
        fmt.Printf("  Acceptance Rate: %.1f%%\n", acceptanceRate)
        fmt.Printf("  Restarts: %d\n", stats.RestartCount)
        fmt.Printf("  Exchanges: %d\n", stats.ExchangeCount)
        fmt.Println()
    }
}
```

### Cooling Schedule Comparison

```go
func compareCoolingSchedules() {
    schedules := []struct {
        name     string
        schedule parallelsimulatedannealing.CoolingSchedule
    }{
        {"Linear", parallelsimulatedannealing.LinearCooling},
        {"Exponential", parallelsimulatedannealing.ExponentialCooling},
        {"Logarithmic", parallelsimulatedannealing.LogarithmicCooling},
        {"Adaptive", parallelsimulatedannealing.AdaptiveCooling},
        {"Geometric", parallelsimulatedannealing.GeometricCooling},
        {"Cosine", parallelsimulatedannealing.CosineAnnealing},
    }
    
    // Ackley function - another classic benchmark
    ackley := func(x []float64) float64 {
        n := float64(len(x))
        sum1 := 0.0
        sum2 := 0.0
        
        for _, xi := range x {
            sum1 += xi * xi
            sum2 += math.Cos(2 * math.Pi * xi)
        }
        
        term1 := -20 * math.Exp(-0.2*math.Sqrt(sum1/n))
        term2 := -math.Exp(sum2/n)
        return term1 + term2 + 20 + math.E
    }
    
    for _, s := range schedules {
        config := parallelsimulatedannealing.DefaultSAConfig()
        config.Dimensions = 5
        config.MaxIterations = 15000
        config.CoolingSchedule = s.schedule
        config.InitialTemperature = 100.0
        config.FinalTemperature = 0.01
        config.LowerBound = []float64{-32, -32, -32, -32, -32}
        config.UpperBound = []float64{32, 32, 32, 32, 32}
        
        psa, err := parallelsimulatedannealing.NewParallelSimulatedAnnealing(config, ackley)
        if err != nil {
            log.Printf("Failed to create optimizer for %s cooling: %v", s.name, err)
            continue
        }
        
        solution, err := psa.Optimize()
        if err != nil {
            log.Printf("Optimization failed for %s cooling: %v", s.name, err)
            continue
        }
        
        fmt.Printf("%s Cooling:\n", s.name)
        fmt.Printf("  Final Fitness: %f\n", solution.Fitness)
        fmt.Printf("  Solution: %v\n", solution.Variables)
        
        // Analyze convergence
        stats := psa.GetStatistics()
        if len(stats.ConvergenceHistory) > 0 {
            initialFitness := stats.ConvergenceHistory[0]
            improvement := initialFitness - solution.Fitness
            fmt.Printf("  Improvement: %f\n", improvement)
        }
        fmt.Println()
    }
}
```

### Real-time Monitoring

```go
func monitorOptimization() {
    config := parallelsimulatedannealing.DefaultSAConfig()
    config.Dimensions = 8
    config.MaxIterations = 30000
    config.EnableStatistics = true
    config.NumChains = 4
    
    psa, err := parallelsimulatedannealing.NewParallelSimulatedAnnealing(config, rosenbrockFunction)
    if err != nil {
        log.Fatalf("Failed to create optimizer: %v", err)
    }
    
    // Start optimization in background
    done := make(chan struct{})
    var finalSolution *parallelsimulatedannealing.Solution
    
    go func() {
        defer close(done)
        finalSolution, _ = psa.Optimize()
    }()
    
    // Monitor progress
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()
    
    fmt.Println("Monitoring optimization progress...")
    fmt.Println("Time\tBest Fitness\tAcceptance Rate\tTemperature")
    fmt.Println("----\t------------\t---------------\t-----------")
    
    startTime := time.Now()
    
    for {
        select {
        case <-done:
            fmt.Printf("\nOptimization completed!\n")
            fmt.Printf("Final solution: fitness=%f, vars=%v\n", 
                finalSolution.Fitness, finalSolution.Variables)
            return
            
        case <-ticker.C:
            current := psa.GetBestSolution()
            stats := psa.GetStatistics()
            
            elapsed := time.Since(startTime)
            acceptanceRate := 0.0
            if stats.TotalAcceptances+stats.TotalRejections > 0 {
                acceptanceRate = float64(stats.TotalAcceptances) / 
                    float64(stats.TotalAcceptances+stats.TotalRejections) * 100
            }
            
            avgTemp := stats.AverageTemperature
            if avgTemp == 0 && len(psa.chains) > 0 {
                // Calculate current average temperature
                tempSum := 0.0
                for _, chain := range psa.chains {
                    tempSum += chain.Temperature
                }
                avgTemp = tempSum / float64(len(psa.chains))
            }
            
            fmt.Printf("%.0fs\t%.6f\t\t%.1f%%\t\t%.3f\n", 
                elapsed.Seconds(), current.Fitness, acceptanceRate, avgTemp)
        }
    }
}
```

### Custom Objective Functions

```go
// Traveling Salesman Problem (TSP) example
func tspExample() {
    // Distance matrix for 10 cities
    distances := [][]float64{
        {0, 10, 15, 20, 25, 30, 35, 40, 45, 50},
        {10, 0, 12, 18, 22, 28, 32, 38, 42, 48},
        {15, 12, 0, 14, 19, 24, 29, 34, 39, 44},
        {20, 18, 14, 0, 16, 21, 26, 31, 36, 41},
        {25, 22, 19, 16, 0, 17, 22, 27, 32, 37},
        {30, 28, 24, 21, 17, 0, 18, 23, 28, 33},
        {35, 32, 29, 26, 22, 18, 0, 19, 24, 29},
        {40, 38, 34, 31, 27, 23, 19, 0, 20, 25},
        {45, 42, 39, 36, 32, 28, 24, 20, 0, 21},
        {50, 48, 44, 41, 37, 33, 29, 25, 21, 0},
    }
    
    numCities := len(distances)
    
    // TSP objective function
    tspObjective := func(x []float64) float64 {
        // Convert continuous variables to permutation
        cities := make([]int, numCities)
        for i := range cities {
            cities[i] = i
        }
        
        // Sort cities based on x values
        for i := 0; i < numCities-1; i++ {
            for j := i + 1; j < numCities; j++ {
                if x[cities[i]] > x[cities[j]] {
                    cities[i], cities[j] = cities[j], cities[i]
                }
            }
        }
        
        // Calculate total distance
        totalDistance := 0.0
        for i := 0; i < numCities; i++ {
            from := cities[i]
            to := cities[(i+1)%numCities]
            totalDistance += distances[from][to]
        }
        
        return totalDistance
    }
    
    config := parallelsimulatedannealing.DefaultSAConfig()
    config.Dimensions = numCities
    config.MaxIterations = 20000
    config.InitialTemperature = 1000.0
    config.PerturbationStrat = parallelsimulatedannealing.CauchyPerturbation
    config.ParallelStrategy = parallelsimulatedannealing.IslandModel
    config.NumChains = 8
    config.LowerBound = make([]float64, numCities)
    config.UpperBound = make([]float64, numCities)
    
    for i := range config.LowerBound {
        config.LowerBound[i] = 0.0
        config.UpperBound[i] = float64(numCities)
    }
    
    psa, err := parallelsimulatedannealing.NewParallelSimulatedAnnealing(config, tspObjective)
    if err != nil {
        log.Fatalf("Failed to create TSP optimizer: %v", err)
    }
    
    solution, err := psa.Optimize()
    if err != nil {
        log.Fatalf("TSP optimization failed: %v", err)
    }
    
    fmt.Printf("TSP Solution:\n")
    fmt.Printf("  Total Distance: %f\n", solution.Fitness)
    fmt.Printf("  Route Encoding: %v\n", solution.Variables)
}

// Portfolio optimization example
func portfolioOptimization() {
    // Expected returns and covariance matrix for 5 assets
    expectedReturns := []float64{0.12, 0.10, 0.08, 0.15, 0.09}
    
    // Simplified covariance matrix (diagonal for this example)
    covariance := [][]float64{
        {0.04, 0.01, 0.01, 0.02, 0.01},
        {0.01, 0.03, 0.01, 0.01, 0.01},
        {0.01, 0.01, 0.02, 0.01, 0.01},
        {0.02, 0.01, 0.01, 0.05, 0.02},
        {0.01, 0.01, 0.01, 0.02, 0.03},
    }
    
    riskAversion := 3.0 // Risk aversion parameter
    
    // Portfolio objective: maximize return - risk penalty
    portfolioObjective := func(weights []float64) float64 {
        // Normalize weights to sum to 1
        sum := 0.0
        for _, w := range weights {
            sum += w
        }
        if sum == 0 {
            return 1e6 // Penalty for zero portfolio
        }
        
        normalizedWeights := make([]float64, len(weights))
        for i, w := range weights {
            normalizedWeights[i] = w / sum
        }
        
        // Calculate expected return
        expectedReturn := 0.0
        for i, w := range normalizedWeights {
            expectedReturn += w * expectedReturns[i]
        }
        
        // Calculate portfolio variance
        variance := 0.0
        for i := 0; i < len(normalizedWeights); i++ {
            for j := 0; j < len(normalizedWeights); j++ {
                variance += normalizedWeights[i] * normalizedWeights[j] * covariance[i][j]
            }
        }
        
        // Return negative utility (minimize negative = maximize positive)
        utility := expectedReturn - riskAversion*variance
        return -utility
    }
    
    config := parallelsimulatedannealing.DefaultSAConfig()
    config.Dimensions = len(expectedReturns)
    config.MaxIterations = 15000
    config.InitialTemperature = 10.0
    config.FinalTemperature = 0.001
    config.LowerBound = make([]float64, config.Dimensions)
    config.UpperBound = make([]float64, config.Dimensions)
    
    // Portfolio weights between 0 and 1
    for i := range config.LowerBound {
        config.LowerBound[i] = 0.0
        config.UpperBound[i] = 1.0
    }
    
    psa, err := parallelsimulatedannealing.NewParallelSimulatedAnnealing(config, portfolioObjective)
    if err != nil {
        log.Fatalf("Failed to create portfolio optimizer: %v", err)
    }
    
    solution, err := psa.Optimize()
    if err != nil {
        log.Fatalf("Portfolio optimization failed: %v", err)
    }
    
    // Normalize final weights
    sum := 0.0
    for _, w := range solution.Variables {
        sum += w
    }
    
    fmt.Printf("Optimal Portfolio:\n")
    fmt.Printf("  Utility: %f\n", -solution.Fitness)
    fmt.Printf("  Weights: ")
    for i, w := range solution.Variables {
        normalizedWeight := w / sum
        fmt.Printf("Asset%d: %.3f ", i+1, normalizedWeight)
    }
    fmt.Println()
}
```

## Configuration Options

### SAConfig Fields

#### Problem Definition
- **Dimensions**: Number of optimization variables
- **LowerBound**: Lower bounds for each variable (auto-set to -100 if empty)
- **UpperBound**: Upper bounds for each variable (auto-set to 100 if empty)

#### Temperature Control
- **InitialTemperature**: Starting temperature for annealing process
- **FinalTemperature**: Minimum temperature (stopping criterion)
- **CoolingSchedule**: Temperature reduction strategy
- **CoolingRate**: Rate parameter for cooling schedules

#### Perturbation Control
- **PerturbationStrat**: Strategy for generating neighbor solutions
- **PerturbationSize**: Scale factor for perturbations

#### Parallelization
- **ParallelStrategy**: Approach for parallel processing
- **NumWorkers**: Number of worker goroutines
- **NumChains**: Number of parallel annealing chains
- **RestartInterval**: Iterations between chain restarts
- **IslandExchangeRate**: Frequency of solution exchange between islands

#### Adaptive Mechanisms
- **AdaptiveParameters**: Enable automatic parameter adjustment
- **EnableMemory**: Store elite solutions in memory
- **Tolerance**: Convergence tolerance for early stopping
- **StagnationLimit**: Maximum iterations without improvement
- **ElitismRate**: Proportion of elite solutions to preserve
- **DiversityThreshold**: Minimum diversity to maintain

#### Monitoring and Control
- **MaxIterations**: Maximum number of iterations per chain
- **EnableStatistics**: Comprehensive performance tracking
- **EnableLogging**: Debug and progress logging
- **RandomSeed**: Seed for random number generation

### Cooling Schedule Types

| Schedule | Formula | Characteristics | Best For |
|----------|---------|-----------------|----------|
| Linear | T = T₀(1 - t/T_max) | Steady linear decrease | Simple problems |
| Exponential | T = T₀ × α^t | Fast initial cooling | Most problems |
| Logarithmic | T = T₀ / log(1+t) | Slow cooling | Theoretical optimality |
| Inverse | T = T₀ / (1+t) | Very slow cooling | Fine-tuning |
| Adaptive | Dynamic based on acceptance | Self-adjusting | Unknown landscapes |
| Geometric | T = T₀ × α^t | Similar to exponential | Classic approach |
| Quadratic | T = T₀(1-t²/T_max²) | Accelerating cooling | Time-constrained |
| Cosine | T = T_f + (T₀-T_f)(1+cos(πt/T_max))/2 | Smooth transitions | Continuous problems |

### Perturbation Strategy Types

| Strategy | Distribution | Properties | Applications |
|----------|-------------|------------|--------------|
| Gaussian | Normal | Symmetric, finite variance | General continuous optimization |
| Uniform | Uniform | Bounded, equal probability | Constrained problems |
| Cauchy | Cauchy | Heavy tails, infinite variance | Escaping local optima |
| Levy Flight | Levy | Power-law tails | Multi-scale exploration |
| Adaptive | Variable | Self-tuning step size | Dynamic landscapes |
| Hybrid | Mixed | Combines multiple strategies | Robust optimization |

## Performance Characteristics

### Computational Complexity
- **Time per Iteration**: O(D × F) where D = dimensions, F = function evaluation cost
- **Space Complexity**: O(C × D × M) where C = chains, M = memory size
- **Parallel Speedup**: Near-linear with number of chains for independent strategies
- **Communication Overhead**: Minimal for independent chains, moderate for cooperative strategies

### Convergence Properties
- **Global Optimization**: Probabilistic guarantee with infinite time
- **Convergence Rate**: Depends on cooling schedule and problem landscape
- **Robustness**: High tolerance to noise and irregular objective functions
- **Scalability**: Effective for high-dimensional problems (tested up to 1000+ dimensions)

### Memory Usage
- **Per Chain**: ~O(D) for current/best solutions plus optional memory
- **Statistics**: ~O(I) where I = iteration count for convergence history
- **Communication Buffers**: ~O(C) for inter-chain communication
- **Total Memory**: Typically < 1MB for problems with < 1000 dimensions

## Best Practices

### Problem Setup
1. **Scale Variables**: Normalize variables to similar ranges for better performance
2. **Set Realistic Bounds**: Tight bounds improve convergence speed
3. **Choose Appropriate Dimensions**: SA works well up to ~1000 dimensions
4. **Function Evaluation**: Minimize expensive function calls through caching

### Parameter Tuning
1. **Initial Temperature**: Set to accept ~80-90% of initial moves
2. **Final Temperature**: Set to 0.01-0.001 of initial temperature
3. **Cooling Rate**: 0.85-0.99 for exponential cooling
4. **Chain Count**: 2-10x number of CPU cores for good parallelization

### Strategy Selection
1. **Independent Chains**: Best for embarrassingly parallel problems
2. **Island Model**: Recommended for complex, multimodal landscapes
3. **Temperature Parallel**: Good for unknown problem characteristics
4. **Cooperative Chains**: Use when global information helps local search

### Performance Optimization
1. **Profile Function Evaluation**: Often the bottleneck in optimization
2. **Use Appropriate Data Types**: float64 for most problems, float32 for speed
3. **Minimize Memory Allocation**: Reuse slices and avoid frequent allocations
4. **Monitor Convergence**: Use statistics to detect premature convergence

## Common Use Cases

### Engineering Optimization
- **Circuit Design**: Component sizing and placement optimization
- **Structural Design**: Weight minimization with strength constraints
- **Process Optimization**: Parameter tuning for manufacturing processes
- **Control Systems**: PID controller tuning and system identification

### Machine Learning
- **Hyperparameter Optimization**: Neural network architecture and parameter tuning
- **Feature Selection**: Optimal subset selection for model performance
- **Model Ensemble**: Weight optimization for ensemble methods
- **Training Data Selection**: Active learning and data subset optimization

### Operations Research
- **Scheduling Problems**: Job shop scheduling and resource allocation
- **Routing Problems**: Vehicle routing and network optimization
- **Portfolio Optimization**: Asset allocation and risk management
- **Supply Chain**: Inventory management and distribution optimization

### Scientific Computing
- **Parameter Estimation**: Model fitting and inverse problems
- **Molecular Dynamics**: Protein folding and drug design
- **Image Processing**: Registration and reconstruction problems
- **Signal Processing**: Filter design and system identification

### Game Theory and Economics
- **Nash Equilibrium**: Finding equilibrium points in games
- **Market Simulation**: Price optimization and strategy selection
- **Auction Design**: Mechanism design and bidding strategies
- **Resource Allocation**: Fair division and assignment problems

## Advanced Features

### Adaptive Mechanisms
- **Temperature Adaptation**: Automatic adjustment based on acceptance rates
- **Step Size Control**: Dynamic perturbation scaling
- **Restart Strategies**: Intelligent restart based on convergence metrics
- **Diversity Maintenance**: Population-based diversity preservation

### Statistical Analysis
- **Convergence Analysis**: Detailed tracking of solution evolution
- **Performance Metrics**: Acceptance rates, temperature evolution, chain statistics
- **Diversity Measurement**: Population spread and exploration coverage
- **Runtime Statistics**: Timing analysis and bottleneck identification

### Integration Features
- **Context Support**: Cancellation and timeout handling
- **Callback Functions**: Custom monitoring and intervention
- **Constraint Handling**: Automatic constraint enforcement
- **Multi-objective Support**: Pareto frontier approximation (extension)

## Limitations and Considerations

### Algorithm Limitations
1. **Local Optima**: No guarantee of global optimum in finite time
2. **Parameter Sensitivity**: Performance depends on parameter tuning
3. **Slow Convergence**: May require many iterations for high precision
4. **Function Evaluations**: Limited by expensive objective function calls

### Implementation Considerations
1. **Memory Usage**: Scales with problem size and chain count
2. **Thread Safety**: Careful synchronization for shared data structures
3. **Numerical Precision**: Float64 precision limits for very small differences
4. **Platform Dependencies**: Performance varies across different architectures

### Problem Suitability
- **Best For**: Continuous optimization, noisy functions, black-box problems
- **Avoid For**: Discrete optimization (use specialized variants), linear problems
- **Consider Alternatives**: For convex problems, gradient-based methods may be faster

## Future Enhancements

Planned improvements for future versions:

- **GPU Acceleration**: CUDA/OpenCL support for massive parallelization
- **Distributed Computing**: Multi-node optimization across clusters
- **Advanced Operators**: Problem-specific neighbor generation strategies
- **Multi-objective Optimization**: Native Pareto frontier optimization
- **Machine Learning Integration**: Learned cooling schedules and perturbation strategies
- **Real-time Visualization**: Live plotting of convergence and parameter evolution
- **Benchmark Suite**: Comprehensive test problems and performance comparison
- **Auto-tuning**: Automatic parameter selection based on problem characteristics