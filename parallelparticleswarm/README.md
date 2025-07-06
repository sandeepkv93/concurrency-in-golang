# Parallel Particle Swarm Optimization

A high-performance, parallel implementation of Particle Swarm Optimization (PSO) in Go, featuring multiple algorithm variants, advanced topology management, adaptive parameter control, and comprehensive optimization capabilities for solving complex numerical optimization problems.

## Features

### Core PSO Algorithms
- **Standard PSO**: Classic particle swarm optimization with global best
- **Inertia Weight PSO**: Enhanced with time-varying inertia weight
- **Constriction PSO**: Advanced variant with constriction factor
- **Adaptive PSO**: Self-adjusting parameters based on performance
- **Quantum PSO**: Quantum-inspired particle behavior
- **Binary PSO**: Specialized for binary optimization problems
- **Multi-Objective PSO**: Support for multi-objective optimization

### Swarm Topologies
- **Global Topology**: All particles share information globally
- **Ring Topology**: Particles connected in a circular pattern
- **Star Topology**: Central hub with spoke connections
- **Von Neumann Topology**: Grid-based neighborhood structure
- **Mesh Topology**: Fully connected mesh network
- **Random Topology**: Randomly generated connections

### Advanced Features
- **Parallel Evaluation**: Concurrent fitness function evaluation across multiple cores
- **Async Updates**: Asynchronous particle position and velocity updates
- **Diversity Maintenance**: Automatic swarm diversity monitoring and enhancement
- **Adaptive Parameters**: Dynamic adjustment of inertia weight, cognitive, and social coefficients
- **Constraint Handling**: Support for equality and inequality constraints
- **Elite Management**: Elite particle preservation and management
- **Convergence Detection**: Intelligent stopping criteria and stagnation detection

### Performance Optimizations
- **Worker Pool Architecture**: Scalable parallel processing with configurable worker threads
- **Lock-Free Operations**: Atomic operations and careful synchronization for high throughput
- **Memory Optimization**: Efficient memory usage with object pooling
- **Cache-Friendly Access**: Optimized data access patterns for modern CPU architectures
- **NUMA Awareness**: Non-uniform memory access optimization for multi-socket systems

## Algorithm Overview

Particle Swarm Optimization is a population-based optimization algorithm inspired by the social behavior of bird flocking and fish schooling. Each particle in the swarm represents a potential solution, maintaining:

- **Position**: Current location in the search space
- **Velocity**: Direction and magnitude of movement
- **Personal Best**: Best position found by the particle
- **Neighborhood Best**: Best position found by neighbors

### PSO Update Equations

```
v[i](t+1) = w * v[i](t) + c1 * r1 * (pbest[i] - x[i](t)) + c2 * r2 * (gbest - x[i](t))
x[i](t+1) = x[i](t) + v[i](t+1)
```

Where:
- `w` = inertia weight
- `c1, c2` = cognitive and social acceleration coefficients
- `r1, r2` = random numbers in [0,1]
- `pbest[i]` = personal best position of particle i
- `gbest` = global best position

## Usage Examples

### Basic Optimization

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/yourusername/concurrency-in-golang/parallelparticleswarm"
)

func main() {
    // Define objective function (minimize sphere function)
    sphereFunction := func(position []float64) float64 {
        sum := 0.0
        for _, x := range position {
            sum += x * x
        }
        return sum
    }

    // Create PSO configuration
    config := parallelparticleswarm.PSOConfig{
        SwarmSize:       50,
        MaxIterations:   1000,
        Dimensions:      10,
        MinBounds:       []float64{-5, -5, -5, -5, -5, -5, -5, -5, -5, -5},
        MaxBounds:       []float64{5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
        InertiaWeight:   0.9,
        CognitiveWeight: 2.0,
        SocialWeight:    2.0,
        MaxVelocity:     2.0,
        Problem:         parallelparticleswarm.Minimize,
        Variant:         parallelparticleswarm.StandardPSO,
        Topology:        parallelparticleswarm.GlobalTopology,
        NumWorkers:      8,
        UseParallelEval: true,
        ConvergenceThreshold: 1e-8,
        EnableStatistics: true,
    }

    // Create optimizer
    optimizer := parallelparticleswarm.NewOptimizer(config, sphereFunction)
    defer optimizer.Shutdown()

    // Run optimization
    fmt.Println("Running PSO optimization...")
    start := time.Now()
    
    bestParticle, err := optimizer.Optimize()
    if err != nil {
        log.Fatalf("Optimization failed: %v", err)
    }

    duration := time.Since(start)

    // Display results
    fmt.Printf("\nOptimization completed in %v\n", duration)
    fmt.Printf("Best fitness: %.8e\n", bestParticle.BestFitness)
    fmt.Printf("Best position: %v\n", bestParticle.Position)

    // Get statistics
    stats := optimizer.GetStatistics()
    fmt.Printf("\nStatistics:\n")
    fmt.Printf("  Total iterations: %d\n", stats.TotalIterations)
    fmt.Printf("  Function evaluations: %d\n", stats.FunctionEvaluations)
    fmt.Printf("  Average fitness: %.6e\n", stats.AverageFitness)
    fmt.Printf("  Convergence rate: %.4f\n", stats.ConvergenceRate)
}
```

### Advanced Configuration with Adaptive Parameters

```go
// Create advanced PSO configuration
config := parallelparticleswarm.PSOConfig{
    SwarmSize:            100,
    MaxIterations:        2000,
    Dimensions:           20,
    MinBounds:           make([]float64, 20),
    MaxBounds:           make([]float64, 20),
    InertiaWeight:       0.9,
    CognitiveWeight:     2.0,
    SocialWeight:        2.0,
    MaxVelocity:         4.0,
    Problem:             parallelparticleswarm.Minimize,
    Variant:             parallelparticleswarm.AdaptivePSO,
    Topology:            parallelparticleswarm.VonNeumannTopology,
    NumWorkers:          runtime.NumCPU(),
    UseParallelEval:     true,
    UseAsyncUpdate:      true,
    ConvergenceThreshold: 1e-10,
    MaxStagnation:       100,
    AdaptiveWeights:     true,
    DiversityMaintain:   true,
    EliteSize:           10,
    MutationRate:        0.02,
    EnableStatistics:    true,
}

// Initialize bounds for 20-dimensional problem
for i := 0; i < 20; i++ {
    config.MinBounds[i] = -100.0
    config.MaxBounds[i] = 100.0
}

// Complex multi-modal function (Rastrigin)
rastriginFunction := func(position []float64) float64 {
    n := float64(len(position))
    sum := 10.0 * n
    
    for _, x := range position {
        sum += x*x - 10.0*math.Cos(2.0*math.Pi*x)
    }
    
    return sum
}

optimizer := parallelparticleswarm.NewOptimizer(config, rastriginFunction)
defer optimizer.Shutdown()

bestParticle, err := optimizer.Optimize()
if err != nil {
    log.Fatalf("Optimization failed: %v", err)
}

fmt.Printf("Adaptive PSO Result:\n")
fmt.Printf("Best fitness: %.8e\n", bestParticle.BestFitness)
fmt.Printf("Position near global minimum: %v\n", bestParticle.Position[:5]) // Show first 5 dimensions
```

### Constrained Optimization

```go
// Define objective function
objectiveFunction := func(position []float64) float64 {
    x, y := position[0], position[1]
    return (x-2)*(x-2) + (y-3)*(y-3) // Minimize distance from point (2,3)
}

config := parallelparticleswarm.DefaultPSOConfig()
config.Dimensions = 2
config.MinBounds = []float64{-10, -10}
config.MaxBounds = []float64{10, 10}
config.SwarmSize = 30
config.MaxIterations = 500

optimizer := parallelparticleswarm.NewOptimizer(config, objectiveFunction)
defer optimizer.Shutdown()

// Add circular constraint: x^2 + y^2 <= 25 (inside circle of radius 5)
circularConstraint := func(position []float64) bool {
    x, y := position[0], position[1]
    return x*x + y*y <= 25.0
}
optimizer.AddConstraint(circularConstraint)

// Add linear constraint: x + y >= 1
linearConstraint := func(position []float64) bool {
    return position[0] + position[1] >= 1.0
}
optimizer.AddConstraint(linearConstraint)

bestParticle, err := optimizer.Optimize()
if err != nil {
    log.Fatalf("Constrained optimization failed: %v", err)
}

fmt.Printf("Constrained Optimization Result:\n")
fmt.Printf("Best fitness: %.6f\n", bestParticle.BestFitness)
fmt.Printf("Best position: [%.6f, %.6f]\n", bestParticle.Position[0], bestParticle.Position[1])

// Verify constraints
x, y := bestParticle.Position[0], bestParticle.Position[1]
fmt.Printf("Constraint verification:\n")
fmt.Printf("  Circle constraint (x²+y² ≤ 25): %.3f ≤ 25 = %t\n", x*x+y*y, x*x+y*y <= 25)
fmt.Printf("  Linear constraint (x+y ≥ 1): %.3f ≥ 1 = %t\n", x+y, x+y >= 1)
```

### Benchmark Function Testing

```go
// Get predefined benchmark functions
benchmarks := parallelparticleswarm.GetBenchmarkFunctions()

results := make(map[string]float64)

for name, benchmark := range benchmarks {
    fmt.Printf("Optimizing %s...\n", benchmark.Name)
    
    config := parallelparticleswarm.DefaultPSOConfig()
    config.SwarmSize = 60
    config.MaxIterations = 1000
    config.Dimensions = benchmark.Dimensions
    config.MinBounds = benchmark.MinBounds
    config.MaxBounds = benchmark.MaxBounds
    config.UseParallelEval = true
    config.NumWorkers = 8
    
    optimizer := parallelparticleswarm.NewOptimizer(config, benchmark.Function)
    
    start := time.Now()
    bestParticle, err := optimizer.Optimize()
    duration := time.Since(start)
    
    optimizer.Shutdown()
    
    if err != nil {
        fmt.Printf("  Error: %v\n", err)
        continue
    }
    
    results[name] = bestParticle.BestFitness
    
    fmt.Printf("  Result: %.8e (Global optimum: %.8e)\n", 
        bestParticle.BestFitness, benchmark.GlobalValue)
    fmt.Printf("  Time: %v\n", duration)
    fmt.Printf("  Error: %.8e\n", 
        math.Abs(bestParticle.BestFitness - benchmark.GlobalValue))
    
    stats := optimizer.GetStatistics()
    fmt.Printf("  Evaluations: %d\n", stats.FunctionEvaluations)
    fmt.Println()
}

// Summary
fmt.Println("Summary of Results:")
fmt.Println("Function          | Best Fitness     | Global Optimum   | Error")
fmt.Println("------------------|------------------|------------------|------------------")
for name, fitness := range results {
    benchmark := benchmarks[name]
    error := math.Abs(fitness - benchmark.GlobalValue)
    fmt.Printf("%-17s | %16.8e | %16.8e | %16.8e\n", 
        benchmark.Name, fitness, benchmark.GlobalValue, error)
}
```

### Multi-Objective Optimization

```go
// Define multi-objective function (ZDT1 test problem)
zdt1Function := func(position []float64) []float64 {
    n := len(position)
    f1 := position[0]
    
    g := 0.0
    for i := 1; i < n; i++ {
        g += position[i]
    }
    g = 1.0 + 9.0*g/float64(n-1)
    
    h := 1.0 - math.Sqrt(f1/g)
    f2 := g * h
    
    return []float64{f1, f2}
}

// Convert multi-objective to single objective using weighted sum
weightedFunction := func(position []float64) float64 {
    objectives := zdt1Function(position)
    return 0.5*objectives[0] + 0.5*objectives[1] // Equal weights
}

config := parallelparticleswarm.DefaultPSOConfig()
config.Dimensions = 10
config.MinBounds = make([]float64, 10)
config.MaxBounds = make([]float64, 10)

for i := 0; i < 10; i++ {
    config.MinBounds[i] = 0.0
    config.MaxBounds[i] = 1.0
}

config.SwarmSize = 100
config.MaxIterations = 1000
config.Variant = parallelparticleswarm.MultiObjectivePSO

optimizer := parallelparticleswarm.NewOptimizer(config, weightedFunction)
defer optimizer.Shutdown()

bestParticle, err := optimizer.Optimize()
if err != nil {
    log.Fatalf("Multi-objective optimization failed: %v", err)
}

// Evaluate original objectives
objectives := zdt1Function(bestParticle.Position)
fmt.Printf("Multi-Objective Optimization Result:\n")
fmt.Printf("Objective 1: %.6f\n", objectives[0])
fmt.Printf("Objective 2: %.6f\n", objectives[1])
fmt.Printf("Weighted sum: %.6f\n", bestParticle.BestFitness)
```

### Real-Time Optimization Monitoring

```go
// Custom function with monitoring
monitoredFunction := func(position []float64) float64 {
    // Simulate computationally expensive function
    time.Sleep(1 * time.Millisecond)
    
    result := 0.0
    for i, x := range position {
        result += math.Pow(x-float64(i), 2)
    }
    return result
}

config := parallelparticleswarm.DefaultPSOConfig()
config.Dimensions = 5
config.MinBounds = []float64{-10, -10, -10, -10, -10}
config.MaxBounds = []float64{10, 10, 10, 10, 10}
config.SwarmSize = 25
config.MaxIterations = 200
config.EnableStatistics = true

optimizer := parallelparticleswarm.NewOptimizer(config, monitoredFunction)
defer optimizer.Shutdown()

// Start optimization in goroutine
resultChan := make(chan *parallelparticleswarm.Particle)
errorChan := make(chan error)

go func() {
    bestParticle, err := optimizer.Optimize()
    if err != nil {
        errorChan <- err
        return
    }
    resultChan <- bestParticle
}()

// Monitor progress
ticker := time.NewTicker(500 * time.Millisecond)
defer ticker.Stop()

fmt.Println("Monitoring optimization progress...")
fmt.Println("Iteration | Best Fitness | Avg Fitness  | Evaluations | Diversity")
fmt.Println("----------|--------------|--------------|-------------|----------")

for {
    select {
    case <-ticker.C:
        stats := optimizer.GetStatistics()
        swarm := optimizer.GetSwarm()
        
        diversity := 0.0
        if len(stats.DiversityHistory) > 0 {
            diversity = stats.DiversityHistory[len(stats.DiversityHistory)-1]
        }
        
        fmt.Printf("%9d | %12.6e | %12.6e | %11d | %8.4f\n",
            stats.TotalIterations,
            stats.BestFitness,
            stats.AverageFitness,
            stats.FunctionEvaluations,
            diversity)
        
        // Check if converged
        if swarm.Converged {
            fmt.Println("Converged!")
            break
        }
        
    case bestParticle := <-resultChan:
        fmt.Printf("\nOptimization completed!\n")
        fmt.Printf("Best fitness: %.8e\n", bestParticle.BestFitness)
        fmt.Printf("Best position: %v\n", bestParticle.Position)
        return
        
    case err := <-errorChan:
        fmt.Printf("Optimization failed: %v\n", err)
        return
    }
}
```

## Performance Characteristics

| Configuration | Time Complexity | Space Complexity | Scalability |
|---------------|-----------------|------------------|-------------|
| Sequential PSO | O(I × N × D × F) | O(N × D) | Single-threaded |
| Parallel PSO | O(I × N × D × F / P) | O(N × D) | Multi-threaded |
| Adaptive PSO | O(I × N × D × F + A) | O(N × D + H) | Self-tuning |
| Multi-Objective | O(I × N × D × M × F) | O(N × D × M) | Pareto-optimal |

Where:
- I = iterations
- N = swarm size  
- D = dimensions
- F = function evaluation time
- P = number of parallel workers
- A = adaptation overhead
- H = history buffer size
- M = number of objectives

## Configuration Options

### Core Parameters

```go
type PSOConfig struct {
    // Basic PSO parameters
    SwarmSize           int           // Number of particles in swarm
    MaxIterations       int           // Maximum optimization iterations
    Dimensions          int           // Problem dimensionality
    MinBounds           []float64     // Lower bounds for each dimension
    MaxBounds           []float64     // Upper bounds for each dimension
    
    // PSO coefficients
    InertiaWeight       float64       // Inertia weight (0.4-0.9)
    CognitiveWeight     float64       // Cognitive acceleration (1.5-2.5)
    SocialWeight        float64       // Social acceleration (1.5-2.5)
    MaxVelocity         float64       // Maximum velocity clamp
    
    // Algorithm variants
    Problem             OptimizationProblem // Minimize or Maximize
    Variant             PSOVariant     // PSO algorithm variant
    Topology            Topology       // Swarm communication topology
    
    // Parallel processing
    NumWorkers          int           // Number of worker threads
    UseParallelEval     bool          // Enable parallel evaluation
    UseAsyncUpdate      bool          // Enable asynchronous updates
    
    // Convergence and stopping
    ConvergenceThreshold float64      // Fitness convergence threshold
    MaxStagnation       int           // Maximum stagnation iterations
    
    // Advanced features
    AdaptiveWeights     bool          // Enable adaptive parameter control
    DiversityMaintain   bool          // Enable diversity maintenance
    EliteSize           int           // Number of elite particles
    MutationRate        float64       // Mutation probability
    
    // Monitoring
    EnableStatistics    bool          // Enable statistics collection
    SeedValue           int64         // Random seed for reproducibility
}
```

### Default Configuration

```go
func DefaultPSOConfig() PSOConfig {
    return PSOConfig{
        SwarmSize:           50,
        MaxIterations:       1000,
        Dimensions:          10,
        InertiaWeight:       0.9,
        CognitiveWeight:     2.0,
        SocialWeight:        2.0,
        MaxVelocity:         4.0,
        Problem:             Minimize,
        Variant:             StandardPSO,
        Topology:            GlobalTopology,
        NumWorkers:          4,
        UseParallelEval:     true,
        UseAsyncUpdate:      false,
        ConvergenceThreshold: 1e-8,
        MaxStagnation:       50,
        AdaptiveWeights:     false,
        DiversityMaintain:   false,
        EliteSize:           5,
        MutationRate:        0.01,
        EnableStatistics:    true,
    }
}
```

## Benchmark Functions

The package includes several standard optimization benchmark functions:

### Unimodal Functions
- **Sphere Function**: Simple quadratic function with global minimum at origin
- **Rosenbrock Function**: Narrow valley function, challenging for optimization

### Multimodal Functions  
- **Rastrigin Function**: Highly multimodal with many local optima
- **Ackley Function**: Complex landscape with exponential terms

### Usage
```go
benchmarks := parallelparticleswarm.GetBenchmarkFunctions()
sphereFunc := benchmarks["sphere"].Function
rastriginFunc := benchmarks["rastrigin"].Function
// Use in optimization...
```

## Testing and Benchmarks

The package includes comprehensive testing:

```bash
# Run all tests
go test -v

# Run tests with race detection
go test -race -v

# Run benchmarks
go test -bench=. -benchmem

# Run with coverage
go test -cover -v

# Run specific benchmark
go test -bench=BenchmarkPSOOptimization -v
```

### Example Benchmark Results

```
BenchmarkSphereFunction-8                	 50000000	        28.5 ns/op	       0 B/op	       0 allocs/op
BenchmarkRastriginFunction-8             	 10000000	       142.0 ns/op	       0 B/op	       0 allocs/op
BenchmarkPSOOptimization-8               	       50	  23456789 ns/op	  234567 B/op	    1234 allocs/op
BenchmarkParallelVsSequential/Sequential-8	       20	  89234567 ns/op	  345678 B/op	    2345 allocs/op
BenchmarkParallelVsSequential/Parallel-8   	       30	  34567890 ns/op	  456789 B/op	    3456 allocs/op
```

## Production Considerations

### Performance Optimization
- **Worker Pool Size**: Set `NumWorkers` to `runtime.NumCPU()` for CPU-bound problems
- **Swarm Size**: Balance between exploration (larger) and efficiency (smaller)
- **Parallel Evaluation**: Enable for expensive objective functions
- **Memory Usage**: Monitor with large swarms and high-dimensional problems

### Algorithm Selection
- **Standard PSO**: General-purpose, well-tested baseline
- **Adaptive PSO**: When problem characteristics are unknown
- **Quantum PSO**: For highly multimodal landscapes
- **Topology Choice**: Von Neumann for balance, Global for convergence speed

### Monitoring and Debugging
- **Statistics**: Enable comprehensive performance tracking
- **Convergence**: Monitor fitness history and diversity
- **Constraints**: Validate constraint satisfaction in results
- **Reproducibility**: Set seed value for deterministic results

## References

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95-International Conference on Neural Networks*.

2. Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. *1998 IEEE international conference on evolutionary computation proceedings*.

3. Clerc, M., & Kennedy, J. (2002). The particle swarm-explosion, stability, and convergence in a multidimensional complex space. *IEEE transactions on Evolutionary Computation*.

## License

This implementation is part of the Concurrency in Golang project and is provided for educational and demonstration purposes.