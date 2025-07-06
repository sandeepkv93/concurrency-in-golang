# Parallel Sudoku Solver

A high-performance, concurrent Sudoku solver implementation in Go featuring multiple solving strategies, parallel processing capabilities, and comprehensive puzzle generation tools.

## Features

### Core Solving Engine
- **Multiple Strategies**: Backtracking, Constraint Propagation, Hybrid, and Brute Force approaches
- **Parallel Processing**: Multi-worker architecture for concurrent solving attempts
- **Advanced Heuristics**: Intelligent value ordering and constraint-based optimizations
- **Pruning Techniques**: Early termination of unsolvable branches
- **Context Support**: Graceful cancellation and timeout handling
- **Real-time Statistics**: Comprehensive performance metrics and solving analytics

### Solving Strategies
- **Backtracking**: Classic recursive backtracking with parallel worker distribution
- **Constraint Propagation**: Logic-based constraint satisfaction approach
- **Hybrid Strategy**: Combines multiple approaches for optimal performance
- **Brute Force**: Exhaustive search for small puzzles with high parallelization

### Puzzle Management
- **Validation**: Comprehensive board validation and integrity checking
- **Generation**: Automatic puzzle creation with configurable difficulty levels
- **Difficulty Levels**: Easy, Medium, Hard, and Expert puzzle generation
- **Board Visualization**: Pretty-printed board representations with Unicode characters

## Usage Examples

### Basic Sudoku Solving

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"
    
    "github.com/yourusername/concurrency-in-golang/parallelsudokusolver"
)

func main() {
    // Configure the solver
    config := parallelsudokusolver.SolverConfig{
        Strategy:       parallelsudokusolver.HybridStrategy,
        MaxWorkers:     8,
        TimeLimit:      30 * time.Second,
        UseHeuristics:  true,
        EnablePruning:  true,
        LogProgress:    true,
    }
    
    // Create solver
    solver := parallelsudokusolver.NewParallelSudokuSolver(config)
    
    // Define a puzzle (0 represents empty cells)
    puzzle := parallelsudokusolver.SudokuBoard{
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9},
    }
    
    fmt.Println("Original Puzzle:")
    fmt.Println(puzzle.PrettyString())
    
    // Solve the puzzle
    ctx := context.Background()
    solution, stats, err := solver.Solve(ctx, puzzle)
    if err != nil {
        log.Fatalf("Failed to solve puzzle: %v", err)
    }
    
    if solution == nil {
        fmt.Println("No solution found!")
        return
    }
    
    fmt.Println("Solution:")
    fmt.Println(solution.PrettyString())
    
    // Display solving statistics
    fmt.Printf("Solving Statistics:\n")
    fmt.Printf("  Strategy: %s\n", getStrategyName(stats.Strategy))
    fmt.Printf("  Workers Used: %d\n", stats.WorkersUsed)
    fmt.Printf("  Solve Time: %v\n", stats.AverageSolveTime)
    fmt.Printf("  Backtrack Operations: %d\n", stats.BacktrackCount)
    fmt.Printf("  Constraint Checks: %d\n", stats.ConstraintChecks)
    fmt.Printf("  Success Rate: %.2f%%\n", 
        float64(stats.SuccessfulSolves)/float64(stats.TotalAttempts)*100)
}

func getStrategyName(strategy parallelsudokusolver.SolverStrategy) string {
    switch strategy {
    case parallelsudokusolver.BacktrackingStrategy:
        return "Backtracking"
    case parallelsudokusolver.ConstraintPropagationStrategy:
        return "Constraint Propagation"
    case parallelsudokusolver.HybridStrategy:
        return "Hybrid"
    case parallelsudokusolver.BruteForceStrategy:
        return "Brute Force"
    default:
        return "Unknown"
    }
}
```

### Strategy Comparison

```go
// Compare different solving strategies
func compareStrategies(puzzle parallelsudokusolver.SudokuBoard) {
    strategies := []parallelsudokusolver.SolverStrategy{
        parallelsudokusolver.BacktrackingStrategy,
        parallelsudokusolver.ConstraintPropagationStrategy,
        parallelsudokusolver.HybridStrategy,
    }
    
    fmt.Println("Strategy Comparison Results:")
    
    for _, strategy := range strategies {
        config := parallelsudokusolver.SolverConfig{
            Strategy:       strategy,
            MaxWorkers:     4,
            TimeLimit:      15 * time.Second,
            UseHeuristics:  true,
            EnablePruning:  true,
        }
        
        solver := parallelsudokusolver.NewParallelSudokuSolver(config)
        
        start := time.Now()
        ctx := context.Background()
        solution, stats, err := solver.Solve(ctx, puzzle)
        duration := time.Since(start)
        
        strategyName := getStrategyName(strategy)
        
        if err != nil || solution == nil {
            fmt.Printf("%s: FAILED (%v) in %v\n", strategyName, err, duration)
        } else {
            fmt.Printf("%s: SUCCESS in %v\n", strategyName, duration)
            fmt.Printf("  Backtrack Operations: %d\n", stats.BacktrackCount)
            fmt.Printf("  Constraint Checks: %d\n", stats.ConstraintChecks)
            fmt.Printf("  Heuristic Evaluations: %d\n", stats.HeuristicEvaluations)
        }
        fmt.Println()
    }
}
```

### Concurrent Puzzle Solving

```go
// Solve multiple puzzles concurrently
func solvePuzzlesBatch(puzzles []parallelsudokusolver.SudokuBoard) {
    config := parallelsudokusolver.SolverConfig{
        Strategy:       parallelsudokusolver.HybridStrategy,
        MaxWorkers:     2, // Limited per solver to allow multiple solvers
        TimeLimit:      20 * time.Second,
        UseHeuristics:  true,
        EnablePruning:  true,
    }
    
    var wg sync.WaitGroup
    results := make(chan SolveResult, len(puzzles))
    
    type SolveResult struct {
        Index    int
        Solution *parallelsudokusolver.SudokuBoard
        Stats    *parallelsudokusolver.SolverStats
        Error    error
        Duration time.Duration
    }
    
    for i, puzzle := range puzzles {
        wg.Add(1)
        go func(index int, p parallelsudokusolver.SudokuBoard) {
            defer wg.Done()
            
            solver := parallelsudokusolver.NewParallelSudokuSolver(config)
            
            start := time.Now()
            ctx := context.Background()
            solution, stats, err := solver.Solve(ctx, p)
            duration := time.Since(start)
            
            results <- SolveResult{
                Index:    index,
                Solution: solution,
                Stats:    stats,
                Error:    err,
                Duration: duration,
            }
        }(i, puzzle)
    }
    
    // Close results channel when all work is done
    go func() {
        wg.Wait()
        close(results)
    }()
    
    // Collect and display results
    solved := 0
    totalTime := time.Duration(0)
    
    for result := range results {
        if result.Error == nil && result.Solution != nil {
            solved++
            totalTime += result.Duration
            fmt.Printf("Puzzle %d: SOLVED in %v\n", result.Index+1, result.Duration)
        } else {
            fmt.Printf("Puzzle %d: FAILED - %v\n", result.Index+1, result.Error)
        }
    }
    
    fmt.Printf("\nBatch Results: %d/%d solved\n", solved, len(puzzles))
    if solved > 0 {
        fmt.Printf("Average solve time: %v\n", totalTime/time.Duration(solved))
    }
}
```

### Puzzle Generation

```go
// Generate puzzles of different difficulties
func generatePuzzleSet() {
    difficulties := []parallelsudokusolver.DifficultyLevel{
        parallelsudokusolver.Easy,
        parallelsudokusolver.Medium,
        parallelsudokusolver.Hard,
        parallelsudokusolver.Expert,
    }
    
    difficultyNames := []string{"Easy", "Medium", "Hard", "Expert"}
    
    for i, difficulty := range difficulties {
        fmt.Printf("\n%s Puzzle:\n", difficultyNames[i])
        
        // Generate puzzle with specific seed for reproducibility
        seed := time.Now().UnixNano() + int64(i)
        puzzle := parallelsudokusolver.GeneratePuzzle(difficulty, seed)
        
        fmt.Println(puzzle.PrettyString())
        
        // Count empty cells
        emptyCells := 0
        for row := 0; row < 9; row++ {
            for col := 0; col < 9; col++ {
                if puzzle[row][col] == 0 {
                    emptyCells++
                }
            }
        }
        
        fmt.Printf("Empty cells: %d\n", emptyCells)
        
        // Test solvability
        config := parallelsudokusolver.SolverConfig{
            Strategy:       parallelsudokusolver.HybridStrategy,
            MaxWorkers:     4,
            TimeLimit:      30 * time.Second,
            UseHeuristics:  true,
        }
        
        solver := parallelsudokusolver.NewParallelSudokuSolver(config)
        
        start := time.Now()
        ctx := context.Background()
        solution, _, err := solver.Solve(ctx, puzzle)
        duration := time.Since(start)
        
        if err != nil || solution == nil {
            fmt.Printf("WARNING: Generated puzzle could not be solved! (%v)\n", err)
        } else {
            fmt.Printf("Solve time: %v\n", duration)
        }
    }
}
```

### Advanced Configuration and Optimization

```go
// Performance-optimized configuration for different scenarios
func getOptimizedConfig(scenario string) parallelsudokusolver.SolverConfig {
    switch scenario {
    case "speed":
        // Optimized for fastest solving
        return parallelsudokusolver.SolverConfig{
            Strategy:       parallelsudokusolver.BacktrackingStrategy,
            MaxWorkers:     runtime.NumCPU(),
            TimeLimit:      5 * time.Second,
            UseHeuristics:  true,
            EnablePruning:  true,
            LogProgress:    false,
        }
        
    case "accuracy":
        // Optimized for finding solutions to difficult puzzles
        return parallelsudokusolver.SolverConfig{
            Strategy:       parallelsudokusolver.HybridStrategy,
            MaxWorkers:     runtime.NumCPU() / 2,
            TimeLimit:      60 * time.Second,
            UseHeuristics:  true,
            EnablePruning:  true,
            LogProgress:    true,
        }
        
    case "low_resource":
        // Optimized for limited resources
        return parallelsudokusolver.SolverConfig{
            Strategy:       parallelsudokusolver.ConstraintPropagationStrategy,
            MaxWorkers:     2,
            TimeLimit:      30 * time.Second,
            UseHeuristics:  false,
            EnablePruning:  false,
            LogProgress:    false,
        }
        
    default:
        // Balanced configuration
        return parallelsudokusolver.SolverConfig{
            Strategy:       parallelsudokusolver.HybridStrategy,
            MaxWorkers:     4,
            TimeLimit:      20 * time.Second,
            UseHeuristics:  true,
            EnablePruning:  true,
            LogProgress:    false,
        }
    }
}

// Benchmark different configurations
func benchmarkConfigurations(puzzle parallelsudokusolver.SudokuBoard) {
    scenarios := []string{"speed", "accuracy", "low_resource", "balanced"}
    
    fmt.Println("Configuration Benchmark Results:")
    
    for _, scenario := range scenarios {
        config := getOptimizedConfig(scenario)
        solver := parallelsudokusolver.NewParallelSudokuSolver(config)
        
        start := time.Now()
        ctx := context.Background()
        solution, stats, err := solver.Solve(ctx, puzzle)
        duration := time.Since(start)
        
        fmt.Printf("\n%s Configuration:\n", scenario)
        fmt.Printf("  Duration: %v\n", duration)
        
        if err != nil || solution == nil {
            fmt.Printf("  Result: FAILED (%v)\n", err)
        } else {
            fmt.Printf("  Result: SUCCESS\n")
            fmt.Printf("  Backtrack Count: %d\n", stats.BacktrackCount)
            fmt.Printf("  Constraint Checks: %d\n", stats.ConstraintChecks)
            
            // Calculate efficiency metrics
            if duration > 0 {
                operationsPerSecond := float64(stats.BacktrackCount + stats.ConstraintChecks) / duration.Seconds()
                fmt.Printf("  Operations/sec: %.0f\n", operationsPerSecond)
            }
        }
    }
}
```

### Real-time Progress Monitoring

```go
// Monitor solving progress in real-time
func solveWithProgressMonitoring(puzzle parallelsudokusolver.SudokuBoard) {
    config := parallelsudokusolver.SolverConfig{
        Strategy:       parallelsudokusolver.HybridStrategy,
        MaxWorkers:     4,
        TimeLimit:      60 * time.Second,
        UseHeuristics:  true,
        EnablePruning:  true,
        LogProgress:    true,
    }
    
    solver := parallelsudokusolver.NewParallelSudokuSolver(config)
    
    // Start progress monitoring in a separate goroutine
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    
    go func() {
        ticker := time.NewTicker(1 * time.Second)
        defer ticker.Stop()
        
        for {
            select {
            case <-ticker.C:
                stats := solver.GetStats()
                if stats.TotalAttempts > 0 {
                    fmt.Printf("Progress: %d attempts, %d backtracks, %d constraint checks\n",
                        stats.TotalAttempts, stats.BacktrackCount, stats.ConstraintChecks)
                }
                
            case <-ctx.Done():
                return
            }
        }
    }()
    
    fmt.Println("Solving puzzle with real-time monitoring...")
    fmt.Println(puzzle.PrettyString())
    
    start := time.Now()
    solution, stats, err := solver.Solve(ctx, puzzle)
    duration := time.Since(start)
    
    cancel() // Stop progress monitoring
    
    fmt.Printf("\nSolving completed in %v\n", duration)
    
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    if solution == nil {
        fmt.Println("No solution found!")
        return
    }
    
    fmt.Println("Solution found:")
    fmt.Println(solution.PrettyString())
    
    // Final statistics
    fmt.Printf("Final Statistics:\n")
    fmt.Printf("  Total Attempts: %d\n", stats.TotalAttempts)
    fmt.Printf("  Successful Solves: %d\n", stats.SuccessfulSolves)
    fmt.Printf("  Failed Solves: %d\n", stats.FailedSolves)
    fmt.Printf("  Backtrack Operations: %d\n", stats.BacktrackCount)
    fmt.Printf("  Constraint Checks: %d\n", stats.ConstraintChecks)
    fmt.Printf("  Pruning Operations: %d\n", stats.PruningOperations)
    fmt.Printf("  Heuristic Evaluations: %d\n", stats.HeuristicEvaluations)
}
```

### Batch Processing and Analysis

```go
// Process and analyze multiple puzzles
func analyzeMultiplePuzzles() {
    // Generate test puzzles
    var puzzles []parallelsudokusolver.SudokuBoard
    
    for i := 0; i < 10; i++ {
        difficulty := parallelsudokusolver.DifficultyLevel(i % 4) // Cycle through difficulties
        puzzle := parallelsudokusolver.GeneratePuzzle(difficulty, int64(i))
        puzzles = append(puzzles, puzzle)
    }
    
    // Test different strategies
    strategies := []parallelsudokusolver.SolverStrategy{
        parallelsudokusolver.BacktrackingStrategy,
        parallelsudokusolver.ConstraintPropagationStrategy,
        parallelsudokusolver.HybridStrategy,
    }
    
    strategyNames := []string{"Backtracking", "Constraint Propagation", "Hybrid"}
    
    fmt.Println("Multi-Puzzle Analysis Results:")
    fmt.Println("================================")
    
    for i, strategy := range strategies {
        fmt.Printf("\n%s Strategy:\n", strategyNames[i])
        
        config := parallelsudokusolver.SolverConfig{
            Strategy:       strategy,
            MaxWorkers:     4,
            TimeLimit:      20 * time.Second,
            UseHeuristics:  true,
            EnablePruning:  true,
        }
        
        var totalTime time.Duration
        var solved int
        var totalBacktracks int64
        var totalConstraints int64
        
        for j, puzzle := range puzzles {
            solver := parallelsudokusolver.NewParallelSudokuSolver(config)
            
            start := time.Now()
            ctx := context.Background()
            solution, stats, err := solver.Solve(ctx, puzzle)
            duration := time.Since(start)
            
            if err == nil && solution != nil {
                solved++
                totalTime += duration
                totalBacktracks += stats.BacktrackCount
                totalConstraints += stats.ConstraintChecks
                fmt.Printf("  Puzzle %d: ✓ %v\n", j+1, duration)
            } else {
                fmt.Printf("  Puzzle %d: ✗ %v\n", j+1, err)
            }
        }
        
        fmt.Printf("Summary:\n")
        fmt.Printf("  Solved: %d/%d (%.1f%%)\n", solved, len(puzzles), 
            float64(solved)/float64(len(puzzles))*100)
        
        if solved > 0 {
            avgTime := totalTime / time.Duration(solved)
            fmt.Printf("  Average time: %v\n", avgTime)
            fmt.Printf("  Average backtracks: %.0f\n", float64(totalBacktracks)/float64(solved))
            fmt.Printf("  Average constraints: %.0f\n", float64(totalConstraints)/float64(solved))
        }
    }
}
```

## Architecture

### Core Components

1. **ParallelSudokuSolver**: Main solver coordinator
   - Strategy selection and configuration
   - Worker pool management
   - Statistics collection and analysis
   - Context and timeout handling

2. **SolverWorker**: Individual solving worker
   - Task processing and result generation
   - Strategy-specific solving logic
   - Local statistics accumulation
   - Error handling and reporting

3. **SudokuBoard**: Board representation and operations
   - 9x9 grid with validation
   - Pretty printing and visualization
   - Cell and constraint management

4. **Solving Strategies**: Multiple algorithmic approaches
   - Backtracking with parallel distribution
   - Constraint propagation logic
   - Hybrid approach optimization
   - Brute force for small problems

### Concurrency Model

- **Worker Pool Architecture**: Fixed number of workers processing solving tasks
- **Channel-based Communication**: Work distribution and result collection
- **Parallel Strategy Execution**: Multiple workers explore different solution paths
- **Context-based Cancellation**: Graceful termination and timeout handling
- **Lock-free Statistics**: Atomic operations for performance metrics

### Algorithm Implementation

#### Backtracking Strategy
```
1. Identify empty cells and sort by constraint count
2. Distribute initial value assignments across workers
3. Each worker performs recursive backtracking
4. Apply heuristics for value ordering (if enabled)
5. Use pruning to eliminate impossible branches
6. Return first valid solution found
```

#### Constraint Propagation
```
1. Analyze current board state
2. Identify cells with single possible values
3. Fill determined cells iteratively
4. Detect constraint violations early
5. Fall back to backtracking if stuck
```

#### Hybrid Strategy
```
1. Apply constraint propagation preprocessing
2. Use single-threaded backtracking for small problems
3. Use parallel backtracking for complex problems
4. Combine heuristics and pruning optimizations
```

## Configuration Options

### SolverConfig Parameters

```go
type SolverConfig struct {
    Strategy       SolverStrategy // Solving algorithm
    MaxWorkers     int           // Parallel workers (1-CPU cores)
    TimeLimit      time.Duration // Maximum solving time
    UseHeuristics  bool         // Enable intelligent value ordering
    EnablePruning  bool         // Enable branch pruning
    LogProgress    bool         // Enable progress logging
    RandomSeed     int64        // Seed for reproducible results
}
```

### Performance Tuning Guidelines

- **Workers**: 
  - CPU-bound: Use CPU core count
  - Memory-limited: Use fewer workers (2-4)
  - I/O bound: Can exceed CPU cores

- **Strategy Selection**:
  - Easy puzzles: Constraint Propagation
  - Medium puzzles: Hybrid Strategy  
  - Hard puzzles: Backtracking or Hybrid
  - Very large search spaces: Brute Force (if small enough)

- **Optimization Flags**:
  - Heuristics: Generally beneficial for complex puzzles
  - Pruning: Reduces search space but adds overhead
  - Progress Logging: Useful for debugging but impacts performance

## Testing

Run the comprehensive test suite:

```bash
go test -v ./parallelsudokusolver/
```

Run benchmarks:

```bash
go test -bench=. ./parallelsudokusolver/
```

Run race condition detection:

```bash
go test -race ./parallelsudokusolver/
```

### Test Coverage

- Solver creation and configuration validation
- Board validation and puzzle generation
- Multiple solving strategies and algorithms
- Concurrent solving and worker coordination
- Statistics collection and accuracy
- Context cancellation and timeout handling
- Error scenarios and edge cases
- Performance benchmarking under various conditions

## Performance Characteristics

### Computational Complexity

| Strategy | Time Complexity | Space Complexity | Parallelization |
|----------|----------------|------------------|-----------------|
| Backtracking | O(9^n) worst case | O(n) | Excellent |
| Constraint Propagation | O(n³) per iteration | O(n) | Limited |
| Hybrid | O(n³ + 9^m) | O(n) | Good |
| Brute Force | O(9^n) | O(n) | Perfect |

Where n = number of empty cells, m = cells after constraint propagation.

### Typical Performance

| Puzzle Difficulty | Empty Cells | Solve Time | Success Rate |
|------------------|-------------|------------|--------------|
| Easy (35 empty)  | 35          | 1-10ms     | >99%         |
| Medium (45 empty)| 45          | 10-100ms   | >95%         |
| Hard (55 empty)  | 55          | 100ms-1s   | >90%         |
| Expert (65 empty)| 65          | 1s-10s     | >80%         |

### Scaling Characteristics

- **Linear speedup** with worker count up to CPU cores
- **Diminishing returns** beyond optimal worker count
- **Memory usage** scales with worker count and puzzle complexity
- **Best performance** achieved with Hybrid strategy for most puzzles

## Use Cases

1. **Puzzle Games**: Real-time Sudoku solving for mobile apps and games
2. **Educational Tools**: Teaching constraint satisfaction and backtracking algorithms
3. **Puzzle Generation**: Creating valid Sudoku puzzles with guaranteed solutions
4. **Algorithm Research**: Comparing different constraint satisfaction approaches
5. **Performance Testing**: Benchmarking parallel computing architectures
6. **AI Training**: Generating training data for machine learning models
7. **Competitive Programming**: Solving optimization problems with similar constraints

## Limitations

This implementation focuses on classic 9x9 Sudoku puzzles:

- No support for variant Sudoku types (Killer, Samurai, etc.)
- No advanced constraint types beyond standard Sudoku rules
- No puzzle difficulty estimation beyond empty cell count
- No solution uniqueness verification for generated puzzles
- No persistent storage or database integration

## Future Enhancements

### Algorithm Improvements
- **Advanced Heuristics**: Implement more sophisticated value ordering
- **Dancing Links**: Exact cover algorithm implementation
- **SAT Solver Integration**: Boolean satisfiability approach
- **Machine Learning**: Neural network-based solving guidance

### Feature Extensions
- **Variant Support**: Killer Sudoku, Samurai, and other variants
- **Solution Verification**: Ensure generated puzzles have unique solutions
- **Difficulty Analysis**: Sophisticated difficulty rating algorithms
- **Interactive Solving**: Step-by-step solution explanation

### Performance Optimizations
- **GPU Acceleration**: CUDA/OpenCL implementation for massive parallelization
- **SIMD Instructions**: Vectorized operations for constraint checking
- **Memory Optimization**: Reduced memory allocation and garbage collection
- **Distributed Solving**: Multi-machine solving for extremely difficult puzzles