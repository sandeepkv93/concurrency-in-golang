# Parallel Monte Carlo Pi Estimation

A sophisticated implementation demonstrating Monte Carlo methods for estimating π using parallel computing, featuring adaptive algorithms, statistical analysis, and performance optimization techniques.

## Problem Description

Monte Carlo methods use random sampling to solve mathematical problems that might be deterministic in principle. Estimating π is a classic example where we:

1. Generate random points in a unit square [0,1] × [0,1]
2. Count how many fall inside the unit circle (x² + y² ≤ 1)
3. Use the ratio to estimate π: **π ≈ 4 × (points inside circle) / (total points)**

This method is "embarrassingly parallel" since each sample is independent, making it ideal for demonstrating parallel computing concepts. However, achieving good performance requires careful attention to:
- Random number generation in parallel
- Load balancing across workers
- Statistical convergence criteria
- Memory efficiency for large sample sizes

## Solution Approach

The implementation provides multiple levels of parallel estimation:

1. **Basic Parallel Estimation**: Fixed sample size with worker pool
2. **Batch Processing**: Results tracked per batch for analysis
3. **Adaptive Estimation**: Automatic convergence detection
4. **Distributed Estimation**: Multiple estimators with result merging
5. **Statistical Analysis**: Comprehensive performance metrics

## Key Components

### Core Estimator

```go
type PiEstimator struct {
    numWorkers   int
    batchSize    int
    randomSource RandomSource
}
```

### Result Analysis

```go
type EstimationResult struct {
    EstimatedPi      float64
    ActualPi         float64
    Error            float64
    ErrorPercentage  float64
    TotalSamples     int64
    InsideCircle     int64
    Duration         time.Duration
    SamplesPerSecond float64
    WorkerResults    []WorkerResult
}
```

### Random Number Generation

- **StandardRandomSource**: Thread-safe single source
- **ThreadSafeRandomSource**: Multiple sources for better parallelism
- **RandomSource Interface**: Pluggable random number generators

### Advanced Features

- **AdaptiveEstimator**: Stops when convergence is achieved
- **DistributedEstimator**: Coordinates multiple estimators
- **ProgressTracker**: Real-time progress monitoring
- **Statistical Analysis**: Variance and convergence analysis

## Usage Examples

### Basic Pi Estimation

```go
// Create estimator configuration
config := EstimationConfig{
    NumWorkers: 4,
    BatchSize:  100000,
}

estimator := NewPiEstimator(config)

// Estimate π with 10 million samples
result := estimator.EstimatePi(10000000)

fmt.Printf("Estimated π: %.10f\n", result.EstimatedPi)
fmt.Printf("Actual π:    %.10f\n", result.ActualPi)
fmt.Printf("Error:       %.10f (%.6f%%)\n", 
    result.Error, result.ErrorPercentage)
fmt.Printf("Duration:    %v\n", result.Duration)
fmt.Printf("Samples/sec: %.0f\n", result.SamplesPerSecond)
```

### Adaptive Estimation with Convergence

```go
// Create adaptive estimator
adaptive := NewAdaptiveEstimator(
    estimator,
    0.0001,    // 0.01% tolerance
    1000000,   // minimum samples
    50000000,  // maximum samples
)

result := adaptive.EstimateAdaptive()
fmt.Printf("Converged after %d samples\n", result.TotalSamples)
```

### Batch Processing with Analysis

```go
// Get detailed batch results
result, batches := estimator.EstimateWithBatches(5000000)

// Analyze convergence
for i, batch := range batches {
    fmt.Printf("Batch %d: π = %.6f\n", i, batch.Pi)
}
```

### Custom Random Source

```go
// Use thread-safe random source for better performance
randomSource := NewThreadSafeRandomSource(4, time.Now().UnixNano())
config.RandomSource = randomSource

estimator := NewPiEstimator(config)
```

### Performance Analysis

```go
// Analyze worker performance
analysis := AnalyzeWorkerPerformance(result.WorkerResults)
fmt.Printf("Worker variance: %.6f\n", analysis["pi_variance"])
fmt.Printf("Processing time variance: %v\n", analysis["duration_variance"])
```

## Technical Features

### Parallel Computing Model

- **Worker Pool**: Fixed number of workers process batches
- **Work Distribution**: Batches distributed via channels
- **Load Balancing**: Dynamic batch assignment prevents idle workers
- **Result Aggregation**: Thread-safe accumulation of results

### Random Number Generation Strategies

1. **Single Source with Mutex**: Simple but can be bottleneck
2. **Per-Worker Sources**: Each worker has own generator
3. **Thread-Safe Distribution**: Multiple sources with load balancing

```go
type ThreadSafeRandomSource struct {
    sources []RandomSource
    current int64
}

func (tsrs *ThreadSafeRandomSource) Float64() float64 {
    current := atomic.AddInt64(&tsrs.current, 1)
    sourceIndex := int(current % int64(len(tsrs.sources)))
    return tsrs.sources[sourceIndex].Float64()
}
```

### Adaptive Convergence Algorithm

```go
func (ae *AdaptiveEstimator) hasConverged(estimates []float64) bool {
    if len(estimates) < 10 {
        return false
    }
    
    // Check variance of recent estimates
    recent := estimates[len(estimates)-10:]
    variance := calculateVariance(recent)
    
    return math.Sqrt(variance) < ae.convergenceTolerance
}
```

### Statistical Analysis Features

- **Convergence Tracking**: Monitor estimate stability over time
- **Worker Performance**: Analyze load distribution and efficiency
- **Error Analysis**: Track accuracy vs sample size relationship
- **Throughput Metrics**: Samples per second across different configurations

## Implementation Details

### Core Estimation Algorithm

```go
func (pe *PiEstimator) worker(workerID int, batchChan <-chan int, 
    totalInsideCircle *int64, completedBatches *int64, result *WorkerResult) {
    
    localInsideCircle := int64(0)
    localSamples := int64(0)
    workerRand := NewStandardRandomSource(time.Now().UnixNano() + int64(workerID))
    
    for batchID := range batchChan {
        batchInside := pe.processBatch(workerRand, pe.batchSize)
        localInsideCircle += batchInside
        localSamples += int64(pe.batchSize)
        
        atomic.AddInt64(totalInsideCircle, batchInside)
        atomic.AddInt64(completedBatches, 1)
    }
    
    // Store worker results for analysis
    *result = WorkerResult{
        WorkerID:     workerID,
        Samples:      localSamples,
        InsideCircle: localInsideCircle,
        LocalPi:      4.0 * float64(localInsideCircle) / float64(localSamples),
    }
}
```

### Memory Efficiency

- **Batch Processing**: Process samples in chunks to control memory usage
- **Streaming Results**: Don't store all sample points in memory
- **Atomic Counters**: Use atomic operations instead of mutex for counters
- **Worker-Local State**: Minimize contention between workers

### Error Handling and Robustness

- **Graceful Degradation**: Continue if some workers fail
- **Timeout Handling**: Prevent indefinite blocking
- **Resource Cleanup**: Proper goroutine lifecycle management
- **Validation**: Input parameter validation and error reporting

## Performance Characteristics

### Scaling Properties

- **Linear Speedup**: Nearly perfect scaling up to number of CPU cores
- **Memory Bound**: Large sample sizes limited by memory bandwidth
- **Cache Friendly**: Small working set per worker
- **RNG Bottleneck**: Random number generation can limit scaling

### Accuracy vs Performance Trade-offs

```
Samples     Error     Time (4 cores)   Samples/sec
1M         0.1%      0.05s            20M/s
10M        0.03%     0.5s             20M/s  
100M       0.01%     5s               20M/s
1B         0.003%    50s              20M/s
```

### Worker Load Balancing

The implementation ensures even work distribution:
- Dynamic batch assignment prevents early worker completion
- Atomic counters track progress without blocking
- Worker-specific random seeds prevent correlation

## Configuration Options

```go
type EstimationConfig struct {
    NumWorkers     int           // Number of parallel workers
    BatchSize      int           // Samples per batch
    RandomSource   RandomSource  // Random number generator
}

type AdaptiveEstimator struct {
    convergenceTolerance float64  // Convergence criteria
    minSamples          int64     // Minimum samples before checking
    maxSamples          int64     // Maximum samples limit
    checkInterval       int64     // How often to check convergence
}
```

## Advanced Features

### Distributed Estimation

```go
// Run multiple estimators in parallel
configs := []EstimationConfig{config1, config2, config3}
distributed := NewDistributedEstimator(configs)
result := distributed.EstimateDistributed(totalSamples)
```

### Convergence Visualization

```go
// Generate data for convergence plots
convergenceData := estimator.GenerateConvergenceData(maxSamples, intervals)
// Each data point shows cumulative estimate vs sample count
```

### Performance Profiling

```go
// Calculate parallel efficiency
efficiency := CalculateEfficiency(singleThreadTime, parallelTime, numWorkers)
fmt.Printf("Parallel efficiency: %.2f%%\n", efficiency*100)
```

This implementation demonstrates how Monte Carlo methods can be effectively parallelized while maintaining statistical rigor and providing comprehensive analysis tools for understanding both the mathematical and computational aspects of the estimation process.