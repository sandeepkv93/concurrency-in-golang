package parallelmontecarlopiestimation

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// PiEstimator handles parallel Monte Carlo Pi estimation
type PiEstimator struct {
	numWorkers   int
	batchSize    int
	randomSource RandomSource
}

// RandomSource provides random number generation interface
type RandomSource interface {
	Float64() float64
	Seed(seed int64)
}

// EstimationConfig holds configuration for Pi estimation
type EstimationConfig struct {
	NumWorkers   int
	BatchSize    int
	RandomSource RandomSource
}

// EstimationResult contains the results of Pi estimation
type EstimationResult struct {
	EstimatedPi      float64
	ActualPi         float64
	Error            float64
	ErrorPercentage  float64
	TotalSamples     int64
	InsideCircle     int64
	Duration         time.Duration
	SamplesPerSecond float64
	Batches          int
	WorkerResults    []WorkerResult
}

// WorkerResult contains per-worker estimation results
type WorkerResult struct {
	WorkerID     int
	Samples      int64
	InsideCircle int64
	Duration     time.Duration
	LocalPi      float64
}

// BatchResult represents results from a single batch
type BatchResult struct {
	BatchID      int
	Samples      int64
	InsideCircle int64
	Duration     time.Duration
	Pi           float64
}

// AdaptiveEstimator performs adaptive estimation with convergence criteria
type AdaptiveEstimator struct {
	estimator         *PiEstimator
	convergenceTolerance float64
	minSamples        int64
	maxSamples        int64
	checkInterval     int64
}

// DistributedEstimator coordinates multiple estimators
type DistributedEstimator struct {
	estimators []PiEstimator
	merger     ResultMerger
}

// ResultMerger combines results from multiple estimators
type ResultMerger interface {
	Merge(results []EstimationResult) EstimationResult
}

// StandardRandomSource provides standard Go random number generation
type StandardRandomSource struct {
	rng *rand.Rand
	mu  sync.Mutex
}

// ThreadSafeRandomSource provides thread-safe random number generation
type ThreadSafeRandomSource struct {
	sources []RandomSource
	current int64
}

// NewPiEstimator creates a new Pi estimator
func NewPiEstimator(config EstimationConfig) *PiEstimator {
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 100000
	}
	if config.RandomSource == nil {
		config.RandomSource = NewStandardRandomSource(time.Now().UnixNano())
	}

	return &PiEstimator{
		numWorkers:   config.NumWorkers,
		batchSize:    config.BatchSize,
		randomSource: config.RandomSource,
	}
}

// EstimatePi estimates Pi using Monte Carlo method with specified number of samples
func (pe *PiEstimator) EstimatePi(totalSamples int64) *EstimationResult {
	start := time.Now()

	// Calculate batches
	totalBatches := int((totalSamples + int64(pe.batchSize) - 1) / int64(pe.batchSize))
	actualSamples := int64(totalBatches) * int64(pe.batchSize)

	// Shared counters
	var totalInsideCircle int64
	var completedBatches int64

	// Worker results
	workerResults := make([]WorkerResult, pe.numWorkers)
	var wg sync.WaitGroup

	// Create batch channel
	batchChan := make(chan int, totalBatches)

	// Start workers
	for i := 0; i < pe.numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			pe.worker(workerID, batchChan, &totalInsideCircle, &completedBatches, &workerResults[workerID])
		}(i)
	}

	// Send batches to workers
	for i := 0; i < totalBatches; i++ {
		batchChan <- i
	}
	close(batchChan)

	// Wait for completion
	wg.Wait()

	duration := time.Since(start)
	insideCircle := atomic.LoadInt64(&totalInsideCircle)

	// Calculate Pi
	estimatedPi := 4.0 * float64(insideCircle) / float64(actualSamples)
	actualPi := math.Pi
	error := math.Abs(estimatedPi - actualPi)
	errorPercentage := (error / actualPi) * 100

	return &EstimationResult{
		EstimatedPi:      estimatedPi,
		ActualPi:         actualPi,
		Error:            error,
		ErrorPercentage:  errorPercentage,
		TotalSamples:     actualSamples,
		InsideCircle:     insideCircle,
		Duration:         duration,
		SamplesPerSecond: float64(actualSamples) / duration.Seconds(),
		Batches:          totalBatches,
		WorkerResults:    workerResults,
	}
}

func (pe *PiEstimator) worker(workerID int, batchChan <-chan int, totalInsideCircle *int64, completedBatches *int64, result *WorkerResult) {
	start := time.Now()
	localInsideCircle := int64(0)
	localSamples := int64(0)

	// Create worker-specific random source
	workerRand := NewStandardRandomSource(time.Now().UnixNano() + int64(workerID))

	for _ = range batchChan {
		batchInside := pe.processBatch(workerRand, pe.batchSize)
		localInsideCircle += batchInside
		localSamples += int64(pe.batchSize)

		atomic.AddInt64(totalInsideCircle, batchInside)
		atomic.AddInt64(completedBatches, 1)
	}

	duration := time.Since(start)
	localPi := 4.0 * float64(localInsideCircle) / float64(localSamples)

	*result = WorkerResult{
		WorkerID:     workerID,
		Samples:      localSamples,
		InsideCircle: localInsideCircle,
		Duration:     duration,
		LocalPi:      localPi,
	}
}

func (pe *PiEstimator) processBatch(rng RandomSource, batchSize int) int64 {
	insideCircle := int64(0)

	for i := 0; i < batchSize; i++ {
		x := rng.Float64()
		y := rng.Float64()

		// Check if point is inside unit circle
		if x*x+y*y <= 1.0 {
			insideCircle++
		}
	}

	return insideCircle
}

// EstimateWithBatches estimates Pi and provides batch-level results
func (pe *PiEstimator) EstimateWithBatches(totalSamples int64) (*EstimationResult, []BatchResult) {
	start := time.Now()

	// Calculate batches
	totalBatches := int((totalSamples + int64(pe.batchSize) - 1) / int64(pe.batchSize))
	actualSamples := int64(totalBatches) * int64(pe.batchSize)

	// Channels for batch results
	batchResultChan := make(chan BatchResult, totalBatches)
	batchChan := make(chan int, totalBatches)

	// Shared counters
	var totalInsideCircle int64
	workerResults := make([]WorkerResult, pe.numWorkers)
	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < pe.numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			pe.batchWorker(workerID, batchChan, batchResultChan, &totalInsideCircle, &workerResults[workerID])
		}(i)
	}

	// Send batches
	for i := 0; i < totalBatches; i++ {
		batchChan <- i
	}
	close(batchChan)

	// Wait for workers
	wg.Wait()
	close(batchResultChan)

	// Collect batch results
	batchResults := make([]BatchResult, 0, totalBatches)
	for batch := range batchResultChan {
		batchResults = append(batchResults, batch)
	}

	duration := time.Since(start)
	insideCircle := atomic.LoadInt64(&totalInsideCircle)

	// Calculate final result
	estimatedPi := 4.0 * float64(insideCircle) / float64(actualSamples)
	actualPi := math.Pi
	error := math.Abs(estimatedPi - actualPi)
	errorPercentage := (error / actualPi) * 100

	result := &EstimationResult{
		EstimatedPi:      estimatedPi,
		ActualPi:         actualPi,
		Error:            error,
		ErrorPercentage:  errorPercentage,
		TotalSamples:     actualSamples,
		InsideCircle:     insideCircle,
		Duration:         duration,
		SamplesPerSecond: float64(actualSamples) / duration.Seconds(),
		Batches:          totalBatches,
		WorkerResults:    workerResults,
	}

	return result, batchResults
}

func (pe *PiEstimator) batchWorker(workerID int, batchChan <-chan int, resultChan chan<- BatchResult, totalInsideCircle *int64, result *WorkerResult) {
	start := time.Now()
	localInsideCircle := int64(0)
	localSamples := int64(0)

	// Worker-specific random source
	workerRand := NewStandardRandomSource(time.Now().UnixNano() + int64(workerID))
	batchCounter := 0

	for _ = range batchChan {
		batchID := batchCounter
		batchCounter++
		batchStart := time.Now()
		batchInside := pe.processBatch(workerRand, pe.batchSize)
		batchDuration := time.Since(batchStart)

		localInsideCircle += batchInside
		localSamples += int64(pe.batchSize)

		atomic.AddInt64(totalInsideCircle, batchInside)

		batchPi := 4.0 * float64(batchInside) / float64(pe.batchSize)

		resultChan <- BatchResult{
			BatchID:      batchID,
			Samples:      int64(pe.batchSize),
			InsideCircle: batchInside,
			Duration:     batchDuration,
			Pi:           batchPi,
		}
	}

	duration := time.Since(start)
	localPi := 4.0 * float64(localInsideCircle) / float64(localSamples)

	*result = WorkerResult{
		WorkerID:     workerID,
		Samples:      localSamples,
		InsideCircle: localInsideCircle,
		Duration:     duration,
		LocalPi:      localPi,
	}
}

// NewAdaptiveEstimator creates an adaptive Pi estimator
func NewAdaptiveEstimator(estimator *PiEstimator, tolerance float64, minSamples, maxSamples int64) *AdaptiveEstimator {
	if tolerance <= 0 {
		tolerance = 0.001 // 0.1% default tolerance
	}
	if minSamples <= 0 {
		minSamples = 100000
	}
	if maxSamples <= 0 {
		maxSamples = 100000000 // 100M samples max
	}

	return &AdaptiveEstimator{
		estimator:         estimator,
		convergenceTolerance: tolerance,
		minSamples:        minSamples,
		maxSamples:        maxSamples,
		checkInterval:     minSamples / 10, // Check every 10% of minimum samples
	}
}

// EstimateAdaptive estimates Pi with adaptive convergence
func (ae *AdaptiveEstimator) EstimateAdaptive() *EstimationResult {
	start := time.Now()
	var totalInsideCircle int64
	var totalSamples int64

	estimates := make([]float64, 0)
	batchSize := int64(ae.estimator.batchSize)

	for totalSamples < ae.maxSamples {
		// Process a batch
		batchResult := ae.estimator.EstimatePi(batchSize)
		totalInsideCircle += batchResult.InsideCircle
		totalSamples += batchResult.TotalSamples

		// Calculate current estimate
		currentPi := 4.0 * float64(totalInsideCircle) / float64(totalSamples)
		estimates = append(estimates, currentPi)

		// Check for convergence after minimum samples
		if totalSamples >= ae.minSamples && totalSamples%ae.checkInterval == 0 {
			if ae.hasConverged(estimates) {
				break
			}
		}
	}

	duration := time.Since(start)
	estimatedPi := 4.0 * float64(totalInsideCircle) / float64(totalSamples)
	actualPi := math.Pi
	error := math.Abs(estimatedPi - actualPi)
	errorPercentage := (error / actualPi) * 100

	return &EstimationResult{
		EstimatedPi:      estimatedPi,
		ActualPi:         actualPi,
		Error:            error,
		ErrorPercentage:  errorPercentage,
		TotalSamples:     totalSamples,
		InsideCircle:     totalInsideCircle,
		Duration:         duration,
		SamplesPerSecond: float64(totalSamples) / duration.Seconds(),
	}
}

func (ae *AdaptiveEstimator) hasConverged(estimates []float64) bool {
	if len(estimates) < 10 {
		return false
	}

	// Check last 10 estimates for stability
	recent := estimates[len(estimates)-10:]
	mean := 0.0
	for _, est := range recent {
		mean += est
	}
	mean /= float64(len(recent))

	// Calculate variance
	variance := 0.0
	for _, est := range recent {
		diff := est - mean
		variance += diff * diff
	}
	variance /= float64(len(recent))

	// Check if variance is within tolerance
	return math.Sqrt(variance) < ae.convergenceTolerance
}

// Standard Random Source Implementation

func NewStandardRandomSource(seed int64) *StandardRandomSource {
	return &StandardRandomSource{
		rng: rand.New(rand.NewSource(seed)),
	}
}

func (srs *StandardRandomSource) Float64() float64 {
	srs.mu.Lock()
	defer srs.mu.Unlock()
	return srs.rng.Float64()
}

func (srs *StandardRandomSource) Seed(seed int64) {
	srs.mu.Lock()
	defer srs.mu.Unlock()
	srs.rng.Seed(seed)
}

// Thread-Safe Random Source Implementation

func NewThreadSafeRandomSource(numSources int, baseSeed int64) *ThreadSafeRandomSource {
	if numSources <= 0 {
		numSources = runtime.NumCPU()
	}

	sources := make([]RandomSource, numSources)
	for i := 0; i < numSources; i++ {
		sources[i] = NewStandardRandomSource(baseSeed + int64(i))
	}

	return &ThreadSafeRandomSource{
		sources: sources,
	}
}

func (tsrs *ThreadSafeRandomSource) Float64() float64 {
	// Use atomic counter to distribute load across sources
	current := atomic.AddInt64(&tsrs.current, 1)
	sourceIndex := int(current % int64(len(tsrs.sources)))
	return tsrs.sources[sourceIndex].Float64()
}

func (tsrs *ThreadSafeRandomSource) Seed(seed int64) {
	for i, source := range tsrs.sources {
		source.Seed(seed + int64(i))
	}
}

// Standard Result Merger Implementation

type StandardResultMerger struct{}

func (srm StandardResultMerger) Merge(results []EstimationResult) EstimationResult {
	if len(results) == 0 {
		return EstimationResult{}
	}

	var totalSamples int64
	var totalInsideCircle int64
	var totalDuration time.Duration

	for _, result := range results {
		totalSamples += result.TotalSamples
		totalInsideCircle += result.InsideCircle
		if result.Duration > totalDuration {
			totalDuration = result.Duration
		}
	}

	estimatedPi := 4.0 * float64(totalInsideCircle) / float64(totalSamples)
	actualPi := math.Pi
	error := math.Abs(estimatedPi - actualPi)
	errorPercentage := (error / actualPi) * 100

	return EstimationResult{
		EstimatedPi:      estimatedPi,
		ActualPi:         actualPi,
		Error:            error,
		ErrorPercentage:  errorPercentage,
		TotalSamples:     totalSamples,
		InsideCircle:     totalInsideCircle,
		Duration:         totalDuration,
		SamplesPerSecond: float64(totalSamples) / totalDuration.Seconds(),
	}
}

// NewDistributedEstimator creates a distributed estimator
func NewDistributedEstimator(configs []EstimationConfig) *DistributedEstimator {
	estimators := make([]PiEstimator, len(configs))
	for i, config := range configs {
		estimators[i] = *NewPiEstimator(config)
	}

	return &DistributedEstimator{
		estimators: estimators,
		merger:     StandardResultMerger{},
	}
}

// EstimateDistributed performs distributed Pi estimation
func (de *DistributedEstimator) EstimateDistributed(totalSamples int64) *EstimationResult {
	samplesPerEstimator := totalSamples / int64(len(de.estimators))
	results := make([]EstimationResult, len(de.estimators))
	var wg sync.WaitGroup

	// Run estimators in parallel
	for i := range de.estimators {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = *de.estimators[idx].EstimatePi(samplesPerEstimator)
		}(i)
	}

	wg.Wait()
	result := de.merger.Merge(results)
	return &result
}

// Visualization and Statistics

// GenerateConvergenceData generates data showing Pi estimation convergence
func (pe *PiEstimator) GenerateConvergenceData(maxSamples int64, intervals int) []EstimationResult {
	intervalSize := maxSamples / int64(intervals)
	results := make([]EstimationResult, intervals)

	var cumulativeInside int64
	var cumulativeSamples int64

	for i := 0; i < intervals; i++ {
		// Estimate for this interval
		intervalResult := pe.EstimatePi(intervalSize)
		
		// Update cumulative counters
		cumulativeInside += intervalResult.InsideCircle
		cumulativeSamples += intervalResult.TotalSamples

		// Calculate cumulative Pi estimate
		cumulativePi := 4.0 * float64(cumulativeInside) / float64(cumulativeSamples)
		actualPi := math.Pi
		error := math.Abs(cumulativePi - actualPi)

		results[i] = EstimationResult{
			EstimatedPi:     cumulativePi,
			ActualPi:        actualPi,
			Error:           error,
			ErrorPercentage: (error / actualPi) * 100,
			TotalSamples:    cumulativeSamples,
			InsideCircle:    cumulativeInside,
		}
	}

	return results
}

// AnalyzeWorkerPerformance analyzes performance across workers
func AnalyzeWorkerPerformance(results []WorkerResult) map[string]interface{} {
	if len(results) == 0 {
		return nil
	}

	analysis := make(map[string]interface{})

	// Calculate statistics
	totalSamples := int64(0)
	totalDuration := time.Duration(0)
	minDuration := results[0].Duration
	maxDuration := results[0].Duration

	piEstimates := make([]float64, len(results))

	for i, result := range results {
		totalSamples += result.Samples
		totalDuration += result.Duration
		piEstimates[i] = result.LocalPi

		if result.Duration < minDuration {
			minDuration = result.Duration
		}
		if result.Duration > maxDuration {
			maxDuration = result.Duration
		}
	}

	avgDuration := totalDuration / time.Duration(len(results))

	// Calculate Pi estimate variance
	meanPi := 0.0
	for _, pi := range piEstimates {
		meanPi += pi
	}
	meanPi /= float64(len(piEstimates))

	variance := 0.0
	for _, pi := range piEstimates {
		diff := pi - meanPi
		variance += diff * diff
	}
	variance /= float64(len(piEstimates))

	analysis["total_workers"] = len(results)
	analysis["total_samples"] = totalSamples
	analysis["avg_duration"] = avgDuration
	analysis["min_duration"] = minDuration
	analysis["max_duration"] = maxDuration
	analysis["duration_variance"] = maxDuration - minDuration
	analysis["pi_variance"] = variance
	analysis["pi_std_dev"] = math.Sqrt(variance)

	return analysis
}

// CalculateEfficiency calculates parallel efficiency
func CalculateEfficiency(singleThreadTime, parallelTime time.Duration, numWorkers int) float64 {
	if parallelTime == 0 {
		return 0
	}
	
	speedup := float64(singleThreadTime) / float64(parallelTime)
	efficiency := speedup / float64(numWorkers)
	return efficiency
}

// Example demonstrates parallel Monte Carlo Pi estimation
func Example() {
	fmt.Println("=== Parallel Monte Carlo Pi Estimation Example ===")

	// Create estimator
	config := EstimationConfig{
		NumWorkers: 4,
		BatchSize:  100000,
	}

	estimator := NewPiEstimator(config)

	// Estimate Pi with different sample sizes
	sampleSizes := []int64{1000000, 10000000, 100000000}

	for _, samples := range sampleSizes {
		fmt.Printf("\nEstimating Pi with %d samples...\n", samples)
		
		result := estimator.EstimatePi(samples)
		
		fmt.Printf("Results:\n")
		fmt.Printf("  Estimated Pi: %.10f\n", result.EstimatedPi)
		fmt.Printf("  Actual Pi:    %.10f\n", result.ActualPi)
		fmt.Printf("  Error:        %.10f (%.6f%%)\n", result.Error, result.ErrorPercentage)
		fmt.Printf("  Duration:     %v\n", result.Duration)
		fmt.Printf("  Samples/sec:  %.0f\n", result.SamplesPerSecond)
		fmt.Printf("  Workers:      %d\n", len(result.WorkerResults))

		// Show worker performance
		fmt.Printf("  Worker Performance:\n")
		for _, worker := range result.WorkerResults {
			fmt.Printf("    Worker %d: %.6f (%.0f samples/sec)\n",
				worker.WorkerID, worker.LocalPi, 
				float64(worker.Samples)/worker.Duration.Seconds())
		}
	}

	// Demonstrate adaptive estimation
	fmt.Println("\nAdaptive Estimation:")
	adaptive := NewAdaptiveEstimator(estimator, 0.0001, 1000000, 50000000)
	adaptiveResult := adaptive.EstimateAdaptive()
	
	fmt.Printf("  Adaptive Pi:  %.10f\n", adaptiveResult.EstimatedPi)
	fmt.Printf("  Error:        %.10f (%.6f%%)\n", adaptiveResult.Error, adaptiveResult.ErrorPercentage)
	fmt.Printf("  Samples used: %d\n", adaptiveResult.TotalSamples)
	fmt.Printf("  Duration:     %v\n", adaptiveResult.Duration)
}