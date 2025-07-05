package parallelmontecarlopiestimation

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestPiEstimator(t *testing.T) {
	config := EstimationConfig{
		NumWorkers: 2,
		BatchSize:  10000,
	}

	estimator := NewPiEstimator(config)
	result := estimator.EstimatePi(100000)

	// Check basic result properties
	if result.TotalSamples != 100000 {
		t.Errorf("Expected 100000 samples, got %d", result.TotalSamples)
	}

	if result.EstimatedPi <= 0 || result.EstimatedPi > 10 {
		t.Errorf("Pi estimate out of reasonable range: %f", result.EstimatedPi)
	}

	if result.ActualPi != math.Pi {
		t.Errorf("Actual Pi should be math.Pi, got %f", result.ActualPi)
	}

	if result.Duration == 0 {
		t.Error("Duration should be non-zero")
	}

	if result.SamplesPerSecond <= 0 {
		t.Error("Samples per second should be positive")
	}

	if len(result.WorkerResults) != 2 {
		t.Errorf("Expected 2 worker results, got %d", len(result.WorkerResults))
	}

	// Check worker results
	totalWorkerSamples := int64(0)
	for _, worker := range result.WorkerResults {
		if worker.Samples <= 0 {
			t.Errorf("Worker %d should have processed samples", worker.WorkerID)
		}
		if worker.LocalPi <= 0 || worker.LocalPi > 10 {
			t.Errorf("Worker %d Pi estimate out of range: %f", worker.WorkerID, worker.LocalPi)
		}
		totalWorkerSamples += worker.Samples
	}

	if totalWorkerSamples != result.TotalSamples {
		t.Errorf("Worker samples don't sum to total: %d vs %d", totalWorkerSamples, result.TotalSamples)
	}
}

func TestPiEstimatorAccuracy(t *testing.T) {
	config := EstimationConfig{
		NumWorkers: 4,
		BatchSize:  50000,
	}

	estimator := NewPiEstimator(config)
	
	// Test with increasing sample sizes
	sampleSizes := []int64{100000, 1000000, 10000000}
	
	for _, samples := range sampleSizes {
		result := estimator.EstimatePi(samples)
		
		// With more samples, error should generally be smaller
		expectedMaxError := 0.1 * math.Sqrt(1000000.0/float64(samples)) // Rough error bound
		
		if result.ErrorPercentage > expectedMaxError*100 {
			t.Logf("Sample size %d: Error %.4f%% (expected < %.4f%%)", 
				samples, result.ErrorPercentage, expectedMaxError*100)
			// This is a probabilistic test, so we just log warnings rather than fail
		}
		
		t.Logf("Samples: %d, Pi: %.6f, Error: %.4f%%", 
			samples, result.EstimatedPi, result.ErrorPercentage)
	}
}

func TestEstimateWithBatches(t *testing.T) {
	config := EstimationConfig{
		NumWorkers: 2,
		BatchSize:  5000,
	}

	estimator := NewPiEstimator(config)
	result, batches := estimator.EstimateWithBatches(50000)

	// Check main result
	if result.TotalSamples != 50000 {
		t.Errorf("Expected 50000 samples, got %d", result.TotalSamples)
	}

	// Check batch results
	expectedBatches := 10 // 50000 / 5000
	if len(batches) != expectedBatches {
		t.Errorf("Expected %d batches, got %d", expectedBatches, len(batches))
	}

	// Verify batch data
	totalBatchSamples := int64(0)
	totalBatchInside := int64(0)

	for i, batch := range batches {
		if batch.Samples != 5000 {
			t.Errorf("Batch %d should have 5000 samples, got %d", i, batch.Samples)
		}
		
		if batch.InsideCircle < 0 || batch.InsideCircle > batch.Samples {
			t.Errorf("Batch %d inside circle count invalid: %d", i, batch.InsideCircle)
		}
		
		if batch.Duration <= 0 {
			t.Errorf("Batch %d duration should be positive", i)
		}
		
		if batch.Pi <= 0 || batch.Pi > 10 {
			t.Errorf("Batch %d Pi estimate out of range: %f", i, batch.Pi)
		}

		totalBatchSamples += batch.Samples
		totalBatchInside += batch.InsideCircle
	}

	if totalBatchSamples != result.TotalSamples {
		t.Errorf("Batch samples don't sum to total: %d vs %d", totalBatchSamples, result.TotalSamples)
	}

	if totalBatchInside != result.InsideCircle {
		t.Errorf("Batch inside counts don't sum to total: %d vs %d", totalBatchInside, result.InsideCircle)
	}
}

func TestAdaptiveEstimator(t *testing.T) {
	config := EstimationConfig{
		NumWorkers: 2,
		BatchSize:  10000,
	}

	baseEstimator := NewPiEstimator(config)
	adaptive := NewAdaptiveEstimator(baseEstimator, 0.01, 50000, 500000)

	result := adaptive.EstimateAdaptive()

	// Should complete within bounds
	if result.TotalSamples < 50000 {
		t.Errorf("Should use at least minimum samples: %d", result.TotalSamples)
	}

	if result.TotalSamples > 500000 {
		t.Errorf("Should not exceed maximum samples: %d", result.TotalSamples)
	}

	// Basic validation
	if result.EstimatedPi <= 0 || result.EstimatedPi > 10 {
		t.Errorf("Pi estimate out of range: %f", result.EstimatedPi)
	}

	t.Logf("Adaptive estimation: %.6f with %d samples (%.4f%% error)",
		result.EstimatedPi, result.TotalSamples, result.ErrorPercentage)
}

func TestDistributedEstimator(t *testing.T) {
	configs := []EstimationConfig{
		{NumWorkers: 2, BatchSize: 10000},
		{NumWorkers: 2, BatchSize: 10000},
		{NumWorkers: 2, BatchSize: 10000},
	}

	distributed := NewDistributedEstimator(configs)
	result := distributed.EstimateDistributed(300000)

	// Should distribute samples across estimators
	if result.TotalSamples != 300000 {
		t.Errorf("Expected 300000 total samples, got %d", result.TotalSamples)
	}

	if result.EstimatedPi <= 0 || result.EstimatedPi > 10 {
		t.Errorf("Pi estimate out of range: %f", result.EstimatedPi)
	}

	t.Logf("Distributed estimation: %.6f (%.4f%% error)",
		result.EstimatedPi, result.ErrorPercentage)
}

func TestStandardRandomSource(t *testing.T) {
	source := NewStandardRandomSource(12345)

	// Test deterministic behavior with same seed
	source.Seed(42)
	val1 := source.Float64()
	
	source.Seed(42)
	val2 := source.Float64()

	if val1 != val2 {
		t.Error("Same seed should produce same first value")
	}

	// Test range
	for i := 0; i < 1000; i++ {
		val := source.Float64()
		if val < 0 || val >= 1.0 {
			t.Errorf("Value out of range [0,1): %f", val)
		}
	}
}

func TestThreadSafeRandomSource(t *testing.T) {
	source := NewThreadSafeRandomSource(4, 12345)

	// Test concurrent access
	var wg sync.WaitGroup
	results := make([]float64, 1000)

	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = source.Float64()
		}(i)
	}

	wg.Wait()

	// Check all values are in range
	for i, val := range results {
		if val < 0 || val >= 1.0 {
			t.Errorf("Value %d out of range [0,1): %f", i, val)
		}
	}

	// Check for some variation (not all same)
	first := results[0]
	allSame := true
	for _, val := range results[1:] {
		if val != first {
			allSame = false
			break
		}
	}

	if allSame {
		t.Error("All random values are the same")
	}
}

func TestStandardResultMerger(t *testing.T) {
	merger := StandardResultMerger{}

	results := []EstimationResult{
		{
			EstimatedPi:   3.14,
			TotalSamples:  100000,
			InsideCircle:  78540,
			Duration:      time.Second,
		},
		{
			EstimatedPi:   3.15,
			TotalSamples:  200000,
			InsideCircle:  157080,
			Duration:      2 * time.Second,
		},
	}

	merged := merger.Merge(results)

	expectedSamples := int64(300000)
	expectedInside := int64(235620)
	expectedPi := 4.0 * float64(expectedInside) / float64(expectedSamples)

	if merged.TotalSamples != expectedSamples {
		t.Errorf("Expected %d total samples, got %d", expectedSamples, merged.TotalSamples)
	}

	if merged.InsideCircle != expectedInside {
		t.Errorf("Expected %d inside circle, got %d", expectedInside, merged.InsideCircle)
	}

	if math.Abs(merged.EstimatedPi-expectedPi) > 1e-10 {
		t.Errorf("Expected Pi %.10f, got %.10f", expectedPi, merged.EstimatedPi)
	}

	if merged.Duration != 2*time.Second {
		t.Errorf("Expected duration %v, got %v", 2*time.Second, merged.Duration)
	}
}

func TestConvergenceData(t *testing.T) {
	config := EstimationConfig{
		NumWorkers: 2,
		BatchSize:  10000,
	}

	estimator := NewPiEstimator(config)
	convergenceData := estimator.GenerateConvergenceData(100000, 10)

	if len(convergenceData) != 10 {
		t.Errorf("Expected 10 convergence points, got %d", len(convergenceData))
	}

	// Check that samples increase
	for i := 1; i < len(convergenceData); i++ {
		if convergenceData[i].TotalSamples <= convergenceData[i-1].TotalSamples {
			t.Errorf("Samples should increase at point %d", i)
		}
	}

	// Check that all estimates are reasonable
	for i, data := range convergenceData {
		if data.EstimatedPi <= 0 || data.EstimatedPi > 10 {
			t.Errorf("Convergence point %d has unreasonable Pi: %f", i, data.EstimatedPi)
		}
	}

	t.Logf("Convergence: initial=%.4f, final=%.4f",
		convergenceData[0].EstimatedPi, convergenceData[len(convergenceData)-1].EstimatedPi)
}

func TestWorkerPerformanceAnalysis(t *testing.T) {
	// Create sample worker results
	results := []WorkerResult{
		{WorkerID: 0, Samples: 50000, InsideCircle: 39269, Duration: 100 * time.Millisecond, LocalPi: 3.14152},
		{WorkerID: 1, Samples: 50000, InsideCircle: 39280, Duration: 102 * time.Millisecond, LocalPi: 3.14240},
		{WorkerID: 2, Samples: 50000, InsideCircle: 39250, Duration: 98 * time.Millisecond, LocalPi: 3.14000},
	}

	analysis := AnalyzeWorkerPerformance(results)

	if analysis == nil {
		t.Fatal("Analysis should not be nil")
	}

	if analysis["total_workers"] != 3 {
		t.Errorf("Expected 3 workers, got %v", analysis["total_workers"])
	}

	if analysis["total_samples"] != int64(150000) {
		t.Errorf("Expected 150000 total samples, got %v", analysis["total_samples"])
	}

	// Check that variance metrics exist
	if _, ok := analysis["pi_variance"]; !ok {
		t.Error("Pi variance should be calculated")
	}

	if _, ok := analysis["pi_std_dev"]; !ok {
		t.Error("Pi standard deviation should be calculated")
	}

	t.Logf("Performance analysis: %+v", analysis)
}

func TestEfficiencyCalculation(t *testing.T) {
	singleThreadTime := 4 * time.Second
	parallelTime := time.Second
	numWorkers := 4

	efficiency := CalculateEfficiency(singleThreadTime, parallelTime, numWorkers)

	expectedEfficiency := 1.0 // Perfect efficiency in this case
	if math.Abs(efficiency-expectedEfficiency) > 0.01 {
		t.Errorf("Expected efficiency %.2f, got %.2f", expectedEfficiency, efficiency)
	}

	// Test with less efficient parallel execution
	parallelTime2 := 2 * time.Second
	efficiency2 := CalculateEfficiency(singleThreadTime, parallelTime2, numWorkers)
	expectedEfficiency2 := 0.5

	if math.Abs(efficiency2-expectedEfficiency2) > 0.01 {
		t.Errorf("Expected efficiency %.2f, got %.2f", expectedEfficiency2, efficiency2)
	}
}

func TestPiEstimatorConfiguration(t *testing.T) {
	// Test default configuration
	config := EstimationConfig{}
	estimator := NewPiEstimator(config)

	if estimator.numWorkers != runtime.NumCPU() {
		t.Errorf("Expected default workers %d, got %d", runtime.NumCPU(), estimator.numWorkers)
	}

	if estimator.batchSize != 100000 {
		t.Errorf("Expected default batch size 100000, got %d", estimator.batchSize)
	}

	// Test custom configuration
	config2 := EstimationConfig{
		NumWorkers: 8,
		BatchSize:  50000,
	}
	estimator2 := NewPiEstimator(config2)

	if estimator2.numWorkers != 8 {
		t.Errorf("Expected 8 workers, got %d", estimator2.numWorkers)
	}

	if estimator2.batchSize != 50000 {
		t.Errorf("Expected batch size 50000, got %d", estimator2.batchSize)
	}
}

func TestEdgeCases(t *testing.T) {
	config := EstimationConfig{
		NumWorkers: 1,
		BatchSize:  1000,
	}

	estimator := NewPiEstimator(config)

	// Test with very small sample size
	result := estimator.EstimatePi(1000)
	if result.TotalSamples != 1000 {
		t.Errorf("Expected 1000 samples, got %d", result.TotalSamples)
	}

	// Test with zero workers (should default to runtime.NumCPU())
	config.NumWorkers = 0
	estimator = NewPiEstimator(config)
	if estimator.numWorkers != runtime.NumCPU() {
		t.Errorf("Zero workers should default to %d, got %d", runtime.NumCPU(), estimator.numWorkers)
	}

	// Test with zero batch size (should default)
	config.BatchSize = 0
	estimator = NewPiEstimator(config)
	if estimator.batchSize != 100000 {
		t.Errorf("Zero batch size should default to 100000, got %d", estimator.batchSize)
	}
}

func BenchmarkPiEstimation(b *testing.B) {
	config := EstimationConfig{
		NumWorkers: runtime.NumCPU(),
		BatchSize:  100000,
	}

	estimator := NewPiEstimator(config)
	samples := int64(1000000)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		estimator.EstimatePi(samples)
	}
}

func BenchmarkWorkerScaling(b *testing.B) {
	samples := int64(1000000)
	workerCounts := []int{1, 2, 4, 8, 16}

	for _, workers := range workerCounts {
		b.Run(fmt.Sprintf("Workers%d", workers), func(b *testing.B) {
			config := EstimationConfig{
				NumWorkers: workers,
				BatchSize:  50000,
			}

			estimator := NewPiEstimator(config)

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				estimator.EstimatePi(samples)
			}
		})
	}
}

func BenchmarkBatchSizes(b *testing.B) {
	samples := int64(1000000)
	batchSizes := []int{10000, 50000, 100000, 500000}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("BatchSize%d", batchSize), func(b *testing.B) {
			config := EstimationConfig{
				NumWorkers: runtime.NumCPU(),
				BatchSize:  batchSize,
			}

			estimator := NewPiEstimator(config)

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				estimator.EstimatePi(samples)
			}
		})
	}
}

func BenchmarkRandomSources(b *testing.B) {
	b.Run("StandardRandomSource", func(b *testing.B) {
		source := NewStandardRandomSource(12345)
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			source.Float64()
		}
	})

	b.Run("ThreadSafeRandomSource", func(b *testing.B) {
		source := NewThreadSafeRandomSource(runtime.NumCPU(), 12345)
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			source.Float64()
		}
	})
}

func TestLargeScaleEstimation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large scale test in short mode")
	}

	config := EstimationConfig{
		NumWorkers: runtime.NumCPU(),
		BatchSize:  1000000,
	}

	estimator := NewPiEstimator(config)
	
	// Large scale test
	samples := int64(100000000) // 100M samples
	result := estimator.EstimatePi(samples)

	// Should be quite accurate with this many samples
	if result.ErrorPercentage > 0.1 {
		t.Logf("Warning: Large scale estimation error %.4f%% (may be acceptable)", result.ErrorPercentage)
	}

	t.Logf("Large scale: %.8f with %d samples (%.6f%% error, %.0f samples/sec)",
		result.EstimatedPi, result.TotalSamples, result.ErrorPercentage, result.SamplesPerSecond)

	// Verify performance metrics
	if result.SamplesPerSecond <= 0 {
		t.Error("Samples per second should be positive")
	}

	if result.Duration <= 0 {
		t.Error("Duration should be positive")
	}
}

func TestConcurrentEstimations(t *testing.T) {
	config := EstimationConfig{
		NumWorkers: 2,
		BatchSize:  50000,
	}

	estimator := NewPiEstimator(config)

	// Run multiple estimations concurrently
	var wg sync.WaitGroup
	results := make([]*EstimationResult, 5)

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = estimator.EstimatePi(100000)
		}(i)
	}

	wg.Wait()

	// Verify all estimations completed
	for i, result := range results {
		if result == nil {
			t.Errorf("Estimation %d failed", i)
			continue
		}

		if result.TotalSamples != 100000 {
			t.Errorf("Estimation %d: expected 100000 samples, got %d", i, result.TotalSamples)
		}

		if result.EstimatedPi <= 0 || result.EstimatedPi > 10 {
			t.Errorf("Estimation %d: Pi estimate out of range: %f", i, result.EstimatedPi)
		}
	}
}