package parallelkmeans

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestNewKMeansClusterer(t *testing.T) {
	config := ClusteringConfig{
		K:             3,
		MaxIterations: 50,
		Tolerance:     1e-6,
		NumWorkers:    4,
		BatchSize:     100,
		RandomSeed:    12345,
	}

	clusterer := NewKMeansClusterer(config)

	if clusterer.config.K != 3 {
		t.Errorf("Expected K=3, got %d", clusterer.config.K)
	}

	if clusterer.config.MaxIterations != 50 {
		t.Errorf("Expected MaxIterations=50, got %d", clusterer.config.MaxIterations)
	}

	if clusterer.config.NumWorkers != 4 {
		t.Errorf("Expected NumWorkers=4, got %d", clusterer.config.NumWorkers)
	}
}

func TestDefaultConfiguration(t *testing.T) {
	config := ClusteringConfig{}
	clusterer := NewKMeansClusterer(config)

	if clusterer.config.K != 3 {
		t.Errorf("Expected default K=3, got %d", clusterer.config.K)
	}

	if clusterer.config.MaxIterations != 100 {
		t.Errorf("Expected default MaxIterations=100, got %d", clusterer.config.MaxIterations)
	}

	if clusterer.config.NumWorkers != runtime.NumCPU() {
		t.Errorf("Expected default NumWorkers=%d, got %d", runtime.NumCPU(), clusterer.config.NumWorkers)
	}
}

func TestSimpleClustering(t *testing.T) {
	// Create simple 2D points with clear clusters
	points := []Point{
		{1.0, 1.0}, {1.1, 1.1}, {0.9, 0.9}, {1.0, 0.8},
		{5.0, 5.0}, {5.1, 5.1}, {4.9, 4.9}, {5.0, 4.8},
		{9.0, 9.0}, {9.1, 9.1}, {8.9, 8.9}, {9.0, 8.8},
	}

	config := ClusteringConfig{
		K:                3,
		MaxIterations:    50,
		Tolerance:        1e-6,
		InitMethod:       KMeansPlusPlusInit,
		DistanceMetric:   EuclideanDistance,
		NumWorkers:       2,
		BatchSize:        4,
		ConvergenceCheck: true,
		RandomSeed:       12345,
	}

	clusterer := NewKMeansClusterer(config)
	result, err := clusterer.Fit(points)

	if err != nil {
		t.Fatalf("Clustering failed: %v", err)
	}

	if result == nil {
		t.Fatal("Expected clustering result, got nil")
	}

	if len(result.Assignments) != len(points) {
		t.Errorf("Expected %d assignments, got %d", len(points), len(result.Assignments))
	}

	if len(result.Clusters) != 3 {
		t.Errorf("Expected 3 clusters, got %d", len(result.Clusters))
	}

	// Check that all assignments are valid
	for i, assignment := range result.Assignments {
		if assignment < 0 || assignment >= config.K {
			t.Errorf("Invalid assignment for point %d: %d", i, assignment)
		}
	}

	// Check that SSE is reasonable
	if result.SSE < 0 {
		t.Errorf("SSE should be non-negative, got %f", result.SSE)
	}

	t.Logf("Clustering completed in %d iterations", result.Stats.TotalIterations)
	t.Logf("SSE: %f", result.SSE)
	t.Logf("Converged: %v", result.Converged)
}

func TestDifferentInitMethods(t *testing.T) {
	points := GenerateClusteredData(3, 50, 2, 1.0, 12345)

	initMethods := []InitMethod{
		RandomInit,
		KMeansPlusPlusInit,
		ForgyInit,
		RandomPartitionInit,
	}

	methodNames := []string{
		"RandomInit",
		"KMeansPlusPlusInit", 
		"ForgyInit",
		"RandomPartitionInit",
	}

	for i, method := range initMethods {
		t.Run(methodNames[i], func(t *testing.T) {
			config := ClusteringConfig{
				K:             3,
				MaxIterations: 50,
				InitMethod:    method,
				NumWorkers:    2,
				RandomSeed:    12345,
			}

			clusterer := NewKMeansClusterer(config)
			result, err := clusterer.Fit(points)

			if err != nil {
				t.Fatalf("Clustering with %s failed: %v", methodNames[i], err)
			}

			if len(result.Assignments) != len(points) {
				t.Errorf("Expected %d assignments, got %d", len(points), len(result.Assignments))
			}

			t.Logf("%s: SSE=%.4f, Iterations=%d", 
				methodNames[i], result.SSE, result.Stats.TotalIterations)
		})
	}
}

func TestDifferentDistanceMetrics(t *testing.T) {
	points := GenerateRandomPoints(100, 3, 12345)

	distanceFuncs := []DistanceFunc{
		EuclideanDistance,
		ManhattanDistance,
		CosineDistance,
	}

	distanceNames := []string{
		"Euclidean",
		"Manhattan", 
		"Cosine",
	}

	for i, distanceFunc := range distanceFuncs {
		t.Run(distanceNames[i], func(t *testing.T) {
			config := ClusteringConfig{
				K:              3,
				MaxIterations:  30,
				DistanceMetric: distanceFunc,
				NumWorkers:     2,
				RandomSeed:     12345,
			}

			clusterer := NewKMeansClusterer(config)
			result, err := clusterer.Fit(points)

			if err != nil {
				t.Fatalf("Clustering with %s distance failed: %v", distanceNames[i], err)
			}

			if len(result.Assignments) != len(points) {
				t.Errorf("Expected %d assignments, got %d", len(points), len(result.Assignments))
			}

			t.Logf("%s: SSE=%.4f, Iterations=%d", 
				distanceNames[i], result.SSE, result.Stats.TotalIterations)
		})
	}
}

func TestConcurrentClustering(t *testing.T) {
	points := GenerateClusteredData(4, 25, 3, 2.0, 12345)

	config := ClusteringConfig{
		K:             4,
		MaxIterations: 30,
		NumWorkers:    4,
		BatchSize:     25,
		RandomSeed:    12345,
	}

	numGoroutines := 5
	var wg sync.WaitGroup
	results := make([]*ClusteringResult, numGoroutines)
	errors := make([]error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()

			clusterer := NewKMeansClusterer(config)
			result, err := clusterer.Fit(points)

			results[index] = result
			errors[index] = err
		}(i)
	}

	wg.Wait()

	// Check all completed successfully
	for i := 0; i < numGoroutines; i++ {
		if errors[i] != nil {
			t.Errorf("Goroutine %d failed: %v", i, errors[i])
		}

		if results[i] == nil {
			t.Errorf("Goroutine %d returned nil result", i)
		}
	}

	t.Logf("All %d concurrent clustering operations completed successfully", numGoroutines)
}

func TestFitPredict(t *testing.T) {
	points := GenerateClusteredData(3, 30, 2, 1.5, 12345)

	config := ClusteringConfig{
		K:             3,
		MaxIterations: 50,
		NumWorkers:    2,
		RandomSeed:    12345,
	}

	clusterer := NewKMeansClusterer(config)
	assignments, err := clusterer.FitPredict(points)

	if err != nil {
		t.Fatalf("FitPredict failed: %v", err)
	}

	if len(assignments) != len(points) {
		t.Errorf("Expected %d assignments, got %d", len(points), len(assignments))
	}

	// Test prediction on new points
	newPoints := []Point{
		{0.0, 0.0},
		{5.0, 5.0},
		{10.0, 10.0},
	}

	newAssignments, err := clusterer.Predict(newPoints)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	if len(newAssignments) != len(newPoints) {
		t.Errorf("Expected %d new assignments, got %d", len(newPoints), len(newAssignments))
	}

	for _, assignment := range newAssignments {
		if assignment < 0 || assignment >= config.K {
			t.Errorf("Invalid assignment: %d", assignment)
		}
	}
}

func TestPredictWithoutFit(t *testing.T) {
	config := ClusteringConfig{K: 3}
	clusterer := NewKMeansClusterer(config)

	points := []Point{{1, 2}, {3, 4}}
	_, err := clusterer.Predict(points)

	if err == nil {
		t.Error("Expected error when predicting without fitting, got nil")
	}
}

func TestEmptyDataset(t *testing.T) {
	config := ClusteringConfig{K: 3}
	clusterer := NewKMeansClusterer(config)

	var points []Point
	_, err := clusterer.Fit(points)

	if err == nil {
		t.Error("Expected error for empty dataset, got nil")
	}
}

func TestInsufficientData(t *testing.T) {
	config := ClusteringConfig{K: 5}
	clusterer := NewKMeansClusterer(config)

	points := []Point{{1, 2}, {3, 4}} // Only 2 points for K=5
	_, err := clusterer.Fit(points)

	if err == nil {
		t.Error("Expected error when points < K, got nil")
	}
}

func TestInconsistentDimensions(t *testing.T) {
	config := ClusteringConfig{K: 2}
	clusterer := NewKMeansClusterer(config)

	points := []Point{
		{1, 2},
		{3, 4, 5}, // Different dimension
	}

	_, err := clusterer.Fit(points)
	if err == nil {
		t.Error("Expected error for inconsistent dimensions, got nil")
	}
}

func TestDistanceFunctions(t *testing.T) {
	p1 := Point{1, 2, 3}
	p2 := Point{4, 5, 6}

	// Test Euclidean distance
	euclidean := EuclideanDistance(p1, p2)
	expected := math.Sqrt(9 + 9 + 9) // sqrt(27)
	if math.Abs(euclidean-expected) > 1e-9 {
		t.Errorf("Euclidean distance: expected %f, got %f", expected, euclidean)
	}

	// Test Manhattan distance
	manhattan := ManhattanDistance(p1, p2)
	expectedManhattan := 3.0 + 3.0 + 3.0 // 9
	if math.Abs(manhattan-expectedManhattan) > 1e-9 {
		t.Errorf("Manhattan distance: expected %f, got %f", expectedManhattan, manhattan)
	}

	// Test Cosine distance
	cosine := CosineDistance(p1, p2)
	// Cosine similarity = (1*4 + 2*5 + 3*6) / (sqrt(14) * sqrt(77))
	// = 32 / sqrt(1078) ≈ 0.9746
	// Cosine distance = 1 - cosine similarity ≈ 0.0254
	if cosine < 0 || cosine > 2 {
		t.Errorf("Cosine distance should be between 0 and 2, got %f", cosine)
	}
}

func TestDistanceFunctionsWithInvalidInput(t *testing.T) {
	p1 := Point{1, 2}
	p2 := Point{3, 4, 5} // Different dimension

	// All distance functions should handle dimension mismatch
	euclidean := EuclideanDistance(p1, p2)
	if !math.IsInf(euclidean, 1) {
		t.Error("Expected infinity for mismatched dimensions in Euclidean distance")
	}

	manhattan := ManhattanDistance(p1, p2)
	if !math.IsInf(manhattan, 1) {
		t.Error("Expected infinity for mismatched dimensions in Manhattan distance")
	}

	cosine := CosineDistance(p1, p2)
	if !math.IsInf(cosine, 1) {
		t.Error("Expected infinity for mismatched dimensions in Cosine distance")
	}
}

func TestGenerateRandomPoints(t *testing.T) {
	numPoints := 100
	dimensions := 3
	seed := int64(12345)

	points := GenerateRandomPoints(numPoints, dimensions, seed)

	if len(points) != numPoints {
		t.Errorf("Expected %d points, got %d", numPoints, len(points))
	}

	for i, point := range points {
		if len(point) != dimensions {
			t.Errorf("Point %d has dimension %d, expected %d", i, len(point), dimensions)
		}
	}

	// Test reproducibility
	points2 := GenerateRandomPoints(numPoints, dimensions, seed)
	for i := 0; i < numPoints; i++ {
		for j := 0; j < dimensions; j++ {
			if points[i][j] != points2[i][j] {
				t.Error("Random point generation is not reproducible")
				break
			}
		}
	}
}

func TestGenerateClusteredData(t *testing.T) {
	numClusters := 3
	pointsPerCluster := 50
	dimensions := 2
	spread := 1.0
	seed := int64(12345)

	points := GenerateClusteredData(numClusters, pointsPerCluster, dimensions, spread, seed)

	expectedPoints := numClusters * pointsPerCluster
	if len(points) != expectedPoints {
		t.Errorf("Expected %d points, got %d", expectedPoints, len(points))
	}

	for i, point := range points {
		if len(point) != dimensions {
			t.Errorf("Point %d has dimension %d, expected %d", i, len(point), dimensions)
		}
	}
}

func TestEvaluateClustering(t *testing.T) {
	points := []Point{
		{1, 1}, {1.1, 1.1}, {0.9, 0.9},
		{5, 5}, {5.1, 5.1}, {4.9, 4.9},
	}

	assignments := []int{0, 0, 0, 1, 1, 1}
	centroids := []Point{{1, 1}, {5, 5}}

	metrics := EvaluateClustering(points, assignments, centroids, EuclideanDistance)

	if metrics == nil {
		t.Fatal("Expected metrics, got nil")
	}

	if wcss, exists := metrics["WCSS"]; !exists || wcss < 0 {
		t.Errorf("Expected non-negative WCSS, got %f", wcss)
	}

	if numClusters, exists := metrics["NumClusters"]; !exists || numClusters != 2 {
		t.Errorf("Expected 2 clusters, got %f", numClusters)
	}
}

func TestClusteringWithDifferentWorkerCounts(t *testing.T) {
	points := GenerateClusteredData(4, 50, 3, 1.5, 12345)

	workerCounts := []int{1, 2, 4, 8}

	for _, workers := range workerCounts {
		t.Run(fmt.Sprintf("Workers_%d", workers), func(t *testing.T) {
			config := ClusteringConfig{
				K:             4,
				MaxIterations: 30,
				NumWorkers:    workers,
				BatchSize:     50,
				RandomSeed:    12345,
			}

			clusterer := NewKMeansClusterer(config)
			result, err := clusterer.Fit(points)

			if err != nil {
				t.Fatalf("Clustering with %d workers failed: %v", workers, err)
			}

			if len(result.Assignments) != len(points) {
				t.Errorf("Expected %d assignments, got %d", len(points), len(result.Assignments))
			}

			// Check parallel efficiency
			if result.Stats.ParallelEfficiency < 0 || result.Stats.ParallelEfficiency > 1 {
				t.Errorf("Parallel efficiency should be between 0 and 1, got %f", 
					result.Stats.ParallelEfficiency)
			}

			t.Logf("%d workers: SSE=%.4f, Efficiency=%.2f%%", 
				workers, result.SSE, result.Stats.ParallelEfficiency*100)
		})
	}
}

func TestGetCentroids(t *testing.T) {
	points := GenerateClusteredData(3, 30, 2, 1.0, 12345)

	config := ClusteringConfig{
		K:             3,
		MaxIterations: 30,
		NumWorkers:    2,
		RandomSeed:    12345,
	}

	clusterer := NewKMeansClusterer(config)
	_, err := clusterer.Fit(points)

	if err != nil {
		t.Fatalf("Clustering failed: %v", err)
	}

	centroids := clusterer.GetCentroids()

	if len(centroids) != 3 {
		t.Errorf("Expected 3 centroids, got %d", len(centroids))
	}

	for i, centroid := range centroids {
		if len(centroid) != 2 {
			t.Errorf("Centroid %d has dimension %d, expected 2", i, len(centroid))
		}
	}
}

func TestGetStats(t *testing.T) {
	points := GenerateRandomPoints(100, 3, 12345)

	config := ClusteringConfig{
		K:             3,
		MaxIterations: 20,
		NumWorkers:    4,
		Verbose:       false,
		RandomSeed:    12345,
	}

	clusterer := NewKMeansClusterer(config)
	result, err := clusterer.Fit(points)

	if err != nil {
		t.Fatalf("Clustering failed: %v", err)
	}

	stats := clusterer.GetStats()

	if stats.TotalPoints != 100 {
		t.Errorf("Expected 100 total points, got %d", stats.TotalPoints)
	}

	if stats.TotalIterations <= 0 {
		t.Errorf("Expected positive iterations, got %d", stats.TotalIterations)
	}

	if stats.ConvergenceTime <= 0 {
		t.Errorf("Expected positive convergence time, got %v", stats.ConvergenceTime)
	}

	if len(stats.WorkerUtilization) != 4 {
		t.Errorf("Expected 4 worker utilization values, got %d", len(stats.WorkerUtilization))
	}

	// Compare with result stats
	if stats.TotalIterations != result.Stats.TotalIterations {
		t.Error("Stats from GetStats() don't match result stats")
	}
}

func TestContextCancellation(t *testing.T) {
	points := GenerateRandomPoints(1000, 5, 12345)

	config := ClusteringConfig{
		K:             5,
		MaxIterations: 1000, // Large number to ensure we can cancel
		NumWorkers:    2,
		RandomSeed:    12345,
	}

	clusterer := NewKMeansClusterer(config)

	// Start clustering in goroutine
	done := make(chan struct{})
	var result *ClusteringResult
	var err error

	go func() {
		result, err = clusterer.Fit(points)
		close(done)
	}()

	// Cancel after short delay
	time.Sleep(50 * time.Millisecond)
	clusterer.cancel()

	// Wait for completion
	select {
	case <-done:
		// Check if cancellation was handled properly
		if err != nil && err == context.Canceled {
			t.Log("Clustering properly cancelled")
		} else if err == nil && result != nil {
			t.Log("Clustering completed before cancellation")
		} else {
			t.Errorf("Unexpected error after cancellation: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Error("Clustering did not respond to cancellation within timeout")
	}
}

func TestLargeDataset(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large dataset test in short mode")
	}

	points := GenerateClusteredData(5, 1000, 10, 2.0, 12345)

	config := ClusteringConfig{
		K:                5,
		MaxIterations:    50,
		NumWorkers:       runtime.NumCPU(),
		BatchSize:        500,
		ConvergenceCheck: true,
		RandomSeed:       12345,
	}

	start := time.Now()
	clusterer := NewKMeansClusterer(config)
	result, err := clusterer.Fit(points)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Large dataset clustering failed: %v", err)
	}

	t.Logf("Large dataset (%d points) clustered in %v", len(points), duration)
	t.Logf("SSE: %.4f, Iterations: %d", result.SSE, result.Stats.TotalIterations)
	t.Logf("Parallel Efficiency: %.2f%%", result.Stats.ParallelEfficiency*100)

	// Verify results
	if len(result.Assignments) != len(points) {
		t.Errorf("Expected %d assignments, got %d", len(points), len(result.Assignments))
	}

	if len(result.Clusters) != 5 {
		t.Errorf("Expected 5 clusters, got %d", len(result.Clusters))
	}
}

func BenchmarkSmallDataset(b *testing.B) {
	points := GenerateClusteredData(3, 100, 2, 1.0, 12345)
	config := ClusteringConfig{
		K:             3,
		MaxIterations: 50,
		NumWorkers:    runtime.NumCPU(),
		RandomSeed:    12345,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clusterer := NewKMeansClusterer(config)
		_, err := clusterer.Fit(points)
		if err != nil {
			b.Fatalf("Benchmark clustering failed: %v", err)
		}
	}
}

func BenchmarkMediumDataset(b *testing.B) {
	points := GenerateClusteredData(5, 1000, 5, 2.0, 12345)
	config := ClusteringConfig{
		K:             5,
		MaxIterations: 30,
		NumWorkers:    runtime.NumCPU(),
		BatchSize:     500,
		RandomSeed:    12345,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clusterer := NewKMeansClusterer(config)
		_, err := clusterer.Fit(points)
		if err != nil {
			b.Fatalf("Benchmark clustering failed: %v", err)
		}
	}
}

func BenchmarkDistanceFunctions(b *testing.B) {
	p1 := Point{1, 2, 3, 4, 5}
	p2 := Point{6, 7, 8, 9, 10}

	b.Run("Euclidean", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			EuclideanDistance(p1, p2)
		}
	})

	b.Run("Manhattan", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ManhattanDistance(p1, p2)
		}
	})

	b.Run("Cosine", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			CosineDistance(p1, p2)
		}
	})
}

func BenchmarkParallelScaling(b *testing.B) {
	points := GenerateClusteredData(4, 500, 3, 1.5, 12345)

	workerCounts := []int{1, 2, 4, 8}

	for _, workers := range workerCounts {
		b.Run(fmt.Sprintf("Workers_%d", workers), func(b *testing.B) {
			config := ClusteringConfig{
				K:             4,
				MaxIterations: 20,
				NumWorkers:    workers,
				BatchSize:     250,
				RandomSeed:    12345,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				clusterer := NewKMeansClusterer(config)
				_, err := clusterer.Fit(points)
				if err != nil {
					b.Fatalf("Benchmark clustering failed: %v", err)
				}
			}
		})
	}
}

