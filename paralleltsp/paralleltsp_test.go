package paralleltsp

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"
)

func TestDefaultTSPConfig(t *testing.T) {
	config := DefaultTSPConfig()
	
	if config.Algorithm != GeneticAlgorithm {
		t.Errorf("Expected default algorithm to be GeneticAlgorithm, got %v", config.Algorithm)
	}
	
	if config.ParallelStrategy != PopulationBased {
		t.Errorf("Expected default parallel strategy to be PopulationBased, got %v", config.ParallelStrategy)
	}
	
	if config.MaxIterations != 10000 {
		t.Errorf("Expected default max iterations to be 10000, got %d", config.MaxIterations)
	}
	
	if config.TimeLimit != 5*time.Minute {
		t.Errorf("Expected default time limit to be 5 minutes, got %v", config.TimeLimit)
	}
	
	if config.PopulationSize != 100 {
		t.Errorf("Expected default population size to be 100, got %d", config.PopulationSize)
	}
	
	if config.MutationRate != 0.1 {
		t.Errorf("Expected default mutation rate to be 0.1, got %f", config.MutationRate)
	}
	
	if config.CrossoverRate != 0.8 {
		t.Errorf("Expected default crossover rate to be 0.8, got %f", config.CrossoverRate)
	}
	
	if !config.EnableCaching {
		t.Error("Expected caching to be enabled by default")
	}
	
	if !config.EnableStatistics {
		t.Error("Expected statistics to be enabled by default")
	}
	
	if config.DistanceMetric != Euclidean {
		t.Errorf("Expected default distance metric to be Euclidean, got %v", config.DistanceMetric)
	}
}

func TestNewParallelTSP(t *testing.T) {
	cities := createTestCities(5)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.MaxIterations = 100
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create ParallelTSP: %v", err)
	}
	
	if tsp == nil {
		t.Fatal("TSP instance should not be nil")
	}
	
	if len(tsp.cities) != len(cities) {
		t.Errorf("Expected %d cities, got %d", len(cities), len(tsp.cities))
	}
	
	if tsp.distances == nil {
		t.Error("Distance matrix should be initialized")
	}
	
	if len(tsp.distances) != len(cities) {
		t.Errorf("Expected distance matrix of size %d, got %d", len(cities), len(tsp.distances))
	}
	
	if tsp.statistics == nil {
		t.Error("Statistics should be initialized")
	}
	
	if tsp.ctx == nil {
		t.Error("Context should be initialized")
	}
}

func TestInvalidConfigurations(t *testing.T) {
	testCases := []struct {
		name   string
		cities []City
		config func() TSPConfig
	}{
		{
			name:   "No cities",
			cities: []City{},
			config: DefaultTSPConfig,
		},
		{
			name:   "Too few cities",
			cities: createTestCities(2),
			config: DefaultTSPConfig,
		},
		{
			name:   "Invalid distance matrix dimensions",
			cities: createTestCities(3),
			config: func() TSPConfig {
				cfg := DefaultTSPConfig()
				cfg.DistanceMatrix = [][]float64{{0, 1}, {1, 0}} // Wrong size
				return cfg
			},
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := tc.config()
			config.Cities = tc.cities
			
			_, err := NewParallelTSP(config)
			if err == nil {
				t.Error("Expected error for invalid configuration")
			}
		})
	}
}

func TestDistanceCalculation(t *testing.T) {
	city1 := City{ID: 0, X: 0, Y: 0}
	city2 := City{ID: 1, X: 3, Y: 4}
	
	testCases := []struct {
		name     string
		metric   DistanceMetric
		expected float64
	}{
		{"Euclidean", Euclidean, 5.0},
		{"Manhattan", Manhattan, 7.0},
		{"Chebyshev", Chebyshev, 4.0},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := DefaultTSPConfig()
			config.Cities = []City{city1, city2, {ID: 2, X: 1, Y: 1}}
			config.DistanceMetric = tc.metric
			
			tsp, err := NewParallelTSP(config)
			if err != nil {
				t.Fatalf("Failed to create TSP: %v", err)
			}
			
			distance := tsp.calculateDistance(city1, city2)
			if math.Abs(distance-tc.expected) > 1e-9 {
				t.Errorf("Expected distance %f, got %f", tc.expected, distance)
			}
		})
	}
}

func TestHaversineDistance(t *testing.T) {
	config := DefaultTSPConfig()
	config.Cities = []City{
		{ID: 0, Lat: 40.7128, Lon: -74.0060}, // New York
		{ID: 1, Lat: 34.0522, Lon: -118.2437}, // Los Angeles
		{ID: 2, Lat: 41.8781, Lon: -87.6298},  // Chicago
	}
	config.DistanceMetric = Haversine
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	// Distance from New York to Los Angeles should be approximately 3944 km
	distance := tsp.calculateDistance(config.Cities[0], config.Cities[1])
	expectedDistance := 3944.0 // Approximate distance in km
	
	if math.Abs(distance-expectedDistance) > 100 { // Allow 100km tolerance
		t.Errorf("Expected distance around %f km, got %f km", expectedDistance, distance)
	}
}

func TestCustomDistanceFunction(t *testing.T) {
	config := DefaultTSPConfig()
	config.Cities = createTestCities(4)
	config.DistanceMetric = Custom
	config.CustomDistanceFunc = func(c1, c2 City) float64 {
		// Custom distance: just the sum of coordinates difference
		return math.Abs(c1.X-c2.X) + math.Abs(c1.Y-c2.Y) + 10 // +10 for testing
	}
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	city1 := config.Cities[0]
	city2 := config.Cities[1]
	
	distance := tsp.calculateDistance(city1, city2)
	expected := math.Abs(city1.X-city2.X) + math.Abs(city1.Y-city2.Y) + 10
	
	if math.Abs(distance-expected) > 1e-9 {
		t.Errorf("Expected custom distance %f, got %f", expected, distance)
	}
}

func TestNearestNeighborAlgorithm(t *testing.T) {
	cities := createTestCities(5)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.Algorithm = NearestNeighbor
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	tour, err := tsp.nearestNeighborTour()
	if err != nil {
		t.Fatalf("Nearest neighbor algorithm failed: %v", err)
	}
	
	if tour == nil {
		t.Fatal("Tour should not be nil")
	}
	
	if len(tour.Cities) != len(cities) {
		t.Errorf("Expected tour length %d, got %d", len(cities), len(tour.Cities))
	}
	
	if tour.Distance <= 0 {
		t.Error("Tour distance should be positive")
	}
	
	if tour.Algorithm != "NearestNeighbor" {
		t.Errorf("Expected algorithm name 'NearestNeighbor', got '%s'", tour.Algorithm)
	}
	
	if !tour.IsValid {
		t.Error("Tour should be marked as valid")
	}
	
	// Verify all cities are visited exactly once
	visited := make(map[int]bool)
	for _, cityID := range tour.Cities {
		if visited[cityID] {
			t.Errorf("City %d visited multiple times", cityID)
		}
		visited[cityID] = true
		
		if cityID < 0 || cityID >= len(cities) {
			t.Errorf("Invalid city ID %d", cityID)
		}
	}
	
	if len(visited) != len(cities) {
		t.Errorf("Expected %d unique cities, got %d", len(cities), len(visited))
	}
}

func TestTourDistanceCalculation(t *testing.T) {
	cities := []City{
		{ID: 0, X: 0, Y: 0},
		{ID: 1, X: 1, Y: 0},
		{ID: 2, X: 1, Y: 1},
		{ID: 3, X: 0, Y: 1},
	}
	
	config := DefaultTSPConfig()
	config.Cities = cities
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	// Square tour: 0 -> 1 -> 2 -> 3 -> 0
	tour := []int{0, 1, 2, 3}
	distance := tsp.calculateTourDistance(tour)
	
	// Expected: 1 + 1 + 1 + 1 = 4
	expected := 4.0
	if math.Abs(distance-expected) > 1e-9 {
		t.Errorf("Expected tour distance %f, got %f", expected, distance)
	}
}

func TestParallelSolving(t *testing.T) {
	cities := createTestCities(8)
	
	testCases := []struct {
		name     string
		strategy ParallelStrategy
		algorithm TSPAlgorithm
	}{
		{"Independent Runs", IndependentRuns, NearestNeighbor},
		{"Worker Pool", WorkerPool, NearestNeighbor},
		{"Hybrid Parallel", HybridParallel, NearestNeighbor},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := DefaultTSPConfig()
			config.Cities = cities
			config.Algorithm = tc.algorithm
			config.ParallelStrategy = tc.strategy
			config.MaxIterations = 100
			config.NumWorkers = 4
			config.TimeLimit = 5 * time.Second
			
			tsp, err := NewParallelTSP(config)
			if err != nil {
				t.Fatalf("Failed to create TSP: %v", err)
			}
			
			tour, err := tsp.Solve()
			if err != nil {
				t.Fatalf("Failed to solve TSP: %v", err)
			}
			
			if tour == nil {
				t.Fatal("Solution should not be nil")
			}
			
			if len(tour.Cities) != len(cities) {
				t.Errorf("Expected tour length %d, got %d", len(cities), len(tour.Cities))
			}
			
			if tour.Distance <= 0 {
				t.Error("Tour distance should be positive")
			}
			
			// Verify solution validity
			validateTour(t, tour, len(cities))
		})
	}
}

func TestGeneticAlgorithmComponents(t *testing.T) {
	cities := createTestCities(6)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.Algorithm = GeneticAlgorithm
	config.PopulationSize = 20
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	// Test population initialization
	if len(tsp.populations) == 0 {
		t.Error("Populations should be initialized")
	}
	
	population := tsp.populations[0]
	if len(population.Tours) != config.PopulationSize {
		t.Errorf("Expected population size %d, got %d", config.PopulationSize, len(population.Tours))
	}
	
	if population.BestTour == nil {
		t.Error("Population should have a best tour")
	}
	
	// Verify all tours in population are valid
	for i, tour := range population.Tours {
		if tour == nil {
			t.Errorf("Tour %d should not be nil", i)
			continue
		}
		
		validateTour(t, tour, len(cities))
	}
}

func TestIslandModelInitialization(t *testing.T) {
	cities := createTestCities(6)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.Algorithm = GeneticAlgorithm
	config.ParallelStrategy = IslandModel
	config.IslandCount = 3
	config.PopulationSize = 15
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	if len(tsp.islands) != config.IslandCount {
		t.Errorf("Expected %d islands, got %d", config.IslandCount, len(tsp.islands))
	}
	
	for i, island := range tsp.islands {
		if island == nil {
			t.Errorf("Island %d should not be nil", i)
			continue
		}
		
		if island.ID != i {
			t.Errorf("Expected island ID %d, got %d", i, island.ID)
		}
		
		if island.Population == nil {
			t.Errorf("Island %d should have a population", i)
			continue
		}
		
		if len(island.Population.Tours) != config.PopulationSize {
			t.Errorf("Island %d expected population size %d, got %d", 
				i, config.PopulationSize, len(island.Population.Tours))
		}
	}
}

func TestAntColonyInitialization(t *testing.T) {
	cities := createTestCities(5)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.Algorithm = AntColonyOptimization
	config.AntCount = 10
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	if tsp.antColony == nil {
		t.Fatal("Ant colony should be initialized")
	}
	
	colony := tsp.antColony
	
	if len(colony.Pheromones) != len(cities) {
		t.Errorf("Expected pheromone matrix size %d, got %d", len(cities), len(colony.Pheromones))
	}
	
	if len(colony.Ants) != config.AntCount {
		t.Errorf("Expected %d ants, got %d", config.AntCount, len(colony.Ants))
	}
	
	// Verify pheromone matrix initialization
	for i, row := range colony.Pheromones {
		if len(row) != len(cities) {
			t.Errorf("Pheromone row %d should have %d elements, got %d", i, len(cities), len(row))
		}
		
		for j, pheromone := range row {
			if i != j && pheromone <= 0 {
				t.Errorf("Pheromone[%d][%d] should be positive, got %f", i, j, pheromone)
			}
		}
	}
	
	// Verify ant initialization
	for i, ant := range colony.Ants {
		if ant == nil {
			t.Errorf("Ant %d should not be nil", i)
			continue
		}
		
		if ant.ID != i {
			t.Errorf("Expected ant ID %d, got %d", i, ant.ID)
		}
		
		if ant.CurrentCity < 0 || ant.CurrentCity >= len(cities) {
			t.Errorf("Ant %d current city %d is out of range", i, ant.CurrentCity)
		}
		
		if len(ant.Visited) != len(cities) {
			t.Errorf("Ant %d visited array should have %d elements, got %d", 
				i, len(cities), len(ant.Visited))
		}
	}
}

func TestDistanceCaching(t *testing.T) {
	cities := createTestCities(4)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.EnableCaching = true
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	city1 := cities[0]
	city2 := cities[1]
	
	// First call should miss cache
	dist1 := tsp.calculateDistance(city1, city2)
	
	// Second call should hit cache
	dist2 := tsp.calculateDistance(city1, city2)
	
	if math.Abs(dist1-dist2) > 1e-9 {
		t.Errorf("Cached distance should be same: %f vs %f", dist1, dist2)
	}
	
	stats := tsp.GetStatistics()
	if stats.CacheHits == 0 {
		t.Error("Should have cache hits")
	}
	
	if stats.CacheMisses == 0 {
		t.Error("Should have cache misses")
	}
}

func TestStatisticsCollection(t *testing.T) {
	cities := createTestCities(5)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.EnableStatistics = true
	config.Algorithm = NearestNeighbor
	config.MaxIterations = 10
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	// Solve the problem
	_, err = tsp.Solve()
	if err != nil {
		t.Fatalf("Failed to solve TSP: %v", err)
	}
	
	stats := tsp.GetStatistics()
	if stats == nil {
		t.Fatal("Statistics should not be nil")
	}
	
	if stats.BestDistance <= 0 {
		t.Error("Best distance should be positive")
	}
	
	if stats.TotalExecutionTime <= 0 {
		t.Error("Execution time should be positive")
	}
	
	if stats.DistanceCalculations == 0 {
		t.Error("Should have distance calculations")
	}
	
	if len(stats.ConvergenceHistory) == 0 {
		t.Error("Should have convergence history")
	}
	
	if stats.AlgorithmPerformance == nil {
		t.Error("Algorithm performance map should be initialized")
	}
}

func TestConcurrentAccess(t *testing.T) {
	cities := createTestCities(6)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.MaxIterations = 50
	config.NumWorkers = 4
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	// Test concurrent access to best tour and statistics
	var wg sync.WaitGroup
	numGoroutines := 10
	
	// Start solver in background
	go func() {
		tsp.Solve()
	}()
	
	// Multiple goroutines accessing data concurrently
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			for j := 0; j < 10; j++ {
				// These should not cause race conditions
				bestTour := tsp.GetBestTour()
				_ = bestTour
				
				stats := tsp.GetStatistics()
				_ = stats
				
				isRunning := tsp.IsRunning()
				_ = isRunning
				
				time.Sleep(time.Millisecond)
			}
		}()
	}
	
	wg.Wait()
}

func TestContextCancellation(t *testing.T) {
	cities := createTestCities(8)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.MaxIterations = 100000 // Large number to ensure timeout
	config.TimeLimit = 100 * time.Millisecond // Short timeout
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	start := time.Now()
	tour, err := tsp.Solve()
	elapsed := time.Since(start)
	
	// Should complete within reasonable time due to timeout
	if elapsed > 500*time.Millisecond {
		t.Errorf("Solver took too long: %v", elapsed)
	}
	
	// Should still return a valid solution even if timed out
	if tour != nil {
		validateTour(t, tour, len(cities))
	}
}

func TestStopMethod(t *testing.T) {
	cities := createTestCities(10)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.MaxIterations = 100000 // Large number
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	// Start solving in background
	done := make(chan struct{})
	go func() {
		defer close(done)
		tsp.Solve()
	}()
	
	// Stop after short time
	time.Sleep(50 * time.Millisecond)
	tsp.Stop()
	
	// Should complete quickly after stop
	select {
	case <-done:
		// Good, it stopped
	case <-time.After(1 * time.Second):
		t.Error("Solver did not stop within reasonable time")
	}
}

func TestRandomNeighborGeneration(t *testing.T) {
	cities := createTestCities(6)
	config := DefaultTSPConfig()
	config.Cities = cities
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		t.Fatalf("Failed to create TSP: %v", err)
	}
	
	// Create initial tour
	original := &Tour{
		Cities:    []int{0, 1, 2, 3, 4, 5},
		Distance:  100.0,
		Algorithm: "Test",
		Timestamp: time.Now(),
		IsValid:   true,
	}
	
	// Generate neighbor
	neighbor := tsp.generateRandomNeighbor(original)
	
	if neighbor == nil {
		t.Fatal("Neighbor should not be nil")
	}
	
	if len(neighbor.Cities) != len(original.Cities) {
		t.Errorf("Neighbor should have same number of cities: %d vs %d", 
			len(neighbor.Cities), len(original.Cities))
	}
	
	// Should be different tour (with very high probability)
	same := true
	for i := range original.Cities {
		if original.Cities[i] != neighbor.Cities[i] {
			same = false
			break
		}
	}
	
	// Validate neighbor tour
	validateTour(t, neighbor, len(cities))
}

func TestMultipleAlgorithmComparison(t *testing.T) {
	cities := createTestCities(6)
	algorithms := []TSPAlgorithm{
		NearestNeighbor,
		TwoOpt,
		SimulatedAnnealing,
	}
	
	results := make(map[TSPAlgorithm]*Tour)
	
	for _, alg := range algorithms {
		config := DefaultTSPConfig()
		config.Cities = cities
		config.Algorithm = alg
		config.MaxIterations = 100
		config.TimeLimit = 2 * time.Second
		
		tsp, err := NewParallelTSP(config)
		if err != nil {
			t.Fatalf("Failed to create TSP for algorithm %v: %v", alg, err)
		}
		
		tour, err := tsp.Solve()
		if err != nil {
			t.Errorf("Algorithm %v failed: %v", alg, err)
			continue
		}
		
		if tour == nil {
			t.Errorf("Algorithm %v returned nil tour", alg)
			continue
		}
		
		validateTour(t, tour, len(cities))
		results[alg] = tour
		
		t.Logf("Algorithm %v: distance = %f", alg, tour.Distance)
	}
	
	if len(results) == 0 {
		t.Fatal("No algorithms succeeded")
	}
	
	// All results should be valid tours
	for alg, tour := range results {
		if tour.Distance <= 0 {
			t.Errorf("Algorithm %v produced invalid distance: %f", alg, tour.Distance)
		}
	}
}

func TestPerformanceBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}
	
	cityCounts := []int{5, 10, 15, 20}
	
	for _, n := range cityCounts {
		t.Run(fmt.Sprintf("Cities_%d", n), func(t *testing.T) {
			cities := createTestCities(n)
			config := DefaultTSPConfig()
			config.Cities = cities
			config.Algorithm = NearestNeighbor
			config.MaxIterations = 1000
			
			tsp, err := NewParallelTSP(config)
			if err != nil {
				t.Fatalf("Failed to create TSP: %v", err)
			}
			
			start := time.Now()
			tour, err := tsp.Solve()
			elapsed := time.Since(start)
			
			if err != nil {
				t.Fatalf("Failed to solve TSP: %v", err)
			}
			
			if tour == nil {
				t.Fatal("Tour should not be nil")
			}
			
			t.Logf("Cities: %d, Distance: %f, Time: %v", n, tour.Distance, elapsed)
			
			// Sanity check: more cities should generally take longer
			// but this isn't a strict requirement due to algorithm variance
			if elapsed > 10*time.Second {
				t.Errorf("Solving took unexpectedly long: %v", elapsed)
			}
		})
	}
}

// Helper functions

func createTestCities(count int) []City {
	cities := make([]City, count)
	
	// Create cities in a rough circle to make the problem interesting
	for i := 0; i < count; i++ {
		angle := 2 * math.Pi * float64(i) / float64(count)
		radius := 10.0 + rand.Float64()*5.0 // Add some randomness
		
		cities[i] = City{
			ID:   i,
			Name: fmt.Sprintf("City_%d", i),
			X:    radius * math.Cos(angle),
			Y:    radius * math.Sin(angle),
		}
	}
	
	return cities
}

func validateTour(t *testing.T, tour *Tour, expectedCityCount int) {
	t.Helper()
	
	if tour == nil {
		t.Fatal("Tour should not be nil")
		return
	}
	
	if len(tour.Cities) != expectedCityCount {
		t.Errorf("Expected tour length %d, got %d", expectedCityCount, len(tour.Cities))
		return
	}
	
	// Check all cities are present exactly once
	visited := make(map[int]bool)
	for _, cityID := range tour.Cities {
		if cityID < 0 || cityID >= expectedCityCount {
			t.Errorf("Invalid city ID %d", cityID)
		}
		
		if visited[cityID] {
			t.Errorf("City %d visited multiple times", cityID)
		}
		visited[cityID] = true
	}
	
	if len(visited) != expectedCityCount {
		t.Errorf("Expected %d unique cities, got %d", expectedCityCount, len(visited))
	}
	
	if tour.Distance <= 0 {
		t.Errorf("Tour distance should be positive, got %f", tour.Distance)
	}
	
	if !tour.IsValid {
		t.Error("Tour should be marked as valid")
	}
	
	if tour.Algorithm == "" {
		t.Error("Tour should have algorithm name")
	}
	
	if tour.Timestamp.IsZero() {
		t.Error("Tour should have timestamp")
	}
}

// Benchmark tests

func BenchmarkNearestNeighbor(b *testing.B) {
	cities := createTestCities(20)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.Algorithm = NearestNeighbor
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		tsp, err := NewParallelTSP(config)
		if err != nil {
			b.Fatalf("Failed to create TSP: %v", err)
		}
		
		_, err = tsp.nearestNeighborTour()
		if err != nil {
			b.Fatalf("Failed to solve: %v", err)
		}
	}
}

func BenchmarkDistanceCalculation(b *testing.B) {
	city1 := City{ID: 0, X: 0, Y: 0}
	city2 := City{ID: 1, X: 3, Y: 4}
	
	config := DefaultTSPConfig()
	config.Cities = []City{city1, city2, {ID: 2, X: 1, Y: 1}}
	config.EnableCaching = false // Test raw performance
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		b.Fatalf("Failed to create TSP: %v", err)
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		tsp.calculateDistance(city1, city2)
	}
}

func BenchmarkDistanceCalculationWithCache(b *testing.B) {
	city1 := City{ID: 0, X: 0, Y: 0}
	city2 := City{ID: 1, X: 3, Y: 4}
	
	config := DefaultTSPConfig()
	config.Cities = []City{city1, city2, {ID: 2, X: 1, Y: 1}}
	config.EnableCaching = true
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		b.Fatalf("Failed to create TSP: %v", err)
	}
	
	// Warm up cache
	tsp.calculateDistance(city1, city2)
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		tsp.calculateDistance(city1, city2)
	}
}

func BenchmarkParallelSolving(b *testing.B) {
	cities := createTestCities(15)
	config := DefaultTSPConfig()
	config.Cities = cities
	config.Algorithm = NearestNeighbor
	config.ParallelStrategy = IndependentRuns
	config.NumWorkers = 4
	config.MaxIterations = 100
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		tsp, err := NewParallelTSP(config)
		if err != nil {
			b.Fatalf("Failed to create TSP: %v", err)
		}
		
		_, err = tsp.Solve()
		if err != nil {
			b.Fatalf("Failed to solve: %v", err)
		}
	}
}

func BenchmarkTourDistanceCalculation(b *testing.B) {
	cities := createTestCities(50)
	config := DefaultTSPConfig()
	config.Cities = cities
	
	tsp, err := NewParallelTSP(config)
	if err != nil {
		b.Fatalf("Failed to create TSP: %v", err)
	}
	
	tour := make([]int, len(cities))
	for i := range tour {
		tour[i] = i
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		tsp.calculateTourDistance(tour)
	}
}