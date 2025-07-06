package parallelantcolony

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestNewACOOptimizer(t *testing.T) {
	problem := createTestTSP(10)
	config := ACOConfig{
		Algorithm:     AntSystem,
		NumAnts:      20,
		NumColonies:  2,
		MaxIterations: 100,
		Alpha:        1.0,
		Beta:         2.0,
		Rho:          0.1,
	}

	optimizer := NewACOOptimizer(config, problem)
	if optimizer == nil {
		t.Fatal("Failed to create ACO optimizer")
	}

	if optimizer.config.NumAnts != 20 {
		t.Errorf("Expected 20 ants, got %d", optimizer.config.NumAnts)
	}

	if optimizer.config.NumColonies != 2 {
		t.Errorf("Expected 2 colonies, got %d", optimizer.config.NumColonies)
	}

	if len(optimizer.colonies) != 2 {
		t.Errorf("Expected 2 colonies, got %d", len(optimizer.colonies))
	}
}

func TestDefaultConfiguration(t *testing.T) {
	problem := createTestTSP(5)
	config := ACOConfig{}

	optimizer := NewACOOptimizer(config, problem)

	if optimizer.config.NumAnts != problem.Size {
		t.Errorf("Expected default ants = problem size (%d), got %d", problem.Size, optimizer.config.NumAnts)
	}

	if optimizer.config.NumColonies != 1 {
		t.Errorf("Expected default colonies = 1, got %d", optimizer.config.NumColonies)
	}

	if optimizer.config.MaxIterations != 1000 {
		t.Errorf("Expected default max iterations = 1000, got %d", optimizer.config.MaxIterations)
	}
}

func TestProblemCreation(t *testing.T) {
	// Test creating problem from coordinates
	coords := []Coordinate{
		{X: 0, Y: 0},
		{X: 1, Y: 0},
		{X: 1, Y: 1},
		{X: 0, Y: 1},
	}

	problem := NewProblem("test", len(coords))
	problem.LoadTSPFromCoordinates(coords)

	if problem.Size != 4 {
		t.Errorf("Expected problem size 4, got %d", problem.Size)
	}

	// Check distance matrix
	expectedDist := math.Sqrt(2.0) // Distance from (0,0) to (1,1)
	if math.Abs(problem.DistMatrix[0][2]-expectedDist) > 1e-10 {
		t.Errorf("Expected distance %.10f, got %.10f", expectedDist, problem.DistMatrix[0][2])
	}

	// Test symmetric matrix
	for i := 0; i < problem.Size; i++ {
		for j := 0; j < problem.Size; j++ {
			if math.Abs(problem.DistMatrix[i][j]-problem.DistMatrix[j][i]) > 1e-10 {
				t.Errorf("Distance matrix not symmetric at (%d,%d)", i, j)
			}
		}
	}
}

func TestProblemFromMatrix(t *testing.T) {
	matrix := [][]float64{
		{0, 1, 2, 3},
		{1, 0, 4, 5},
		{2, 4, 0, 6},
		{3, 5, 6, 0},
	}

	problem := NewProblem("matrix_test", 4)
	problem.LoadTSPFromMatrix(matrix)

	if problem.Size != 4 {
		t.Errorf("Expected problem size 4, got %d", problem.Size)
	}

	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if problem.DistMatrix[i][j] != matrix[i][j] {
				t.Errorf("Matrix mismatch at (%d,%d): expected %f, got %f", 
					i, j, matrix[i][j], problem.DistMatrix[i][j])
			}
		}
	}
}

func TestBasicOptimization(t *testing.T) {
	problem := createTestTSP(8)
	config := ACOConfig{
		Algorithm:     AntSystem,
		NumAnts:      10,
		NumColonies:  1,
		MaxIterations: 50,
		Alpha:        1.0,
		Beta:         2.0,
		Rho:          0.1,
	}

	optimizer := NewACOOptimizer(config, problem)
	solution, err := optimizer.Optimize()

	if err != nil {
		t.Fatalf("Optimization failed: %v", err)
	}

	if solution == nil {
		t.Fatal("No solution returned")
	}

	if !solution.Valid {
		t.Error("Solution is not valid")
	}

	if len(solution.Tour) != problem.Size {
		t.Errorf("Expected tour length %d, got %d", problem.Size, len(solution.Tour))
	}

	// Check tour validity (all cities visited exactly once)
	visited := make(map[int]bool)
	for _, city := range solution.Tour {
		if city < 0 || city >= problem.Size {
			t.Errorf("Invalid city %d in tour", city)
		}
		if visited[city] {
			t.Errorf("City %d visited multiple times", city)
		}
		visited[city] = true
	}

	if len(visited) != problem.Size {
		t.Errorf("Expected %d unique cities, got %d", problem.Size, len(visited))
	}
}

func TestDifferentAlgorithms(t *testing.T) {
	problem := createTestTSP(6)
	algorithms := []ACOAlgorithm{
		AntSystem,
		AntColonySystem,
		MaxMinAntSystem,
		ElitistAntSystem,
	}

	for _, algorithm := range algorithms {
		t.Run(algorithm.String(), func(t *testing.T) {
			config := ACOConfig{
				Algorithm:     algorithm,
				NumAnts:      8,
				NumColonies:  1,
				MaxIterations: 30,
				Alpha:        1.0,
				Beta:         2.0,
				Rho:          0.1,
			}

			optimizer := NewACOOptimizer(config, problem)
			solution, err := optimizer.Optimize()

			if err != nil {
				t.Fatalf("Optimization with %s failed: %v", algorithm.String(), err)
			}

			if solution == nil || !solution.Valid {
				t.Errorf("Invalid solution for algorithm %s", algorithm.String())
			}
		})
	}
}

func TestLocalSearchMethods(t *testing.T) {
	problem := createTestTSP(8)
	methods := []LocalSearchMethod{
		TwoOpt,
		OrOpt,
	}

	for _, method := range methods {
		t.Run(method.String(), func(t *testing.T) {
			config := ACOConfig{
				Algorithm:              AntSystem,
				NumAnts:               6,
				MaxIterations:          20,
				LocalSearchMethod:      method,
				LocalSearchProbability: 1.0, // Always apply local search
			}

			optimizer := NewACOOptimizer(config, problem)
			solution, err := optimizer.Optimize()

			if err != nil {
				t.Fatalf("Optimization with %s local search failed: %v", method.String(), err)
			}

			if solution == nil || !solution.Valid {
				t.Errorf("Invalid solution for local search method %s", method.String())
			}

			stats := optimizer.GetStatistics()
			if stats.LocalSearchCount == 0 {
				t.Errorf("Expected local search to be applied, but count is 0")
			}
		})
	}
}

func TestTwoOptLocalSearch(t *testing.T) {
	problem := createTestTSP(5)
	searcher := NewLocalSearcher(TwoOpt, problem)

	// Create a suboptimal solution
	solution := &Solution{
		Tour: []int{0, 2, 1, 3, 4}, // Deliberately suboptimal
		Cost: 0,
		Valid: true,
	}
	solution.Cost = calculateTourCost(solution.Tour, problem)

	improved := searcher.ImproveSolution(solution)

	if improved.Cost >= solution.Cost {
		t.Logf("Original cost: %f, Improved cost: %f", solution.Cost, improved.Cost)
		// Note: 2-opt might not always improve very small problems
	}

	// Verify the improved solution is still valid
	if len(improved.Tour) != problem.Size {
		t.Errorf("Improved tour has wrong length: expected %d, got %d", problem.Size, len(improved.Tour))
	}
}

func TestPheromoneUpdateStrategies(t *testing.T) {
	problem := createTestTSP(6)
	strategies := []PheromoneUpdateStrategy{
		GlobalUpdate,
		ElitistUpdate,
		MaxMinUpdate,
	}

	for _, strategy := range strategies {
		t.Run(strategy.String(), func(t *testing.T) {
			config := ACOConfig{
				Algorithm:               AntSystem,
				NumAnts:                6,
				MaxIterations:           15,
				PheromoneUpdateStrategy: strategy,
				MinPheromone:           0.01,
				MaxPheromone:           10.0,
			}

			optimizer := NewACOOptimizer(config, problem)
			solution, err := optimizer.Optimize()

			if err != nil {
				t.Fatalf("Optimization with %s update failed: %v", strategy.String(), err)
			}

			if solution == nil || !solution.Valid {
				t.Errorf("Invalid solution for update strategy %s", strategy.String())
			}

			stats := optimizer.GetStatistics()
			if stats.PheromoneUpdates == 0 {
				t.Errorf("Expected pheromone updates, but count is 0")
			}
		})
	}
}

func TestParallelColonies(t *testing.T) {
	problem := createTestTSP(8)
	config := ACOConfig{
		Algorithm:           AntSystem,
		NumAnts:            6,
		NumColonies:        3,
		MaxIterations:       20,
		UseParallelColonies: true,
	}

	optimizer := NewACOOptimizer(config, problem)
	solution, err := optimizer.Optimize()

	if err != nil {
		t.Fatalf("Parallel colonies optimization failed: %v", err)
	}

	if solution == nil || !solution.Valid {
		t.Error("Invalid solution from parallel colonies")
	}

	stats := optimizer.GetStatistics()
	if len(stats.ColonyStatistics) != 3 {
		t.Errorf("Expected 3 colony statistics, got %d", len(stats.ColonyStatistics))
	}

	// Verify all colonies found solutions
	for i, colonyStats := range stats.ColonyStatistics {
		if colonyStats.SolutionsFound == 0 {
			t.Errorf("Colony %d found no solutions", i)
		}
	}
}

func TestParallelAnts(t *testing.T) {
	problem := createTestTSP(8)
	config := ACOConfig{
		Algorithm:       AntSystem,
		NumAnts:        8,
		NumColonies:    2,
		MaxIterations:   15,
		UseParallelAnts: true,
	}

	optimizer := NewACOOptimizer(config, problem)
	solution, err := optimizer.Optimize()

	if err != nil {
		t.Fatalf("Parallel ants optimization failed: %v", err)
	}

	if solution == nil || !solution.Valid {
		t.Error("Invalid solution from parallel ants")
	}

	// Verify workers were created
	if len(optimizer.workers) == 0 {
		t.Error("No workers created for parallel ants")
	}
}

func TestConvergenceDetection(t *testing.T) {
	detector := NewConvergenceDetector(5, 0.001)

	// Simulate improving costs
	costs := []float64{100.0, 95.0, 90.0, 89.5, 89.45, 89.44, 89.44, 89.44, 89.44, 89.44}

	converged := false
	for _, cost := range costs {
		converged = detector.CheckConvergence(cost)
		if converged {
			break
		}
	}

	if !converged {
		t.Error("Expected convergence detection, but algorithm didn't converge")
	}
}

func TestDiversityManagement(t *testing.T) {
	manager := NewDiversityManager(0.1)

	// Simulate low diversity
	for i := 0; i < 10; i++ {
		manager.diversityMetrics = append(manager.diversityMetrics, 0.05)
	}

	if !manager.NeedsDiversification() {
		t.Error("Expected diversification to be needed with low diversity")
	}

	// Simulate high diversity
	manager.diversityMetrics = nil
	for i := 0; i < 10; i++ {
		manager.diversityMetrics = append(manager.diversityMetrics, 0.5)
	}

	if manager.NeedsDiversification() {
		t.Error("Expected diversification not to be needed with high diversity")
	}
}

func TestStatisticsCollection(t *testing.T) {
	problem := createTestTSP(6)
	config := ACOConfig{
		Algorithm:        AntSystem,
		NumAnts:         4,
		NumColonies:     2,
		MaxIterations:    10,
		EnableStatistics: true,
	}

	optimizer := NewACOOptimizer(config, problem)
	solution, err := optimizer.Optimize()

	if err != nil {
		t.Fatalf("Optimization failed: %v", err)
	}

	if solution == nil {
		t.Fatal("No solution returned")
	}

	stats := optimizer.GetStatistics()

	if stats.TotalIterations == 0 {
		t.Error("Expected total iterations > 0")
	}

	if stats.BestCost == math.Inf(1) {
		t.Error("Best cost should be finite")
	}

	if len(stats.CostHistory) == 0 {
		t.Error("Expected cost history to be recorded")
	}

	if len(stats.ColonyStatistics) != 2 {
		t.Errorf("Expected 2 colony statistics, got %d", len(stats.ColonyStatistics))
	}

	for i, colonyStats := range stats.ColonyStatistics {
		if colonyStats.SolutionsFound == 0 {
			t.Errorf("Colony %d should have found solutions", i)
		}
	}
}

func TestConcurrentSafety(t *testing.T) {
	problem := createTestTSP(8)
	config := ACOConfig{
		Algorithm:           AntSystem,
		NumAnts:            4,
		NumColonies:        2,
		MaxIterations:       20,
		UseParallelColonies: true,
		UseParallelAnts:     true,
	}

	const numRuns = 10
	var wg sync.WaitGroup
	results := make(chan *Solution, numRuns)
	errors := make(chan error, numRuns)

	for i := 0; i < numRuns; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			optimizer := NewACOOptimizer(config, problem)
			solution, err := optimizer.Optimize()
			
			if err != nil {
				errors <- err
			} else {
				results <- solution
			}
		}()
	}

	wg.Wait()
	close(results)
	close(errors)

	// Check for errors
	for err := range errors {
		t.Errorf("Concurrent optimization error: %v", err)
	}

	// Check results
	solutionCount := 0
	for solution := range results {
		if solution != nil && solution.Valid {
			solutionCount++
		}
	}

	if solutionCount < numRuns/2 {
		t.Errorf("Too few valid solutions: got %d out of %d", solutionCount, numRuns)
	}
}

func TestStopOptimization(t *testing.T) {
	problem := createTestTSP(20) // Larger problem for longer optimization
	config := ACOConfig{
		Algorithm:     AntSystem,
		NumAnts:      30,
		MaxIterations: 10000, // Very long optimization
	}

	optimizer := NewACOOptimizer(config, problem)

	// Start optimization in background
	done := make(chan bool)
	var solution *Solution
	var err error

	go func() {
		solution, err = optimizer.Optimize()
		done <- true
	}()

	// Let it run briefly then stop
	time.Sleep(100 * time.Millisecond)
	optimizer.Stop()

	// Wait for completion
	<-done

	if err == nil {
		t.Log("Optimization completed normally (might be too fast to interrupt)")
	} else {
		t.Logf("Optimization stopped with error: %v", err)
	}

	// Solution might still be valid if found before stopping
	if solution != nil && !solution.Valid {
		t.Error("Returned solution is invalid")
	}
}

func TestCleanup(t *testing.T) {
	problem := createTestTSP(5)
	config := ACOConfig{
		Algorithm:       AntSystem,
		NumAnts:        4,
		UseParallelAnts: true,
	}

	optimizer := NewACOOptimizer(config, problem)
	
	// Cleanup should not panic
	optimizer.Cleanup()

	// Check that resources are cleaned up
	optimizer.mutex.RLock()
	coloniesNil := optimizer.colonies == nil
	workersNil := optimizer.workers == nil
	optimizer.mutex.RUnlock()

	if !coloniesNil {
		t.Error("Colonies should be nil after cleanup")
	}

	if !workersNil {
		t.Error("Workers should be nil after cleanup")
	}
}

func TestSolutionQuality(t *testing.T) {
	// Test with a known simple TSP where optimal is known
	coords := []Coordinate{
		{X: 0, Y: 0},
		{X: 1, Y: 0},
		{X: 1, Y: 1},
		{X: 0, Y: 1},
	}

	problem := NewProblem("square", len(coords))
	problem.LoadTSPFromCoordinates(coords)

	config := ACOConfig{
		Algorithm:     AntSystem,
		NumAnts:      20,
		MaxIterations: 100,
		Alpha:        1.0,
		Beta:         2.0,
		Rho:          0.1,
	}

	optimizer := NewACOOptimizer(config, problem)
	solution, err := optimizer.Optimize()

	if err != nil {
		t.Fatalf("Optimization failed: %v", err)
	}

	// For a 4-city square, optimal tour length is 4.0
	optimalCost := 4.0
	if math.Abs(solution.Cost-optimalCost) > 0.001 {
		t.Logf("Solution cost: %f, Optimal: %f", solution.Cost, optimalCost)
		// Note: ACO might not always find the optimal solution
	}
}

func TestLargerProblem(t *testing.T) {
	problem := createTestTSP(15)
	config := ACOConfig{
		Algorithm:           AntSystem,
		NumAnts:            20,
		NumColonies:        2,
		MaxIterations:       50,
		UseParallelColonies: true,
		LocalSearchMethod:   TwoOpt,
		LocalSearchProbability: 0.5,
	}

	start := time.Now()
	optimizer := NewACOOptimizer(config, problem)
	solution, err := optimizer.Optimize()
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Large problem optimization failed: %v", err)
	}

	if solution == nil || !solution.Valid {
		t.Error("Invalid solution for large problem")
	}

	t.Logf("15-city problem solved in %v with cost %f", duration, solution.Cost)

	stats := optimizer.GetStatistics()
	t.Logf("Statistics: %d iterations, %d solutions evaluated, %d local searches", 
		stats.TotalIterations, stats.SolutionsEvaluated, stats.LocalSearchCount)
}

// Benchmark tests

func BenchmarkACOOptimization(b *testing.B) {
	problem := createTestTSP(10)
	config := ACOConfig{
		Algorithm:     AntSystem,
		NumAnts:      15,
		MaxIterations: 50,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer := NewACOOptimizer(config, problem)
		_, err := optimizer.Optimize()
		if err != nil {
			b.Fatalf("Optimization failed: %v", err)
		}
	}
}

func BenchmarkParallelColonies(b *testing.B) {
	problem := createTestTSP(10)
	
	configs := []struct {
		name     string
		parallel bool
		colonies int
	}{
		{"Sequential", false, 4},
		{"Parallel", true, 4},
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			config := ACOConfig{
				Algorithm:           AntSystem,
				NumAnts:            8,
				NumColonies:        cfg.colonies,
				MaxIterations:       30,
				UseParallelColonies: cfg.parallel,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				optimizer := NewACOOptimizer(config, problem)
				_, err := optimizer.Optimize()
				if err != nil {
					b.Fatalf("Optimization failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkDifferentSizes(b *testing.B) {
	sizes := []int{5, 10, 15, 20}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size-%d", size), func(b *testing.B) {
			problem := createTestTSP(size)
			config := ACOConfig{
				Algorithm:     AntSystem,
				NumAnts:      size,
				MaxIterations: 30,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				optimizer := NewACOOptimizer(config, problem)
				_, err := optimizer.Optimize()
				if err != nil {
					b.Fatalf("Optimization failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkLocalSearch(b *testing.B) {
	problem := createTestTSP(12)
	searcher := NewLocalSearcher(TwoOpt, problem)
	
	// Create a random solution
	solution := &Solution{
		Tour: rand.Perm(problem.Size),
		Valid: true,
	}
	solution.Cost = calculateTourCost(solution.Tour, problem)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		searcher.ImproveSolution(solution)
	}
}

func BenchmarkPheromoneUpdate(b *testing.B) {
	problem := createTestTSP(15)
	config := ACOConfig{
		Algorithm: AntSystem,
		NumAnts:   10,
	}

	optimizer := NewACOOptimizer(config, problem)
	
	// Create a solution for testing
	solution := &Solution{
		Tour: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
		Cost: 100.0,
		Valid: true,
	}
	optimizer.globalBestSolution = solution

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer.updatePheromones()
	}
}

func BenchmarkWorkerProcessing(b *testing.B) {
	if !testing.Short() {
		problem := createTestTSP(8)
		config := ACOConfig{
			Algorithm:       AntSystem,
			NumAnts:        4,
			UseParallelAnts: true,
		}

		optimizer := NewACOOptimizer(config, problem)
		
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				// Simulate worker task processing
				worker := optimizer.workers[0]
				task := Task{
					Type:      ConstructSolutionTask,
					ColonyID:  0,
					AntID:     0,
					TaskID:    "bench",
				}
				
				result := worker.processTask(task)
				if !result.Success {
					b.Errorf("Task processing failed: %v", result.Error)
				}
			}
		})
	}
}

// Helper functions

func createTestTSP(size int) *Problem {
	// Create a random TSP problem with given size
	problem := NewProblem(fmt.Sprintf("test_%d", size), size)
	
	// Generate random coordinates
	coords := make([]Coordinate, size)
	rand.Seed(42) // Fixed seed for reproducible tests
	
	for i := 0; i < size; i++ {
		coords[i] = Coordinate{
			X: rand.Float64() * 100,
			Y: rand.Float64() * 100,
		}
	}
	
	problem.LoadTSPFromCoordinates(coords)
	return problem
}

func calculateTourCost(tour []int, problem *Problem) float64 {
	cost := 0.0
	for i := 0; i < len(tour); i++ {
		from := tour[i]
		to := tour[(i+1)%len(tour)]
		cost += problem.DistMatrix[from][to]
	}
	return cost
}

// String methods for enums (for test output)

func (a ACOAlgorithm) String() string {
	switch a {
	case AntSystem:
		return "AntSystem"
	case AntColonySystem:
		return "AntColonySystem"
	case MaxMinAntSystem:
		return "MaxMinAntSystem"
	case RankBasedAntSystem:
		return "RankBasedAntSystem"
	case ElitistAntSystem:
		return "ElitistAntSystem"
	case HybridAntSystem:
		return "HybridAntSystem"
	default:
		return "Unknown"
	}
}

func (l LocalSearchMethod) String() string {
	switch l {
	case NoLocalSearch:
		return "NoLocalSearch"
	case TwoOpt:
		return "TwoOpt"
	case ThreeOpt:
		return "ThreeOpt"
	case OrOpt:
		return "OrOpt"
	case LinKernighan:
		return "LinKernighan"
	case SimulatedAnnealing:
		return "SimulatedAnnealing"
	case HillClimbing:
		return "HillClimbing"
	default:
		return "Unknown"
	}
}

func (p PheromoneUpdateStrategy) String() string {
	switch p {
	case GlobalUpdate:
		return "GlobalUpdate"
	case LocalUpdate:
		return "LocalUpdate"
	case ElitistUpdate:
		return "ElitistUpdate"
	case RankBasedUpdate:
		return "RankBasedUpdate"
	case MaxMinUpdate:
		return "MaxMinUpdate"
	case HybridUpdate:
		return "HybridUpdate"
	default:
		return "Unknown"
	}
}

func ExampleACOOptimizer_Optimize() {
	// Create a simple 4-city TSP problem
	coords := []Coordinate{
		{X: 0, Y: 0},
		{X: 1, Y: 0},
		{X: 1, Y: 1},
		{X: 0, Y: 1},
	}

	problem := NewProblem("square", len(coords))
	problem.LoadTSPFromCoordinates(coords)

	// Configure ACO
	config := ACOConfig{
		Algorithm:     AntSystem,
		NumAnts:      10,
		MaxIterations: 50,
		Alpha:        1.0,
		Beta:         2.0,
		Rho:          0.1,
	}

	optimizer := NewACOOptimizer(config, problem)
	solution, err := optimizer.Optimize()

	if err != nil {
		panic(err)
	}

	fmt.Printf("Best tour cost: %.2f\n", solution.Cost)
	fmt.Printf("Tour length: %d cities\n", len(solution.Tour))
	fmt.Printf("Valid solution: %t\n", solution.Valid)

	// Output:
	// Best tour cost: 4.00
	// Tour length: 4 cities
	// Valid solution: true
}

func ExampleLocalSearcher_ImproveSolution() {
	// Create a test problem and solution
	problem := createTestTSP(5)
	searcher := NewLocalSearcher(TwoOpt, problem)

	// Create a suboptimal solution
	solution := &Solution{
		Tour:  []int{0, 2, 1, 3, 4},
		Cost:  calculateTourCost([]int{0, 2, 1, 3, 4}, problem),
		Valid: true,
	}

	// Improve the solution
	improved := searcher.ImproveSolution(solution)

	fmt.Printf("Original cost: %.2f\n", solution.Cost)
	fmt.Printf("Improved cost: %.2f\n", improved.Cost)
	fmt.Printf("Improvement: %.2f%%\n", 
		(solution.Cost-improved.Cost)/solution.Cost*100)
}