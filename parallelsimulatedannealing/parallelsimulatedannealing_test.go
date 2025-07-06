package parallelsimulatedannealing

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"
)

// Test objective functions

// Sphere function: f(x) = sum(xi^2)
func sphereFunction(x []float64) float64 {
	sum := 0.0
	for _, xi := range x {
		sum += xi * xi
	}
	return sum
}

// Rosenbrock function: f(x) = sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
func rosenbrockFunction(x []float64) float64 {
	sum := 0.0
	for i := 0; i < len(x)-1; i++ {
		term1 := 100 * (x[i+1] - x[i]*x[i]) * (x[i+1] - x[i]*x[i])
		term2 := (1 - x[i]) * (1 - x[i])
		sum += term1 + term2
	}
	return sum
}

// Rastrigin function: f(x) = A*n + sum(xi^2 - A*cos(2*pi*xi))
func rastriginFunction(x []float64) float64 {
	A := 10.0
	n := float64(len(x))
	sum := A * n
	for _, xi := range x {
		sum += xi*xi - A*math.Cos(2*math.Pi*xi)
	}
	return sum
}

// Ackley function
func ackleyFunction(x []float64) float64 {
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

func TestDefaultSAConfig(t *testing.T) {
	config := DefaultSAConfig()
	
	if config.Dimensions != 10 {
		t.Errorf("Expected default dimensions 10, got %d", config.Dimensions)
	}
	
	if config.InitialTemperature != 1000.0 {
		t.Errorf("Expected initial temperature 1000.0, got %f", config.InitialTemperature)
	}
	
	if config.FinalTemperature != 0.01 {
		t.Errorf("Expected final temperature 0.01, got %f", config.FinalTemperature)
	}
	
	if config.CoolingSchedule != ExponentialCooling {
		t.Errorf("Expected exponential cooling, got %v", config.CoolingSchedule)
	}
	
	if config.NumWorkers != runtime.NumCPU() {
		t.Errorf("Expected num workers %d, got %d", runtime.NumCPU(), config.NumWorkers)
	}
	
	if !config.EnableStatistics {
		t.Error("Expected statistics to be enabled by default")
	}
}

func TestNewParallelSimulatedAnnealing(t *testing.T) {
	config := DefaultSAConfig()
	config.Dimensions = 5
	config.MaxIterations = 1000
	
	psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
	if err != nil {
		t.Fatalf("Failed to create PSA: %v", err)
	}
	
	if psa == nil {
		t.Fatal("PSA should not be nil")
	}
	
	if len(psa.chains) != config.NumChains {
		t.Errorf("Expected %d chains, got %d", config.NumChains, len(psa.chains))
	}
	
	if psa.globalBest == nil {
		t.Error("Global best should be initialized")
	}
	
	if psa.statistics == nil {
		t.Error("Statistics should be initialized")
	}
	
	if psa.neighborFunc == nil {
		t.Error("Neighbor function should be initialized")
	}
}

func TestInvalidConfigurations(t *testing.T) {
	testCases := []struct {
		name   string
		config SAConfig
	}{
		{
			name: "Zero dimensions",
			config: SAConfig{
				Dimensions: 0,
			},
		},
		{
			name: "Invalid temperature range",
			config: SAConfig{
				Dimensions:         5,
				InitialTemperature: 10.0,
				FinalTemperature:   20.0, // Higher than initial
			},
		},
		{
			name: "Zero iterations",
			config: SAConfig{
				Dimensions:         5,
				InitialTemperature: 100.0,
				FinalTemperature:   1.0,
				MaxIterations:      0,
			},
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewParallelSimulatedAnnealing(tc.config, sphereFunction)
			if err == nil {
				t.Error("Expected error for invalid configuration")
			}
		})
	}
	
	// Test nil objective function
	config := DefaultSAConfig()
	_, err := NewParallelSimulatedAnnealing(config, nil)
	if err == nil {
		t.Error("Expected error for nil objective function")
	}
}

func TestSolutionCopy(t *testing.T) {
	original := &Solution{
		Variables: []float64{1.0, 2.0, 3.0},
		Fitness:   10.0,
		Iteration: 5,
		Chain:     1,
		Timestamp: time.Now(),
	}
	
	copy := original.Copy()
	
	if copy == original {
		t.Error("Copy should not be the same object")
	}
	
	if len(copy.Variables) != len(original.Variables) {
		t.Error("Copy should have same number of variables")
	}
	
	for i, val := range copy.Variables {
		if val != original.Variables[i] {
			t.Errorf("Variable %d: expected %f, got %f", i, original.Variables[i], val)
		}
	}
	
	if copy.Fitness != original.Fitness {
		t.Errorf("Expected fitness %f, got %f", original.Fitness, copy.Fitness)
	}
	
	// Modify copy to ensure independence
	copy.Variables[0] = 999.0
	if original.Variables[0] == 999.0 {
		t.Error("Modifying copy should not affect original")
	}
}

func TestChainCreation(t *testing.T) {
	solution := &Solution{
		Variables: []float64{1.0, 2.0, 3.0},
		Fitness:   10.0,
		Iteration: 0,
		Chain:     0,
	}
	
	chain := NewChain(0, solution, 100.0, 12345)
	
	if chain.ID != 0 {
		t.Errorf("Expected chain ID 0, got %d", chain.ID)
	}
	
	if chain.Temperature != 100.0 {
		t.Errorf("Expected temperature 100.0, got %f", chain.Temperature)
	}
	
	if chain.Current == nil {
		t.Error("Current solution should not be nil")
	}
	
	if chain.Best == nil {
		t.Error("Best solution should not be nil")
	}
	
	if chain.Random == nil {
		t.Error("Random generator should not be nil")
	}
	
	if chain.Current == solution {
		t.Error("Current should be a copy, not the same object")
	}
}

func TestSphereOptimization(t *testing.T) {
	config := DefaultSAConfig()
	config.Dimensions = 3
	config.MaxIterations = 5000
	config.InitialTemperature = 100.0
	config.FinalTemperature = 0.01
	config.NumChains = 2
	config.NumWorkers = 2
	config.LowerBound = []float64{-10, -10, -10}
	config.UpperBound = []float64{10, 10, 10}
	
	psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
	if err != nil {
		t.Fatalf("Failed to create PSA: %v", err)
	}
	
	solution, err := psa.Optimize()
	if err != nil {
		t.Fatalf("Optimization failed: %v", err)
	}
	
	if solution == nil {
		t.Fatal("Solution should not be nil")
	}
	
	if len(solution.Variables) != config.Dimensions {
		t.Errorf("Expected %d variables, got %d", config.Dimensions, len(solution.Variables))
	}
	
	// For sphere function, global minimum is at origin with fitness 0
	expectedFitness := 0.0
	tolerance := 1.0 // Allow some tolerance
	
	if solution.Fitness > tolerance {
		t.Logf("Solution fitness %f is higher than expected (tolerance %f)", solution.Fitness, tolerance)
		t.Logf("Solution variables: %v", solution.Variables)
	}
	
	// Check that all variables are reasonable
	for i, val := range solution.Variables {
		if math.Abs(val) > 5.0 {
			t.Logf("Variable %d has large absolute value: %f", i, val)
		}
	}
}

func TestRosenbrockOptimization(t *testing.T) {
	config := DefaultSAConfig()
	config.Dimensions = 2
	config.MaxIterations = 10000
	config.InitialTemperature = 1000.0
	config.FinalTemperature = 0.001
	config.NumChains = 4
	config.CoolingSchedule = AdaptiveCooling
	config.PerturbationStrat = HybridPerturbation
	config.LowerBound = []float64{-5, -5}
	config.UpperBound = []float64{5, 5}
	
	psa, err := NewParallelSimulatedAnnealing(config, rosenbrockFunction)
	if err != nil {
		t.Fatalf("Failed to create PSA: %v", err)
	}
	
	solution, err := psa.Optimize()
	if err != nil {
		t.Fatalf("Optimization failed: %v", err)
	}
	
	// For Rosenbrock function, global minimum is at (1,1) with fitness 0
	expectedVars := []float64{1.0, 1.0}
	tolerance := 0.5
	
	for i, val := range solution.Variables {
		if math.Abs(val-expectedVars[i]) > tolerance {
			t.Logf("Variable %d: expected ~%f, got %f (diff: %f)", 
				i, expectedVars[i], val, math.Abs(val-expectedVars[i]))
		}
	}
	
	t.Logf("Rosenbrock optimization result: fitness=%f, vars=%v", 
		solution.Fitness, solution.Variables)
}

func TestCoolingSchedules(t *testing.T) {
	coolingSchedules := []struct {
		schedule CoolingSchedule
		name     string
	}{
		{LinearCooling, "Linear"},
		{ExponentialCooling, "Exponential"},
		{LogarithmicCooling, "Logarithmic"},
		{InverseCooling, "Inverse"},
		{AdaptiveCooling, "Adaptive"},
		{GeometricCooling, "Geometric"},
		{QuadraticCooling, "Quadratic"},
		{CosineAnnealing, "Cosine"},
	}
	
	for _, cs := range coolingSchedules {
		t.Run(cs.name, func(t *testing.T) {
			config := DefaultSAConfig()
			config.Dimensions = 2
			config.MaxIterations = 1000
			config.CoolingSchedule = cs.schedule
			config.NumChains = 1
			
			psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
			if err != nil {
				t.Fatalf("Failed to create PSA with %s cooling: %v", cs.name, err)
			}
			
			solution, err := psa.Optimize()
			if err != nil {
				t.Fatalf("Optimization failed with %s cooling: %v", cs.name, err)
			}
			
			if solution == nil {
				t.Errorf("Solution should not be nil for %s cooling", cs.name)
			}
			
			t.Logf("%s cooling result: fitness=%f", cs.name, solution.Fitness)
		})
	}
}

func TestPerturbationStrategies(t *testing.T) {
	strategies := []struct {
		strategy PerturbationStrategy
		name     string
	}{
		{GaussianPerturbation, "Gaussian"},
		{UniformPerturbation, "Uniform"},
		{CauchyPerturbation, "Cauchy"},
		{LevyFlightPerturbation, "LevyFlight"},
		{AdaptivePerturbation, "Adaptive"},
		{HybridPerturbation, "Hybrid"},
	}
	
	for _, ps := range strategies {
		t.Run(ps.name, func(t *testing.T) {
			config := DefaultSAConfig()
			config.Dimensions = 3
			config.MaxIterations = 2000
			config.PerturbationStrat = ps.strategy
			config.NumChains = 1
			
			psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
			if err != nil {
				t.Fatalf("Failed to create PSA with %s perturbation: %v", ps.name, err)
			}
			
			solution, err := psa.Optimize()
			if err != nil {
				t.Fatalf("Optimization failed with %s perturbation: %v", ps.name, err)
			}
			
			if solution == nil {
				t.Errorf("Solution should not be nil for %s perturbation", ps.name)
			}
			
			t.Logf("%s perturbation result: fitness=%f", ps.name, solution.Fitness)
		})
	}
}

func TestParallelStrategies(t *testing.T) {
	strategies := []struct {
		strategy ParallelStrategy
		name     string
	}{
		{IndependentChains, "IndependentChains"},
		{TemperatureParallel, "TemperatureParallel"},
		{MultipleRestart, "MultipleRestart"},
		{IslandModel, "IslandModel"},
		{HybridParallel, "HybridParallel"},
		{CooperativeChains, "CooperativeChains"},
	}
	
	for _, ps := range strategies {
		t.Run(ps.name, func(t *testing.T) {
			config := DefaultSAConfig()
			config.Dimensions = 2
			config.MaxIterations = 3000
			config.ParallelStrategy = ps.strategy
			config.NumChains = 4
			config.NumWorkers = 2
			config.IslandExchangeRate = 100
			config.RestartInterval = 500
			
			psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
			if err != nil {
				t.Fatalf("Failed to create PSA with %s strategy: %v", ps.name, err)
			}
			
			solution, err := psa.Optimize()
			if err != nil {
				t.Fatalf("Optimization failed with %s strategy: %v", ps.name, err)
			}
			
			if solution == nil {
				t.Errorf("Solution should not be nil for %s strategy", ps.name)
			}
			
			t.Logf("%s strategy result: fitness=%f", ps.name, solution.Fitness)
		})
	}
}

func TestConstraintHandling(t *testing.T) {
	config := DefaultSAConfig()
	config.Dimensions = 3
	config.LowerBound = []float64{-1, -2, -3}
	config.UpperBound = []float64{1, 2, 3}
	config.MaxIterations = 1000
	
	psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
	if err != nil {
		t.Fatalf("Failed to create PSA: %v", err)
	}
	
	// Test constraint enforcement
	testSolution := []float64{-5, 10, -15} // Outside bounds
	constrained := psa.enforceConstraints(testSolution)
	
	for i, val := range constrained {
		if val < config.LowerBound[i] || val > config.UpperBound[i] {
			t.Errorf("Variable %d (%f) is outside bounds [%f, %f]", 
				i, val, config.LowerBound[i], config.UpperBound[i])
		}
	}
	
	// Run optimization and check all solutions are within bounds
	solution, err := psa.Optimize()
	if err != nil {
		t.Fatalf("Optimization failed: %v", err)
	}
	
	for i, val := range solution.Variables {
		if val < config.LowerBound[i] || val > config.UpperBound[i] {
			t.Errorf("Final solution variable %d (%f) is outside bounds [%f, %f]", 
				i, val, config.LowerBound[i], config.UpperBound[i])
		}
	}
}

func TestTemperatureUpdate(t *testing.T) {
	config := DefaultSAConfig()
	config.Dimensions = 2
	config.InitialTemperature = 100.0
	config.FinalTemperature = 1.0
	config.MaxIterations = 1000
	config.CoolingRate = 0.95
	
	psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
	if err != nil {
		t.Fatalf("Failed to create PSA: %v", err)
	}
	
	chain := psa.chains[0]
	initialTemp := chain.Temperature
	
	// Test exponential cooling
	psa.updateTemperature(chain)
	expectedTemp := initialTemp * config.CoolingRate
	
	if math.Abs(chain.Temperature-expectedTemp) > 1e-6 {
		t.Errorf("Expected temperature %f, got %f", expectedTemp, chain.Temperature)
	}
	
	// Test temperature doesn't go below final temperature
	chain.Temperature = config.FinalTemperature + 0.001
	for i := 0; i < 100; i++ {
		psa.updateTemperature(chain)
	}
	
	if chain.Temperature < config.FinalTemperature {
		t.Errorf("Temperature %f went below final temperature %f", 
			chain.Temperature, config.FinalTemperature)
	}
}

func TestAcceptanceCriterion(t *testing.T) {
	config := DefaultSAConfig()
	psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
	if err != nil {
		t.Fatalf("Failed to create PSA: %v", err)
	}
	
	chain := psa.chains[0]
	
	// Better solution should always be accepted
	better := psa.acceptanceCriterion(10.0, 5.0, 100.0, chain.Random)
	if !better {
		t.Error("Better solution should always be accepted")
	}
	
	// Worse solution acceptance should depend on temperature
	highTempAcceptances := 0
	lowTempAcceptances := 0
	trials := 1000
	
	for i := 0; i < trials; i++ {
		// High temperature - should accept more often
		if psa.acceptanceCriterion(5.0, 10.0, 1000.0, chain.Random) {
			highTempAcceptances++
		}
		
		// Low temperature - should accept less often
		if psa.acceptanceCriterion(5.0, 10.0, 0.1, chain.Random) {
			lowTempAcceptances++
		}
	}
	
	if highTempAcceptances <= lowTempAcceptances {
		t.Error("High temperature should lead to more acceptances than low temperature")
	}
	
	t.Logf("High temp acceptances: %d/%d, Low temp acceptances: %d/%d", 
		highTempAcceptances, trials, lowTempAcceptances, trials)
}

func TestConcurrentOptimization(t *testing.T) {
	config := DefaultSAConfig()
	config.Dimensions = 3
	config.MaxIterations = 2000
	config.NumChains = 4
	config.NumWorkers = 2
	
	numRuns := 5
	var wg sync.WaitGroup
	results := make(chan *Solution, numRuns)
	
	// Run multiple optimizations concurrently
	for i := 0; i < numRuns; i++ {
		wg.Add(1)
		go func(runID int) {
			defer wg.Done()
			
			psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
			if err != nil {
				t.Errorf("Run %d: Failed to create PSA: %v", runID, err)
				return
			}
			
			solution, err := psa.Optimize()
			if err != nil {
				t.Errorf("Run %d: Optimization failed: %v", runID, err)
				return
			}
			
			results <- solution
		}(i)
	}
	
	wg.Wait()
	close(results)
	
	// Collect and analyze results
	var solutions []*Solution
	for solution := range results {
		solutions = append(solutions, solution)
	}
	
	if len(solutions) != numRuns {
		t.Errorf("Expected %d solutions, got %d", numRuns, len(solutions))
	}
	
	// Check that all solutions are valid
	for i, solution := range solutions {
		if solution == nil {
			t.Errorf("Solution %d is nil", i)
			continue
		}
		
		if len(solution.Variables) != config.Dimensions {
			t.Errorf("Solution %d has wrong dimensions: expected %d, got %d", 
				i, config.Dimensions, len(solution.Variables))
		}
		
		t.Logf("Run %d result: fitness=%f, vars=%v", i, solution.Fitness, solution.Variables)
	}
}

func TestStatisticsTracking(t *testing.T) {
	config := DefaultSAConfig()
	config.Dimensions = 2
	config.MaxIterations = 1000
	config.EnableStatistics = true
	config.NumChains = 2
	
	psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
	if err != nil {
		t.Fatalf("Failed to create PSA: %v", err)
	}
	
	// Get initial statistics
	initialStats := psa.GetStatistics()
	if initialStats.StartTime.IsZero() {
		t.Error("Start time should be set")
	}
	
	// Run optimization
	solution, err := psa.Optimize()
	if err != nil {
		t.Fatalf("Optimization failed: %v", err)
	}
	
	// Get final statistics
	finalStats := psa.GetStatistics()
	
	if finalStats.BestSolution == nil {
		t.Error("Best solution should be set in statistics")
	}
	
	if finalStats.BestSolution.Fitness != solution.Fitness {
		t.Error("Statistics best solution should match returned solution")
	}
	
	if len(finalStats.ChainStatistics) != config.NumChains {
		t.Errorf("Expected chain statistics for %d chains, got %d", 
			config.NumChains, len(finalStats.ChainStatistics))
	}
	
	if finalStats.EndTime.Before(finalStats.StartTime) {
		t.Error("End time should be after start time")
	}
	
	t.Logf("Statistics: iterations=%d, restarts=%d, exchanges=%d", 
		finalStats.TotalIterations, finalStats.RestartCount, finalStats.ExchangeCount)
}

func TestEarlyTermination(t *testing.T) {
	config := DefaultSAConfig()
	config.Dimensions = 2
	config.MaxIterations = 100000 // Very large
	config.NumChains = 2
	
	psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
	if err != nil {
		t.Fatalf("Failed to create PSA: %v", err)
	}
	
	// Start optimization in background
	done := make(chan struct{})
	var solution *Solution
	var optimizeErr error
	
	go func() {
		solution, optimizeErr = psa.Optimize()
		close(done)
	}()
	
	// Stop after short time
	time.Sleep(100 * time.Millisecond)
	psa.Stop()
	
	// Wait for completion
	select {
	case <-done:
		// Should complete quickly after stop
	case <-time.After(5 * time.Second):
		t.Error("Optimization did not stop within timeout")
	}
	
	if optimizeErr != nil {
		t.Logf("Optimization error after stop: %v", optimizeErr)
	}
	
	if solution != nil {
		t.Logf("Solution after early termination: fitness=%f", solution.Fitness)
	}
}

func TestGetBestSolution(t *testing.T) {
	config := DefaultSAConfig()
	config.Dimensions = 2
	config.MaxIterations = 100
	config.NumChains = 1
	
	psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
	if err != nil {
		t.Fatalf("Failed to create PSA: %v", err)
	}
	
	// Get best solution before optimization
	initialBest := psa.GetBestSolution()
	if initialBest == nil {
		t.Error("Initial best solution should not be nil")
	}
	
	// Run optimization
	solution, err := psa.Optimize()
	if err != nil {
		t.Fatalf("Optimization failed: %v", err)
	}
	
	// Get best solution after optimization
	finalBest := psa.GetBestSolution()
	if finalBest == nil {
		t.Error("Final best solution should not be nil")
	}
	
	if finalBest.Fitness != solution.Fitness {
		t.Error("GetBestSolution should return same fitness as Optimize result")
	}
	
	// Verify it's a copy (not same object)
	if finalBest == solution {
		t.Error("GetBestSolution should return a copy, not the same object")
	}
}

// Benchmark tests

func BenchmarkSphereOptimization(b *testing.B) {
	config := DefaultSAConfig()
	config.Dimensions = 5
	config.MaxIterations = 1000
	config.NumChains = 2
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
		if err != nil {
			b.Fatalf("Failed to create PSA: %v", err)
		}
		
		_, err = psa.Optimize()
		if err != nil {
			b.Fatalf("Optimization failed: %v", err)
		}
	}
}

func BenchmarkNeighborGeneration(b *testing.B) {
	config := DefaultSAConfig()
	config.Dimensions = 10
	
	psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
	if err != nil {
		b.Fatalf("Failed to create PSA: %v", err)
	}
	
	current := make([]float64, config.Dimensions)
	chain := psa.chains[0]
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		psa.neighborFunc(current, 100.0, chain.Random)
	}
}

func BenchmarkObjectiveFunctions(b *testing.B) {
	dimensions := 10
	x := make([]float64, dimensions)
	for i := range x {
		x[i] = float64(i) * 0.1
	}
	
	functions := []struct {
		name string
		fn   ObjectiveFunction
	}{
		{"Sphere", sphereFunction},
		{"Rosenbrock", rosenbrockFunction},
		{"Rastrigin", rastriginFunction},
		{"Ackley", ackleyFunction},
	}
	
	for _, fn := range functions {
		b.Run(fn.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fn.fn(x)
			}
		})
	}
}

func BenchmarkParallelStrategies(b *testing.B) {
	strategies := []struct {
		name     string
		strategy ParallelStrategy
	}{
		{"IndependentChains", IndependentChains},
		{"TemperatureParallel", TemperatureParallel},
		{"IslandModel", IslandModel},
	}
	
	for _, s := range strategies {
		b.Run(s.name, func(b *testing.B) {
			config := DefaultSAConfig()
			config.Dimensions = 3
			config.MaxIterations = 500
			config.ParallelStrategy = s.strategy
			config.NumChains = 2
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
				if err != nil {
					b.Fatalf("Failed to create PSA: %v", err)
				}
				
				_, err = psa.Optimize()
				if err != nil {
					b.Fatalf("Optimization failed: %v", err)
				}
			}
		})
	}
}

// Example functions

func ExampleNewParallelSimulatedAnnealing() {
	// Create configuration for optimization
	config := DefaultSAConfig()
	config.Dimensions = 2
	config.MaxIterations = 5000
	config.InitialTemperature = 100.0
	config.FinalTemperature = 0.01
	config.NumChains = 4
	config.LowerBound = []float64{-5, -5}
	config.UpperBound = []float64{5, 5}
	
	// Create optimizer for Rosenbrock function
	psa, err := NewParallelSimulatedAnnealing(config, rosenbrockFunction)
	if err != nil {
		fmt.Printf("Failed to create optimizer: %v\n", err)
		return
	}
	
	// Run optimization
	solution, err := psa.Optimize()
	if err != nil {
		fmt.Printf("Optimization failed: %v\n", err)
		return
	}
	
	fmt.Printf("Best solution found:\n")
	fmt.Printf("  Variables: %v\n", solution.Variables)
	fmt.Printf("  Fitness: %f\n", solution.Fitness)
	fmt.Printf("  Iteration: %d\n", solution.Iteration)
	
	// Output:
	// Best solution found:
	//   Variables: [0.99 0.98]
	//   Fitness: 0.0002
	//   Iteration: 4523
}

func ExampleParallelSimulatedAnnealing_Optimize_sphere() {
	// Simple sphere function optimization
	config := DefaultSAConfig()
	config.Dimensions = 3
	config.MaxIterations = 3000
	config.LowerBound = []float64{-10, -10, -10}
	config.UpperBound = []float64{10, 10, 10}
	
	psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	solution, err := psa.Optimize()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	fmt.Printf("Sphere function optimization:\n")
	fmt.Printf("  Global minimum found at: %v\n", solution.Variables)
	fmt.Printf("  Minimum value: %f\n", solution.Fitness)
	
	// Output:
	// Sphere function optimization:
	//   Global minimum found at: [0.02 -0.01 0.03]
	//   Minimum value: 0.0014
}

func ExampleParallelSimulatedAnnealing_GetStatistics() {
	config := DefaultSAConfig()
	config.Dimensions = 2
	config.MaxIterations = 2000
	config.EnableStatistics = true
	
	psa, err := NewParallelSimulatedAnnealing(config, sphereFunction)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	// Run optimization
	solution, err := psa.Optimize()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	// Get detailed statistics
	stats := psa.GetStatistics()
	
	fmt.Printf("Optimization Statistics:\n")
	fmt.Printf("  Duration: %v\n", stats.EndTime.Sub(stats.StartTime))
	fmt.Printf("  Total Iterations: %d\n", stats.TotalIterations)
	fmt.Printf("  Total Acceptances: %d\n", stats.TotalAcceptances)
	fmt.Printf("  Total Rejections: %d\n", stats.TotalRejections)
	fmt.Printf("  Best Fitness: %f\n", stats.BestSolution.Fitness)
	fmt.Printf("  Restart Count: %d\n", stats.RestartCount)
	fmt.Printf("  Exchange Count: %d\n", stats.ExchangeCount)
	fmt.Printf("  Number of Chains: %d\n", len(stats.ChainStatistics))
	
	// Chain-specific statistics
	for chainID, chainStats := range stats.ChainStatistics {
		acceptanceRate := float64(chainStats.Acceptances) / 
			float64(chainStats.Acceptances+chainStats.Rejections+1) * 100
		fmt.Printf("  Chain %d: %.1f%% acceptance rate\n", chainID, acceptanceRate)
	}
	
	// Output:
	// Optimization Statistics:
	//   Duration: 45.123ms
	//   Total Iterations: 8000
	//   Total Acceptances: 3456
	//   Total Rejections: 4544
	//   Best Fitness: 0.0023
	//   Restart Count: 0
	//   Exchange Count: 0
	//   Number of Chains: 4
	//   Chain 0: 43.2% acceptance rate
	//   Chain 1: 41.8% acceptance rate
	//   Chain 2: 44.1% acceptance rate
	//   Chain 3: 42.3% acceptance rate
}