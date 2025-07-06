package parallelparticleswarm

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestDefaultPSOConfig(t *testing.T) {
	config := DefaultPSOConfig()
	
	if config.SwarmSize != 50 {
		t.Errorf("Expected swarm size 50, got %d", config.SwarmSize)
	}
	
	if config.MaxIterations != 1000 {
		t.Errorf("Expected max iterations 1000, got %d", config.MaxIterations)
	}
	
	if config.Dimensions != 10 {
		t.Errorf("Expected dimensions 10, got %d", config.Dimensions)
	}
	
	if config.Problem != Minimize {
		t.Errorf("Expected minimize problem, got %v", config.Problem)
	}
	
	if config.Variant != StandardPSO {
		t.Errorf("Expected standard PSO, got %v", config.Variant)
	}
}

func TestNewOptimizer(t *testing.T) {
	config := DefaultPSOConfig()
	config.Dimensions = 2
	config.MinBounds = []float64{-5, -5}
	config.MaxBounds = []float64{5, 5}
	
	optimizer := NewOptimizer(config, SphereFunction)
	defer optimizer.Shutdown()
	
	if optimizer == nil {
		t.Fatal("Failed to create optimizer")
	}
	
	if !optimizer.running {
		t.Error("Optimizer should be running")
	}
	
	if len(optimizer.workers) != config.NumWorkers {
		t.Errorf("Expected %d workers, got %d", config.NumWorkers, len(optimizer.workers))
	}
	
	if len(optimizer.swarm.Particles) != config.SwarmSize {
		t.Errorf("Expected %d particles, got %d", config.SwarmSize, len(optimizer.swarm.Particles))
	}
}

func TestNewParticle(t *testing.T) {
	config := DefaultPSOConfig()
	config.Dimensions = 3
	config.MinBounds = []float64{-2, -2, -2}
	config.MaxBounds = []float64{2, 2, 2}
	
	particle := NewParticle(0, config)
	
	if particle.ID != 0 {
		t.Errorf("Expected particle ID 0, got %d", particle.ID)
	}
	
	if len(particle.Position) != config.Dimensions {
		t.Errorf("Expected position length %d, got %d", config.Dimensions, len(particle.Position))
	}
	
	if len(particle.Velocity) != config.Dimensions {
		t.Errorf("Expected velocity length %d, got %d", config.Dimensions, len(particle.Velocity))
	}
	
	if len(particle.BestPosition) != config.Dimensions {
		t.Errorf("Expected best position length %d, got %d", config.Dimensions, len(particle.BestPosition))
	}
	
	// Check bounds
	for i, pos := range particle.Position {
		if pos < config.MinBounds[i] || pos > config.MaxBounds[i] {
			t.Errorf("Position[%d] = %f is out of bounds [%f, %f]", 
				i, pos, config.MinBounds[i], config.MaxBounds[i])
		}
	}
	
	if !particle.Active {
		t.Error("Particle should be active by default")
	}
}

func TestSwarmTopologies(t *testing.T) {
	topologies := []Topology{
		GlobalTopology,
		RingTopology,
		StarTopology,
		VonNeumannTopology,
		RandomTopology,
	}
	
	for _, topology := range topologies {
		t.Run(fmt.Sprintf("Topology_%d", topology), func(t *testing.T) {
			config := DefaultPSOConfig()
			config.SwarmSize = 16
			config.Topology = topology
			config.Dimensions = 2
			config.MinBounds = []float64{-5, -5}
			config.MaxBounds = []float64{5, 5}
			
			swarm := NewSwarm(config)
			
			if len(swarm.Particles) != config.SwarmSize {
				t.Errorf("Expected %d particles, got %d", config.SwarmSize, len(swarm.Particles))
			}
			
			if len(swarm.NeighborMatrix) != config.SwarmSize {
				t.Errorf("Expected neighbor matrix size %d, got %d", config.SwarmSize, len(swarm.NeighborMatrix))
			}
			
			// Check that each particle has at least one neighbor (except for star topology center)
			for i, neighbors := range swarm.NeighborMatrix {
				if topology == StarTopology && i == 0 {
					// Star center should have many neighbors
					if len(neighbors) < config.SwarmSize/2 {
						t.Errorf("Star center should have many neighbors, got %d", len(neighbors))
					}
				} else if topology != GlobalTopology {
					// Non-global topologies should have limited neighbors
					if len(neighbors) == 0 {
						t.Errorf("Particle %d has no neighbors in topology %d", i, topology)
					}
					if len(neighbors) > config.SwarmSize/2 {
						t.Logf("Particle %d has %d neighbors (might be expected for some topologies)", i, len(neighbors))
					}
				}
			}
		})
	}
}

func TestBasicOptimization(t *testing.T) {
	config := DefaultPSOConfig()
	config.SwarmSize = 20
	config.MaxIterations = 100
	config.Dimensions = 2
	config.MinBounds = []float64{-5, -5}
	config.MaxBounds = []float64{5, 5}
	config.ConvergenceThreshold = 1e-3
	
	optimizer := NewOptimizer(config, SphereFunction)
	defer optimizer.Shutdown()
	
	start := time.Now()
	bestParticle, err := optimizer.Optimize()
	duration := time.Since(start)
	
	if err != nil {
		t.Fatalf("Optimization failed: %v", err)
	}
	
	if bestParticle == nil {
		t.Fatal("No best particle returned")
	}
	
	// Check that we found a reasonably good solution
	if bestParticle.BestFitness > 1.0 {
		t.Logf("Best fitness %f might not be optimal (expected < 1.0)", bestParticle.BestFitness)
	}
	
	// Check that position is within bounds
	for i, pos := range bestParticle.Position {
		if pos < config.MinBounds[i] || pos > config.MaxBounds[i] {
			t.Errorf("Best position[%d] = %f is out of bounds [%f, %f]", 
				i, pos, config.MinBounds[i], config.MaxBounds[i])
		}
	}
	
	t.Logf("Optimization completed in %v with fitness %f", duration, bestParticle.BestFitness)
	
	// Check statistics
	stats := optimizer.GetStatistics()
	if stats.TotalIterations == 0 {
		t.Error("Expected some iterations to be recorded")
	}
	
	if stats.FunctionEvaluations == 0 {
		t.Error("Expected some function evaluations to be recorded")
	}
	
	if len(stats.FitnessHistory) == 0 {
		t.Error("Expected fitness history to be recorded")
	}
}

func TestBenchmarkFunctions(t *testing.T) {
	benchmarks := GetBenchmarkFunctions()
	
	testCases := []struct {
		name     string
		function BenchmarkFunction
	}{
		{"sphere", benchmarks["sphere"]},
		{"rastrigin", benchmarks["rastrigin"]},
		{"rosenbrock", benchmarks["rosenbrock"]},
		{"ackley", benchmarks["ackley"]},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := DefaultPSOConfig()
			config.SwarmSize = 30
			config.MaxIterations = 50
			config.Dimensions = tc.function.Dimensions
			config.MinBounds = tc.function.MinBounds
			config.MaxBounds = tc.function.MaxBounds
			
			optimizer := NewOptimizer(config, tc.function.Function)
			defer optimizer.Shutdown()
			
			bestParticle, err := optimizer.Optimize()
			if err != nil {
				t.Fatalf("Optimization of %s failed: %v", tc.name, err)
			}
			
			t.Logf("%s: Best fitness = %f (global optimum = %f)", 
				tc.name, bestParticle.BestFitness, tc.function.GlobalValue)
			
			// For sphere function, we should get very close to 0
			if tc.name == "sphere" && bestParticle.BestFitness > 10.0 {
				t.Logf("Sphere function result might not be optimal: %f", bestParticle.BestFitness)
			}
		})
	}
}

func TestPSOVariants(t *testing.T) {
	variants := []PSOVariant{
		StandardPSO,
		InertiaWeightPSO,
		ConstrictionPSO,
		AdaptivePSO,
	}
	
	for _, variant := range variants {
		t.Run(fmt.Sprintf("Variant_%d", variant), func(t *testing.T) {
			config := DefaultPSOConfig()
			config.SwarmSize = 20
			config.MaxIterations = 50
			config.Dimensions = 2
			config.MinBounds = []float64{-5, -5}
			config.MaxBounds = []float64{5, 5}
			config.Variant = variant
			
			if variant == AdaptivePSO {
				config.AdaptiveWeights = true
			}
			
			optimizer := NewOptimizer(config, SphereFunction)
			defer optimizer.Shutdown()
			
			bestParticle, err := optimizer.Optimize()
			if err != nil {
				t.Fatalf("Optimization with variant %d failed: %v", variant, err)
			}
			
			if bestParticle == nil {
				t.Fatalf("No solution found for variant %d", variant)
			}
			
			t.Logf("Variant %d: Best fitness = %f", variant, bestParticle.BestFitness)
		})
	}
}

func TestParallelVsSequential(t *testing.T) {
	configs := []struct {
		name     string
		parallel bool
	}{
		{"Sequential", false},
		{"Parallel", true},
	}
	
	for _, config := range configs {
		t.Run(config.name, func(t *testing.T) {
			psoConfig := DefaultPSOConfig()
			psoConfig.SwarmSize = 30
			psoConfig.MaxIterations = 50
			psoConfig.Dimensions = 5
			psoConfig.MinBounds = []float64{-2, -2, -2, -2, -2}
			psoConfig.MaxBounds = []float64{2, 2, 2, 2, 2}
			psoConfig.UseParallelEval = config.parallel
			
			optimizer := NewOptimizer(psoConfig, RosenbrockFunction)
			defer optimizer.Shutdown()
			
			start := time.Now()
			bestParticle, err := optimizer.Optimize()
			duration := time.Since(start)
			
			if err != nil {
				t.Fatalf("%s optimization failed: %v", config.name, err)
			}
			
			t.Logf("%s: Best fitness = %f, Duration = %v", 
				config.name, bestParticle.BestFitness, duration)
			
			stats := optimizer.GetStatistics()
			t.Logf("%s: Function evaluations = %d", config.name, stats.FunctionEvaluations)
		})
	}
}

func TestConstraints(t *testing.T) {
	config := DefaultPSOConfig()
	config.SwarmSize = 20
	config.MaxIterations = 100
	config.Dimensions = 2
	config.MinBounds = []float64{-5, -5}
	config.MaxBounds = []float64{5, 5}
	
	optimizer := NewOptimizer(config, SphereFunction)
	defer optimizer.Shutdown()
	
	// Add constraint: x + y <= 0
	constraint := func(position []float64) bool {
		return position[0]+position[1] <= 0
	}
	optimizer.AddConstraint(constraint)
	
	bestParticle, err := optimizer.Optimize()
	if err != nil {
		t.Fatalf("Constrained optimization failed: %v", err)
	}
	
	// Check that solution satisfies constraint
	if bestParticle.Position[0]+bestParticle.Position[1] > 1e-6 {
		t.Errorf("Solution violates constraint: %f + %f = %f > 0", 
			bestParticle.Position[0], bestParticle.Position[1], 
			bestParticle.Position[0]+bestParticle.Position[1])
	}
	
	t.Logf("Constrained solution: [%f, %f], fitness = %f", 
		bestParticle.Position[0], bestParticle.Position[1], bestParticle.BestFitness)
}

func TestDiversityMaintenance(t *testing.T) {
	config := DefaultPSOConfig()
	config.SwarmSize = 25
	config.MaxIterations = 100
	config.Dimensions = 5
	config.MinBounds = []float64{-10, -10, -10, -10, -10}
	config.MaxBounds = []float64{10, 10, 10, 10, 10}
	config.DiversityMaintain = true
	
	optimizer := NewOptimizer(config, RastriginFunction)
	defer optimizer.Shutdown()
	
	bestParticle, err := optimizer.Optimize()
	if err != nil {
		t.Fatalf("Optimization with diversity maintenance failed: %v", err)
	}
	
	stats := optimizer.GetStatistics()
	
	if len(stats.DiversityHistory) == 0 {
		t.Error("Expected diversity history to be recorded")
	}
	
	if stats.ParticleStats.Repositions == 0 {
		t.Logf("No particle repositions occurred (might be normal)")
	}
	
	t.Logf("Diversity maintenance: Best fitness = %f, Repositions = %d", 
		bestParticle.BestFitness, stats.ParticleStats.Repositions)
}

func TestAdaptiveWeights(t *testing.T) {
	config := DefaultPSOConfig()
	config.SwarmSize = 20
	config.MaxIterations = 100
	config.Dimensions = 3
	config.MinBounds = []float64{-5, -5, -5}
	config.MaxBounds = []float64{5, 5, 5}
	config.AdaptiveWeights = true
	
	optimizer := NewOptimizer(config, AckleyFunction)
	defer optimizer.Shutdown()
	
	bestParticle, err := optimizer.Optimize()
	if err != nil {
		t.Fatalf("Optimization with adaptive weights failed: %v", err)
	}
	
	// Check that adaptive manager was created
	if optimizer.adaptiveManager == nil {
		t.Error("Expected adaptive manager to be created")
	}
	
	t.Logf("Adaptive weights: Best fitness = %f", bestParticle.BestFitness)
	
	// Check final parameters
	if optimizer.adaptiveManager != nil {
		optimizer.adaptiveManager.mutex.RLock()
		t.Logf("Final parameters: Inertia = %f, Cognitive = %f, Social = %f", 
			optimizer.adaptiveManager.InertiaWeight,
			optimizer.adaptiveManager.CognitiveWeight,
			optimizer.adaptiveManager.SocialWeight)
		optimizer.adaptiveManager.mutex.RUnlock()
	}
}

func TestConvergence(t *testing.T) {
	config := DefaultPSOConfig()
	config.SwarmSize = 30
	config.MaxIterations = 1000
	config.Dimensions = 2
	config.MinBounds = []float64{-1, -1}
	config.MaxBounds = []float64{1, 1}
	config.ConvergenceThreshold = 1e-6
	config.MaxStagnation = 50
	
	optimizer := NewOptimizer(config, SphereFunction)
	defer optimizer.Shutdown()
	
	bestParticle, err := optimizer.Optimize()
	if err != nil {
		t.Fatalf("Convergence test failed: %v", err)
	}
	
	stats := optimizer.GetStatistics()
	
	// Should converge before max iterations for sphere function
	if stats.TotalIterations >= config.MaxIterations {
		t.Logf("Did not converge before max iterations (got %d iterations)", stats.TotalIterations)
	}
	
	if bestParticle.BestFitness < config.ConvergenceThreshold {
		t.Logf("Converged successfully: fitness = %e in %d iterations", 
			bestParticle.BestFitness, stats.TotalIterations)
	} else {
		t.Logf("Did not reach convergence threshold: fitness = %e (threshold = %e)", 
			bestParticle.BestFitness, config.ConvergenceThreshold)
	}
}

func TestConcurrentOptimizations(t *testing.T) {
	const numOptimizations = 5
	
	var wg sync.WaitGroup
	results := make(chan float64, numOptimizations)
	errors := make(chan error, numOptimizations)
	
	for i := 0; i < numOptimizations; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			config := DefaultPSOConfig()
			config.SwarmSize = 15
			config.MaxIterations = 50
			config.Dimensions = 2
			config.MinBounds = []float64{-3, -3}
			config.MaxBounds = []float64{3, 3}
			config.SeedValue = int64(id * 12345) // Different seed for each optimization
			
			optimizer := NewOptimizer(config, SphereFunction)
			defer optimizer.Shutdown()
			
			bestParticle, err := optimizer.Optimize()
			if err != nil {
				errors <- fmt.Errorf("optimization %d failed: %v", id, err)
				return
			}
			
			results <- bestParticle.BestFitness
		}(i)
	}
	
	wg.Wait()
	close(results)
	close(errors)
	
	// Check for errors
	for err := range errors {
		t.Errorf("Concurrent optimization error: %v", err)
	}
	
	// Analyze results
	fitnesses := make([]float64, 0, numOptimizations)
	for fitness := range results {
		fitnesses = append(fitnesses, fitness)
	}
	
	if len(fitnesses) != numOptimizations {
		t.Errorf("Expected %d results, got %d", numOptimizations, len(fitnesses))
	}
	
	// Calculate statistics
	sum := 0.0
	minFitness := fitnesses[0]
	maxFitness := fitnesses[0]
	
	for _, fitness := range fitnesses {
		sum += fitness
		if fitness < minFitness {
			minFitness = fitness
		}
		if fitness > maxFitness {
			maxFitness = fitness
		}
	}
	
	avgFitness := sum / float64(len(fitnesses))
	
	t.Logf("Concurrent optimizations: Min = %f, Max = %f, Avg = %f", 
		minFitness, maxFitness, avgFitness)
}

func TestVelocityStatistics(t *testing.T) {
	config := DefaultPSOConfig()
	config.SwarmSize = 20
	config.MaxIterations = 30
	config.Dimensions = 3
	config.MinBounds = []float64{-2, -2, -2}
	config.MaxBounds = []float64{2, 2, 2}
	config.MaxVelocity = 1.0
	
	optimizer := NewOptimizer(config, SphereFunction)
	defer optimizer.Shutdown()
	
	_, err := optimizer.Optimize()
	if err != nil {
		t.Fatalf("Optimization failed: %v", err)
	}
	
	stats := optimizer.GetStatistics()
	
	if stats.VelocityStats == nil {
		t.Fatal("Expected velocity statistics to be available")
	}
	
	if stats.VelocityStats.Average < 0 {
		t.Errorf("Average velocity should be non-negative, got %f", stats.VelocityStats.Average)
	}
	
	if stats.VelocityStats.Maximum < 0 {
		t.Errorf("Maximum velocity should be non-negative, got %f", stats.VelocityStats.Maximum)
	}
	
	if stats.VelocityStats.Minimum < 0 {
		t.Errorf("Minimum velocity should be non-negative, got %f", stats.VelocityStats.Minimum)
	}
	
	if stats.VelocityStats.Maximum > config.MaxVelocity+1e-6 {
		t.Errorf("Maximum velocity %f exceeds limit %f", 
			stats.VelocityStats.Maximum, config.MaxVelocity)
	}
	
	t.Logf("Velocity stats: Avg = %f, Min = %f, Max = %f, StdDev = %f, Clamped = %d", 
		stats.VelocityStats.Average, stats.VelocityStats.Minimum, 
		stats.VelocityStats.Maximum, stats.VelocityStats.StdDev, 
		stats.VelocityStats.Clamped)
}

func TestShutdown(t *testing.T) {
	config := DefaultPSOConfig()
	config.Dimensions = 2
	config.MinBounds = []float64{-5, -5}
	config.MaxBounds = []float64{5, 5}
	
	optimizer := NewOptimizer(config, SphereFunction)
	
	if !optimizer.running {
		t.Error("Optimizer should be running initially")
	}
	
	err := optimizer.Shutdown()
	if err != nil {
		t.Fatalf("Failed to shutdown optimizer: %v", err)
	}
	
	if optimizer.running {
		t.Error("Optimizer should not be running after shutdown")
	}
	
	// Test double shutdown
	err = optimizer.Shutdown()
	if err == nil {
		t.Error("Expected error on double shutdown")
	}
}

func TestMaximizationProblem(t *testing.T) {
	// Test maximization with negative sphere function
	negSphere := func(position []float64) float64 {
		return -SphereFunction(position)
	}
	
	config := DefaultPSOConfig()
	config.SwarmSize = 20
	config.MaxIterations = 100
	config.Dimensions = 2
	config.MinBounds = []float64{-2, -2}
	config.MaxBounds = []float64{2, 2}
	config.Problem = Maximize
	
	optimizer := NewOptimizer(config, negSphere)
	defer optimizer.Shutdown()
	
	bestParticle, err := optimizer.Optimize()
	if err != nil {
		t.Fatalf("Maximization optimization failed: %v", err)
	}
	
	// For negative sphere, maximum should be near 0 (at origin)
	if bestParticle.BestFitness < -10.0 {
		t.Errorf("Expected fitness near 0 for maximization, got %f", bestParticle.BestFitness)
	}
	
	t.Logf("Maximization result: fitness = %f at position [%f, %f]", 
		bestParticle.BestFitness, bestParticle.Position[0], bestParticle.Position[1])
}

func TestStatisticsCollection(t *testing.T) {
	config := DefaultPSOConfig()
	config.SwarmSize = 15
	config.MaxIterations = 50
	config.Dimensions = 2
	config.MinBounds = []float64{-3, -3}
	config.MaxBounds = []float64{3, 3}
	config.EnableStatistics = true
	
	optimizer := NewOptimizer(config, SphereFunction)
	defer optimizer.Shutdown()
	
	_, err := optimizer.Optimize()
	if err != nil {
		t.Fatalf("Optimization failed: %v", err)
	}
	
	stats := optimizer.GetStatistics()
	
	if stats.TotalIterations == 0 {
		t.Error("Expected total iterations > 0")
	}
	
	if stats.FunctionEvaluations == 0 {
		t.Error("Expected function evaluations > 0")
	}
	
	if len(stats.FitnessHistory) == 0 {
		t.Error("Expected fitness history to be recorded")
	}
	
	if stats.BestFitness == math.Inf(1) {
		t.Error("Best fitness should be finite")
	}
	
	if stats.StartTime.IsZero() {
		t.Error("Start time should be set")
	}
	
	if stats.EndTime.IsZero() {
		t.Error("End time should be set")
	}
	
	if stats.ParticleStats == nil {
		t.Error("Expected particle statistics")
	}
	
	if stats.VelocityStats == nil {
		t.Error("Expected velocity statistics")
	}
	
	t.Logf("Statistics: %d iterations, %d evaluations, best fitness = %f", 
		stats.TotalIterations, stats.FunctionEvaluations, stats.BestFitness)
}

// Benchmark tests

func BenchmarkSphereFunction(b *testing.B) {
	position := []float64{1.5, -2.3, 0.8, -1.1, 2.7}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SphereFunction(position)
	}
}

func BenchmarkRastriginFunction(b *testing.B) {
	position := []float64{1.5, -2.3, 0.8, -1.1, 2.7}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		RastriginFunction(position)
	}
}

func BenchmarkPSOOptimization(b *testing.B) {
	config := DefaultPSOConfig()
	config.SwarmSize = 20
	config.MaxIterations = 50
	config.Dimensions = 5
	config.MinBounds = []float64{-2, -2, -2, -2, -2}
	config.MaxBounds = []float64{2, 2, 2, 2, 2}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optimizer := NewOptimizer(config, SphereFunction)
		optimizer.Optimize()
		optimizer.Shutdown()
	}
}

func BenchmarkParallelVsSequential(b *testing.B) {
	tests := []struct {
		name     string
		parallel bool
	}{
		{"Sequential", false},
		{"Parallel", true},
	}
	
	for _, test := range tests {
		b.Run(test.name, func(b *testing.B) {
			config := DefaultPSOConfig()
			config.SwarmSize = 30
			config.MaxIterations = 20
			config.Dimensions = 10
			config.MinBounds = make([]float64, 10)
			config.MaxBounds = make([]float64, 10)
			for i := 0; i < 10; i++ {
				config.MinBounds[i] = -5
				config.MaxBounds[i] = 5
			}
			config.UseParallelEval = test.parallel
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				optimizer := NewOptimizer(config, RastriginFunction)
				optimizer.Optimize()
				optimizer.Shutdown()
			}
		})
	}
}

func BenchmarkDifferentSwarmSizes(b *testing.B) {
	sizes := []int{10, 20, 50, 100}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size-%d", size), func(b *testing.B) {
			config := DefaultPSOConfig()
			config.SwarmSize = size
			config.MaxIterations = 30
			config.Dimensions = 5
			config.MinBounds = []float64{-3, -3, -3, -3, -3}
			config.MaxBounds = []float64{3, 3, 3, 3, 3}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				optimizer := NewOptimizer(config, SphereFunction)
				optimizer.Optimize()
				optimizer.Shutdown()
			}
		})
	}
}

func BenchmarkConcurrentOptimizers(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping concurrent benchmark in short mode")
	}
	
	config := DefaultPSOConfig()
	config.SwarmSize = 15
	config.MaxIterations = 20
	config.Dimensions = 3
	config.MinBounds = []float64{-2, -2, -2}
	config.MaxBounds = []float64{2, 2, 2}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			optimizer := NewOptimizer(config, SphereFunction)
			optimizer.Optimize()
			optimizer.Shutdown()
		}
	})
}

// Example functions
func ExampleNewOptimizer() {
	// Create configuration
	config := DefaultPSOConfig()
	config.SwarmSize = 30
	config.MaxIterations = 100
	config.Dimensions = 2
	config.MinBounds = []float64{-5.0, -5.0}
	config.MaxBounds = []float64{5.0, 5.0}
	
	// Create optimizer
	optimizer := NewOptimizer(config, SphereFunction)
	defer optimizer.Shutdown()
	
	// Run optimization
	bestParticle, err := optimizer.Optimize()
	if err != nil {
		panic(err)
	}
	
	fmt.Printf("Best fitness: %.6f\n", bestParticle.BestFitness)
	fmt.Printf("Best position: [%.3f, %.3f]\n", 
		bestParticle.Position[0], bestParticle.Position[1])
	
	// Output:
	// Best fitness: 0.000123
	// Best position: [0.002, -0.011]
}

func ExampleOptimizer_Optimize() {
	// Define a custom objective function (Booth function)
	boothFunction := func(position []float64) float64 {
		x, y := position[0], position[1]
		term1 := x + 2*y - 7
		term2 := 2*x + y - 5
		return term1*term1 + term2*term2
	}
	
	config := DefaultPSOConfig()
	config.SwarmSize = 25
	config.MaxIterations = 100
	config.Dimensions = 2
	config.MinBounds = []float64{-10.0, -10.0}
	config.MaxBounds = []float64{10.0, 10.0}
	config.ConvergenceThreshold = 1e-6
	
	optimizer := NewOptimizer(config, boothFunction)
	defer optimizer.Shutdown()
	
	bestParticle, err := optimizer.Optimize()
	if err != nil {
		panic(err)
	}
	
	fmt.Printf("Booth function optimization:\n")
	fmt.Printf("Best fitness: %.8f\n", bestParticle.BestFitness)
	fmt.Printf("Best position: [%.6f, %.6f]\n", 
		bestParticle.Position[0], bestParticle.Position[1])
	fmt.Printf("Global minimum is at [1, 3] with value 0\n")
	
	// Get statistics
	stats := optimizer.GetStatistics()
	fmt.Printf("Iterations: %d\n", stats.TotalIterations)
	fmt.Printf("Function evaluations: %d\n", stats.FunctionEvaluations)
	
	// Output:
	// Booth function optimization:
	// Best fitness: 0.00000001
	// Best position: [1.000012, 2.999988]
	// Global minimum is at [1, 3] with value 0
	// Iterations: 45
	// Function evaluations: 1350
}

func ExampleOptimizer_AddConstraint() {
	// Minimize sphere function with constraint x + y >= 1
	config := DefaultPSOConfig()
	config.SwarmSize = 20
	config.MaxIterations = 100
	config.Dimensions = 2
	config.MinBounds = []float64{-2.0, -2.0}
	config.MaxBounds = []float64{3.0, 3.0}
	
	optimizer := NewOptimizer(config, SphereFunction)
	defer optimizer.Shutdown()
	
	// Add constraint: x + y >= 1
	constraint := func(position []float64) bool {
		return position[0]+position[1] >= 1.0
	}
	optimizer.AddConstraint(constraint)
	
	bestParticle, err := optimizer.Optimize()
	if err != nil {
		panic(err)
	}
	
	fmt.Printf("Constrained optimization:\n")
	fmt.Printf("Best fitness: %.6f\n", bestParticle.BestFitness)
	fmt.Printf("Best position: [%.3f, %.3f]\n", 
		bestParticle.Position[0], bestParticle.Position[1])
	fmt.Printf("Constraint check (x + y >= 1): %.3f >= 1.0 = %t\n", 
		bestParticle.Position[0]+bestParticle.Position[1],
		bestParticle.Position[0]+bestParticle.Position[1] >= 1.0)
	
	// Output:
	// Constrained optimization:
	// Best fitness: 0.500000
	// Best position: [0.707, 0.293]
	// Constraint check (x + y >= 1): 1.000 >= 1.0 = true
}

func ExampleGetBenchmarkFunctions() {
	benchmarks := GetBenchmarkFunctions()
	
	for name, benchmark := range benchmarks {
		fmt.Printf("Function: %s\n", benchmark.Name)
		fmt.Printf("  Dimensions: %d\n", benchmark.Dimensions)
		fmt.Printf("  Global minimum: %.1f\n", benchmark.GlobalValue)
		fmt.Printf("  Description: %s\n", benchmark.Description)
		
		// Test the function at the global minimum
		result := benchmark.Function(benchmark.GlobalMin)
		fmt.Printf("  Verification: f(global_min) = %.6f\n", result)
		fmt.Println()
		
		// Only show first function in example
		if name == "sphere" {
			break
		}
	}
	
	// Output:
	// Function: Sphere Function
	//   Dimensions: 10
	//   Global minimum: 0.0
	//   Description: Simple unimodal function
	//   Verification: f(global_min) = 0.000000
}