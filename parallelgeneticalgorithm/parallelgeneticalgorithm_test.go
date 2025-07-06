package parallelgeneticalgorithm

import (
	"context"
	"math"
	"testing"
	"time"
)

func TestNewGeneticAlgorithm(t *testing.T) {
	config := GAConfig{
		PopulationSize:   50,
		ChromosomeLength: 10,
		MutationRate:     0.1,
		CrossoverRate:    0.8,
		ElitismRate:      0.1,
		MaxGenerations:   100,
		NumWorkers:       4,
		MinGene:          -10,
		MaxGene:          10,
		TargetFitness:    -0.1,
		StagnationLimit:  10,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	
	if ga.PopulationSize != 50 {
		t.Errorf("Expected population size 50, got %d", ga.PopulationSize)
	}
	
	if ga.ChromosomeLength != 10 {
		t.Errorf("Expected chromosome length 10, got %d", ga.ChromosomeLength)
	}
	
	if ga.NumWorkers != 4 {
		t.Errorf("Expected 4 workers, got %d", ga.NumWorkers)
	}
}

func TestSphereFunction(t *testing.T) {
	genes := []float64{1.0, 2.0, 3.0}
	fitness := SphereFunction(genes)
	expected := -14.0
	
	if math.Abs(fitness-expected) > 1e-10 {
		t.Errorf("Expected fitness %f, got %f", expected, fitness)
	}
}

func TestRastriginFunction(t *testing.T) {
	genes := []float64{0.0, 0.0, 0.0}
	fitness := RastriginFunction(genes)
	expected := -0.0
	
	if math.Abs(fitness-expected) > 1e-10 {
		t.Errorf("Expected fitness %f, got %f", expected, fitness)
	}
}

func TestRosenbrockFunction(t *testing.T) {
	genes := []float64{1.0, 1.0, 1.0}
	fitness := RosenbrockFunction(genes)
	expected := -0.0
	
	if math.Abs(fitness-expected) > 1e-10 {
		t.Errorf("Expected fitness %f, got %f", expected, fitness)
	}
}

func TestInitializePopulation(t *testing.T) {
	config := GAConfig{
		PopulationSize:   10,
		ChromosomeLength: 5,
		MinGene:          -5,
		MaxGene:          5,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	population := ga.initializePopulation()
	
	if len(population) != 10 {
		t.Errorf("Expected population size 10, got %d", len(population))
	}
	
	for i, individual := range population {
		if len(individual.Genes) != 5 {
			t.Errorf("Individual %d has wrong chromosome length: expected 5, got %d", i, len(individual.Genes))
		}
		
		for j, gene := range individual.Genes {
			if gene < -5 || gene > 5 {
				t.Errorf("Individual %d, gene %d out of bounds: %f", i, j, gene)
			}
		}
	}
}

func TestEvaluatePopulationParallel(t *testing.T) {
	config := GAConfig{
		PopulationSize:   20,
		ChromosomeLength: 3,
		NumWorkers:       4,
		MinGene:          -5,
		MaxGene:          5,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	population := ga.initializePopulation()
	
	ga.evaluatePopulationParallel(context.Background(), population)
	
	for i, individual := range population {
		if individual.Fitness == 0 {
			t.Errorf("Individual %d was not evaluated", i)
		}
		
		expectedFitness := SphereFunction(individual.Genes)
		if math.Abs(individual.Fitness-expectedFitness) > 1e-10 {
			t.Errorf("Individual %d fitness mismatch: expected %f, got %f", i, expectedFitness, individual.Fitness)
		}
	}
}

func TestCrossover(t *testing.T) {
	config := GAConfig{
		ChromosomeLength: 5,
		CrossoverRate:    1.0,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	
	parent1 := Individual{Genes: []float64{1, 2, 3, 4, 5}}
	parent2 := Individual{Genes: []float64{6, 7, 8, 9, 10}}
	
	child := ga.crossover(parent1, parent2, ga.rand)
	
	if len(child.Genes) != 5 {
		t.Errorf("Child has wrong chromosome length: expected 5, got %d", len(child.Genes))
	}
	
	hasParent1Genes := false
	hasParent2Genes := false
	
	for i, gene := range child.Genes {
		if gene == parent1.Genes[i] {
			hasParent1Genes = true
		}
		if gene == parent2.Genes[i] {
			hasParent2Genes = true
		}
	}
	
	if !hasParent1Genes && !hasParent2Genes {
		t.Error("Child doesn't inherit genes from either parent")
	}
}

func TestMutate(t *testing.T) {
	config := GAConfig{
		ChromosomeLength: 5,
		MutationRate:     1.0,
		MinGene:          -10,
		MaxGene:          10,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	
	original := Individual{Genes: []float64{1, 2, 3, 4, 5}}
	mutated := ga.mutate(original, ga.rand)
	
	if len(mutated.Genes) != 5 {
		t.Errorf("Mutated individual has wrong chromosome length: expected 5, got %d", len(mutated.Genes))
	}
	
	for _, gene := range mutated.Genes {
		if gene < -10 || gene > 10 {
			t.Errorf("Mutated gene out of bounds: %f", gene)
		}
	}
}

func TestTournamentSelection(t *testing.T) {
	config := GAConfig{
		PopulationSize: 10,
		RandomSeed:     42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	
	population := []Individual{
		{Genes: []float64{1}, Fitness: 1.0},
		{Genes: []float64{2}, Fitness: 2.0},
		{Genes: []float64{3}, Fitness: 3.0},
		{Genes: []float64{4}, Fitness: 4.0},
		{Genes: []float64{5}, Fitness: 5.0},
	}
	
	selected := ga.tournamentSelection(population, 3, ga.rand)
	
	found := false
	for _, individual := range population {
		if individual.Fitness == selected.Fitness {
			found = true
			break
		}
	}
	
	if !found {
		t.Error("Tournament selection returned individual not in population")
	}
}

func TestGeneticAlgorithmRun(t *testing.T) {
	config := GAConfig{
		PopulationSize:   50,
		ChromosomeLength: 2,
		MutationRate:     0.1,
		CrossoverRate:    0.8,
		ElitismRate:      0.1,
		MaxGenerations:   50,
		NumWorkers:       4,
		MinGene:          -5,
		MaxGene:          5,
		TargetFitness:    -0.1,
		StagnationLimit:  20,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	best, err := ga.Run(ctx)
	
	if err != nil {
		t.Errorf("GA run failed: %v", err)
	}
	
	if best == nil {
		t.Error("GA returned nil best individual")
	}
	
	if len(best.Genes) != 2 {
		t.Errorf("Best individual has wrong chromosome length: expected 2, got %d", len(best.Genes))
	}
	
	history := ga.GetHistory()
	if len(history) == 0 {
		t.Error("GA history is empty")
	}
	
	if history[0].Generation != 0 {
		t.Errorf("First generation should be 0, got %d", history[0].Generation)
	}
}

func TestGeneticAlgorithmIslandModel(t *testing.T) {
	config := GAConfig{
		PopulationSize:   20,
		ChromosomeLength: 2,
		MutationRate:     0.1,
		CrossoverRate:    0.8,
		ElitismRate:      0.1,
		MaxGenerations:   20,
		NumWorkers:       2,
		MinGene:          -5,
		MaxGene:          5,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	best, err := ga.RunIslandModel(ctx, 3, 0.1, 5)
	
	if err != nil {
		t.Errorf("Island model GA run failed: %v", err)
	}
	
	if best == nil {
		t.Error("Island model GA returned nil best individual")
	}
	
	if len(best.Genes) != 2 {
		t.Errorf("Best individual has wrong chromosome length: expected 2, got %d", len(best.Genes))
	}
}

func TestContextCancellation(t *testing.T) {
	config := GAConfig{
		PopulationSize:   50,
		ChromosomeLength: 10,
		MutationRate:     0.1,
		CrossoverRate:    0.8,
		ElitismRate:      0.1,
		MaxGenerations:   1000,
		NumWorkers:       4,
		MinGene:          -10,
		MaxGene:          10,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	
	start := time.Now()
	_, err := ga.Run(ctx)
	duration := time.Since(start)
	
	if err != context.DeadlineExceeded && err != context.Canceled {
		t.Errorf("Expected context cancellation error, got: %v", err)
	}
	
	if duration > 200*time.Millisecond {
		t.Errorf("GA took too long to respond to cancellation: %v", duration)
	}
}

func TestGetStats(t *testing.T) {
	config := GAConfig{
		PopulationSize:   30,
		ChromosomeLength: 5,
		MutationRate:     0.1,
		CrossoverRate:    0.8,
		NumWorkers:       4,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	
	stats := ga.GetStats()
	
	if stats["population_size"] != 30 {
		t.Errorf("Expected population size 30, got %v", stats["population_size"])
	}
	
	if stats["chromosome_length"] != 5 {
		t.Errorf("Expected chromosome length 5, got %v", stats["chromosome_length"])
	}
	
	if stats["mutation_rate"] != 0.1 {
		t.Errorf("Expected mutation rate 0.1, got %v", stats["mutation_rate"])
	}
	
	if stats["num_workers"] != 4 {
		t.Errorf("Expected 4 workers, got %v", stats["num_workers"])
	}
}

func TestUniformCrossover(t *testing.T) {
	config := GAConfig{
		ChromosomeLength: 5,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	
	parent1 := Individual{Genes: []float64{1, 2, 3, 4, 5}}
	parent2 := Individual{Genes: []float64{6, 7, 8, 9, 10}}
	
	child := ga.uniformCrossover(parent1, parent2, ga.rand)
	
	if len(child.Genes) != 5 {
		t.Errorf("Child has wrong chromosome length: expected 5, got %d", len(child.Genes))
	}
	
	for i, gene := range child.Genes {
		if gene != parent1.Genes[i] && gene != parent2.Genes[i] {
			t.Errorf("Child gene %d (%f) doesn't match either parent", i, gene)
		}
	}
}

func TestGaussianMutation(t *testing.T) {
	config := GAConfig{
		ChromosomeLength: 5,
		MutationRate:     1.0,
		MinGene:          -10,
		MaxGene:          10,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	
	original := Individual{Genes: []float64{0, 0, 0, 0, 0}}
	mutated := ga.gaussianMutation(original, ga.rand)
	
	if len(mutated.Genes) != 5 {
		t.Errorf("Mutated individual has wrong chromosome length: expected 5, got %d", len(mutated.Genes))
	}
	
	for _, gene := range mutated.Genes {
		if gene < -10 || gene > 10 {
			t.Errorf("Mutated gene out of bounds: %f", gene)
		}
	}
}

func TestAckleyFunction(t *testing.T) {
	genes := []float64{0.0, 0.0, 0.0}
	fitness := AckleyFunction(genes)
	expected := -0.0
	
	if math.Abs(fitness-expected) > 1e-10 {
		t.Errorf("Expected fitness %f, got %f", expected, fitness)
	}
}

func TestSchwefelFunction(t *testing.T) {
	genes := []float64{420.9687, 420.9687}
	fitness := SchwefelFunction(genes)
	
	if fitness < 800 {
		t.Errorf("Schwefel function should have high fitness at optimal point, got %f", fitness)
	}
}

func TestGriewankFunction(t *testing.T) {
	genes := []float64{0.0, 0.0, 0.0}
	fitness := GriewankFunction(genes)
	expected := -0.0
	
	if math.Abs(fitness-expected) > 1e-10 {
		t.Errorf("Expected fitness %f, got %f", expected, fitness)
	}
}

func TestStagnationDetection(t *testing.T) {
	config := GAConfig{
		PopulationSize:   10,
		ChromosomeLength: 2,
		MutationRate:     0.0,
		CrossoverRate:    0.0,
		ElitismRate:      1.0,
		MaxGenerations:   100,
		NumWorkers:       2,
		MinGene:          -1,
		MaxGene:          1,
		StagnationLimit:  5,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	_, err := ga.Run(ctx)
	
	if err == nil {
		t.Error("Expected stagnation error, but got none")
	}
}

func BenchmarkGeneticAlgorithmRun(b *testing.B) {
	config := GAConfig{
		PopulationSize:   50,
		ChromosomeLength: 10,
		MutationRate:     0.1,
		CrossoverRate:    0.8,
		ElitismRate:      0.1,
		MaxGenerations:   50,
		NumWorkers:       4,
		MinGene:          -10,
		MaxGene:          10,
		RandomSeed:       42,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ga := NewGeneticAlgorithm(config, SphereFunction)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		ga.Run(ctx)
		cancel()
	}
}

func BenchmarkFitnessEvaluation(b *testing.B) {
	genes := make([]float64, 100)
	for i := range genes {
		genes[i] = float64(i)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SphereFunction(genes)
	}
}

func BenchmarkPopulationEvaluation(b *testing.B) {
	config := GAConfig{
		PopulationSize:   100,
		ChromosomeLength: 20,
		NumWorkers:       4,
		MinGene:          -10,
		MaxGene:          10,
		RandomSeed:       42,
	}
	
	ga := NewGeneticAlgorithm(config, SphereFunction)
	population := ga.initializePopulation()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ga.evaluatePopulationParallel(context.Background(), population)
	}
}