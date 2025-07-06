package parallelgeneticalgorithm

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

type Individual struct {
	Genes   []float64
	Fitness float64
}

type Population struct {
	Individuals []Individual
	Generation  int
	BestFitness float64
	AvgFitness  float64
}

type FitnessFunction func([]float64) float64

type GeneticAlgorithm struct {
	PopulationSize    int
	ChromosomeLength  int
	MutationRate      float64
	CrossoverRate     float64
	ElitismRate       float64
	MaxGenerations    int
	NumWorkers        int
	FitnessFunc       FitnessFunction
	MinGene           float64
	MaxGene           float64
	TargetFitness     float64
	StagnationLimit   int
	
	bestIndividual    Individual
	history           []Population
	mutex             sync.RWMutex
	rand              *rand.Rand
}

type GAConfig struct {
	PopulationSize    int
	ChromosomeLength  int
	MutationRate      float64
	CrossoverRate     float64
	ElitismRate       float64
	MaxGenerations    int
	NumWorkers        int
	MinGene           float64
	MaxGene           float64
	TargetFitness     float64
	StagnationLimit   int
	RandomSeed        int64
}

func NewGeneticAlgorithm(config GAConfig, fitnessFunc FitnessFunction) *GeneticAlgorithm {
	if config.RandomSeed == 0 {
		config.RandomSeed = time.Now().UnixNano()
	}
	
	return &GeneticAlgorithm{
		PopulationSize:    config.PopulationSize,
		ChromosomeLength:  config.ChromosomeLength,
		MutationRate:      config.MutationRate,
		CrossoverRate:     config.CrossoverRate,
		ElitismRate:       config.ElitismRate,
		MaxGenerations:    config.MaxGenerations,
		NumWorkers:        config.NumWorkers,
		FitnessFunc:       fitnessFunc,
		MinGene:           config.MinGene,
		MaxGene:           config.MaxGene,
		TargetFitness:     config.TargetFitness,
		StagnationLimit:   config.StagnationLimit,
		history:           make([]Population, 0),
		rand:              rand.New(rand.NewSource(config.RandomSeed)),
	}
}

func (ga *GeneticAlgorithm) Run(ctx context.Context) (*Individual, error) {
	population := ga.initializePopulation()
	
	ga.evaluatePopulationParallel(ctx, population)
	ga.updateBest(population)
	ga.recordHistory(population)
	
	stagnationCount := 0
	previousBest := population[0].Fitness
	
	for generation := 0; generation < ga.MaxGenerations; generation++ {
		select {
		case <-ctx.Done():
			return &ga.bestIndividual, ctx.Err()
		default:
		}
		
		newPopulation := ga.evolveGenerationParallel(ctx, population)
		
		ga.evaluatePopulationParallel(ctx, newPopulation)
		
		sort.Slice(newPopulation, func(i, j int) bool {
			return newPopulation[i].Fitness > newPopulation[j].Fitness
		})
		
		if newPopulation[0].Fitness > ga.bestIndividual.Fitness {
			ga.updateBest(newPopulation)
			stagnationCount = 0
		} else {
			stagnationCount++
		}
		
		if ga.TargetFitness > 0 && newPopulation[0].Fitness >= ga.TargetFitness {
			return &ga.bestIndividual, nil
		}
		
		if ga.StagnationLimit > 0 && stagnationCount >= ga.StagnationLimit {
			return &ga.bestIndividual, fmt.Errorf("evolution stagnated after %d generations", stagnationCount)
		}
		
		population = newPopulation
		ga.recordHistory(population)
		
		if math.Abs(newPopulation[0].Fitness-previousBest) < 1e-10 {
			stagnationCount++
		}
		previousBest = newPopulation[0].Fitness
	}
	
	return &ga.bestIndividual, nil
}

func (ga *GeneticAlgorithm) RunIslandModel(ctx context.Context, numIslands int, migrationRate float64, migrationInterval int) (*Individual, error) {
	if numIslands < 2 {
		return ga.Run(ctx)
	}
	
	islands := make([][]Individual, numIslands)
	for i := range islands {
		islands[i] = ga.initializePopulation()
		ga.evaluatePopulationParallel(ctx, islands[i])
	}
	
	var wg sync.WaitGroup
	results := make(chan Individual, numIslands)
	
	for island := 0; island < numIslands; island++ {
		wg.Add(1)
		go func(islandID int) {
			defer wg.Done()
			
			population := islands[islandID]
			localGA := *ga
			localGA.rand = rand.New(rand.NewSource(time.Now().UnixNano() + int64(islandID)))
			
			for generation := 0; generation < ga.MaxGenerations; generation++ {
				select {
				case <-ctx.Done():
					return
				default:
				}
				
				newPopulation := localGA.evolveGenerationParallel(ctx, population)
				localGA.evaluatePopulationParallel(ctx, newPopulation)
				
				sort.Slice(newPopulation, func(i, j int) bool {
					return newPopulation[i].Fitness > newPopulation[j].Fitness
				})
				
				if generation%migrationInterval == 0 && generation > 0 {
					localGA.performMigration(islands, islandID, migrationRate)
				}
				
				population = newPopulation
				islands[islandID] = population
			}
			
			results <- population[0]
		}(island)
	}
	
	go func() {
		wg.Wait()
		close(results)
	}()
	
	var bestOverall Individual
	bestOverall.Fitness = -math.Inf(1)
	
	for result := range results {
		if result.Fitness > bestOverall.Fitness {
			bestOverall = result
		}
	}
	
	return &bestOverall, nil
}

func (ga *GeneticAlgorithm) initializePopulation() []Individual {
	population := make([]Individual, ga.PopulationSize)
	
	for i := range population {
		individual := Individual{
			Genes: make([]float64, ga.ChromosomeLength),
		}
		
		for j := range individual.Genes {
			individual.Genes[j] = ga.MinGene + ga.rand.Float64()*(ga.MaxGene-ga.MinGene)
		}
		
		population[i] = individual
	}
	
	return population
}

func (ga *GeneticAlgorithm) evaluatePopulationParallel(ctx context.Context, population []Individual) {
	jobs := make(chan int, len(population))
	var wg sync.WaitGroup
	
	for i := 0; i < ga.NumWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case idx, ok := <-jobs:
					if !ok {
						return
					}
					population[idx].Fitness = ga.FitnessFunc(population[idx].Genes)
				case <-ctx.Done():
					return
				}
			}
		}()
	}
	
	for i := range population {
		jobs <- i
	}
	close(jobs)
	
	wg.Wait()
}

func (ga *GeneticAlgorithm) evolveGenerationParallel(ctx context.Context, population []Individual) []Individual {
	sort.Slice(population, func(i, j int) bool {
		return population[i].Fitness > population[j].Fitness
	})
	
	eliteCount := int(float64(ga.PopulationSize) * ga.ElitismRate)
	newPopulation := make([]Individual, ga.PopulationSize)
	
	for i := 0; i < eliteCount; i++ {
		newPopulation[i] = population[i]
	}
	
	jobs := make(chan int, ga.PopulationSize-eliteCount)
	var wg sync.WaitGroup
	
	for i := 0; i < ga.NumWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			localRand := rand.New(rand.NewSource(time.Now().UnixNano()))
			
			for {
				select {
				case idx, ok := <-jobs:
					if !ok {
						return
					}
					
					parent1 := ga.selectParent(population, localRand)
					parent2 := ga.selectParent(population, localRand)
					
					child := ga.crossover(parent1, parent2, localRand)
					child = ga.mutate(child, localRand)
					
					newPopulation[idx] = child
				case <-ctx.Done():
					return
				}
			}
		}()
	}
	
	for i := eliteCount; i < ga.PopulationSize; i++ {
		jobs <- i
	}
	close(jobs)
	
	wg.Wait()
	return newPopulation
}

func (ga *GeneticAlgorithm) selectParent(population []Individual, r *rand.Rand) Individual {
	return ga.tournamentSelection(population, 3, r)
}

func (ga *GeneticAlgorithm) tournamentSelection(population []Individual, tournamentSize int, r *rand.Rand) Individual {
	best := population[r.Intn(len(population))]
	
	for i := 1; i < tournamentSize; i++ {
		candidate := population[r.Intn(len(population))]
		if candidate.Fitness > best.Fitness {
			best = candidate
		}
	}
	
	return best
}

func (ga *GeneticAlgorithm) crossover(parent1, parent2 Individual, r *rand.Rand) Individual {
	child := Individual{
		Genes: make([]float64, ga.ChromosomeLength),
	}
	
	if r.Float64() < ga.CrossoverRate {
		crossoverPoint := r.Intn(ga.ChromosomeLength)
		
		for i := 0; i < crossoverPoint; i++ {
			child.Genes[i] = parent1.Genes[i]
		}
		for i := crossoverPoint; i < ga.ChromosomeLength; i++ {
			child.Genes[i] = parent2.Genes[i]
		}
	} else {
		if r.Float64() < 0.5 {
			copy(child.Genes, parent1.Genes)
		} else {
			copy(child.Genes, parent2.Genes)
		}
	}
	
	return child
}

func (ga *GeneticAlgorithm) uniformCrossover(parent1, parent2 Individual, r *rand.Rand) Individual {
	child := Individual{
		Genes: make([]float64, ga.ChromosomeLength),
	}
	
	for i := range child.Genes {
		if r.Float64() < 0.5 {
			child.Genes[i] = parent1.Genes[i]
		} else {
			child.Genes[i] = parent2.Genes[i]
		}
	}
	
	return child
}

func (ga *GeneticAlgorithm) mutate(individual Individual, r *rand.Rand) Individual {
	mutated := Individual{
		Genes: make([]float64, len(individual.Genes)),
	}
	copy(mutated.Genes, individual.Genes)
	
	for i := range mutated.Genes {
		if r.Float64() < ga.MutationRate {
			mutated.Genes[i] = ga.MinGene + r.Float64()*(ga.MaxGene-ga.MinGene)
		}
	}
	
	return mutated
}

func (ga *GeneticAlgorithm) gaussianMutation(individual Individual, r *rand.Rand) Individual {
	mutated := Individual{
		Genes: make([]float64, len(individual.Genes)),
	}
	copy(mutated.Genes, individual.Genes)
	
	for i := range mutated.Genes {
		if r.Float64() < ga.MutationRate {
			gaussian := r.NormFloat64() * 0.1
			mutated.Genes[i] += gaussian
			
			if mutated.Genes[i] < ga.MinGene {
				mutated.Genes[i] = ga.MinGene
			} else if mutated.Genes[i] > ga.MaxGene {
				mutated.Genes[i] = ga.MaxGene
			}
		}
	}
	
	return mutated
}

func (ga *GeneticAlgorithm) performMigration(islands [][]Individual, islandID int, migrationRate float64) {
	numIslands := len(islands)
	if numIslands < 2 {
		return
	}
	
	migrantCount := int(float64(ga.PopulationSize) * migrationRate)
	if migrantCount == 0 {
		migrantCount = 1
	}
	
	targetIsland := (islandID + 1) % numIslands
	
	sort.Slice(islands[islandID], func(i, j int) bool {
		return islands[islandID][i].Fitness > islands[islandID][j].Fitness
	})
	
	for i := 0; i < migrantCount; i++ {
		replaceIndex := ga.PopulationSize - 1 - i
		islands[targetIsland][replaceIndex] = islands[islandID][i]
	}
}

func (ga *GeneticAlgorithm) updateBest(population []Individual) {
	ga.mutex.Lock()
	defer ga.mutex.Unlock()
	
	sort.Slice(population, func(i, j int) bool {
		return population[i].Fitness > population[j].Fitness
	})
	
	if population[0].Fitness > ga.bestIndividual.Fitness {
		ga.bestIndividual = Individual{
			Genes:   make([]float64, len(population[0].Genes)),
			Fitness: population[0].Fitness,
		}
		copy(ga.bestIndividual.Genes, population[0].Genes)
	}
}

func (ga *GeneticAlgorithm) recordHistory(population []Individual) {
	ga.mutex.Lock()
	defer ga.mutex.Unlock()
	
	sort.Slice(population, func(i, j int) bool {
		return population[i].Fitness > population[j].Fitness
	})
	
	var totalFitness float64
	for _, individual := range population {
		totalFitness += individual.Fitness
	}
	
	ga.history = append(ga.history, Population{
		Individuals: append([]Individual{}, population...),
		Generation:  len(ga.history),
		BestFitness: population[0].Fitness,
		AvgFitness:  totalFitness / float64(len(population)),
	})
}

func (ga *GeneticAlgorithm) GetBestIndividual() Individual {
	ga.mutex.RLock()
	defer ga.mutex.RUnlock()
	
	return Individual{
		Genes:   append([]float64{}, ga.bestIndividual.Genes...),
		Fitness: ga.bestIndividual.Fitness,
	}
}

func (ga *GeneticAlgorithm) GetHistory() []Population {
	ga.mutex.RLock()
	defer ga.mutex.RUnlock()
	
	return append([]Population{}, ga.history...)
}

func (ga *GeneticAlgorithm) GetStats() map[string]interface{} {
	ga.mutex.RLock()
	defer ga.mutex.RUnlock()
	
	stats := map[string]interface{}{
		"generations":       len(ga.history),
		"best_fitness":      ga.bestIndividual.Fitness,
		"population_size":   ga.PopulationSize,
		"chromosome_length": ga.ChromosomeLength,
		"mutation_rate":     ga.MutationRate,
		"crossover_rate":    ga.CrossoverRate,
		"num_workers":       ga.NumWorkers,
	}
	
	if len(ga.history) > 0 {
		stats["avg_fitness"] = ga.history[len(ga.history)-1].AvgFitness
	}
	
	return stats
}

func SphereFunction(genes []float64) float64 {
	sum := 0.0
	for _, gene := range genes {
		sum += gene * gene
	}
	return -sum
}

func RastriginFunction(genes []float64) float64 {
	n := float64(len(genes))
	sum := 0.0
	for _, gene := range genes {
		sum += gene*gene - 10*math.Cos(2*math.Pi*gene)
	}
	return -(10*n + sum)
}

func RosenbrockFunction(genes []float64) float64 {
	sum := 0.0
	for i := 0; i < len(genes)-1; i++ {
		a := genes[i]
		b := genes[i+1]
		sum += 100*(b-a*a)*(b-a*a) + (1-a)*(1-a)
	}
	return -sum
}

func AckleyFunction(genes []float64) float64 {
	n := float64(len(genes))
	sum1 := 0.0
	sum2 := 0.0
	
	for _, gene := range genes {
		sum1 += gene * gene
		sum2 += math.Cos(2 * math.Pi * gene)
	}
	
	result := -20*math.Exp(-0.2*math.Sqrt(sum1/n)) - math.Exp(sum2/n) + 20 + math.E
	return -result
}

func SchwefelFunction(genes []float64) float64 {
	sum := 0.0
	for _, gene := range genes {
		sum += gene * math.Sin(math.Sqrt(math.Abs(gene)))
	}
	return sum
}

func GriewankFunction(genes []float64) float64 {
	sum := 0.0
	product := 1.0
	
	for i, gene := range genes {
		sum += gene * gene
		product *= math.Cos(gene / math.Sqrt(float64(i+1)))
	}
	
	return -(sum/4000 - product + 1)
}