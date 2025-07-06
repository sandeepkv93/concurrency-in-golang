package paralleltsp

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// TSPAlgorithm defines different TSP solving algorithms
type TSPAlgorithm int

const (
	NearestNeighbor TSPAlgorithm = iota
	TwoOpt
	GeneticAlgorithm
	SimulatedAnnealing
	ChristofidesAlgorithm
	AntColonyOptimization
	BranchAndBound
	DynamicProgramming
	LinKernighan
	HybridApproach
)

// DistanceMetric defines different distance calculation methods
type DistanceMetric int

const (
	Euclidean DistanceMetric = iota
	Manhattan
	Chebyshev
	Haversine
	Custom
)

// ParallelStrategy defines parallel processing approaches
type ParallelStrategy int

const (
	IndependentRuns ParallelStrategy = iota
	PopulationBased
	IslandModel
	HybridParallel
	WorkerPool
	DivideAndConquer
)

// City represents a city with coordinates
type City struct {
	ID   int     `json:"id"`
	Name string  `json:"name"`
	X    float64 `json:"x"`
	Y    float64 `json:"y"`
	Lat  float64 `json:"latitude,omitempty"`
	Lon  float64 `json:"longitude,omitempty"`
}

// Tour represents a solution to the TSP
type Tour struct {
	Cities      []int     `json:"cities"`
	Distance    float64   `json:"distance"`
	Algorithm   string    `json:"algorithm"`
	Timestamp   time.Time `json:"timestamp"`
	Iteration   int       `json:"iteration"`
	ElapsedTime time.Duration `json:"elapsed_time"`
	IsValid     bool      `json:"is_valid"`
}

// TSPConfig holds configuration for the TSP solver
type TSPConfig struct {
	// Problem Definition
	Cities         []City        `json:"cities"`
	DistanceMatrix [][]float64   `json:"distance_matrix,omitempty"`
	DistanceMetric DistanceMetric `json:"distance_metric"`
	
	// Algorithm Configuration
	Algorithm        TSPAlgorithm    `json:"algorithm"`
	ParallelStrategy ParallelStrategy `json:"parallel_strategy"`
	MaxIterations    int             `json:"max_iterations"`
	TimeLimit        time.Duration   `json:"time_limit"`
	
	// Parallel Processing
	NumWorkers     int `json:"num_workers"`
	NumPopulations int `json:"num_populations"`
	PopulationSize int `json:"population_size"`
	IslandCount    int `json:"island_count"`
	
	// Algorithm-specific Parameters
	MutationRate      float64 `json:"mutation_rate"`
	CrossoverRate     float64 `json:"crossover_rate"`
	ElitismRate       float64 `json:"elitism_rate"`
	InitialTemp       float64 `json:"initial_temperature"`
	CoolingRate       float64 `json:"cooling_rate"`
	AntCount          int     `json:"ant_count"`
	PheromoneEvap     float64 `json:"pheromone_evaporation"`
	Alpha             float64 `json:"alpha"`
	Beta              float64 `json:"beta"`
	
	// Optimization Settings
	EnableTwoOpt      bool    `json:"enable_two_opt"`
	EnableThreeOpt    bool    `json:"enable_three_opt"`
	EnableOrOpt       bool    `json:"enable_or_opt"`
	ImprovementThresh float64 `json:"improvement_threshold"`
	StagnationLimit   int     `json:"stagnation_limit"`
	
	// Performance Settings
	EnableCaching     bool `json:"enable_caching"`
	EnableStatistics  bool `json:"enable_statistics"`
	EnableLogging     bool `json:"enable_logging"`
	RandomSeed        int64 `json:"random_seed"`
	
	// Custom Distance Function
	CustomDistanceFunc func(City, City) float64 `json:"-"`
}

// Population represents a population of tours for genetic algorithms
type Population struct {
	Tours     []*Tour  `json:"tours"`
	BestTour  *Tour    `json:"best_tour"`
	Generation int     `json:"generation"`
	Diversity  float64 `json:"diversity"`
}

// Island represents an island in island model parallel GA
type Island struct {
	ID         int         `json:"id"`
	Population *Population `json:"population"`
	BestTour   *Tour       `json:"best_tour"`
	Migrants   []*Tour     `json:"migrants"`
	mutex      sync.RWMutex
}

// AntColony represents ant colony optimization state
type AntColony struct {
	Pheromones  [][]float64 `json:"pheromones"`
	Ants        []*Ant      `json:"ants"`
	BestTour    *Tour       `json:"best_tour"`
	Iteration   int         `json:"iteration"`
	mutex       sync.RWMutex
}

// Ant represents an ant in ACO
type Ant struct {
	ID          int     `json:"id"`
	CurrentCity int     `json:"current_city"`
	Visited     []bool  `json:"visited"`
	Tour        []int   `json:"tour"`
	Distance    float64 `json:"distance"`
}

// TSPStatistics holds performance statistics
type TSPStatistics struct {
	TotalIterations      int64         `json:"total_iterations"`
	BestDistance         float64       `json:"best_distance"`
	AverageDistance      float64       `json:"average_distance"`
	WorstDistance        float64       `json:"worst_distance"`
	ImprovementCount     int64         `json:"improvement_count"`
	StagnationCount      int64         `json:"stagnation_count"`
	TotalExecutionTime   time.Duration `json:"total_execution_time"`
	ConvergenceHistory   []float64     `json:"convergence_history"`
	AlgorithmPerformance map[string]TSPAlgStats `json:"algorithm_performance"`
	PopulationDiversity  []float64     `json:"population_diversity"`
	DistanceCalculations int64         `json:"distance_calculations"`
	CacheHits           int64         `json:"cache_hits"`
	CacheMisses         int64         `json:"cache_misses"`
	ParallelEfficiency  float64       `json:"parallel_efficiency"`
	mutex               sync.RWMutex
}

// TSPAlgStats holds algorithm-specific statistics
type TSPAlgStats struct {
	Executions       int64         `json:"executions"`
	BestDistance     float64       `json:"best_distance"`
	AverageDistance  float64       `json:"average_distance"`
	AverageTime      time.Duration `json:"average_time"`
	SuccessRate      float64       `json:"success_rate"`
}

// ParallelTSP represents the parallel TSP solver
type ParallelTSP struct {
	config       TSPConfig
	cities       []City
	distances    [][]float64
	bestTour     *Tour
	populations  []*Population
	islands      []*Island
	antColony    *AntColony
	statistics   *TSPStatistics
	
	// Control and synchronization
	ctx          context.Context
	cancel       context.CancelFunc
	running      int32
	workers      sync.WaitGroup
	resultChan   chan *Tour
	
	// Caching
	distanceCache sync.Map
	tourCache     sync.Map
	
	// Thread safety
	mutex        sync.RWMutex
}

// DefaultTSPConfig returns a default configuration
func DefaultTSPConfig() TSPConfig {
	return TSPConfig{
		Algorithm:         GeneticAlgorithm,
		ParallelStrategy:  PopulationBased,
		MaxIterations:     10000,
		TimeLimit:         5 * time.Minute,
		NumWorkers:        runtime.NumCPU(),
		NumPopulations:    4,
		PopulationSize:    100,
		IslandCount:       4,
		MutationRate:      0.1,
		CrossoverRate:     0.8,
		ElitismRate:       0.2,
		InitialTemp:       1000.0,
		CoolingRate:       0.95,
		AntCount:          20,
		PheromoneEvap:     0.5,
		Alpha:             1.0,
		Beta:              2.0,
		EnableTwoOpt:      true,
		EnableThreeOpt:    false,
		EnableOrOpt:       true,
		ImprovementThresh: 0.001,
		StagnationLimit:   1000,
		EnableCaching:     true,
		EnableStatistics:  true,
		EnableLogging:     false,
		DistanceMetric:    Euclidean,
		RandomSeed:        time.Now().UnixNano(),
	}
}

// NewParallelTSP creates a new parallel TSP solver
func NewParallelTSP(config TSPConfig) (*ParallelTSP, error) {
	if len(config.Cities) < 3 {
		return nil, errors.New("need at least 3 cities for TSP")
	}
	
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	
	if config.PopulationSize <= 0 {
		config.PopulationSize = 100
	}
	
	if config.MaxIterations <= 0 {
		config.MaxIterations = 10000
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	tsp := &ParallelTSP{
		config:     config,
		cities:     config.Cities,
		ctx:        ctx,
		cancel:     cancel,
		resultChan: make(chan *Tour, config.NumWorkers*2),
		statistics: &TSPStatistics{
			AlgorithmPerformance: make(map[string]TSPAlgStats),
		},
	}
	
	// Initialize random seed
	if config.RandomSeed != 0 {
		rand.Seed(config.RandomSeed)
	}
	
	// Calculate distance matrix
	err := tsp.calculateDistanceMatrix()
	if err != nil {
		return nil, fmt.Errorf("failed to calculate distance matrix: %v", err)
	}
	
	// Initialize algorithm-specific data structures
	err = tsp.initializeAlgorithmData()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize algorithm data: %v", err)
	}
	
	return tsp, nil
}

// calculateDistanceMatrix computes the distance matrix between all cities
func (tsp *ParallelTSP) calculateDistanceMatrix() error {
	n := len(tsp.cities)
	tsp.distances = make([][]float64, n)
	for i := range tsp.distances {
		tsp.distances[i] = make([]float64, n)
	}
	
	// Use custom distance matrix if provided
	if tsp.config.DistanceMatrix != nil {
		if len(tsp.config.DistanceMatrix) != n || len(tsp.config.DistanceMatrix[0]) != n {
			return errors.New("distance matrix dimensions don't match number of cities")
		}
		tsp.distances = tsp.config.DistanceMatrix
		return nil
	}
	
	// Calculate distances in parallel
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, tsp.config.NumWorkers)
	
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			wg.Add(1)
			go func(i, j int) {
				defer wg.Done()
				semaphore <- struct{}{}
				defer func() { <-semaphore }()
				
				if i == j {
					tsp.distances[i][j] = 0
				} else {
					dist := tsp.calculateDistance(tsp.cities[i], tsp.cities[j])
					tsp.distances[i][j] = dist
					tsp.distances[j][i] = dist
				}
				
				atomic.AddInt64(&tsp.statistics.DistanceCalculations, 1)
			}(i, j)
		}
	}
	
	wg.Wait()
	return nil
}

// calculateDistance computes distance between two cities
func (tsp *ParallelTSP) calculateDistance(city1, city2 City) float64 {
	// Check cache first
	if tsp.config.EnableCaching {
		key := fmt.Sprintf("%d-%d", city1.ID, city2.ID)
		if dist, ok := tsp.distanceCache.Load(key); ok {
			atomic.AddInt64(&tsp.statistics.CacheHits, 1)
			return dist.(float64)
		}
		atomic.AddInt64(&tsp.statistics.CacheMisses, 1)
	}
	
	var dist float64
	
	switch tsp.config.DistanceMetric {
	case Euclidean:
		dx := city1.X - city2.X
		dy := city1.Y - city2.Y
		dist = math.Sqrt(dx*dx + dy*dy)
		
	case Manhattan:
		dist = math.Abs(city1.X-city2.X) + math.Abs(city1.Y-city2.Y)
		
	case Chebyshev:
		dist = math.Max(math.Abs(city1.X-city2.X), math.Abs(city1.Y-city2.Y))
		
	case Haversine:
		if city1.Lat == 0 || city1.Lon == 0 || city2.Lat == 0 || city2.Lon == 0 {
			// Fall back to Euclidean if lat/lon not provided
			dx := city1.X - city2.X
			dy := city1.Y - city2.Y
			dist = math.Sqrt(dx*dx + dy*dy)
		} else {
			dist = tsp.haversineDistance(city1.Lat, city1.Lon, city2.Lat, city2.Lon)
		}
		
	case Custom:
		if tsp.config.CustomDistanceFunc != nil {
			dist = tsp.config.CustomDistanceFunc(city1, city2)
		} else {
			// Fall back to Euclidean
			dx := city1.X - city2.X
			dy := city1.Y - city2.Y
			dist = math.Sqrt(dx*dx + dy*dy)
		}
		
	default:
		dx := city1.X - city2.X
		dy := city1.Y - city2.Y
		dist = math.Sqrt(dx*dx + dy*dy)
	}
	
	// Cache the result
	if tsp.config.EnableCaching {
		key := fmt.Sprintf("%d-%d", city1.ID, city2.ID)
		tsp.distanceCache.Store(key, dist)
	}
	
	return dist
}

// haversineDistance calculates the great circle distance between two points
func (tsp *ParallelTSP) haversineDistance(lat1, lon1, lat2, lon2 float64) float64 {
	const earthRadius = 6371.0 // Earth radius in kilometers
	
	// Convert degrees to radians
	lat1Rad := lat1 * math.Pi / 180
	lon1Rad := lon1 * math.Pi / 180
	lat2Rad := lat2 * math.Pi / 180
	lon2Rad := lon2 * math.Pi / 180
	
	deltaLat := lat2Rad - lat1Rad
	deltaLon := lon2Rad - lon1Rad
	
	a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
		math.Cos(lat1Rad)*math.Cos(lat2Rad)*
			math.Sin(deltaLon/2)*math.Sin(deltaLon/2)
	
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	
	return earthRadius * c
}

// initializeAlgorithmData initializes data structures for specific algorithms
func (tsp *ParallelTSP) initializeAlgorithmData() error {
	switch tsp.config.Algorithm {
	case GeneticAlgorithm:
		return tsp.initializePopulations()
	case AntColonyOptimization:
		return tsp.initializeAntColony()
	default:
		// Most algorithms don't need special initialization
		return nil
	}
}

// initializePopulations creates initial populations for genetic algorithm
func (tsp *ParallelTSP) initializePopulations() error {
	numPops := tsp.config.NumPopulations
	if tsp.config.ParallelStrategy == IslandModel {
		numPops = tsp.config.IslandCount
	}
	
	tsp.populations = make([]*Population, numPops)
	
	for i := 0; i < numPops; i++ {
		pop, err := tsp.createRandomPopulation(tsp.config.PopulationSize)
		if err != nil {
			return fmt.Errorf("failed to create population %d: %v", i, err)
		}
		tsp.populations[i] = pop
	}
	
	// Initialize islands if using island model
	if tsp.config.ParallelStrategy == IslandModel {
		tsp.islands = make([]*Island, tsp.config.IslandCount)
		for i := 0; i < tsp.config.IslandCount; i++ {
			tsp.islands[i] = &Island{
				ID:         i,
				Population: tsp.populations[i],
				Migrants:   make([]*Tour, 0),
			}
		}
	}
	
	return nil
}

// createRandomPopulation creates a random population of tours
func (tsp *ParallelTSP) createRandomPopulation(size int) (*Population, error) {
	population := &Population{
		Tours:      make([]*Tour, size),
		Generation: 0,
	}
	
	n := len(tsp.cities)
	
	for i := 0; i < size; i++ {
		// Create random tour
		tour := make([]int, n)
		for j := 0; j < n; j++ {
			tour[j] = j
		}
		
		// Shuffle the tour (except first city)
		for j := 1; j < n; j++ {
			k := rand.Intn(n-1) + 1
			tour[j], tour[k] = tour[k], tour[j]
		}
		
		distance := tsp.calculateTourDistance(tour)
		population.Tours[i] = &Tour{
			Cities:    tour,
			Distance:  distance,
			Algorithm: "RandomInit",
			Timestamp: time.Now(),
			IsValid:   true,
		}
	}
	
	// Find best tour in population
	population.BestTour = tsp.findBestTour(population.Tours)
	
	return population, nil
}

// initializeAntColony initializes ant colony optimization data
func (tsp *ParallelTSP) initializeAntColony() error {
	n := len(tsp.cities)
	
	// Initialize pheromone matrix
	pheromones := make([][]float64, n)
	for i := range pheromones {
		pheromones[i] = make([]float64, n)
		for j := range pheromones[i] {
			if i != j {
				pheromones[i][j] = 1.0 // Initial pheromone level
			}
		}
	}
	
	// Create ants
	ants := make([]*Ant, tsp.config.AntCount)
	for i := 0; i < tsp.config.AntCount; i++ {
		ants[i] = &Ant{
			ID:          i,
			CurrentCity: rand.Intn(n),
			Visited:     make([]bool, n),
			Tour:        make([]int, 0, n),
		}
	}
	
	tsp.antColony = &AntColony{
		Pheromones: pheromones,
		Ants:       ants,
		Iteration:  0,
	}
	
	return nil
}

// Solve executes the TSP solver using the configured algorithm and parallel strategy
func (tsp *ParallelTSP) Solve() (*Tour, error) {
	if atomic.LoadInt32(&tsp.running) == 1 {
		return nil, errors.New("solver is already running")
	}
	
	atomic.StoreInt32(&tsp.running, 1)
	defer atomic.StoreInt32(&tsp.running, 0)
	
	startTime := time.Now()
	defer func() {
		tsp.statistics.TotalExecutionTime = time.Since(startTime)
	}()
	
	// Set timeout if specified
	if tsp.config.TimeLimit > 0 {
		var cancel context.CancelFunc
		tsp.ctx, cancel = context.WithTimeout(tsp.ctx, tsp.config.TimeLimit)
		defer cancel()
	}
	
	// Choose solving strategy based on parallel approach
	switch tsp.config.ParallelStrategy {
	case IndependentRuns:
		return tsp.solveIndependentRuns()
	case PopulationBased:
		return tsp.solvePopulationBased()
	case IslandModel:
		return tsp.solveIslandModel()
	case HybridParallel:
		return tsp.solveHybridParallel()
	case WorkerPool:
		return tsp.solveWorkerPool()
	case DivideAndConquer:
		return tsp.solveDivideAndConquer()
	default:
		return tsp.solveSequential()
	}
}

// solveIndependentRuns runs multiple independent algorithm instances
func (tsp *ParallelTSP) solveIndependentRuns() (*Tour, error) {
	numRuns := tsp.config.NumWorkers
	results := make(chan *Tour, numRuns)
	
	// Start independent runs
	for i := 0; i < numRuns; i++ {
		tsp.workers.Add(1)
		go func(runID int) {
			defer tsp.workers.Done()
			
			// Create a copy of the solver for this run
			solver := tsp.copyForRun(runID)
			
			// Run the algorithm
			tour, err := solver.runSingleAlgorithm()
			if err == nil && tour != nil {
				tour.Algorithm = fmt.Sprintf("%s-Run%d", tsp.algorithmName(), runID)
				results <- tour
			}
		}(i)
	}
	
	// Wait for all runs to complete
	go func() {
		tsp.workers.Wait()
		close(results)
	}()
	
	// Collect results and find best
	var bestTour *Tour
	resultCount := 0
	
	for tour := range results {
		resultCount++
		if bestTour == nil || tour.Distance < bestTour.Distance {
			bestTour = tour
			tsp.updateBestTour(bestTour)
		}
		
		// Update statistics
		atomic.AddInt64(&tsp.statistics.TotalIterations, int64(tour.Iteration))
	}
	
	if bestTour == nil {
		return nil, errors.New("no valid solutions found")
	}
	
	// Calculate parallel efficiency
	if resultCount > 0 {
		tsp.statistics.ParallelEfficiency = float64(resultCount) / float64(numRuns)
	}
	
	return bestTour, nil
}

// solvePopulationBased runs genetic algorithm with multiple populations
func (tsp *ParallelTSP) solvePopulationBased() (*Tour, error) {
	if tsp.config.Algorithm != GeneticAlgorithm {
		return nil, errors.New("population-based strategy requires genetic algorithm")
	}
	
	numPops := len(tsp.populations)
	results := make(chan *Tour, numPops)
	
	// Evolve each population in parallel
	for i := 0; i < numPops; i++ {
		tsp.workers.Add(1)
		go func(popIndex int) {
			defer tsp.workers.Done()
			
			population := tsp.populations[popIndex]
			
			for gen := 0; gen < tsp.config.MaxIterations; gen++ {
				select {
				case <-tsp.ctx.Done():
					return
				default:
				}
				
				// Evolve population
				newPop, err := tsp.evolvePopulation(population)
				if err != nil {
					continue
				}
				
				population = newPop
				population.Generation = gen
				
				// Update best tour
				if population.BestTour != nil {
					tsp.updateBestTour(population.BestTour)
				}
				
				// Check for convergence
				if tsp.checkConvergence(population) {
					break
				}
			}
			
			if population.BestTour != nil {
				results <- population.BestTour
			}
		}(i)
	}
	
	// Wait for all populations
	go func() {
		tsp.workers.Wait()
		close(results)
	}()
	
	// Find best result
	var bestTour *Tour
	for tour := range results {
		if bestTour == nil || tour.Distance < bestTour.Distance {
			bestTour = tour
		}
	}
	
	return bestTour, nil
}

// solveIslandModel implements island model genetic algorithm
func (tsp *ParallelTSP) solveIslandModel() (*Tour, error) {
	if tsp.config.Algorithm != GeneticAlgorithm {
		return nil, errors.New("island model requires genetic algorithm")
	}
	
	migrationInterval := tsp.config.MaxIterations / 10 // Migrate every 10% of iterations
	
	// Start evolution on each island
	for i := 0; i < len(tsp.islands); i++ {
		tsp.workers.Add(1)
		go func(islandID int) {
			defer tsp.workers.Done()
			
			island := tsp.islands[islandID]
			
			for gen := 0; gen < tsp.config.MaxIterations; gen++ {
				select {
				case <-tsp.ctx.Done():
					return
				default:
				}
				
				// Evolve island population
				newPop, err := tsp.evolvePopulation(island.Population)
				if err != nil {
					continue
				}
				
				island.mutex.Lock()
				island.Population = newPop
				island.Population.Generation = gen
				
				// Update island best
				if newPop.BestTour != nil {
					if island.BestTour == nil || newPop.BestTour.Distance < island.BestTour.Distance {
						island.BestTour = newPop.BestTour
						tsp.updateBestTour(island.BestTour)
					}
				}
				island.mutex.Unlock()
				
				// Migration
				if gen%migrationInterval == 0 && gen > 0 {
					tsp.performMigration(islandID)
				}
			}
		}(i)
	}
	
	tsp.workers.Wait()
	
	// Find best tour across all islands
	var bestTour *Tour
	for _, island := range tsp.islands {
		island.mutex.RLock()
		if island.BestTour != nil {
			if bestTour == nil || island.BestTour.Distance < bestTour.Distance {
				bestTour = island.BestTour
			}
		}
		island.mutex.RUnlock()
	}
	
	return bestTour, nil
}

// solveHybridParallel combines multiple algorithms and strategies
func (tsp *ParallelTSP) solveHybridParallel() (*Tour, error) {
	algorithms := []TSPAlgorithm{
		NearestNeighbor,
		TwoOpt,
		GeneticAlgorithm,
		SimulatedAnnealing,
	}
	
	results := make(chan *Tour, len(algorithms))
	
	// Run different algorithms in parallel
	for _, alg := range algorithms {
		tsp.workers.Add(1)
		go func(algorithm TSPAlgorithm) {
			defer tsp.workers.Done()
			
			// Create solver copy with different algorithm
			config := tsp.config
			config.Algorithm = algorithm
			
			solver, err := NewParallelTSP(config)
			if err != nil {
				return
			}
			
			tour, err := solver.runSingleAlgorithm()
			if err == nil && tour != nil {
				results <- tour
			}
		}(alg)
	}
	
	go func() {
		tsp.workers.Wait()
		close(results)
	}()
	
	// Collect and improve results
	var bestTour *Tour
	allTours := make([]*Tour, 0, len(algorithms))
	
	for tour := range results {
		allTours = append(allTours, tour)
		if bestTour == nil || tour.Distance < bestTour.Distance {
			bestTour = tour
		}
	}
	
	// Apply local search to best tours
	if len(allTours) > 0 {
		improvedTour := tsp.applyLocalSearch(bestTour)
		if improvedTour != nil && improvedTour.Distance < bestTour.Distance {
			bestTour = improvedTour
		}
	}
	
	return bestTour, nil
}

// solveWorkerPool uses a worker pool to process tour improvements
func (tsp *ParallelTSP) solveWorkerPool() (*Tour, error) {
	// Start with a greedy initial tour
	initialTour, err := tsp.nearestNeighborTour()
	if err != nil {
		return nil, err
	}
	
	currentBest := initialTour
	tsp.updateBestTour(currentBest)
	
	// Create job queue for improvements
	jobQueue := make(chan *Tour, tsp.config.NumWorkers*2)
	resultQueue := make(chan *Tour, tsp.config.NumWorkers*2)
	
	// Start workers
	for i := 0; i < tsp.config.NumWorkers; i++ {
		tsp.workers.Add(1)
		go func() {
			defer tsp.workers.Done()
			
			for tour := range jobQueue {
				// Apply various improvements
				improved := tsp.applyLocalSearch(tour)
				if improved != nil {
					resultQueue <- improved
				}
			}
		}()
	}
	
	// Generate variations of current best
	go func() {
		defer close(jobQueue)
		
		for iter := 0; iter < tsp.config.MaxIterations; iter++ {
			select {
			case <-tsp.ctx.Done():
				return
			default:
			}
			
			// Generate variations
			variations := tsp.generateTourVariations(currentBest, 10)
			for _, variation := range variations {
				select {
				case jobQueue <- variation:
				case <-tsp.ctx.Done():
					return
				}
			}
		}
	}()
	
	// Collect improvements
	go func() {
		tsp.workers.Wait()
		close(resultQueue)
	}()
	
	for improved := range resultQueue {
		if improved.Distance < currentBest.Distance {
			currentBest = improved
			tsp.updateBestTour(currentBest)
		}
	}
	
	return currentBest, nil
}

// solveDivideAndConquer applies divide and conquer approach
func (tsp *ParallelTSP) solveDivideAndConquer() (*Tour, error) {
	n := len(tsp.cities)
	
	// For small problems, solve directly
	if n <= 10 {
		return tsp.solveSequential()
	}
	
	// Divide cities into clusters
	numClusters := min(tsp.config.NumWorkers, n/3)
	clusters := tsp.divideCitiesIntoClusters(numClusters)
	
	// Solve each cluster
	clusterTours := make([]*Tour, numClusters)
	var wg sync.WaitGroup
	
	for i, cluster := range clusters {
		wg.Add(1)
		go func(clusterID int, cities []int) {
			defer wg.Done()
			
			// Create sub-problem
			subConfig := tsp.config
			subConfig.Cities = make([]City, len(cities))
			for j, cityID := range cities {
				subConfig.Cities[j] = tsp.cities[cityID]
			}
			
			subSolver, err := NewParallelTSP(subConfig)
			if err != nil {
				return
			}
			
			tour, err := subSolver.solveSequential()
			if err == nil {
				// Map back to original city IDs
				mappedTour := make([]int, len(tour.Cities))
				for j, subCityID := range tour.Cities {
					mappedTour[j] = cities[subCityID]
				}
				
				clusterTours[clusterID] = &Tour{
					Cities:    mappedTour,
					Distance:  tsp.calculateTourDistance(mappedTour),
					Algorithm: "DivideConquer",
					Timestamp: time.Now(),
					IsValid:   true,
				}
			}
		}(i, cluster)
	}
	
	wg.Wait()
	
	// Combine cluster tours
	return tsp.combineClusters(clusterTours)
}

// solveSequential runs the configured algorithm sequentially
func (tsp *ParallelTSP) solveSequential() (*Tour, error) {
	return tsp.runSingleAlgorithm()
}

// Helper methods for TSP algorithms

// runSingleAlgorithm executes a single algorithm
func (tsp *ParallelTSP) runSingleAlgorithm() (*Tour, error) {
	switch tsp.config.Algorithm {
	case NearestNeighbor:
		return tsp.nearestNeighborTour()
	case TwoOpt:
		return tsp.twoOptTour()
	case GeneticAlgorithm:
		return tsp.geneticAlgorithmTour()
	case SimulatedAnnealing:
		return tsp.simulatedAnnealingTour()
	case ChristofidesAlgorithm:
		return tsp.christofidesTour()
	case AntColonyOptimization:
		return tsp.antColonyTour()
	case BranchAndBound:
		return tsp.branchAndBoundTour()
	case DynamicProgramming:
		return tsp.dynamicProgrammingTour()
	case LinKernighan:
		return tsp.linKernighanTour()
	case HybridApproach:
		return tsp.hybridTour()
	default:
		return tsp.nearestNeighborTour()
	}
}

// nearestNeighborTour implements the nearest neighbor algorithm
func (tsp *ParallelTSP) nearestNeighborTour() (*Tour, error) {
	n := len(tsp.cities)
	if n == 0 {
		return nil, errors.New("no cities to solve")
	}
	
	visited := make([]bool, n)
	tour := make([]int, 0, n)
	current := 0
	tour = append(tour, current)
	visited[current] = true
	
	for len(tour) < n {
		nearest := -1
		minDist := math.Inf(1)
		
		for next := 0; next < n; next++ {
			if !visited[next] && tsp.distances[current][next] < minDist {
				minDist = tsp.distances[current][next]
				nearest = next
			}
		}
		
		if nearest == -1 {
			break
		}
		
		tour = append(tour, nearest)
		visited[nearest] = true
		current = nearest
	}
	
	distance := tsp.calculateTourDistance(tour)
	
	return &Tour{
		Cities:    tour,
		Distance:  distance,
		Algorithm: "NearestNeighbor",
		Timestamp: time.Now(),
		IsValid:   true,
	}, nil
}

// calculateTourDistance calculates the total distance of a tour
func (tsp *ParallelTSP) calculateTourDistance(tour []int) float64 {
	if len(tour) < 2 {
		return 0
	}
	
	distance := 0.0
	for i := 0; i < len(tour); i++ {
		from := tour[i]
		to := tour[(i+1)%len(tour)]
		distance += tsp.distances[from][to]
	}
	
	return distance
}

// Additional helper methods would be implemented here...

// Utility methods

// algorithmName returns the name of the current algorithm
func (tsp *ParallelTSP) algorithmName() string {
	switch tsp.config.Algorithm {
	case NearestNeighbor:
		return "NearestNeighbor"
	case TwoOpt:
		return "TwoOpt"
	case GeneticAlgorithm:
		return "GeneticAlgorithm"
	case SimulatedAnnealing:
		return "SimulatedAnnealing"
	case ChristofidesAlgorithm:
		return "Christofides"
	case AntColonyOptimization:
		return "AntColony"
	case BranchAndBound:
		return "BranchAndBound"
	case DynamicProgramming:
		return "DynamicProgramming"
	case LinKernighan:
		return "LinKernighan"
	case HybridApproach:
		return "Hybrid"
	default:
		return "Unknown"
	}
}

// updateBestTour updates the global best tour
func (tsp *ParallelTSP) updateBestTour(tour *Tour) {
	tsp.mutex.Lock()
	defer tsp.mutex.Unlock()
	
	if tsp.bestTour == nil || tour.Distance < tsp.bestTour.Distance {
		tsp.bestTour = tour
		
		// Update statistics
		if tsp.config.EnableStatistics {
			tsp.statistics.BestDistance = tour.Distance
			tsp.statistics.ConvergenceHistory = append(tsp.statistics.ConvergenceHistory, tour.Distance)
			atomic.AddInt64(&tsp.statistics.ImprovementCount, 1)
		}
	}
}

// GetBestTour returns the current best tour
func (tsp *ParallelTSP) GetBestTour() *Tour {
	tsp.mutex.RLock()
	defer tsp.mutex.RUnlock()
	return tsp.bestTour
}

// GetStatistics returns current statistics
func (tsp *ParallelTSP) GetStatistics() *TSPStatistics {
	tsp.statistics.mutex.RLock()
	defer tsp.statistics.mutex.RUnlock()
	
	// Return a copy to avoid race conditions
	stats := *tsp.statistics
	return &stats
}

// IsRunning returns whether the solver is currently running
func (tsp *ParallelTSP) IsRunning() bool {
	return atomic.LoadInt32(&tsp.running) == 1
}

// Stop stops the solver
func (tsp *ParallelTSP) Stop() {
	tsp.cancel()
}

// Placeholder implementations for the remaining algorithms
// These would be fully implemented in a complete version

func (tsp *ParallelTSP) twoOptTour() (*Tour, error) {
	// Start with nearest neighbor, then improve with 2-opt
	initial, err := tsp.nearestNeighborTour()
	if err != nil {
		return nil, err
	}
	
	return tsp.applyTwoOpt(initial), nil
}

func (tsp *ParallelTSP) applyTwoOpt(tour *Tour) *Tour {
	// Implement 2-opt improvement
	// This is a simplified version - full implementation would be more complex
	return tour
}

func (tsp *ParallelTSP) geneticAlgorithmTour() (*Tour, error) {
	if len(tsp.populations) == 0 {
		return nil, errors.New("no populations initialized")
	}
	
	population := tsp.populations[0]
	
	for gen := 0; gen < tsp.config.MaxIterations; gen++ {
		select {
		case <-tsp.ctx.Done():
			break
		default:
		}
		
		newPop, err := tsp.evolvePopulation(population)
		if err != nil {
			continue
		}
		
		population = newPop
		
		if tsp.checkConvergence(population) {
			break
		}
	}
	
	return population.BestTour, nil
}

func (tsp *ParallelTSP) simulatedAnnealingTour() (*Tour, error) {
	// Start with random tour
	initial, err := tsp.nearestNeighborTour()
	if err != nil {
		return nil, err
	}
	
	current := initial
	best := initial
	temp := tsp.config.InitialTemp
	
	for iter := 0; iter < tsp.config.MaxIterations && temp > 0.01; iter++ {
		// Generate neighbor
		neighbor := tsp.generateRandomNeighbor(current)
		
		// Accept or reject
		delta := neighbor.Distance - current.Distance
		if delta < 0 || rand.Float64() < math.Exp(-delta/temp) {
			current = neighbor
			if neighbor.Distance < best.Distance {
				best = neighbor
			}
		}
		
		// Cool down
		temp *= tsp.config.CoolingRate
	}
	
	return best, nil
}

func (tsp *ParallelTSP) christofidesTour() (*Tour, error) {
	// Placeholder for Christofides algorithm
	return tsp.nearestNeighborTour()
}

func (tsp *ParallelTSP) antColonyTour() (*Tour, error) {
	// Placeholder for Ant Colony Optimization
	return tsp.nearestNeighborTour()
}

func (tsp *ParallelTSP) branchAndBoundTour() (*Tour, error) {
	// Placeholder for Branch and Bound
	return tsp.nearestNeighborTour()
}

func (tsp *ParallelTSP) dynamicProgrammingTour() (*Tour, error) {
	// Placeholder for Dynamic Programming (Held-Karp)
	return tsp.nearestNeighborTour()
}

func (tsp *ParallelTSP) linKernighanTour() (*Tour, error) {
	// Placeholder for Lin-Kernighan heuristic
	return tsp.nearestNeighborTour()
}

func (tsp *ParallelTSP) hybridTour() (*Tour, error) {
	// Combine multiple algorithms
	return tsp.nearestNeighborTour()
}

// Additional placeholder methods for genetic algorithm operations
func (tsp *ParallelTSP) evolvePopulation(pop *Population) (*Population, error) {
	// Placeholder for population evolution
	return pop, nil
}

func (tsp *ParallelTSP) checkConvergence(pop *Population) bool {
	// Placeholder for convergence check
	return false
}

func (tsp *ParallelTSP) performMigration(islandID int) {
	// Placeholder for island migration
}

func (tsp *ParallelTSP) copyForRun(runID int) *ParallelTSP {
	// Placeholder for creating solver copy
	return tsp
}

func (tsp *ParallelTSP) applyLocalSearch(tour *Tour) *Tour {
	// Placeholder for local search improvement
	return tour
}

func (tsp *ParallelTSP) generateTourVariations(tour *Tour, count int) []*Tour {
	// Placeholder for generating tour variations
	return []*Tour{tour}
}

func (tsp *ParallelTSP) divideCitiesIntoClusters(numClusters int) [][]int {
	// Placeholder for city clustering
	n := len(tsp.cities)
	clusterSize := n / numClusters
	clusters := make([][]int, numClusters)
	
	for i := 0; i < numClusters; i++ {
		start := i * clusterSize
		end := start + clusterSize
		if i == numClusters-1 {
			end = n
		}
		
		cluster := make([]int, end-start)
		for j := start; j < end; j++ {
			cluster[j-start] = j
		}
		clusters[i] = cluster
	}
	
	return clusters
}

func (tsp *ParallelTSP) combineClusters(tours []*Tour) (*Tour, error) {
	// Placeholder for combining cluster tours
	if len(tours) == 0 {
		return nil, errors.New("no tours to combine")
	}
	return tours[0], nil
}

func (tsp *ParallelTSP) generateRandomNeighbor(tour *Tour) *Tour {
	// Simple 2-opt neighbor generation
	n := len(tour.Cities)
	if n < 4 {
		return tour
	}
	
	newCities := make([]int, n)
	copy(newCities, tour.Cities)
	
	// Pick two random positions and reverse the segment
	i := rand.Intn(n-1) + 1
	j := rand.Intn(n-1) + 1
	if i > j {
		i, j = j, i
	}
	
	// Reverse segment
	for k := 0; k < (j-i+1)/2; k++ {
		newCities[i+k], newCities[j-k] = newCities[j-k], newCities[i+k]
	}
	
	return &Tour{
		Cities:    newCities,
		Distance:  tsp.calculateTourDistance(newCities),
		Algorithm: "Neighbor",
		Timestamp: time.Now(),
		IsValid:   true,
	}
}

func (tsp *ParallelTSP) findBestTour(tours []*Tour) *Tour {
	if len(tours) == 0 {
		return nil
	}
	
	best := tours[0]
	for _, tour := range tours[1:] {
		if tour.Distance < best.Distance {
			best = tour
		}
	}
	
	return best
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}