package parallelantcolony

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// ACOAlgorithm defines different ACO algorithm variants
type ACOAlgorithm int

const (
	AntSystem ACOAlgorithm = iota
	AntColonySystem
	MaxMinAntSystem
	RankBasedAntSystem
	ElitistAntSystem
	HybridAntSystem
)

// OptimizationProblem defines the type of optimization problem
type OptimizationProblem int

const (
	TravelingSalesman OptimizationProblem = iota
	GraphColoring
	VehicleRouting
	JobShopScheduling
	QuadraticAssignment
	SetCovering
)

// PheromoneUpdateStrategy defines pheromone update methods
type PheromoneUpdateStrategy int

const (
	GlobalUpdate PheromoneUpdateStrategy = iota
	LocalUpdate
	ElitistUpdate
	RankBasedUpdate
	MaxMinUpdate
	HybridUpdate
)

// LocalSearchMethod defines local search optimization methods
type LocalSearchMethod int

const (
	NoLocalSearch LocalSearchMethod = iota
	TwoOpt
	ThreeOpt
	OrOpt
	LinKernighan
	SimulatedAnnealing
	HillClimbing
)

// ACOConfig contains configuration for the ACO algorithm
type ACOConfig struct {
	Algorithm               ACOAlgorithm
	Problem                 OptimizationProblem
	NumAnts                 int
	NumColonies             int
	MaxIterations           int
	Alpha                   float64 // Pheromone importance
	Beta                    float64 // Heuristic importance
	Rho                     float64 // Evaporation rate
	Q                       float64 // Pheromone deposit factor
	InitialPheromone        float64
	MinPheromone            float64
	MaxPheromone            float64
	ElitistWeight           float64
	LocalSearchProbability  float64
	LocalSearchMethod       LocalSearchMethod
	PheromoneUpdateStrategy PheromoneUpdateStrategy
	UseParallelAnts         bool
	UseParallelColonies     bool
	MaxStagnation           int
	ConvergenceThreshold    float64
	DiversityThreshold      float64
	EnableStatistics        bool
	SeedValue               int64
}

// Problem represents an optimization problem instance
type Problem struct {
	Name        string
	Size        int
	DistMatrix  [][]float64
	Coordinates []Coordinate
	Constraints map[string]interface{}
	BestKnown   float64
	metadata    map[string]interface{}
}

// Coordinate represents a 2D coordinate
type Coordinate struct {
	X, Y float64
}

// Solution represents a solution to the optimization problem
type Solution struct {
	Tour     []int
	Cost     float64
	Quality  float64
	Iteration int
	ColonyID int
	AntID    int
	Valid    bool
	metadata map[string]interface{}
}

// Ant represents an individual ant in the colony
type Ant struct {
	ID          int
	ColonyID    int
	CurrentCity int
	Tour        []int
	Visited     []bool
	TourCost    float64
	Probability []float64
	LocalPheromone [][]float64
	rng         *rand.Rand
}

// Colony represents a colony of ants
type Colony struct {
	ID                int
	Ants              []*Ant
	BestSolution      *Solution
	BestEverSolution  *Solution
	PheromoneMatrix   [][]float64
	HeuristicMatrix   [][]float64
	LocalSearcher     *LocalSearcher
	Statistics        *ColonyStatistics
	rng               *rand.Rand
	mutex             sync.RWMutex
}

// ACOOptimizer is the main optimizer instance
type ACOOptimizer struct {
	config              ACOConfig
	problem             *Problem
	colonies            []*Colony
	globalBestSolution  *Solution
	iterationBestSolution *Solution
	pheromoneMatrix     [][]float64
	heuristicMatrix     [][]float64
	statistics          *OptimizationStatistics
	localSearcher       *LocalSearcher
	diversityManager    *DiversityManager
	convergenceDetector *ConvergenceDetector
	workers             []*Worker
	taskQueue           chan Task
	resultQueue         chan Result
	ctx                 context.Context
	cancel              context.CancelFunc
	wg                  sync.WaitGroup
	running             bool
	mutex               sync.RWMutex
}

// Worker represents a parallel worker for ant processing
type Worker struct {
	ID          int
	optimizer   *ACOOptimizer
	taskQueue   chan Task
	resultQueue chan Result
	rng         *rand.Rand
	ctx         context.Context
}

// Task represents a work task for parallel processing
type Task struct {
	Type      TaskType
	ColonyID  int
	AntID     int
	Data      interface{}
	Iteration int
	TaskID    string
}

// TaskType defines different types of tasks
type TaskType int

const (
	ConstructSolutionTask TaskType = iota
	LocalSearchTask
	PheromoneUpdateTask
	EvaluationTask
)

// Result represents the result of a task
type Result struct {
	TaskID    string
	Type      TaskType
	Solution  *Solution
	Success   bool
	Error     error
	Duration  time.Duration
	Data      interface{}
}

// LocalSearcher handles local search optimization
type LocalSearcher struct {
	method    LocalSearchMethod
	problem   *Problem
	rng       *rand.Rand
	mutex     sync.Mutex
}

// DiversityManager manages population diversity
type DiversityManager struct {
	solutions        []*Solution
	diversityMetrics []float64
	threshold        float64
	mutex            sync.RWMutex
}

// ConvergenceDetector detects algorithm convergence
type ConvergenceDetector struct {
	bestCosts        []float64
	stagnationCount  int
	maxStagnation    int
	threshold        float64
	converged        bool
	mutex            sync.RWMutex
}

// OptimizationStatistics contains overall optimization statistics
type OptimizationStatistics struct {
	TotalIterations      int
	BestCost             float64
	AverageCost          float64
	WorstCost            float64
	ConvergenceIteration int
	TotalTime            time.Duration
	SolutionsEvaluated   int64
	PheromoneUpdates     int64
	LocalSearchCount     int64
	DiversityHistory     []float64
	CostHistory          []float64
	ColonyStatistics     []*ColonyStatistics
	WorkerUtilization    []float64
	mutex                sync.RWMutex
}

// ColonyStatistics contains statistics for individual colonies
type ColonyStatistics struct {
	ColonyID            int
	IterationCount      int
	BestCost            float64
	AverageCost         float64
	SolutionsFound      int64
	LocalSearchUsage    int64
	PheromoneLevel      float64
	DiversityMeasure    float64
	ConvergenceRate     float64
	mutex               sync.RWMutex
}

// NewACOOptimizer creates a new ACO optimizer
func NewACOOptimizer(config ACOConfig, problem *Problem) *ACOOptimizer {
	if config.NumAnts <= 0 {
		config.NumAnts = problem.Size
	}
	if config.NumColonies <= 0 {
		config.NumColonies = 1
	}
	if config.MaxIterations <= 0 {
		config.MaxIterations = 1000
	}
	if config.Alpha == 0 {
		config.Alpha = 1.0
	}
	if config.Beta == 0 {
		config.Beta = 2.0
	}
	if config.Rho == 0 {
		config.Rho = 0.1
	}
	if config.Q == 0 {
		config.Q = 100.0
	}
	if config.InitialPheromone == 0 {
		config.InitialPheromone = 1.0
	}
	if config.MaxStagnation == 0 {
		config.MaxStagnation = 100
	}

	ctx, cancel := context.WithCancel(context.Background())

	optimizer := &ACOOptimizer{
		config:              config,
		problem:             problem,
		ctx:                 ctx,
		cancel:              cancel,
		statistics:          NewOptimizationStatistics(config.NumColonies),
		diversityManager:    NewDiversityManager(config.DiversityThreshold),
		convergenceDetector: NewConvergenceDetector(config.MaxStagnation, config.ConvergenceThreshold),
	}

	optimizer.initializeMatrices()
	optimizer.initializeColonies()
	optimizer.initializeWorkers()

	if config.LocalSearchMethod != NoLocalSearch {
		optimizer.localSearcher = NewLocalSearcher(config.LocalSearchMethod, problem)
	}

	return optimizer
}

// NewProblem creates a new optimization problem
func NewProblem(name string, size int) *Problem {
	return &Problem{
		Name:        name,
		Size:        size,
		DistMatrix:  make([][]float64, size),
		Coordinates: make([]Coordinate, size),
		Constraints: make(map[string]interface{}),
		metadata:    make(map[string]interface{}),
	}
}

// NewOptimizationStatistics creates new optimization statistics
func NewOptimizationStatistics(numColonies int) *OptimizationStatistics {
	stats := &OptimizationStatistics{
		BestCost:         math.Inf(1),
		AverageCost:      0.0,
		WorstCost:        0.0,
		DiversityHistory: make([]float64, 0),
		CostHistory:      make([]float64, 0),
		ColonyStatistics: make([]*ColonyStatistics, numColonies),
		WorkerUtilization: make([]float64, 0),
	}

	for i := 0; i < numColonies; i++ {
		stats.ColonyStatistics[i] = &ColonyStatistics{
			ColonyID: i,
			BestCost: math.Inf(1),
		}
	}

	return stats
}

// NewDiversityManager creates a new diversity manager
func NewDiversityManager(threshold float64) *DiversityManager {
	return &DiversityManager{
		solutions:        make([]*Solution, 0),
		diversityMetrics: make([]float64, 0),
		threshold:        threshold,
	}
}

// NewConvergenceDetector creates a new convergence detector
func NewConvergenceDetector(maxStagnation int, threshold float64) *ConvergenceDetector {
	return &ConvergenceDetector{
		bestCosts:       make([]float64, 0),
		maxStagnation:   maxStagnation,
		threshold:       threshold,
		stagnationCount: 0,
		converged:       false,
	}
}

// NewLocalSearcher creates a new local searcher
func NewLocalSearcher(method LocalSearchMethod, problem *Problem) *LocalSearcher {
	return &LocalSearcher{
		method:  method,
		problem: problem,
		rng:     rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// LoadTSPFromCoordinates loads a TSP problem from coordinates
func (p *Problem) LoadTSPFromCoordinates(coords []Coordinate) {
	p.Size = len(coords)
	p.Coordinates = coords
	p.DistMatrix = make([][]float64, p.Size)

	for i := 0; i < p.Size; i++ {
		p.DistMatrix[i] = make([]float64, p.Size)
		for j := 0; j < p.Size; j++ {
			if i == j {
				p.DistMatrix[i][j] = 0
			} else {
				dx := coords[i].X - coords[j].X
				dy := coords[i].Y - coords[j].Y
				p.DistMatrix[i][j] = math.Sqrt(dx*dx + dy*dy)
			}
		}
	}
}

// LoadTSPFromMatrix loads a TSP problem from distance matrix
func (p *Problem) LoadTSPFromMatrix(matrix [][]float64) {
	p.Size = len(matrix)
	p.DistMatrix = make([][]float64, p.Size)
	for i := 0; i < p.Size; i++ {
		p.DistMatrix[i] = make([]float64, p.Size)
		copy(p.DistMatrix[i], matrix[i])
	}
}

// initializeMatrices initializes pheromone and heuristic matrices
func (opt *ACOOptimizer) initializeMatrices() {
	size := opt.problem.Size
	
	// Initialize pheromone matrix
	opt.pheromoneMatrix = make([][]float64, size)
	for i := 0; i < size; i++ {
		opt.pheromoneMatrix[i] = make([]float64, size)
		for j := 0; j < size; j++ {
			opt.pheromoneMatrix[i][j] = opt.config.InitialPheromone
		}
	}

	// Initialize heuristic matrix (inverse of distance)
	opt.heuristicMatrix = make([][]float64, size)
	for i := 0; i < size; i++ {
		opt.heuristicMatrix[i] = make([]float64, size)
		for j := 0; j < size; j++ {
			if i != j {
				opt.heuristicMatrix[i][j] = 1.0 / opt.problem.DistMatrix[i][j]
			} else {
				opt.heuristicMatrix[i][j] = 0
			}
		}
	}
}

// initializeColonies initializes all colonies
func (opt *ACOOptimizer) initializeColonies() {
	opt.colonies = make([]*Colony, opt.config.NumColonies)
	
	for i := 0; i < opt.config.NumColonies; i++ {
		colony := &Colony{
			ID:              i,
			Ants:            make([]*Ant, opt.config.NumAnts),
			PheromoneMatrix: opt.copyMatrix(opt.pheromoneMatrix),
			HeuristicMatrix: opt.copyMatrix(opt.heuristicMatrix),
			Statistics:      opt.statistics.ColonyStatistics[i],
			rng:             rand.New(rand.NewSource(opt.config.SeedValue + int64(i))),
		}

		// Initialize ants
		for j := 0; j < opt.config.NumAnts; j++ {
			colony.Ants[j] = &Ant{
				ID:          j,
				ColonyID:    i,
				Tour:        make([]int, 0, opt.problem.Size),
				Visited:     make([]bool, opt.problem.Size),
				Probability: make([]float64, opt.problem.Size),
				rng:         rand.New(rand.NewSource(opt.config.SeedValue + int64(i*opt.config.NumAnts+j))),
			}
		}

		if opt.config.LocalSearchMethod != NoLocalSearch {
			colony.LocalSearcher = NewLocalSearcher(opt.config.LocalSearchMethod, opt.problem)
		}

		opt.colonies[i] = colony
	}
}

// initializeWorkers initializes parallel workers if enabled
func (opt *ACOOptimizer) initializeWorkers() {
	if opt.config.UseParallelAnts {
		numWorkers := opt.config.NumColonies * opt.config.NumAnts
		if numWorkers > 100 { // Limit max workers
			numWorkers = 100
		}

		opt.workers = make([]*Worker, numWorkers)
		opt.taskQueue = make(chan Task, numWorkers*2)
		opt.resultQueue = make(chan Result, numWorkers*2)

		for i := 0; i < numWorkers; i++ {
			opt.workers[i] = &Worker{
				ID:          i,
				optimizer:   opt,
				taskQueue:   opt.taskQueue,
				resultQueue: opt.resultQueue,
				rng:         rand.New(rand.NewSource(opt.config.SeedValue + int64(i+1000))),
				ctx:         opt.ctx,
			}
		}
	}
}

// Optimize runs the ACO optimization algorithm
func (opt *ACOOptimizer) Optimize() (*Solution, error) {
	opt.mutex.Lock()
	if opt.running {
		opt.mutex.Unlock()
		return nil, errors.New("optimization is already running")
	}
	opt.running = true
	opt.mutex.Unlock()

	defer func() {
		opt.mutex.Lock()
		opt.running = false
		opt.mutex.Unlock()
	}()

	start := time.Now()
	defer func() {
		opt.statistics.mutex.Lock()
		opt.statistics.TotalTime = time.Since(start)
		opt.statistics.mutex.Unlock()
	}()

	// Start workers if parallel processing is enabled
	if opt.config.UseParallelAnts {
		opt.startWorkers()
		defer opt.stopWorkers()
	}

	// Main optimization loop
	for iteration := 0; iteration < opt.config.MaxIterations; iteration++ {
		select {
		case <-opt.ctx.Done():
			return opt.globalBestSolution, opt.ctx.Err()
		default:
		}

		// Run iteration
		iterationBest, err := opt.runIteration(iteration)
		if err != nil {
			return opt.globalBestSolution, err
		}

		// Update global best
		if iterationBest != nil && (opt.globalBestSolution == nil || iterationBest.Cost < opt.globalBestSolution.Cost) {
			opt.globalBestSolution = opt.copySolution(iterationBest)
			opt.globalBestSolution.Iteration = iteration
		}

		// Update statistics
		opt.updateStatistics(iteration, iterationBest)

		// Check convergence
		if opt.convergenceDetector.CheckConvergence(opt.globalBestSolution.Cost) {
			opt.statistics.mutex.Lock()
			opt.statistics.ConvergenceIteration = iteration
			opt.statistics.mutex.Unlock()
			break
		}

		// Apply diversity management if needed
		if opt.diversityManager.NeedsDiversification() {
			opt.applyDiversification()
		}

		// Update pheromone trails
		opt.updatePheromones()
	}

	return opt.globalBestSolution, nil
}

// runIteration runs a single iteration of the algorithm
func (opt *ACOOptimizer) runIteration(iteration int) (*Solution, error) {
	var iterationBest *Solution

	if opt.config.UseParallelColonies {
		// Parallel colony processing
		iterationBest = opt.runParallelColonies(iteration)
	} else {
		// Sequential colony processing
		for _, colony := range opt.colonies {
			colonyBest := opt.runColony(colony, iteration)
			if iterationBest == nil || (colonyBest != nil && colonyBest.Cost < iterationBest.Cost) {
				iterationBest = colonyBest
			}
		}
	}

	return iterationBest, nil
}

// runParallelColonies runs colonies in parallel
func (opt *ACOOptimizer) runParallelColonies(iteration int) *Solution {
	var wg sync.WaitGroup
	results := make(chan *Solution, opt.config.NumColonies)

	for _, colony := range opt.colonies {
		wg.Add(1)
		go func(c *Colony) {
			defer wg.Done()
			best := opt.runColony(c, iteration)
			results <- best
		}(colony)
	}

	wg.Wait()
	close(results)

	var iterationBest *Solution
	for result := range results {
		if iterationBest == nil || (result != nil && result.Cost < iterationBest.Cost) {
			iterationBest = result
		}
	}

	return iterationBest
}

// runColony runs a single colony for one iteration
func (opt *ACOOptimizer) runColony(colony *Colony, iteration int) *Solution {
	// Reset ants
	for _, ant := range colony.Ants {
		opt.resetAnt(ant)
	}

	if opt.config.UseParallelAnts {
		return opt.runColonyParallel(colony, iteration)
	} else {
		return opt.runColonySequential(colony, iteration)
	}
}

// runColonySequential runs colony ants sequentially
func (opt *ACOOptimizer) runColonySequential(colony *Colony, iteration int) *Solution {
	var colonyBest *Solution

	// Construct solutions for all ants
	for _, ant := range colony.Ants {
		solution := opt.constructSolution(ant, colony)
		
		// Apply local search if configured
		if opt.config.LocalSearchMethod != NoLocalSearch && 
		   colony.rng.Float64() < opt.config.LocalSearchProbability {
			solution = opt.applyLocalSearch(solution, colony.LocalSearcher)
		}

		// Update colony best
		if colonyBest == nil || solution.Cost < colonyBest.Cost {
			colonyBest = opt.copySolution(solution)
		}

		// Update colony statistics
		atomic.AddInt64(&colony.Statistics.SolutionsFound, 1)
	}

	// Update colony best solution
	colony.mutex.Lock()
	if colony.BestSolution == nil || colonyBest.Cost < colony.BestSolution.Cost {
		colony.BestSolution = opt.copySolution(colonyBest)
	}
	if colony.BestEverSolution == nil || colonyBest.Cost < colony.BestEverSolution.Cost {
		colony.BestEverSolution = opt.copySolution(colonyBest)
	}
	colony.mutex.Unlock()

	return colonyBest
}

// runColonyParallel runs colony ants in parallel
func (opt *ACOOptimizer) runColonyParallel(colony *Colony, iteration int) *Solution {
	// Send tasks to workers
	for _, ant := range colony.Ants {
		task := Task{
			Type:      ConstructSolutionTask,
			ColonyID:  colony.ID,
			AntID:     ant.ID,
			Iteration: iteration,
			TaskID:    fmt.Sprintf("construct_%d_%d_%d", iteration, colony.ID, ant.ID),
		}

		select {
		case opt.taskQueue <- task:
		case <-opt.ctx.Done():
			return nil
		}
	}

	// Collect results
	var colonyBest *Solution
	for i := 0; i < len(colony.Ants); i++ {
		select {
		case result := <-opt.resultQueue:
			if result.Success && result.Solution != nil {
				if colonyBest == nil || result.Solution.Cost < colonyBest.Cost {
					colonyBest = opt.copySolution(result.Solution)
				}
				atomic.AddInt64(&colony.Statistics.SolutionsFound, 1)
			}
		case <-opt.ctx.Done():
			return colonyBest
		}
	}

	// Update colony best solution
	colony.mutex.Lock()
	if colony.BestSolution == nil || (colonyBest != nil && colonyBest.Cost < colony.BestSolution.Cost) {
		colony.BestSolution = opt.copySolution(colonyBest)
	}
	if colony.BestEverSolution == nil || (colonyBest != nil && colonyBest.Cost < colony.BestEverSolution.Cost) {
		colony.BestEverSolution = opt.copySolution(colonyBest)
	}
	colony.mutex.Unlock()

	return colonyBest
}

// constructSolution constructs a solution using an ant
func (opt *ACOOptimizer) constructSolution(ant *Ant, colony *Colony) *Solution {
	// Start from random city
	startCity := ant.rng.Intn(opt.problem.Size)
	ant.CurrentCity = startCity
	ant.Tour = append(ant.Tour, startCity)
	ant.Visited[startCity] = true

	// Construct tour
	for len(ant.Tour) < opt.problem.Size {
		nextCity := opt.selectNextCity(ant, colony)
		if nextCity == -1 {
			break // No more cities available
		}

		ant.TourCost += opt.problem.DistMatrix[ant.CurrentCity][nextCity]
		ant.CurrentCity = nextCity
		ant.Tour = append(ant.Tour, nextCity)
		ant.Visited[nextCity] = true
	}

	// Complete tour by returning to start
	if len(ant.Tour) == opt.problem.Size {
		ant.TourCost += opt.problem.DistMatrix[ant.CurrentCity][ant.Tour[0]]
	}

	solution := &Solution{
		Tour:     make([]int, len(ant.Tour)),
		Cost:     ant.TourCost,
		Iteration: 0,
		ColonyID: ant.ColonyID,
		AntID:    ant.ID,
		Valid:    len(ant.Tour) == opt.problem.Size,
	}
	copy(solution.Tour, ant.Tour)

	if solution.Valid {
		solution.Quality = 1.0 / solution.Cost
	}

	return solution
}

// selectNextCity selects the next city for an ant to visit
func (opt *ACOOptimizer) selectNextCity(ant *Ant, colony *Colony) int {
	currentCity := ant.CurrentCity
	
	// Calculate probabilities for each unvisited city
	totalProbability := 0.0
	for i := 0; i < opt.problem.Size; i++ {
		if !ant.Visited[i] {
			pheromone := math.Pow(colony.PheromoneMatrix[currentCity][i], opt.config.Alpha)
			heuristic := math.Pow(colony.HeuristicMatrix[currentCity][i], opt.config.Beta)
			ant.Probability[i] = pheromone * heuristic
			totalProbability += ant.Probability[i]
		} else {
			ant.Probability[i] = 0
		}
	}

	if totalProbability == 0 {
		return -1 // No valid cities
	}

	// Normalize probabilities
	for i := 0; i < opt.problem.Size; i++ {
		ant.Probability[i] /= totalProbability
	}

	// Apply algorithm-specific selection rules
	switch opt.config.Algorithm {
	case AntColonySystem:
		return opt.selectCityACS(ant)
	case MaxMinAntSystem:
		return opt.selectCityMAS(ant)
	default:
		return opt.selectCityRoulette(ant)
	}
}

// selectCityRoulette selects city using roulette wheel selection
func (opt *ACOOptimizer) selectCityRoulette(ant *Ant) int {
	r := ant.rng.Float64()
	cumulative := 0.0
	
	for i := 0; i < opt.problem.Size; i++ {
		if !ant.Visited[i] {
			cumulative += ant.Probability[i]
			if r <= cumulative {
				return i
			}
		}
	}

	// Fallback: select first unvisited city
	for i := 0; i < opt.problem.Size; i++ {
		if !ant.Visited[i] {
			return i
		}
	}
	
	return -1
}

// selectCityACS implements Ant Colony System city selection
func (opt *ACOOptimizer) selectCityACS(ant *Ant) int {
	q0 := 0.9 // Exploitation parameter
	
	if ant.rng.Float64() < q0 {
		// Exploitation: select best city
		bestCity := -1
		bestValue := -1.0
		
		for i := 0; i < opt.problem.Size; i++ {
			if !ant.Visited[i] && ant.Probability[i] > bestValue {
				bestValue = ant.Probability[i]
				bestCity = i
			}
		}
		return bestCity
	} else {
		// Exploration: use roulette wheel
		return opt.selectCityRoulette(ant)
	}
}

// selectCityMAS implements MAX-MIN Ant System city selection
func (opt *ACOOptimizer) selectCityMAS(ant *Ant) int {
	// Similar to roulette wheel but with pheromone bounds
	return opt.selectCityRoulette(ant)
}

// applyLocalSearch applies local search to improve a solution
func (opt *ACOOptimizer) applyLocalSearch(solution *Solution, searcher *LocalSearcher) *Solution {
	if searcher == nil {
		return solution
	}

	improved := searcher.ImproveSolution(solution)
	if improved.Cost < solution.Cost {
		atomic.AddInt64(&opt.statistics.LocalSearchCount, 1)
		return improved
	}
	
	return solution
}

// ImproveSolution improves a solution using local search
func (ls *LocalSearcher) ImproveSolution(solution *Solution) *Solution {
	switch ls.method {
	case TwoOpt:
		return ls.twoOpt(solution)
	case ThreeOpt:
		return ls.threeOpt(solution)
	case OrOpt:
		return ls.orOpt(solution)
	default:
		return solution
	}
}

// twoOpt implements 2-opt local search
func (ls *LocalSearcher) twoOpt(solution *Solution) *Solution {
	best := ls.copySolution(solution)
	improved := true
	
	for improved {
		improved = false
		for i := 1; i < len(best.Tour)-2; i++ {
			for j := i + 1; j < len(best.Tour); j++ {
				if j-i == 1 {
					continue // Skip adjacent edges
				}
				
				newTour := ls.reverse(best.Tour, i, j)
				newCost := ls.calculateTourCost(newTour)
				
				if newCost < best.Cost {
					best.Tour = newTour
					best.Cost = newCost
					improved = true
				}
			}
		}
	}
	
	return best
}

// threeOpt implements 3-opt local search (simplified version)
func (ls *LocalSearcher) threeOpt(solution *Solution) *Solution {
	// Simplified 3-opt implementation
	return ls.twoOpt(solution) // Fallback to 2-opt for now
}

// orOpt implements Or-opt local search
func (ls *LocalSearcher) orOpt(solution *Solution) *Solution {
	best := ls.copySolution(solution)
	improved := true
	
	for improved {
		improved = false
		// Try moving sequences of 1, 2, and 3 cities
		for seqLen := 1; seqLen <= 3; seqLen++ {
			for i := 0; i < len(best.Tour)-seqLen; i++ {
				for j := 0; j < len(best.Tour); j++ {
					if j >= i && j <= i+seqLen {
						continue // Skip overlapping positions
					}
					
					newTour := ls.relocateSequence(best.Tour, i, seqLen, j)
					newCost := ls.calculateTourCost(newTour)
					
					if newCost < best.Cost {
						best.Tour = newTour
						best.Cost = newCost
						improved = true
					}
				}
			}
		}
	}
	
	return best
}

// Helper functions for local search

func (ls *LocalSearcher) reverse(tour []int, i, j int) []int {
	newTour := make([]int, len(tour))
	copy(newTour, tour)
	
	for k := i; k <= j; k++ {
		newTour[k] = tour[i+j-k]
	}
	
	return newTour
}

func (ls *LocalSearcher) relocateSequence(tour []int, start, length, target int) []int {
	newTour := make([]int, 0, len(tour))
	
	// Add cities before the sequence
	for i := 0; i < start; i++ {
		newTour = append(newTour, tour[i])
	}
	
	// Add cities after the sequence but before target
	for i := start + length; i < target; i++ {
		newTour = append(newTour, tour[i])
	}
	
	// Add the relocated sequence
	for i := start; i < start+length; i++ {
		newTour = append(newTour, tour[i])
	}
	
	// Add remaining cities
	for i := target; i < len(tour); i++ {
		newTour = append(newTour, tour[i])
	}
	
	return newTour
}

func (ls *LocalSearcher) calculateTourCost(tour []int) float64 {
	cost := 0.0
	for i := 0; i < len(tour)-1; i++ {
		cost += ls.problem.DistMatrix[tour[i]][tour[i+1]]
	}
	// Add return to start
	cost += ls.problem.DistMatrix[tour[len(tour)-1]][tour[0]]
	return cost
}

func (ls *LocalSearcher) copySolution(solution *Solution) *Solution {
	newSolution := &Solution{
		Tour:     make([]int, len(solution.Tour)),
		Cost:     solution.Cost,
		Quality:  solution.Quality,
		Iteration: solution.Iteration,
		ColonyID: solution.ColonyID,
		AntID:    solution.AntID,
		Valid:    solution.Valid,
	}
	copy(newSolution.Tour, solution.Tour)
	return newSolution
}

// Pheromone update methods

func (opt *ACOOptimizer) updatePheromones() {
	switch opt.config.PheromoneUpdateStrategy {
	case GlobalUpdate:
		opt.globalPheromoneUpdate()
	case LocalUpdate:
		opt.localPheromoneUpdate()
	case ElitistUpdate:
		opt.elitistPheromoneUpdate()
	case MaxMinUpdate:
		opt.maxMinPheromoneUpdate()
	default:
		opt.globalPheromoneUpdate()
	}
	
	atomic.AddInt64(&opt.statistics.PheromoneUpdates, 1)
}

func (opt *ACOOptimizer) globalPheromoneUpdate() {
	// Evaporation
	for i := 0; i < opt.problem.Size; i++ {
		for j := 0; j < opt.problem.Size; j++ {
			opt.pheromoneMatrix[i][j] *= (1.0 - opt.config.Rho)
		}
	}

	// Pheromone deposit by best solutions
	for _, colony := range opt.colonies {
		if colony.BestSolution != nil && colony.BestSolution.Valid {
			deposit := opt.config.Q / colony.BestSolution.Cost
			opt.depositPheromone(colony.BestSolution.Tour, deposit)
		}
	}

	// Global best reinforcement
	if opt.globalBestSolution != nil && opt.globalBestSolution.Valid {
		deposit := opt.config.Q / opt.globalBestSolution.Cost
		opt.depositPheromone(opt.globalBestSolution.Tour, deposit*opt.config.ElitistWeight)
	}
}

func (opt *ACOOptimizer) localPheromoneUpdate() {
	// Local pheromone update during construction (ACS style)
	evaporationRate := 0.1
	for _, colony := range opt.colonies {
		for i := 0; i < opt.problem.Size; i++ {
			for j := 0; j < opt.problem.Size; j++ {
				colony.PheromoneMatrix[i][j] = colony.PheromoneMatrix[i][j]*(1.0-evaporationRate) + 
					evaporationRate*opt.config.InitialPheromone
			}
		}
	}
}

func (opt *ACOOptimizer) elitistPheromoneUpdate() {
	// Evaporation
	for i := 0; i < opt.problem.Size; i++ {
		for j := 0; j < opt.problem.Size; j++ {
			opt.pheromoneMatrix[i][j] *= (1.0 - opt.config.Rho)
		}
	}

	// Only best solutions deposit pheromone
	if opt.globalBestSolution != nil && opt.globalBestSolution.Valid {
		deposit := opt.config.Q / opt.globalBestSolution.Cost
		opt.depositPheromone(opt.globalBestSolution.Tour, deposit)
	}
}

func (opt *ACOOptimizer) maxMinPheromoneUpdate() {
	// Evaporation
	for i := 0; i < opt.problem.Size; i++ {
		for j := 0; j < opt.problem.Size; j++ {
			opt.pheromoneMatrix[i][j] *= (1.0 - opt.config.Rho)
		}
	}

	// Pheromone deposit with bounds
	if opt.globalBestSolution != nil && opt.globalBestSolution.Valid {
		deposit := opt.config.Q / opt.globalBestSolution.Cost
		opt.depositPheromone(opt.globalBestSolution.Tour, deposit)
	}

	// Apply pheromone bounds
	for i := 0; i < opt.problem.Size; i++ {
		for j := 0; j < opt.problem.Size; j++ {
			if opt.pheromoneMatrix[i][j] < opt.config.MinPheromone {
				opt.pheromoneMatrix[i][j] = opt.config.MinPheromone
			}
			if opt.pheromoneMatrix[i][j] > opt.config.MaxPheromone {
				opt.pheromoneMatrix[i][j] = opt.config.MaxPheromone
			}
		}
	}
}

func (opt *ACOOptimizer) depositPheromone(tour []int, amount float64) {
	for i := 0; i < len(tour); i++ {
		from := tour[i]
		to := tour[(i+1)%len(tour)]
		opt.pheromoneMatrix[from][to] += amount
		opt.pheromoneMatrix[to][from] += amount // For symmetric problems
	}
}

// Utility functions

func (opt *ACOOptimizer) resetAnt(ant *Ant) {
	ant.CurrentCity = -1
	ant.Tour = ant.Tour[:0]
	ant.TourCost = 0
	for i := range ant.Visited {
		ant.Visited[i] = false
	}
}

func (opt *ACOOptimizer) copySolution(solution *Solution) *Solution {
	if solution == nil {
		return nil
	}
	
	newSolution := &Solution{
		Tour:     make([]int, len(solution.Tour)),
		Cost:     solution.Cost,
		Quality:  solution.Quality,
		Iteration: solution.Iteration,
		ColonyID: solution.ColonyID,
		AntID:    solution.AntID,
		Valid:    solution.Valid,
	}
	copy(newSolution.Tour, solution.Tour)
	return newSolution
}

func (opt *ACOOptimizer) copyMatrix(matrix [][]float64) [][]float64 {
	copied := make([][]float64, len(matrix))
	for i := range matrix {
		copied[i] = make([]float64, len(matrix[i]))
		copy(copied[i], matrix[i])
	}
	return copied
}

// Worker management

func (opt *ACOOptimizer) startWorkers() {
	for _, worker := range opt.workers {
		opt.wg.Add(1)
		go worker.Start()
	}
}

func (opt *ACOOptimizer) stopWorkers() {
	close(opt.taskQueue)
	opt.wg.Wait()
}

func (w *Worker) Start() {
	defer w.optimizer.wg.Done()
	
	for {
		select {
		case task, ok := <-w.taskQueue:
			if !ok {
				return
			}
			
			result := w.processTask(task)
			
			select {
			case w.resultQueue <- result:
			case <-w.ctx.Done():
				return
			}
			
		case <-w.ctx.Done():
			return
		}
	}
}

func (w *Worker) processTask(task Task) Result {
	start := time.Now()
	result := Result{
		TaskID:   task.TaskID,
		Type:     task.Type,
		Success:  false,
		Duration: 0,
	}
	
	defer func() {
		result.Duration = time.Since(start)
	}()
	
	switch task.Type {
	case ConstructSolutionTask:
		colony := w.optimizer.colonies[task.ColonyID]
		ant := colony.Ants[task.AntID]
		w.optimizer.resetAnt(ant)
		
		solution := w.optimizer.constructSolution(ant, colony)
		result.Solution = solution
		result.Success = solution.Valid
		
	default:
		result.Error = fmt.Errorf("unknown task type: %v", task.Type)
	}
	
	return result
}

// Statistics and monitoring

func (opt *ACOOptimizer) updateStatistics(iteration int, iterationBest *Solution) {
	opt.statistics.mutex.Lock()
	defer opt.statistics.mutex.Unlock()

	opt.statistics.TotalIterations = iteration + 1

	if iterationBest != nil {
		opt.statistics.CostHistory = append(opt.statistics.CostHistory, iterationBest.Cost)
		
		if iterationBest.Cost < opt.statistics.BestCost {
			opt.statistics.BestCost = iterationBest.Cost
		}
		
		// Update average cost
		if len(opt.statistics.CostHistory) > 0 {
			sum := 0.0
			for _, cost := range opt.statistics.CostHistory {
				sum += cost
			}
			opt.statistics.AverageCost = sum / float64(len(opt.statistics.CostHistory))
		}
	}

	// Update diversity measures
	diversity := opt.calculateDiversity()
	opt.statistics.DiversityHistory = append(opt.statistics.DiversityHistory, diversity)
}

func (opt *ACOOptimizer) calculateDiversity() float64 {
	// Simple diversity measure based on solution differences
	if len(opt.colonies) < 2 {
		return 0.0
	}

	totalDiff := 0.0
	comparisons := 0

	for i := 0; i < len(opt.colonies); i++ {
		for j := i + 1; j < len(opt.colonies); j++ {
			if opt.colonies[i].BestSolution != nil && opt.colonies[j].BestSolution != nil {
				diff := opt.calculateSolutionDifference(
					opt.colonies[i].BestSolution,
					opt.colonies[j].BestSolution,
				)
				totalDiff += diff
				comparisons++
			}
		}
	}

	if comparisons == 0 {
		return 0.0
	}

	return totalDiff / float64(comparisons)
}

func (opt *ACOOptimizer) calculateSolutionDifference(sol1, sol2 *Solution) float64 {
	if len(sol1.Tour) != len(sol2.Tour) {
		return 1.0
	}

	differences := 0
	for i := range sol1.Tour {
		if sol1.Tour[i] != sol2.Tour[i] {
			differences++
		}
	}

	return float64(differences) / float64(len(sol1.Tour))
}

// Convergence detection

func (cd *ConvergenceDetector) CheckConvergence(currentBest float64) bool {
	cd.mutex.Lock()
	defer cd.mutex.Unlock()

	cd.bestCosts = append(cd.bestCosts, currentBest)

	if len(cd.bestCosts) < 2 {
		return false
	}

	// Check if improvement is below threshold
	improvement := cd.bestCosts[len(cd.bestCosts)-2] - currentBest
	if improvement < cd.threshold {
		cd.stagnationCount++
	} else {
		cd.stagnationCount = 0
	}

	if cd.stagnationCount >= cd.maxStagnation {
		cd.converged = true
		return true
	}

	return false
}

// Diversification methods

func (dm *DiversityManager) NeedsDiversification() bool {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()

	if len(dm.diversityMetrics) < 10 {
		return false
	}

	// Check if diversity has been low for several iterations
	recent := dm.diversityMetrics[len(dm.diversityMetrics)-5:]
	avgDiversity := 0.0
	for _, d := range recent {
		avgDiversity += d
	}
	avgDiversity /= float64(len(recent))

	return avgDiversity < dm.threshold
}

func (opt *ACOOptimizer) applyDiversification() {
	// Reset some pheromone trails to increase exploration
	resetProbability := 0.1
	
	for i := 0; i < opt.problem.Size; i++ {
		for j := 0; j < opt.problem.Size; j++ {
			if rand.Float64() < resetProbability {
				opt.pheromoneMatrix[i][j] = opt.config.InitialPheromone
			}
		}
	}

	// Reinitialize some ants in random positions
	for _, colony := range opt.colonies {
		for i := 0; i < len(colony.Ants)/4; i++ {
			colony.Ants[i].rng = rand.New(rand.NewSource(time.Now().UnixNano()))
		}
	}
}

// GetStatistics returns current optimization statistics
func (opt *ACOOptimizer) GetStatistics() *OptimizationStatistics {
	opt.statistics.mutex.RLock()
	defer opt.statistics.mutex.RUnlock()

	// Create a copy of statistics
	stats := &OptimizationStatistics{
		TotalIterations:      opt.statistics.TotalIterations,
		BestCost:             opt.statistics.BestCost,
		AverageCost:          opt.statistics.AverageCost,
		WorstCost:            opt.statistics.WorstCost,
		ConvergenceIteration: opt.statistics.ConvergenceIteration,
		TotalTime:            opt.statistics.TotalTime,
		SolutionsEvaluated:   atomic.LoadInt64(&opt.statistics.SolutionsEvaluated),
		PheromoneUpdates:     atomic.LoadInt64(&opt.statistics.PheromoneUpdates),
		LocalSearchCount:     atomic.LoadInt64(&opt.statistics.LocalSearchCount),
		DiversityHistory:     make([]float64, len(opt.statistics.DiversityHistory)),
		CostHistory:          make([]float64, len(opt.statistics.CostHistory)),
		ColonyStatistics:     make([]*ColonyStatistics, len(opt.statistics.ColonyStatistics)),
		WorkerUtilization:    make([]float64, len(opt.statistics.WorkerUtilization)),
	}

	copy(stats.DiversityHistory, opt.statistics.DiversityHistory)
	copy(stats.CostHistory, opt.statistics.CostHistory)
	copy(stats.WorkerUtilization, opt.statistics.WorkerUtilization)

	for i, cs := range opt.statistics.ColonyStatistics {
		stats.ColonyStatistics[i] = &ColonyStatistics{
			ColonyID:         cs.ColonyID,
			IterationCount:   cs.IterationCount,
			BestCost:         cs.BestCost,
			AverageCost:      cs.AverageCost,
			SolutionsFound:   atomic.LoadInt64(&cs.SolutionsFound),
			LocalSearchUsage: atomic.LoadInt64(&cs.LocalSearchUsage),
			PheromoneLevel:   cs.PheromoneLevel,
			DiversityMeasure: cs.DiversityMeasure,
			ConvergenceRate:  cs.ConvergenceRate,
		}
	}

	return stats
}

// Stop stops the optimization process
func (opt *ACOOptimizer) Stop() {
	opt.cancel()
}

// Cleanup releases resources
func (opt *ACOOptimizer) Cleanup() {
	opt.Stop()
	
	opt.mutex.Lock()
	defer opt.mutex.Unlock()
	
	opt.colonies = nil
	opt.workers = nil
	opt.pheromoneMatrix = nil
	opt.heuristicMatrix = nil
}