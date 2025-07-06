package parallelsimulatedannealing

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

// ObjectiveFunction defines the function to optimize
type ObjectiveFunction func([]float64) float64

// NeighborFunction generates a neighbor solution from current solution
type NeighborFunction func([]float64, float64, *rand.Rand) []float64

// CoolingSchedule defines how temperature decreases over time
type CoolingSchedule int

const (
	LinearCooling CoolingSchedule = iota
	ExponentialCooling
	LogarithmicCooling
	InverseCooling
	AdaptiveCooling
	GeometricCooling
	QuadraticCooling
	CosineAnnealing
)

// PerturbationStrategy defines different neighbor generation strategies
type PerturbationStrategy int

const (
	GaussianPerturbation PerturbationStrategy = iota
	UniformPerturbation
	CauchyPerturbation
	LevyFlightPerturbation
	AdaptivePerturbation
	HybridPerturbation
)

// ParallelStrategy defines different parallelization approaches
type ParallelStrategy int

const (
	IndependentChains ParallelStrategy = iota
	TemperatureParallel
	MultipleRestart
	IslandModel
	HybridParallel
	CooperativeChains
)

// SAConfig contains configuration for the simulated annealing algorithm
type SAConfig struct {
	Dimensions           int
	LowerBound          []float64
	UpperBound          []float64
	InitialTemperature  float64
	FinalTemperature    float64
	MaxIterations       int
	CoolingSchedule     CoolingSchedule
	CoolingRate         float64
	PerturbationStrat   PerturbationStrategy
	PerturbationSize    float64
	ParallelStrategy    ParallelStrategy
	NumWorkers          int
	NumChains           int
	RestartInterval     int
	IslandExchangeRate  int
	AdaptiveParameters  bool
	EnableMemory        bool
	EnableStatistics    bool
	EnableLogging       bool
	Tolerance           float64
	StagnationLimit     int
	ElitismRate         float64
	DiversityThreshold  float64
	RandomSeed          int64
}

// DefaultSAConfig returns default simulated annealing configuration
func DefaultSAConfig() SAConfig {
	return SAConfig{
		Dimensions:         10,
		InitialTemperature: 1000.0,
		FinalTemperature:   0.01,
		MaxIterations:      100000,
		CoolingSchedule:    ExponentialCooling,
		CoolingRate:        0.95,
		PerturbationStrat:  GaussianPerturbation,
		PerturbationSize:   1.0,
		ParallelStrategy:   IndependentChains,
		NumWorkers:         runtime.NumCPU(),
		NumChains:          runtime.NumCPU(),
		RestartInterval:    10000,
		IslandExchangeRate: 1000,
		AdaptiveParameters: true,
		EnableMemory:       true,
		EnableStatistics:   true,
		EnableLogging:      false,
		Tolerance:          1e-6,
		StagnationLimit:    5000,
		ElitismRate:        0.1,
		DiversityThreshold: 0.01,
		RandomSeed:         time.Now().UnixNano(),
	}
}

// Solution represents a solution in the optimization space
type Solution struct {
	Variables []float64
	Fitness   float64
	Iteration int
	Chain     int
	Timestamp time.Time
}

// Copy creates a deep copy of the solution
func (s *Solution) Copy() *Solution {
	variables := make([]float64, len(s.Variables))
	copy(variables, s.Variables)
	return &Solution{
		Variables: variables,
		Fitness:   s.Fitness,
		Iteration: s.Iteration,
		Chain:     s.Chain,
		Timestamp: s.Timestamp,
	}
}

// Chain represents a single simulated annealing chain
type Chain struct {
	ID               int
	Current          *Solution
	Best             *Solution
	Temperature      float64
	Iteration        int
	AcceptanceCount  int64
	RejectionCount   int64
	StagnationCount  int
	Random           *rand.Rand
	Memory           []*Solution
	AdaptiveStepSize float64
	DiversityScore   float64
	mutex            sync.RWMutex
}

// NewChain creates a new simulated annealing chain
func NewChain(id int, initialSolution *Solution, temperature float64, seed int64) *Chain {
	return &Chain{
		ID:               id,
		Current:          initialSolution.Copy(),
		Best:             initialSolution.Copy(),
		Temperature:      temperature,
		Iteration:        0,
		AcceptanceCount:  0,
		RejectionCount:   0,
		StagnationCount:  0,
		Random:           rand.New(rand.NewSource(seed + int64(id))),
		Memory:           make([]*Solution, 0),
		AdaptiveStepSize: 1.0,
		DiversityScore:   0.0,
	}
}

// Statistics tracks algorithm performance
type Statistics struct {
	StartTime           time.Time
	EndTime             time.Time
	TotalIterations     int64
	TotalAcceptances    int64
	TotalRejections     int64
	BestSolution        *Solution
	ConvergenceHistory  []float64
	TemperatureHistory  []float64
	AcceptanceRateHist  []float64
	DiversityHistory    []float64
	ChainStatistics     map[int]*ChainStats
	RestartCount        int64
	ExchangeCount       int64
	StagnationPeriods   int64
	AverageTemperature  float64
	FinalTemperature    float64
	mutex               sync.RWMutex
}

// ChainStats tracks individual chain statistics
type ChainStats struct {
	ChainID         int
	Iterations      int64
	Acceptances     int64
	Rejections      int64
	BestFitness     float64
	FinalFitness    float64
	AcceptanceRate  float64
	AverageStepSize float64
	StagnationCount int64
}

// ParallelSimulatedAnnealing implements parallel simulated annealing optimization
type ParallelSimulatedAnnealing struct {
	config            SAConfig
	objective         ObjectiveFunction
	neighborFunc      NeighborFunction
	chains            []*Chain
	globalBest        *Solution
	statistics        *Statistics
	running           bool
	ctx               context.Context
	cancel            context.CancelFunc
	workerPool        chan struct{}
	resultChan        chan *Solution
	exchangeChan      chan *Solution
	restartSignal     chan struct{}
	mutex             sync.RWMutex
	wg                sync.WaitGroup
}

// NewParallelSimulatedAnnealing creates a new parallel simulated annealing optimizer
func NewParallelSimulatedAnnealing(config SAConfig, objective ObjectiveFunction) (*ParallelSimulatedAnnealing, error) {
	if config.Dimensions <= 0 {
		return nil, errors.New("dimensions must be positive")
	}
	
	if config.InitialTemperature <= config.FinalTemperature {
		return nil, errors.New("initial temperature must be greater than final temperature")
	}
	
	if config.MaxIterations <= 0 {
		return nil, errors.New("max iterations must be positive")
	}
	
	if objective == nil {
		return nil, errors.New("objective function cannot be nil")
	}
	
	// Set default bounds if not provided
	if len(config.LowerBound) == 0 {
		config.LowerBound = make([]float64, config.Dimensions)
		for i := range config.LowerBound {
			config.LowerBound[i] = -100.0
		}
	}
	
	if len(config.UpperBound) == 0 {
		config.UpperBound = make([]float64, config.Dimensions)
		for i := range config.UpperBound {
			config.UpperBound[i] = 100.0
		}
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	psa := &ParallelSimulatedAnnealing{
		config:        config,
		objective:     objective,
		chains:        make([]*Chain, config.NumChains),
		workerPool:    make(chan struct{}, config.NumWorkers),
		resultChan:    make(chan *Solution, config.NumChains*2),
		exchangeChan:  make(chan *Solution, config.NumChains),
		restartSignal: make(chan struct{}, 1),
		running:       false,
		ctx:           ctx,
		cancel:        cancel,
		statistics: &Statistics{
			ChainStatistics:    make(map[int]*ChainStats),
			ConvergenceHistory: make([]float64, 0),
			TemperatureHistory: make([]float64, 0),
			AcceptanceRateHist: make([]float64, 0),
			DiversityHistory:   make([]float64, 0),
		},
	}
	
	// Initialize neighbor function
	psa.neighborFunc = psa.createNeighborFunction()
	
	// Initialize chains
	psa.initializeChains()
	
	return psa, nil
}

// initializeChains creates and initializes all chains
func (psa *ParallelSimulatedAnnealing) initializeChains() {
	rand.Seed(psa.config.RandomSeed)
	
	for i := 0; i < psa.config.NumChains; i++ {
		// Generate random initial solution
		variables := make([]float64, psa.config.Dimensions)
		for j := 0; j < psa.config.Dimensions; j++ {
			variables[j] = psa.config.LowerBound[j] + 
				rand.Float64()*(psa.config.UpperBound[j]-psa.config.LowerBound[j])
		}
		
		fitness := psa.objective(variables)
		initialSolution := &Solution{
			Variables: variables,
			Fitness:   fitness,
			Iteration: 0,
			Chain:     i,
			Timestamp: time.Now(),
		}
		
		chain := NewChain(i, initialSolution, psa.config.InitialTemperature, 
			psa.config.RandomSeed+int64(i))
		psa.chains[i] = chain
		
		// Initialize global best
		if psa.globalBest == nil || fitness < psa.globalBest.Fitness {
			psa.globalBest = initialSolution.Copy()
		}
		
		// Initialize chain statistics
		psa.statistics.ChainStatistics[i] = &ChainStats{
			ChainID:     i,
			BestFitness: fitness,
			FinalFitness: fitness,
		}
	}
}

// createNeighborFunction creates the neighbor generation function
func (psa *ParallelSimulatedAnnealing) createNeighborFunction() NeighborFunction {
	switch psa.config.PerturbationStrat {
	case GaussianPerturbation:
		return psa.gaussianNeighbor
	case UniformPerturbation:
		return psa.uniformNeighbor
	case CauchyPerturbation:
		return psa.cauchyNeighbor
	case LevyFlightPerturbation:
		return psa.levyFlightNeighbor
	case AdaptivePerturbation:
		return psa.adaptiveNeighbor
	case HybridPerturbation:
		return psa.hybridNeighbor
	default:
		return psa.gaussianNeighbor
	}
}

// Optimize runs the parallel simulated annealing optimization
func (psa *ParallelSimulatedAnnealing) Optimize() (*Solution, error) {
	psa.mutex.Lock()
	if psa.running {
		psa.mutex.Unlock()
		return nil, errors.New("optimization is already running")
	}
	psa.running = true
	psa.statistics.StartTime = time.Now()
	psa.mutex.Unlock()
	
	defer func() {
		psa.mutex.Lock()
		psa.running = false
		psa.statistics.EndTime = time.Now()
		psa.mutex.Unlock()
	}()
	
	// Start background goroutines
	psa.wg.Add(3)
	go psa.resultCollector()
	go psa.exchangeManager()
	go psa.restartManager()
	
	// Start optimization based on parallel strategy
	switch psa.config.ParallelStrategy {
	case IndependentChains:
		psa.runIndependentChains()
	case TemperatureParallel:
		psa.runTemperatureParallel()
	case MultipleRestart:
		psa.runMultipleRestart()
	case IslandModel:
		psa.runIslandModel()
	case HybridParallel:
		psa.runHybridParallel()
	case CooperativeChains:
		psa.runCooperativeChains()
	default:
		psa.runIndependentChains()
	}
	
	// Wait for completion
	psa.cancel()
	psa.wg.Wait()
	
	// Finalize statistics
	psa.finalizeStatistics()
	
	return psa.globalBest.Copy(), nil
}

// runIndependentChains runs multiple independent annealing chains
func (psa *ParallelSimulatedAnnealing) runIndependentChains() {
	for i := 0; i < psa.config.NumChains; i++ {
		psa.wg.Add(1)
		go func(chainID int) {
			defer psa.wg.Done()
			psa.optimizeChain(psa.chains[chainID])
		}(i)
	}
}

// runTemperatureParallel runs parallel chains with different temperature schedules
func (psa *ParallelSimulatedAnnealing) runTemperatureParallel() {
	tempRange := psa.config.InitialTemperature - psa.config.FinalTemperature
	
	for i := 0; i < psa.config.NumChains; i++ {
		// Assign different initial temperatures
		tempOffset := float64(i) / float64(psa.config.NumChains-1)
		chainTemp := psa.config.InitialTemperature - tempOffset*tempRange*0.5
		psa.chains[i].Temperature = chainTemp
		
		psa.wg.Add(1)
		go func(chainID int) {
			defer psa.wg.Done()
			psa.optimizeChain(psa.chains[chainID])
		}(i)
	}
}

// runMultipleRestart runs chains with periodic restarts
func (psa *ParallelSimulatedAnnealing) runMultipleRestart() {
	for i := 0; i < psa.config.NumChains; i++ {
		psa.wg.Add(1)
		go func(chainID int) {
			defer psa.wg.Done()
			psa.optimizeChainWithRestart(psa.chains[chainID])
		}(i)
	}
}

// runIslandModel runs island model with solution exchange
func (psa *ParallelSimulatedAnnealing) runIslandModel() {
	for i := 0; i < psa.config.NumChains; i++ {
		psa.wg.Add(1)
		go func(chainID int) {
			defer psa.wg.Done()
			psa.optimizeIsland(psa.chains[chainID])
		}(i)
	}
}

// runHybridParallel combines multiple strategies
func (psa *ParallelSimulatedAnnealing) runHybridParallel() {
	// Half independent, half with exchange
	mid := psa.config.NumChains / 2
	
	for i := 0; i < mid; i++ {
		psa.wg.Add(1)
		go func(chainID int) {
			defer psa.wg.Done()
			psa.optimizeChain(psa.chains[chainID])
		}(i)
	}
	
	for i := mid; i < psa.config.NumChains; i++ {
		psa.wg.Add(1)
		go func(chainID int) {
			defer psa.wg.Done()
			psa.optimizeIsland(psa.chains[chainID])
		}(i)
	}
}

// runCooperativeChains runs chains with cooperative search
func (psa *ParallelSimulatedAnnealing) runCooperativeChains() {
	for i := 0; i < psa.config.NumChains; i++ {
		psa.wg.Add(1)
		go func(chainID int) {
			defer psa.wg.Done()
			psa.optimizeCooperative(psa.chains[chainID])
		}(i)
	}
}

// optimizeChain optimizes a single chain
func (psa *ParallelSimulatedAnnealing) optimizeChain(chain *Chain) {
	for chain.Iteration < psa.config.MaxIterations && chain.Temperature > psa.config.FinalTemperature {
		select {
		case <-psa.ctx.Done():
			return
		default:
			psa.performIteration(chain)
			psa.updateTemperature(chain)
			
			if chain.Iteration%1000 == 0 {
				psa.updateStatistics(chain)
			}
		}
	}
}

// optimizeChainWithRestart optimizes a chain with periodic restarts
func (psa *ParallelSimulatedAnnealing) optimizeChainWithRestart(chain *Chain) {
	for chain.Iteration < psa.config.MaxIterations {
		select {
		case <-psa.ctx.Done():
			return
		case <-psa.restartSignal:
			psa.restartChain(chain)
		default:
			psa.performIteration(chain)
			psa.updateTemperature(chain)
			
			if chain.Iteration%psa.config.RestartInterval == 0 {
				psa.restartChain(chain)
			}
			
			if chain.Iteration%1000 == 0 {
				psa.updateStatistics(chain)
			}
		}
	}
}

// optimizeIsland optimizes using island model
func (psa *ParallelSimulatedAnnealing) optimizeIsland(chain *Chain) {
	for chain.Iteration < psa.config.MaxIterations && chain.Temperature > psa.config.FinalTemperature {
		select {
		case <-psa.ctx.Done():
			return
		case migrant := <-psa.exchangeChan:
			psa.handleMigration(chain, migrant)
		default:
			psa.performIteration(chain)
			psa.updateTemperature(chain)
			
			if chain.Iteration%psa.config.IslandExchangeRate == 0 {
				psa.sendMigrant(chain)
			}
			
			if chain.Iteration%1000 == 0 {
				psa.updateStatistics(chain)
			}
		}
	}
}

// optimizeCooperative optimizes using cooperative search
func (psa *ParallelSimulatedAnnealing) optimizeCooperative(chain *Chain) {
	for chain.Iteration < psa.config.MaxIterations && chain.Temperature > psa.config.FinalTemperature {
		select {
		case <-psa.ctx.Done():
			return
		default:
			psa.performCooperativeIteration(chain)
			psa.updateTemperature(chain)
			
			if chain.Iteration%1000 == 0 {
				psa.updateStatistics(chain)
			}
		}
	}
}

// performIteration performs one iteration of simulated annealing
func (psa *ParallelSimulatedAnnealing) performIteration(chain *Chain) {
	// Generate neighbor solution
	neighbor := psa.neighborFunc(chain.Current.Variables, chain.Temperature, chain.Random)
	
	// Ensure neighbor is within bounds
	neighbor = psa.enforceConstraints(neighbor)
	
	// Evaluate neighbor
	neighborFitness := psa.objective(neighbor)
	
	// Accept or reject neighbor
	accepted := psa.acceptanceCriterion(chain.Current.Fitness, neighborFitness, chain.Temperature, chain.Random)
	
	if accepted {
		// Update current solution
		chain.Current.Variables = neighbor
		chain.Current.Fitness = neighborFitness
		chain.Current.Iteration = chain.Iteration
		chain.Current.Timestamp = time.Now()
		
		atomic.AddInt64(&chain.AcceptanceCount, 1)
		
		// Update best solution
		if neighborFitness < chain.Best.Fitness {
			chain.Best = chain.Current.Copy()
			chain.StagnationCount = 0
			
			// Update global best
			psa.updateGlobalBest(chain.Best)
		} else {
			chain.StagnationCount++
		}
		
		// Update adaptive parameters
		if psa.config.AdaptiveParameters {
			psa.updateAdaptiveParameters(chain, true)
		}
	} else {
		atomic.AddInt64(&chain.RejectionCount, 1)
		chain.StagnationCount++
		
		// Update adaptive parameters
		if psa.config.AdaptiveParameters {
			psa.updateAdaptiveParameters(chain, false)
		}
	}
	
	// Store in memory if enabled
	if psa.config.EnableMemory && len(chain.Memory) < 100 {
		chain.Memory = append(chain.Memory, chain.Current.Copy())
	}
	
	chain.Iteration++
	
	// Send result
	select {
	case psa.resultChan <- chain.Current.Copy():
	default:
	}
}

// performCooperativeIteration performs cooperative iteration with global information
func (psa *ParallelSimulatedAnnealing) performCooperativeIteration(chain *Chain) {
	// Standard iteration
	psa.performIteration(chain)
	
	// Cooperative enhancement: bias toward global best region
	if chain.Iteration%100 == 0 && psa.globalBest != nil {
		cooperativeNeighbor := psa.generateCooperativeNeighbor(chain, psa.globalBest)
		cooperativeFitness := psa.objective(cooperativeNeighbor)
		
		if cooperativeFitness < chain.Current.Fitness {
			chain.Current.Variables = cooperativeNeighbor
			chain.Current.Fitness = cooperativeFitness
			
			if cooperativeFitness < chain.Best.Fitness {
				chain.Best = chain.Current.Copy()
				psa.updateGlobalBest(chain.Best)
			}
		}
	}
}

// acceptanceCriterion determines whether to accept a neighbor solution
func (psa *ParallelSimulatedAnnealing) acceptanceCriterion(currentFitness, neighborFitness, temperature float64, rng *rand.Rand) bool {
	if neighborFitness < currentFitness {
		return true // Always accept better solutions
	}
	
	// Accept worse solutions with probability exp(-ΔE/T)
	deltaE := neighborFitness - currentFitness
	probability := math.Exp(-deltaE / temperature)
	return rng.Float64() < probability
}

// updateTemperature updates the temperature according to cooling schedule
func (psa *ParallelSimulatedAnnealing) updateTemperature(chain *Chain) {
	switch psa.config.CoolingSchedule {
	case LinearCooling:
		chain.Temperature = psa.config.InitialTemperature * 
			(1.0 - float64(chain.Iteration)/float64(psa.config.MaxIterations))
	case ExponentialCooling:
		chain.Temperature *= psa.config.CoolingRate
	case LogarithmicCooling:
		chain.Temperature = psa.config.InitialTemperature / 
			math.Log(1.0+float64(chain.Iteration))
	case InverseCooling:
		chain.Temperature = psa.config.InitialTemperature / 
			(1.0 + float64(chain.Iteration))
	case AdaptiveCooling:
		acceptanceRate := float64(chain.AcceptanceCount) / 
			float64(chain.AcceptanceCount+chain.RejectionCount+1)
		if acceptanceRate > 0.6 {
			chain.Temperature *= 0.99 // Cool faster
		} else if acceptanceRate < 0.2 {
			chain.Temperature *= 1.01 // Cool slower
		} else {
			chain.Temperature *= psa.config.CoolingRate
		}
	case GeometricCooling:
		chain.Temperature = psa.config.InitialTemperature * 
			math.Pow(psa.config.CoolingRate, float64(chain.Iteration))
	case QuadraticCooling:
		progress := float64(chain.Iteration) / float64(psa.config.MaxIterations)
		chain.Temperature = psa.config.InitialTemperature * (1.0 - progress*progress)
	case CosineAnnealing:
		progress := float64(chain.Iteration) / float64(psa.config.MaxIterations)
		chain.Temperature = psa.config.FinalTemperature + 
			(psa.config.InitialTemperature-psa.config.FinalTemperature) * 
			(1.0+math.Cos(math.Pi*progress)) / 2.0
	}
	
	// Ensure temperature doesn't go below final temperature
	if chain.Temperature < psa.config.FinalTemperature {
		chain.Temperature = psa.config.FinalTemperature
	}
}

// Neighbor generation functions

func (psa *ParallelSimulatedAnnealing) gaussianNeighbor(current []float64, temperature float64, rng *rand.Rand) []float64 {
	neighbor := make([]float64, len(current))
	stepSize := psa.config.PerturbationSize * math.Sqrt(temperature/psa.config.InitialTemperature)
	
	for i, val := range current {
		neighbor[i] = val + rng.NormFloat64()*stepSize
	}
	
	return neighbor
}

func (psa *ParallelSimulatedAnnealing) uniformNeighbor(current []float64, temperature float64, rng *rand.Rand) []float64 {
	neighbor := make([]float64, len(current))
	stepSize := psa.config.PerturbationSize * temperature / psa.config.InitialTemperature
	
	for i, val := range current {
		neighbor[i] = val + (rng.Float64()-0.5)*2*stepSize
	}
	
	return neighbor
}

func (psa *ParallelSimulatedAnnealing) cauchyNeighbor(current []float64, temperature float64, rng *rand.Rand) []float64 {
	neighbor := make([]float64, len(current))
	stepSize := psa.config.PerturbationSize * temperature / psa.config.InitialTemperature
	
	for i, val := range current {
		// Generate Cauchy distributed random number
		u := rng.Float64()
		cauchy := math.Tan(math.Pi * (u - 0.5))
		neighbor[i] = val + cauchy*stepSize
	}
	
	return neighbor
}

func (psa *ParallelSimulatedAnnealing) levyFlightNeighbor(current []float64, temperature float64, rng *rand.Rand) []float64 {
	neighbor := make([]float64, len(current))
	beta := 1.5 // Levy exponent
	stepSize := psa.config.PerturbationSize * temperature / psa.config.InitialTemperature
	
	for i, val := range current {
		// Generate Levy flight step
		u := rng.NormFloat64()
		v := rng.NormFloat64()
		levyStep := u / math.Pow(math.Abs(v), 1.0/beta)
		neighbor[i] = val + levyStep*stepSize
	}
	
	return neighbor
}

func (psa *ParallelSimulatedAnnealing) adaptiveNeighbor(current []float64, temperature float64, rng *rand.Rand) []float64 {
	// Implementation would adapt step size based on acceptance history
	return psa.gaussianNeighbor(current, temperature, rng)
}

func (psa *ParallelSimulatedAnnealing) hybridNeighbor(current []float64, temperature float64, rng *rand.Rand) []float64 {
	// Mix different perturbation strategies
	switch rng.Intn(3) {
	case 0:
		return psa.gaussianNeighbor(current, temperature, rng)
	case 1:
		return psa.cauchyNeighbor(current, temperature, rng)
	default:
		return psa.levyFlightNeighbor(current, temperature, rng)
	}
}

// Utility functions

func (psa *ParallelSimulatedAnnealing) enforceConstraints(solution []float64) []float64 {
	constrained := make([]float64, len(solution))
	for i, val := range solution {
		if val < psa.config.LowerBound[i] {
			constrained[i] = psa.config.LowerBound[i]
		} else if val > psa.config.UpperBound[i] {
			constrained[i] = psa.config.UpperBound[i]
		} else {
			constrained[i] = val
		}
	}
	return constrained
}

func (psa *ParallelSimulatedAnnealing) updateGlobalBest(solution *Solution) {
	psa.mutex.Lock()
	defer psa.mutex.Unlock()
	
	if psa.globalBest == nil || solution.Fitness < psa.globalBest.Fitness {
		psa.globalBest = solution.Copy()
	}
}

func (psa *ParallelSimulatedAnnealing) updateAdaptiveParameters(chain *Chain, accepted bool) {
	if accepted {
		chain.AdaptiveStepSize *= 1.01 // Increase step size
	} else {
		chain.AdaptiveStepSize *= 0.99 // Decrease step size
	}
	
	// Keep within reasonable bounds
	if chain.AdaptiveStepSize < 0.1 {
		chain.AdaptiveStepSize = 0.1
	} else if chain.AdaptiveStepSize > 10.0 {
		chain.AdaptiveStepSize = 10.0
	}
}

func (psa *ParallelSimulatedAnnealing) generateCooperativeNeighbor(chain *Chain, globalBest *Solution) []float64 {
	neighbor := make([]float64, len(chain.Current.Variables))
	
	for i := 0; i < len(neighbor); i++ {
		// Blend current solution with global best
		alpha := chain.Random.Float64()
		neighbor[i] = alpha*chain.Current.Variables[i] + (1-alpha)*globalBest.Variables[i]
		
		// Add small perturbation
		neighbor[i] += chain.Random.NormFloat64() * 0.1 * psa.config.PerturbationSize
	}
	
	return psa.enforceConstraints(neighbor)
}

// Background management functions

func (psa *ParallelSimulatedAnnealing) resultCollector() {
	defer psa.wg.Done()
	
	for {
		select {
		case <-psa.ctx.Done():
			return
		case solution := <-psa.resultChan:
			psa.updateGlobalBest(solution)
		}
	}
}

func (psa *ParallelSimulatedAnnealing) exchangeManager() {
	defer psa.wg.Done()
	
	ticker := time.NewTicker(time.Duration(psa.config.IslandExchangeRate) * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-psa.ctx.Done():
			return
		case <-ticker.C:
			// Facilitate solution exchange between islands
			psa.performSolutionExchange()
		}
	}
}

func (psa *ParallelSimulatedAnnealing) restartManager() {
	defer psa.wg.Done()
	
	ticker := time.NewTicker(time.Duration(psa.config.RestartInterval) * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-psa.ctx.Done():
			return
		case <-ticker.C:
			select {
			case psa.restartSignal <- struct{}{}:
			default:
			}
		}
	}
}

// Additional methods would continue here...

// GetBestSolution returns the current best solution
func (psa *ParallelSimulatedAnnealing) GetBestSolution() *Solution {
	psa.mutex.RLock()
	defer psa.mutex.RUnlock()
	
	if psa.globalBest == nil {
		return nil
	}
	return psa.globalBest.Copy()
}

// GetStatistics returns current algorithm statistics
func (psa *ParallelSimulatedAnnealing) GetStatistics() Statistics {
	psa.statistics.mutex.RLock()
	defer psa.statistics.mutex.RUnlock()
	
	stats := *psa.statistics
	stats.BestSolution = psa.globalBest.Copy()
	return stats
}

// Stop stops the optimization process
func (psa *ParallelSimulatedAnnealing) Stop() {
	psa.cancel()
}

// Placeholder implementations for remaining functionality
func (psa *ParallelSimulatedAnnealing) handleMigration(chain *Chain, migrant *Solution) {
	// Implementation would handle solution migration between islands
}

func (psa *ParallelSimulatedAnnealing) sendMigrant(chain *Chain) {
	// Implementation would send best solution to exchange channel
	select {
	case psa.exchangeChan <- chain.Best.Copy():
	default:
	}
}

func (psa *ParallelSimulatedAnnealing) restartChain(chain *Chain) {
	// Implementation would restart chain with new random solution
	atomic.AddInt64(&psa.statistics.RestartCount, 1)
}

func (psa *ParallelSimulatedAnnealing) performSolutionExchange() {
	// Implementation would exchange solutions between chains
	atomic.AddInt64(&psa.statistics.ExchangeCount, 1)
}

func (psa *ParallelSimulatedAnnealing) updateStatistics(chain *Chain) {
	// Implementation would update algorithm statistics
}

func (psa *ParallelSimulatedAnnealing) finalizeStatistics() {
	// Implementation would finalize and aggregate statistics
}