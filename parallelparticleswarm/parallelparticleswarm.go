package parallelparticleswarm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// OptimizationProblem defines different types of optimization problems
type OptimizationProblem int

const (
	Minimize OptimizationProblem = iota
	Maximize
)

// PSO Algorithm Variants
type PSOVariant int

const (
	StandardPSO PSOVariant = iota
	InertiaWeightPSO
	ConstrictionPSO
	AdaptivePSO
	QuantumPSO
	BinaryPSO
	MultiObjectivePSO
)

// Topology defines swarm communication topology
type Topology int

const (
	GlobalTopology Topology = iota
	RingTopology
	StarTopology
	MeshTopology
	VonNeumannTopology
	RandomTopology
)

// PSOConfig contains configuration for the PSO algorithm
type PSOConfig struct {
	SwarmSize           int
	MaxIterations       int
	Dimensions          int
	MinBounds           []float64
	MaxBounds           []float64
	InertiaWeight       float64
	CognitiveWeight     float64
	SocialWeight        float64
	MaxVelocity         float64
	Problem             OptimizationProblem
	Variant             PSOVariant
	Topology            Topology
	NumWorkers          int
	UseParallelEval     bool
	UseAsyncUpdate      bool
	ConvergenceThreshold float64
	MaxStagnation       int
	AdaptiveWeights     bool
	DiversityMaintain   bool
	EliteSize           int
	MutationRate        float64
	EnableStatistics    bool
	SeedValue           int64
}

// DefaultPSOConfig returns default PSO configuration
func DefaultPSOConfig() PSOConfig {
	return PSOConfig{
		SwarmSize:           50,
		MaxIterations:       1000,
		Dimensions:          10,
		InertiaWeight:       0.9,
		CognitiveWeight:     2.0,
		SocialWeight:        2.0,
		MaxVelocity:         4.0,
		Problem:             Minimize,
		Variant:             StandardPSO,
		Topology:            GlobalTopology,
		NumWorkers:          4,
		UseParallelEval:     true,
		UseAsyncUpdate:      false,
		ConvergenceThreshold: 1e-8,
		MaxStagnation:       50,
		AdaptiveWeights:     false,
		DiversityMaintain:   false,
		EliteSize:           5,
		MutationRate:        0.01,
		EnableStatistics:    true,
		SeedValue:           time.Now().UnixNano(),
	}
}

// Particle represents a particle in the swarm
type Particle struct {
	ID               int
	Position         []float64
	Velocity         []float64
	BestPosition     []float64
	BestFitness      float64
	CurrentFitness   float64
	NeighborBest     []float64
	NeighborFitness  float64
	ImprovementCount int
	StagnationCount  int
	LastUpdate       time.Time
	Active           bool
	mutex            sync.RWMutex
}

// Swarm represents a collection of particles
type Swarm struct {
	Particles       []*Particle
	GlobalBest      []float64
	GlobalFitness   float64
	Iteration       int
	Topology        Topology
	NeighborMatrix  [][]int
	Elite           []*Particle
	Diversity       float64
	Converged       bool
	StagnationCount int
	mutex           sync.RWMutex
}

// ObjectiveFunction defines the function to optimize
type ObjectiveFunction func(position []float64) float64

// MultiObjectiveFunction defines multi-objective function
type MultiObjectiveFunction func(position []float64) []float64

// ConstraintFunction defines constraint validation
type ConstraintFunction func(position []float64) bool

// Optimizer represents the PSO optimizer
type Optimizer struct {
	config           PSOConfig
	swarm            *Swarm
	objective        ObjectiveFunction
	multiObjective   MultiObjectiveFunction
	constraints      []ConstraintFunction
	workers          []*Worker
	taskQueue        chan Task
	resultQueue      chan Result
	statistics       *Statistics
	convergenceData  *ConvergenceData
	diversityManager *DiversityManager
	adaptiveManager  *AdaptiveManager
	ctx              context.Context
	cancel           context.CancelFunc
	wg               sync.WaitGroup
	running          bool
	mutex            sync.RWMutex
}

// Worker represents a worker for parallel evaluation
type Worker struct {
	id          int
	optimizer   *Optimizer
	taskQueue   chan Task
	resultQueue chan Result
	ctx         context.Context
}

// Task represents a task for worker processing
type Task struct {
	Type      string
	ParticleID int
	Position  []float64
	Data      map[string]interface{}
}

// Result represents the result of a task
type Result struct {
	ParticleID int
	Fitness    float64
	Success    bool
	Error      error
	Data       map[string]interface{}
}

// Statistics tracks PSO optimization metrics
type Statistics struct {
	StartTime           time.Time
	EndTime             time.Time
	TotalIterations     int
	FunctionEvaluations int64
	BestFitness         float64
	WorstFitness        float64
	AverageFitness      float64
	ConvergenceRate     float64
	DiversityHistory    []float64
	FitnessHistory      []float64
	VelocityStats       *VelocityStatistics
	ParticleStats       *ParticleStatistics
	TopologyStats       *TopologyStatistics
	PerformanceMetrics  *PerformanceMetrics
	mutex               sync.RWMutex
}

// VelocityStatistics tracks velocity metrics
type VelocityStatistics struct {
	Average    float64
	Maximum    float64
	Minimum    float64
	StdDev     float64
	Clamped    int64
	ZeroCount  int64
}

// ParticleStatistics tracks particle metrics
type ParticleStatistics struct {
	Improvements     int64
	Stagnations      int64
	Repositions      int64
	BoundaryHits     int64
	DiversityLoss    int64
	EliteChanges     int64
}

// TopologyStatistics tracks topology metrics
type TopologyStatistics struct {
	ConnectionChanges int64
	MessagesPassed    int64
	InfluenceSpread   float64
	ClusteringIndex   float64
}

// PerformanceMetrics tracks performance metrics
type PerformanceMetrics struct {
	EvaluationTime    time.Duration
	UpdateTime        time.Duration
	CommunicationTime time.Duration
	MemoryUsage       int64
	CPUUtilization    float64
	ParallelEfficiency float64
}

// ConvergenceData tracks convergence information
type ConvergenceData struct {
	Threshold        float64
	CurrentValue     float64
	StagnationCount  int
	MaxStagnation    int
	ConvergedAt      int
	ConvergenceSpeed float64
	Plateau          bool
	LastImprovement  time.Time
}

// DiversityManager manages swarm diversity
type DiversityManager struct {
	Threshold       float64
	Current         float64
	History         []float64
	WindowSize      int
	MaintainActive  bool
	DiversityLoss   int
	LastDiversify   time.Time
	mutex           sync.RWMutex
}

// AdaptiveManager manages adaptive parameters
type AdaptiveManager struct {
	InertiaWeight     float64
	CognitiveWeight   float64
	SocialWeight      float64
	MaxVelocity       float64
	AdaptationRate    float64
	PerformanceWindow []float64
	WindowSize        int
	LastAdaptation    time.Time
	mutex             sync.RWMutex
}

// BenchmarkFunction represents a benchmark optimization function
type BenchmarkFunction struct {
	Name        string
	Function    ObjectiveFunction
	Dimensions  int
	MinBounds   []float64
	MaxBounds   []float64
	GlobalMin   []float64
	GlobalValue float64
	Description string
}

// NewOptimizer creates a new PSO optimizer
func NewOptimizer(config PSOConfig, objective ObjectiveFunction) *Optimizer {
	ctx, cancel := context.WithCancel(context.Background())
	
	optimizer := &Optimizer{
		config:          config,
		objective:       objective,
		constraints:     make([]ConstraintFunction, 0),
		taskQueue:       make(chan Task, config.NumWorkers*10),
		resultQueue:     make(chan Result, config.NumWorkers*10),
		statistics:      NewStatistics(),
		convergenceData: NewConvergenceData(config),
		ctx:             ctx,
		cancel:          cancel,
		running:         true,
	}
	
	// Initialize swarm
	optimizer.swarm = NewSwarm(config)
	
	// Initialize managers
	if config.DiversityMaintain {
		optimizer.diversityManager = NewDiversityManager(0.1)
	}
	
	if config.AdaptiveWeights {
		optimizer.adaptiveManager = NewAdaptiveManager(config)
	}
	
	// Initialize workers
	optimizer.initializeWorkers()
	
	return optimizer
}

// NewSwarm creates a new swarm
func NewSwarm(config PSOConfig) *Swarm {
	swarm := &Swarm{
		Particles:      make([]*Particle, config.SwarmSize),
		GlobalBest:     make([]float64, config.Dimensions),
		GlobalFitness:  getWorstFitness(config.Problem),
		Topology:       config.Topology,
		Elite:          make([]*Particle, 0, config.EliteSize),
		Converged:      false,
	}
	
	// Initialize particles
	for i := 0; i < config.SwarmSize; i++ {
		swarm.Particles[i] = NewParticle(i, config)
	}
	
	// Initialize topology
	swarm.initializeTopology(config)
	
	return swarm
}

// NewParticle creates a new particle
func NewParticle(id int, config PSOConfig) *Particle {
	particle := &Particle{
		ID:              id,
		Position:        make([]float64, config.Dimensions),
		Velocity:        make([]float64, config.Dimensions),
		BestPosition:    make([]float64, config.Dimensions),
		BestFitness:     getWorstFitness(config.Problem),
		CurrentFitness:  getWorstFitness(config.Problem),
		NeighborBest:    make([]float64, config.Dimensions),
		NeighborFitness: getWorstFitness(config.Problem),
		Active:          true,
		LastUpdate:      time.Now(),
	}
	
	// Initialize random position
	for i := 0; i < config.Dimensions; i++ {
		particle.Position[i] = config.MinBounds[i] + 
			rand.Float64()*(config.MaxBounds[i]-config.MinBounds[i])
		particle.BestPosition[i] = particle.Position[i]
		
		// Initialize random velocity
		maxVel := config.MaxVelocity
		particle.Velocity[i] = (rand.Float64()*2 - 1) * maxVel
	}
	
	copy(particle.NeighborBest, particle.BestPosition)
	
	return particle
}

// initializeTopology sets up the swarm topology
func (s *Swarm) initializeTopology(config PSOConfig) {
	size := len(s.Particles)
	s.NeighborMatrix = make([][]int, size)
	
	for i := 0; i < size; i++ {
		s.NeighborMatrix[i] = make([]int, 0)
	}
	
	switch config.Topology {
	case GlobalTopology:
		// All particles connected to all others
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				if i != j {
					s.NeighborMatrix[i] = append(s.NeighborMatrix[i], j)
				}
			}
		}
		
	case RingTopology:
		// Each particle connected to immediate neighbors
		for i := 0; i < size; i++ {
			s.NeighborMatrix[i] = append(s.NeighborMatrix[i], (i+1)%size)
			s.NeighborMatrix[i] = append(s.NeighborMatrix[i], (i-1+size)%size)
		}
		
	case StarTopology:
		// One central particle connected to all others
		center := 0
		for i := 1; i < size; i++ {
			s.NeighborMatrix[center] = append(s.NeighborMatrix[center], i)
			s.NeighborMatrix[i] = append(s.NeighborMatrix[i], center)
		}
		
	case VonNeumannTopology:
		// Grid-like neighborhood
		gridSize := int(math.Sqrt(float64(size)))
		for i := 0; i < size; i++ {
			row := i / gridSize
			col := i % gridSize
			
			// Add neighbors (up, down, left, right)
			neighbors := [][]int{
				{(row-1+gridSize)%gridSize, col},
				{(row+1)%gridSize, col},
				{row, (col-1+gridSize)%gridSize},
				{row, (col+1)%gridSize},
			}
			
			for _, neighbor := range neighbors {
				neighborID := neighbor[0]*gridSize + neighbor[1]
				if neighborID < size {
					s.NeighborMatrix[i] = append(s.NeighborMatrix[i], neighborID)
				}
			}
		}
		
	case RandomTopology:
		// Random connections
		connectionsPerParticle := max(2, size/10)
		for i := 0; i < size; i++ {
			connections := 0
			for connections < connectionsPerParticle {
				neighbor := rand.Intn(size)
				if neighbor != i {
					found := false
					for _, existing := range s.NeighborMatrix[i] {
						if existing == neighbor {
							found = true
							break
						}
					}
					if !found {
						s.NeighborMatrix[i] = append(s.NeighborMatrix[i], neighbor)
						connections++
					}
				}
			}
		}
	}
}

// initializeWorkers creates and starts worker goroutines
func (o *Optimizer) initializeWorkers() {
	o.workers = make([]*Worker, o.config.NumWorkers)
	
	for i := 0; i < o.config.NumWorkers; i++ {
		worker := &Worker{
			id:          i,
			optimizer:   o,
			taskQueue:   o.taskQueue,
			resultQueue: o.resultQueue,
			ctx:         o.ctx,
		}
		o.workers[i] = worker
		
		o.wg.Add(1)
		go worker.start(&o.wg)
	}
}

// start starts the worker
func (w *Worker) start(wg *sync.WaitGroup) {
	defer wg.Done()
	
	for {
		select {
		case task := <-w.taskQueue:
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

// processTask processes a task
func (w *Worker) processTask(task Task) Result {
	switch task.Type {
	case "evaluate":
		return w.evaluateParticle(task)
	case "update":
		return w.updateParticle(task)
	default:
		return Result{
			ParticleID: task.ParticleID,
			Success:    false,
			Error:      fmt.Errorf("unknown task type: %s", task.Type),
		}
	}
}

// evaluateParticle evaluates a particle's fitness
func (w *Worker) evaluateParticle(task Task) Result {
	start := time.Now()
	
	// Evaluate objective function
	fitness := w.optimizer.objective(task.Position)
	
	// Apply constraints
	for _, constraint := range w.optimizer.constraints {
		if !constraint(task.Position) {
			// Penalize constraint violations
			if w.optimizer.config.Problem == Minimize {
				fitness = math.Inf(1)
			} else {
				fitness = math.Inf(-1)
			}
			break
		}
	}
	
	duration := time.Since(start)
	
	return Result{
		ParticleID: task.ParticleID,
		Fitness:    fitness,
		Success:    true,
		Data: map[string]interface{}{
			"evaluation_time": duration,
		},
	}
}

// updateParticle updates a particle's position and velocity
func (w *Worker) updateParticle(task Task) Result {
	particleID := task.ParticleID
	if particleID >= len(w.optimizer.swarm.Particles) {
		return Result{
			ParticleID: particleID,
			Success:    false,
			Error:      fmt.Errorf("invalid particle ID: %d", particleID),
		}
	}
	
	particle := w.optimizer.swarm.Particles[particleID]
	config := w.optimizer.config
	
	particle.mutex.Lock()
	defer particle.mutex.Unlock()
	
	// Get parameters
	inertia := config.InertiaWeight
	cognitive := config.CognitiveWeight
	social := config.SocialWeight
	
	// Adaptive parameters
	if w.optimizer.adaptiveManager != nil {
		w.optimizer.adaptiveManager.mutex.RLock()
		inertia = w.optimizer.adaptiveManager.InertiaWeight
		cognitive = w.optimizer.adaptiveManager.CognitiveWeight
		social = w.optimizer.adaptiveManager.SocialWeight
		w.optimizer.adaptiveManager.mutex.RUnlock()
	}
	
	// Update velocity and position
	for i := 0; i < config.Dimensions; i++ {
		r1 := rand.Float64()
		r2 := rand.Float64()
		
		// Standard PSO velocity update
		newVelocity := inertia*particle.Velocity[i] +
			cognitive*r1*(particle.BestPosition[i]-particle.Position[i]) +
			social*r2*(particle.NeighborBest[i]-particle.Position[i])
		
		// Apply velocity clamping
		maxVel := config.MaxVelocity
		if w.optimizer.adaptiveManager != nil {
			w.optimizer.adaptiveManager.mutex.RLock()
			maxVel = w.optimizer.adaptiveManager.MaxVelocity
			w.optimizer.adaptiveManager.mutex.RUnlock()
		}
		
		if newVelocity > maxVel {
			newVelocity = maxVel
			atomic.AddInt64(&w.optimizer.statistics.VelocityStats.Clamped, 1)
		} else if newVelocity < -maxVel {
			newVelocity = -maxVel
			atomic.AddInt64(&w.optimizer.statistics.VelocityStats.Clamped, 1)
		}
		
		particle.Velocity[i] = newVelocity
		
		// Update position
		particle.Position[i] += particle.Velocity[i]
		
		// Apply boundary constraints
		if particle.Position[i] < config.MinBounds[i] {
			particle.Position[i] = config.MinBounds[i]
			particle.Velocity[i] = -particle.Velocity[i] * 0.5 // Bounce back
			atomic.AddInt64(&w.optimizer.statistics.ParticleStats.BoundaryHits, 1)
		} else if particle.Position[i] > config.MaxBounds[i] {
			particle.Position[i] = config.MaxBounds[i]
			particle.Velocity[i] = -particle.Velocity[i] * 0.5 // Bounce back
			atomic.AddInt64(&w.optimizer.statistics.ParticleStats.BoundaryHits, 1)
		}
	}
	
	particle.LastUpdate = time.Now()
	
	return Result{
		ParticleID: particleID,
		Success:    true,
	}
}

// Optimize runs the PSO optimization
func (o *Optimizer) Optimize() (*Particle, error) {
	if !o.running {
		return nil, errors.New("optimizer is not running")
	}
	
	o.statistics.StartTime = time.Now()
	defer func() {
		o.statistics.EndTime = time.Now()
	}()
	
	// Initial evaluation
	o.evaluateSwarm()
	o.updateStatistics()
	
	for iteration := 0; iteration < o.config.MaxIterations; iteration++ {
		o.swarm.Iteration = iteration
		
		// Check for convergence
		if o.checkConvergence() {
			o.swarm.Converged = true
			o.convergenceData.ConvergedAt = iteration
			break
		}
		
		// Update particles
		if o.config.UseParallelEval {
			o.updateSwarmParallel()
		} else {
			o.updateSwarmSequential()
		}
		
		// Update neighborhood best
		o.updateNeighborhoodBest()
		
		// Update global best
		o.updateGlobalBest()
		
		// Maintain diversity if enabled
		if o.config.DiversityMaintain && o.diversityManager != nil {
			o.maintainDiversity()
		}
		
		// Adaptive parameter adjustment
		if o.config.AdaptiveWeights && o.adaptiveManager != nil {
			o.adaptiveManager.updateParameters(o.statistics)
		}
		
		// Update statistics
		o.updateStatistics()
		
		// Check for early stopping
		if o.shouldStop() {
			break
		}
		
		// Check for context cancellation
		select {
		case <-o.ctx.Done():
			return nil, errors.New("optimization cancelled")
		default:
		}
	}
	
	// Find best particle
	bestParticle := o.getBestParticle()
	
	return bestParticle, nil
}

// evaluateSwarm evaluates all particles in the swarm
func (o *Optimizer) evaluateSwarm() {
	if o.config.UseParallelEval {
		o.evaluateSwarmParallel()
	} else {
		o.evaluateSwarmSequential()
	}
}

// evaluateSwarmParallel evaluates particles in parallel
func (o *Optimizer) evaluateSwarmParallel() {
	// Send evaluation tasks
	for i, particle := range o.swarm.Particles {
		o.taskQueue <- Task{
			Type:       "evaluate",
			ParticleID: i,
			Position:   append([]float64(nil), particle.Position...),
		}
	}
	
	// Collect results
	for i := 0; i < len(o.swarm.Particles); i++ {
		select {
		case result := <-o.resultQueue:
			if result.Success {
				particle := o.swarm.Particles[result.ParticleID]
				particle.mutex.Lock()
				particle.CurrentFitness = result.Fitness
				
				// Update personal best
				if o.isBetter(result.Fitness, particle.BestFitness) {
					particle.BestFitness = result.Fitness
					copy(particle.BestPosition, particle.Position)
					particle.ImprovementCount++
					particle.StagnationCount = 0
					atomic.AddInt64(&o.statistics.ParticleStats.Improvements, 1)
				} else {
					particle.StagnationCount++
					atomic.AddInt64(&o.statistics.ParticleStats.Stagnations, 1)
				}
				particle.mutex.Unlock()
				
				atomic.AddInt64(&o.statistics.FunctionEvaluations, 1)
			}
		case <-o.ctx.Done():
			return
		}
	}
}

// evaluateSwarmSequential evaluates particles sequentially
func (o *Optimizer) evaluateSwarmSequential() {
	for _, particle := range o.swarm.Particles {
		particle.mutex.Lock()
		fitness := o.objective(particle.Position)
		particle.CurrentFitness = fitness
		
		// Update personal best
		if o.isBetter(fitness, particle.BestFitness) {
			particle.BestFitness = fitness
			copy(particle.BestPosition, particle.Position)
			particle.ImprovementCount++
			particle.StagnationCount = 0
			atomic.AddInt64(&o.statistics.ParticleStats.Improvements, 1)
		} else {
			particle.StagnationCount++
			atomic.AddInt64(&o.statistics.ParticleStats.Stagnations, 1)
		}
		particle.mutex.Unlock()
		
		atomic.AddInt64(&o.statistics.FunctionEvaluations, 1)
	}
}

// updateSwarmParallel updates particles in parallel
func (o *Optimizer) updateSwarmParallel() {
	// Send update tasks
	for i := range o.swarm.Particles {
		o.taskQueue <- Task{
			Type:       "update",
			ParticleID: i,
		}
	}
	
	// Collect results
	for i := 0; i < len(o.swarm.Particles); i++ {
		select {
		case result := <-o.resultQueue:
			if !result.Success {
				// Handle update error
				continue
			}
		case <-o.ctx.Done():
			return
		}
	}
	
	// Re-evaluate after position updates
	o.evaluateSwarmParallel()
}

// updateSwarmSequential updates particles sequentially
func (o *Optimizer) updateSwarmSequential() {
	for i, particle := range o.swarm.Particles {
		// Update particle
		task := Task{
			Type:       "update",
			ParticleID: i,
		}
		
		worker := o.workers[0] // Use first worker for sequential processing
		worker.processTask(task)
		
		// Re-evaluate
		fitness := o.objective(particle.Position)
		particle.mutex.Lock()
		particle.CurrentFitness = fitness
		
		// Update personal best
		if o.isBetter(fitness, particle.BestFitness) {
			particle.BestFitness = fitness
			copy(particle.BestPosition, particle.Position)
			particle.ImprovementCount++
			particle.StagnationCount = 0
			atomic.AddInt64(&o.statistics.ParticleStats.Improvements, 1)
		} else {
			particle.StagnationCount++
			atomic.AddInt64(&o.statistics.ParticleStats.Stagnations, 1)
		}
		particle.mutex.Unlock()
		
		atomic.AddInt64(&o.statistics.FunctionEvaluations, 1)
	}
}

// updateNeighborhoodBest updates neighborhood best for each particle
func (o *Optimizer) updateNeighborhoodBest() {
	for i, particle := range o.swarm.Particles {
		bestFitness := particle.BestFitness
		bestPosition := particle.BestPosition
		
		// Check neighbors
		for _, neighborID := range o.swarm.NeighborMatrix[i] {
			neighbor := o.swarm.Particles[neighborID]
			neighbor.mutex.RLock()
			if o.isBetter(neighbor.BestFitness, bestFitness) {
				bestFitness = neighbor.BestFitness
				bestPosition = neighbor.BestPosition
			}
			neighbor.mutex.RUnlock()
		}
		
		particle.mutex.Lock()
		particle.NeighborFitness = bestFitness
		copy(particle.NeighborBest, bestPosition)
		particle.mutex.Unlock()
	}
}

// updateGlobalBest updates the global best solution
func (o *Optimizer) updateGlobalBest() {
	o.swarm.mutex.Lock()
	defer o.swarm.mutex.Unlock()
	
	improved := false
	
	for _, particle := range o.swarm.Particles {
		particle.mutex.RLock()
		if o.isBetter(particle.BestFitness, o.swarm.GlobalFitness) {
			o.swarm.GlobalFitness = particle.BestFitness
			copy(o.swarm.GlobalBest, particle.BestPosition)
			improved = true
			o.swarm.StagnationCount = 0
		}
		particle.mutex.RUnlock()
	}
	
	if !improved {
		o.swarm.StagnationCount++
	}
}

// maintainDiversity maintains swarm diversity
func (o *Optimizer) maintainDiversity() {
	if o.diversityManager == nil {
		return
	}
	
	o.diversityManager.mutex.Lock()
	defer o.diversityManager.mutex.Unlock()
	
	// Calculate current diversity
	diversity := o.calculateDiversity()
	o.diversityManager.Current = diversity
	o.diversityManager.History = append(o.diversityManager.History, diversity)
	
	// Keep history window
	if len(o.diversityManager.History) > o.diversityManager.WindowSize {
		o.diversityManager.History = o.diversityManager.History[1:]
	}
	
	// Check if diversity is too low
	if diversity < o.diversityManager.Threshold {
		o.diversityManager.DiversityLoss++
		
		// Apply diversity maintenance strategy
		o.applyDiversityMaintenance()
		o.diversityManager.LastDiversify = time.Now()
	}
}

// calculateDiversity calculates swarm diversity
func (o *Optimizer) calculateDiversity() float64 {
	if len(o.swarm.Particles) < 2 {
		return 0.0
	}
	
	totalDistance := 0.0
	comparisons := 0
	
	for i := 0; i < len(o.swarm.Particles); i++ {
		for j := i + 1; j < len(o.swarm.Particles); j++ {
			distance := o.euclideanDistance(
				o.swarm.Particles[i].Position,
				o.swarm.Particles[j].Position,
			)
			totalDistance += distance
			comparisons++
		}
	}
	
	if comparisons == 0 {
		return 0.0
	}
	
	return totalDistance / float64(comparisons)
}

// applyDiversityMaintenance applies diversity maintenance strategies
func (o *Optimizer) applyDiversityMaintenance() {
	// Strategy 1: Reinitialize worst particles
	numToReinit := max(1, len(o.swarm.Particles)/10)
	
	// Sort particles by fitness (worst first)
	particles := make([]*Particle, len(o.swarm.Particles))
	copy(particles, o.swarm.Particles)
	
	sort.Slice(particles, func(i, j int) bool {
		particles[i].mutex.RLock()
		particles[j].mutex.RLock()
		defer particles[i].mutex.RUnlock()
		defer particles[j].mutex.RUnlock()
		
		if o.config.Problem == Minimize {
			return particles[i].CurrentFitness > particles[j].CurrentFitness
		}
		return particles[i].CurrentFitness < particles[j].CurrentFitness
	})
	
	// Reinitialize worst particles
	for i := 0; i < numToReinit; i++ {
		particle := particles[i]
		particle.mutex.Lock()
		
		// Reinitialize position
		for j := 0; j < o.config.Dimensions; j++ {
			particle.Position[j] = o.config.MinBounds[j] + 
				rand.Float64()*(o.config.MaxBounds[j]-o.config.MinBounds[j])
			
			// Reset velocity
			maxVel := o.config.MaxVelocity
			particle.Velocity[j] = (rand.Float64()*2 - 1) * maxVel
		}
		
		particle.mutex.Unlock()
		atomic.AddInt64(&o.statistics.ParticleStats.Repositions, 1)
	}
}

// checkConvergence checks if the swarm has converged
func (o *Optimizer) checkConvergence() bool {
	o.convergenceData.CurrentValue = o.swarm.GlobalFitness
	
	// Check fitness threshold
	if math.Abs(o.convergenceData.CurrentValue) < o.convergenceData.Threshold {
		return true
	}
	
	// Check stagnation
	if o.swarm.StagnationCount >= o.convergenceData.MaxStagnation {
		o.convergenceData.Plateau = true
		return true
	}
	
	return false
}

// shouldStop checks if optimization should stop early
func (o *Optimizer) shouldStop() bool {
	// Check maximum stagnation
	if o.swarm.StagnationCount >= o.config.MaxStagnation {
		return true
	}
	
	// Check diversity loss
	if o.diversityManager != nil {
		o.diversityManager.mutex.RLock()
		diversityLoss := o.diversityManager.DiversityLoss
		o.diversityManager.mutex.RUnlock()
		
		if diversityLoss > o.config.MaxStagnation/2 {
			return true
		}
	}
	
	return false
}

// updateStatistics updates optimization statistics
func (o *Optimizer) updateStatistics() {
	o.statistics.mutex.Lock()
	defer o.statistics.mutex.Unlock()
	
	o.statistics.TotalIterations = o.swarm.Iteration + 1
	o.statistics.BestFitness = o.swarm.GlobalFitness
	
	// Calculate fitness statistics
	totalFitness := 0.0
	worstFitness := o.swarm.GlobalFitness
	
	for _, particle := range o.swarm.Particles {
		particle.mutex.RLock()
		fitness := particle.CurrentFitness
		totalFitness += fitness
		
		if o.config.Problem == Minimize {
			if fitness > worstFitness {
				worstFitness = fitness
			}
		} else {
			if fitness < worstFitness {
				worstFitness = fitness
			}
		}
		particle.mutex.RUnlock()
	}
	
	o.statistics.AverageFitness = totalFitness / float64(len(o.swarm.Particles))
	o.statistics.WorstFitness = worstFitness
	
	// Update fitness history
	o.statistics.FitnessHistory = append(o.statistics.FitnessHistory, o.statistics.BestFitness)
	
	// Update diversity history
	if o.diversityManager != nil {
		o.diversityManager.mutex.RLock()
		o.statistics.DiversityHistory = append(o.statistics.DiversityHistory, o.diversityManager.Current)
		o.diversityManager.mutex.RUnlock()
	}
	
	// Update velocity statistics
	o.updateVelocityStatistics()
}

// updateVelocityStatistics updates velocity statistics
func (o *Optimizer) updateVelocityStatistics() {
	velocities := make([]float64, 0)
	zeroCount := int64(0)
	
	for _, particle := range o.swarm.Particles {
		particle.mutex.RLock()
		for _, vel := range particle.Velocity {
			velocities = append(velocities, math.Abs(vel))
			if math.Abs(vel) < 1e-10 {
				zeroCount++
			}
		}
		particle.mutex.RUnlock()
	}
	
	if len(velocities) == 0 {
		return
	}
	
	// Calculate statistics
	total := 0.0
	minVel := velocities[0]
	maxVel := velocities[0]
	
	for _, vel := range velocities {
		total += vel
		if vel < minVel {
			minVel = vel
		}
		if vel > maxVel {
			maxVel = vel
		}
	}
	
	avgVel := total / float64(len(velocities))
	
	// Calculate standard deviation
	variance := 0.0
	for _, vel := range velocities {
		variance += (vel - avgVel) * (vel - avgVel)
	}
	stdDev := math.Sqrt(variance / float64(len(velocities)))
	
	o.statistics.VelocityStats.Average = avgVel
	o.statistics.VelocityStats.Minimum = minVel
	o.statistics.VelocityStats.Maximum = maxVel
	o.statistics.VelocityStats.StdDev = stdDev
	o.statistics.VelocityStats.ZeroCount = zeroCount
}

// getBestParticle returns the best particle
func (o *Optimizer) getBestParticle() *Particle {
	bestParticle := &Particle{
		Position:    make([]float64, len(o.swarm.GlobalBest)),
		BestFitness: o.swarm.GlobalFitness,
	}
	
	copy(bestParticle.Position, o.swarm.GlobalBest)
	copy(bestParticle.BestPosition, o.swarm.GlobalBest)
	
	return bestParticle
}

// isBetter checks if fitness1 is better than fitness2
func (o *Optimizer) isBetter(fitness1, fitness2 float64) bool {
	if o.config.Problem == Minimize {
		return fitness1 < fitness2
	}
	return fitness1 > fitness2
}

// euclideanDistance calculates Euclidean distance between two positions
func (o *Optimizer) euclideanDistance(pos1, pos2 []float64) float64 {
	if len(pos1) != len(pos2) {
		return 0.0
	}
	
	sum := 0.0
	for i := 0; i < len(pos1); i++ {
		diff := pos1[i] - pos2[i]
		sum += diff * diff
	}
	
	return math.Sqrt(sum)
}

// AddConstraint adds a constraint function
func (o *Optimizer) AddConstraint(constraint ConstraintFunction) {
	o.mutex.Lock()
	defer o.mutex.Unlock()
	
	o.constraints = append(o.constraints, constraint)
}

// GetStatistics returns current statistics
func (o *Optimizer) GetStatistics() *Statistics {
	o.statistics.mutex.RLock()
	defer o.statistics.mutex.RUnlock()
	
	stats := *o.statistics
	return &stats
}

// GetSwarm returns current swarm state
func (o *Optimizer) GetSwarm() *Swarm {
	o.swarm.mutex.RLock()
	defer o.swarm.mutex.RUnlock()
	
	swarmCopy := *o.swarm
	return &swarmCopy
}

// Shutdown gracefully shuts down the optimizer
func (o *Optimizer) Shutdown() error {
	if !o.running {
		return errors.New("optimizer is not running")
	}
	
	o.running = false
	o.cancel()
	
	// Wait for workers to finish
	o.wg.Wait()
	
	return nil
}

// Helper functions and constructors

// NewStatistics creates new statistics instance
func NewStatistics() *Statistics {
	return &Statistics{
		BestFitness:        math.Inf(1),
		WorstFitness:       math.Inf(-1),
		FitnessHistory:     make([]float64, 0),
		DiversityHistory:   make([]float64, 0),
		VelocityStats:      &VelocityStatistics{},
		ParticleStats:      &ParticleStatistics{},
		TopologyStats:      &TopologyStatistics{},
		PerformanceMetrics: &PerformanceMetrics{},
	}
}

// NewConvergenceData creates new convergence data
func NewConvergenceData(config PSOConfig) *ConvergenceData {
	return &ConvergenceData{
		Threshold:       config.ConvergenceThreshold,
		MaxStagnation:   config.MaxStagnation,
		LastImprovement: time.Now(),
	}
}

// NewDiversityManager creates new diversity manager
func NewDiversityManager(threshold float64) *DiversityManager {
	return &DiversityManager{
		Threshold:     threshold,
		History:       make([]float64, 0),
		WindowSize:    50,
		MaintainActive: true,
		LastDiversify: time.Now(),
	}
}

// NewAdaptiveManager creates new adaptive manager
func NewAdaptiveManager(config PSOConfig) *AdaptiveManager {
	return &AdaptiveManager{
		InertiaWeight:     config.InertiaWeight,
		CognitiveWeight:   config.CognitiveWeight,
		SocialWeight:      config.SocialWeight,
		MaxVelocity:       config.MaxVelocity,
		AdaptationRate:    0.1,
		PerformanceWindow: make([]float64, 0),
		WindowSize:        20,
		LastAdaptation:    time.Now(),
	}
}

// updateParameters updates adaptive parameters
func (am *AdaptiveManager) updateParameters(stats *Statistics) {
	am.mutex.Lock()
	defer am.mutex.Unlock()
	
	// Add current performance to window
	am.PerformanceWindow = append(am.PerformanceWindow, stats.BestFitness)
	
	// Keep window size
	if len(am.PerformanceWindow) > am.WindowSize {
		am.PerformanceWindow = am.PerformanceWindow[1:]
	}
	
	// Calculate performance trend
	if len(am.PerformanceWindow) >= am.WindowSize {
		trend := am.calculateTrend()
		
		// Adjust parameters based on trend
		if trend < 0 { // Improving
			// Increase exploitation
			am.InertiaWeight = math.Max(0.1, am.InertiaWeight-am.AdaptationRate*0.1)
			am.SocialWeight = math.Min(4.0, am.SocialWeight+am.AdaptationRate*0.5)
		} else { // Stagnating
			// Increase exploration
			am.InertiaWeight = math.Min(0.9, am.InertiaWeight+am.AdaptationRate*0.1)
			am.CognitiveWeight = math.Min(4.0, am.CognitiveWeight+am.AdaptationRate*0.3)
		}
		
		am.LastAdaptation = time.Now()
	}
}

// calculateTrend calculates performance trend
func (am *AdaptiveManager) calculateTrend() float64 {
	if len(am.PerformanceWindow) < 2 {
		return 0.0
	}
	
	n := len(am.PerformanceWindow)
	recent := am.PerformanceWindow[n/2:]
	early := am.PerformanceWindow[:n/2]
	
	recentAvg := 0.0
	for _, val := range recent {
		recentAvg += val
	}
	recentAvg /= float64(len(recent))
	
	earlyAvg := 0.0
	for _, val := range early {
		earlyAvg += val
	}
	earlyAvg /= float64(len(early))
	
	return recentAvg - earlyAvg
}

// getWorstFitness returns worst possible fitness value
func getWorstFitness(problem OptimizationProblem) float64 {
	if problem == Minimize {
		return math.Inf(1)
	}
	return math.Inf(-1)
}

// max returns maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// min returns minimum of two values
func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// Benchmark functions

// GetBenchmarkFunctions returns a collection of benchmark optimization functions
func GetBenchmarkFunctions() map[string]BenchmarkFunction {
	return map[string]BenchmarkFunction{
		"sphere": {
			Name:        "Sphere Function",
			Function:    SphereFunction,
			Dimensions:  10,
			MinBounds:   repeatFloat(-5.12, 10),
			MaxBounds:   repeatFloat(5.12, 10),
			GlobalMin:   repeatFloat(0.0, 10),
			GlobalValue: 0.0,
			Description: "Simple unimodal function",
		},
		"rastrigin": {
			Name:        "Rastrigin Function",
			Function:    RastriginFunction,
			Dimensions:  10,
			MinBounds:   repeatFloat(-5.12, 10),
			MaxBounds:   repeatFloat(5.12, 10),
			GlobalMin:   repeatFloat(0.0, 10),
			GlobalValue: 0.0,
			Description: "Highly multimodal function",
		},
		"rosenbrock": {
			Name:        "Rosenbrock Function",
			Function:    RosenbrockFunction,
			Dimensions:  10,
			MinBounds:   repeatFloat(-2.048, 10),
			MaxBounds:   repeatFloat(2.048, 10),
			GlobalMin:   repeatFloat(1.0, 10),
			GlobalValue: 0.0,
			Description: "Valley-shaped function",
		},
		"ackley": {
			Name:        "Ackley Function",
			Function:    AckleyFunction,
			Dimensions:  10,
			MinBounds:   repeatFloat(-32.768, 10),
			MaxBounds:   repeatFloat(32.768, 10),
			GlobalMin:   repeatFloat(0.0, 10),
			GlobalValue: 0.0,
			Description: "Multimodal with many local optima",
		},
	}
}

// SphereFunction implements the sphere function
func SphereFunction(position []float64) float64 {
	sum := 0.0
	for _, x := range position {
		sum += x * x
	}
	return sum
}

// RastriginFunction implements the Rastrigin function
func RastriginFunction(position []float64) float64 {
	n := float64(len(position))
	sum := 10.0 * n
	
	for _, x := range position {
		sum += x*x - 10.0*math.Cos(2.0*math.Pi*x)
	}
	
	return sum
}

// RosenbrockFunction implements the Rosenbrock function
func RosenbrockFunction(position []float64) float64 {
	sum := 0.0
	for i := 0; i < len(position)-1; i++ {
		term1 := position[i+1] - position[i]*position[i]
		term2 := 1.0 - position[i]
		sum += 100.0*term1*term1 + term2*term2
	}
	return sum
}

// AckleyFunction implements the Ackley function
func AckleyFunction(position []float64) float64 {
	n := float64(len(position))
	
	sum1 := 0.0
	sum2 := 0.0
	
	for _, x := range position {
		sum1 += x * x
		sum2 += math.Cos(2.0 * math.Pi * x)
	}
	
	term1 := -20.0 * math.Exp(-0.2*math.Sqrt(sum1/n))
	term2 := -math.Exp(sum2 / n)
	
	return term1 + term2 + 20.0 + math.E
}

// repeatFloat creates a slice with repeated values
func repeatFloat(value float64, count int) []float64 {
	result := make([]float64, count)
	for i := 0; i < count; i++ {
		result[i] = value
	}
	return result
}