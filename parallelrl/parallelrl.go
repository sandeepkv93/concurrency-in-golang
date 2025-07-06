package parallelrl

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

// AgentType defines different types of RL agents
type AgentType int

const (
	QLearning AgentType = iota
	DeepQNetwork
	ActorCritic
	PolicyGradient
	AdvantageActorCritic
	ProximalPolicyOptimization
	SARSA
	MonteCarloTreeSearch
	EvolutionaryStrategy
	MultiAgent
)

// LearningAlgorithm defines the core learning algorithm
type LearningAlgorithm int

const (
	TabularQLearning LearningAlgorithm = iota
	FunctionApproximation
	NeuralNetwork
	EnsembleMethods
	TransferLearning
	MetaLearning
)

// ParallelStrategy defines parallel training strategies
type ParallelStrategy int

const (
	AsyncActorCritic ParallelStrategy = iota
	DistributedTraining
	PopulationBased
	ExperienceReplay
	ModelParallelism
	DataParallelism
	GradientSharing
	ParameterServer
)

// Environment represents the RL environment
type Environment interface {
	Reset() State
	Step(action Action) (State, Reward, bool, map[string]interface{})
	GetActionSpace() []Action
	GetStateSpace() StateSpace
	GetCurrentState() State
	IsTerminal() bool
	Clone() Environment
	Render() string
}

// State represents an environment state
type State interface {
	ToVector() []float64
	Equals(other State) bool
	Hash() string
	Dimension() int
	IsValid() bool
}

// Action represents an action in the environment
type Action interface {
	ToVector() []float64
	Equals(other Action) bool
	IsValid() bool
	String() string
}

// Reward represents a reward signal
type Reward float64

// StateSpace defines the state space characteristics
type StateSpace struct {
	Dimensions []int     `json:"dimensions"`
	Continuous bool      `json:"continuous"`
	Bounds     [][]float64 `json:"bounds,omitempty"`
	Discrete   [][]string  `json:"discrete,omitempty"`
}

// Policy represents an agent's policy
type Policy interface {
	SelectAction(state State) Action
	GetActionProbabilities(state State) map[Action]float64
	Update(experience *Experience) error
	GetParameters() []float64
	SetParameters(params []float64) error
	Clone() Policy
}

// ValueFunction represents a value function
type ValueFunction interface {
	Evaluate(state State) float64
	Update(state State, target float64) error
	GetParameters() []float64
	SetParameters(params []float64) error
	Clone() ValueFunction
}

// Experience represents a single experience tuple
type Experience struct {
	State      State                  `json:"state"`
	Action     Action                 `json:"action"`
	Reward     Reward                 `json:"reward"`
	NextState  State                  `json:"next_state"`
	Done       bool                   `json:"done"`
	Timestamp  time.Time              `json:"timestamp"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// Episode represents a complete episode
type Episode struct {
	ID          string        `json:"id"`
	Experiences []*Experience `json:"experiences"`
	TotalReward Reward        `json:"total_reward"`
	Duration    time.Duration `json:"duration"`
	Steps       int           `json:"steps"`
	Success     bool          `json:"success"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// TrainingConfig holds training configuration
type TrainingConfig struct {
	// Agent Configuration
	AgentType        AgentType        `json:"agent_type"`
	Algorithm        LearningAlgorithm `json:"algorithm"`
	ParallelStrategy ParallelStrategy  `json:"parallel_strategy"`
	
	// Learning Parameters
	LearningRate     float64 `json:"learning_rate"`
	DiscountFactor   float64 `json:"discount_factor"`
	ExplorationRate  float64 `json:"exploration_rate"`
	ExplorationDecay float64 `json:"exploration_decay"`
	MinExploration   float64 `json:"min_exploration"`
	
	// Training Settings
	MaxEpisodes      int           `json:"max_episodes"`
	MaxStepsPerEpisode int         `json:"max_steps_per_episode"`
	TimeLimit        time.Duration `json:"time_limit"`
	TargetReward     float64       `json:"target_reward"`
	
	// Parallel Settings
	NumWorkers       int `json:"num_workers"`
	NumActors        int `json:"num_actors"`
	NumLearners      int `json:"num_learners"`
	BatchSize        int `json:"batch_size"`
	UpdateFrequency  int `json:"update_frequency"`
	
	// Experience Replay
	ReplayBufferSize int     `json:"replay_buffer_size"`
	ReplayBatchSize  int     `json:"replay_batch_size"`
	ReplayStartSize  int     `json:"replay_start_size"`
	PriorityReplay   bool    `json:"priority_replay"`
	ImportanceSampling bool  `json:"importance_sampling"`
	
	// Network Architecture (for neural networks)
	HiddenLayers     []int   `json:"hidden_layers"`
	ActivationFunc   string  `json:"activation_function"`
	Optimizer        string  `json:"optimizer"`
	LossFunction     string  `json:"loss_function"`
	L2Regularization float64 `json:"l2_regularization"`
	DropoutRate      float64 `json:"dropout_rate"`
	
	// Advanced Settings
	TargetNetworkUpdate int     `json:"target_network_update"`
	GradientClipping    float64 `json:"gradient_clipping"`
	NoiseScale          float64 `json:"noise_scale"`
	ParameterNoise      bool    `json:"parameter_noise"`
	
	// Multi-Agent Settings
	NumAgents           int     `json:"num_agents"`
	CommunicationRange  float64 `json:"communication_range"`
	CooperativeReward   bool    `json:"cooperative_reward"`
	CompetitiveReward   bool    `json:"competitive_reward"`
	
	// Monitoring and Logging
	LogInterval       int  `json:"log_interval"`
	SaveInterval      int  `json:"save_interval"`
	EvaluationEpisodes int `json:"evaluation_episodes"`
	EnableLogging     bool `json:"enable_logging"`
	EnableVisualizer  bool `json:"enable_visualizer"`
	
	// Performance Tuning
	Seed             int64 `json:"seed"`
	NumCPUs          int   `json:"num_cpus"`
	UseGPU           bool  `json:"use_gpu"`
	MemoryLimit      int64 `json:"memory_limit"`
}

// Agent represents a reinforcement learning agent
type Agent interface {
	SelectAction(state State) Action
	Learn(experience *Experience) error
	Update(batch []*Experience) error
	Evaluate(env Environment, episodes int) float64
	GetPolicy() Policy
	GetValueFunction() ValueFunction
	GetStatistics() *AgentStatistics
	Save(path string) error
	Load(path string) error
	Clone() Agent
}

// ParallelAgent represents a parallel RL learning system
type ParallelAgent struct {
	config       TrainingConfig
	agents       []Agent
	environment  Environment
	policy       Policy
	valueFunc    ValueFunction
	replayBuffer *ReplayBuffer
	
	// Parallel components
	actors       []*Actor
	learners     []*Learner
	coordinator  *Coordinator
	
	// Statistics and monitoring
	statistics   *TrainingStatistics
	episodeLog   []*Episode
	
	// Synchronization
	ctx          context.Context
	cancel       context.CancelFunc
	running      int32
	mutex        sync.RWMutex
	
	// Communication channels
	experienceChan chan *Experience
	updateChan     chan *ParameterUpdate
	resultChan     chan *EpisodeResult
	
	// Performance tracking
	startTime    time.Time
	totalEpisodes int64
	totalSteps   int64
}

// Actor collects experiences from environment
type Actor struct {
	ID           int           `json:"id"`
	Agent        Agent         `json:"-"`
	Environment  Environment   `json:"-"`
	Policy       Policy        `json:"-"`
	Statistics   *ActorStats   `json:"statistics"`
	ExperienceChan chan *Experience `json:"-"`
	UpdateChan   chan *ParameterUpdate `json:"-"`
	Running      bool          `json:"running"`
	mutex        sync.RWMutex
}

// Learner processes experiences and updates policies
type Learner struct {
	ID           int               `json:"id"`
	Agent        Agent             `json:"-"`
	ReplayBuffer *ReplayBuffer     `json:"-"`
	Statistics   *LearnerStats     `json:"statistics"`
	ExperienceChan chan *Experience `json:"-"`
	UpdateChan   chan *ParameterUpdate `json:"-"`
	Running      bool              `json:"running"`
	mutex        sync.RWMutex
}

// Coordinator manages actors and learners
type Coordinator struct {
	Actors       []*Actor          `json:"actors"`
	Learners     []*Learner        `json:"learners"`
	Statistics   *CoordinatorStats `json:"statistics"`
	Running      bool              `json:"running"`
	mutex        sync.RWMutex
}

// ReplayBuffer stores and samples experiences
type ReplayBuffer struct {
	experiences   []*Experience
	capacity      int
	size          int
	position      int
	priorities    []float64
	alpha         float64 // Priority exponent
	beta          float64 // Importance sampling exponent
	epsilon       float64 // Small constant for numerical stability
	mutex         sync.RWMutex
}

// ParameterUpdate represents a parameter update
type ParameterUpdate struct {
	Parameters []float64              `json:"parameters"`
	Gradients  []float64              `json:"gradients,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	Timestamp  time.Time              `json:"timestamp"`
}

// EpisodeResult represents the result of an episode
type EpisodeResult struct {
	Episode     *Episode `json:"episode"`
	ActorID     int      `json:"actor_id"`
	Performance float64  `json:"performance"`
	Metrics     map[string]float64 `json:"metrics"`
}

// Statistics structures
type AgentStatistics struct {
	TotalEpisodes    int64                  `json:"total_episodes"`
	TotalSteps       int64                  `json:"total_steps"`
	AverageReward    float64                `json:"average_reward"`
	BestReward       float64                `json:"best_reward"`
	WorstReward      float64                `json:"worst_reward"`
	AverageSteps     float64                `json:"average_steps"`
	LearningRate     float64                `json:"learning_rate"`
	ExplorationRate  float64                `json:"exploration_rate"`
	LossHistory      []float64              `json:"loss_history"`
	RewardHistory    []float64              `json:"reward_history"`
	ParameterNorm    float64                `json:"parameter_norm"`
	GradientNorm     float64                `json:"gradient_norm"`
	UpdateCount      int64                  `json:"update_count"`
	LastUpdateTime   time.Time              `json:"last_update_time"`
	mutex            sync.RWMutex
}

type TrainingStatistics struct {
	StartTime        time.Time              `json:"start_time"`
	TotalTrainingTime time.Duration         `json:"total_training_time"`
	EpisodesCompleted int64                 `json:"episodes_completed"`
	StepsCompleted   int64                  `json:"steps_completed"`
	AverageReward    float64                `json:"average_reward"`
	BestEpisodeReward float64               `json:"best_episode_reward"`
	WorstEpisodeReward float64              `json:"worst_episode_reward"`
	RewardTrend      []float64              `json:"reward_trend"`
	LossTrend        []float64              `json:"loss_trend"`
	ExplorationTrend []float64              `json:"exploration_trend"`
	ActorStats       map[int]*ActorStats    `json:"actor_stats"`
	LearnerStats     map[int]*LearnerStats  `json:"learner_stats"`
	ParallelEfficiency float64              `json:"parallel_efficiency"`
	ThroughputEPS    float64                `json:"throughput_eps"` // Episodes per second
	ThroughputSPS    float64                `json:"throughput_sps"` // Steps per second
	MemoryUsage      int64                  `json:"memory_usage"`
	CPUUsage         float64                `json:"cpu_usage"`
	mutex            sync.RWMutex
}

type ActorStats struct {
	EpisodesCompleted int64     `json:"episodes_completed"`
	StepsCompleted    int64     `json:"steps_completed"`
	AverageReward     float64   `json:"average_reward"`
	AverageSteps      float64   `json:"average_steps"`
	LastEpisodeTime   time.Time `json:"last_episode_time"`
	ExperiencesGenerated int64  `json:"experiences_generated"`
	ActionsSelected   map[string]int64 `json:"actions_selected"`
}

type LearnerStats struct {
	UpdatesPerformed  int64     `json:"updates_performed"`
	ExperiencesProcessed int64  `json:"experiences_processed"`
	AverageLoss       float64   `json:"average_loss"`
	LastUpdateTime    time.Time `json:"last_update_time"`
	ParameterNorm     float64   `json:"parameter_norm"`
	GradientNorm      float64   `json:"gradient_norm"`
	LearningRate      float64   `json:"learning_rate"`
}

type CoordinatorStats struct {
	ActiveActors     int       `json:"active_actors"`
	ActiveLearners   int       `json:"active_learners"`
	MessagesExchanged int64    `json:"messages_exchanged"`
	SynchronizationTime time.Duration `json:"synchronization_time"`
	LastSyncTime     time.Time `json:"last_sync_time"`
}

// DefaultTrainingConfig returns a default training configuration
func DefaultTrainingConfig() TrainingConfig {
	return TrainingConfig{
		AgentType:        QLearning,
		Algorithm:        TabularQLearning,
		ParallelStrategy: AsyncActorCritic,
		LearningRate:     0.1,
		DiscountFactor:   0.95,
		ExplorationRate:  1.0,
		ExplorationDecay: 0.995,
		MinExploration:   0.01,
		MaxEpisodes:      10000,
		MaxStepsPerEpisode: 1000,
		TimeLimit:        30 * time.Minute,
		TargetReward:     math.Inf(1),
		NumWorkers:       runtime.NumCPU(),
		NumActors:        4,
		NumLearners:      2,
		BatchSize:        32,
		UpdateFrequency:  4,
		ReplayBufferSize: 100000,
		ReplayBatchSize:  32,
		ReplayStartSize:  1000,
		PriorityReplay:   false,
		ImportanceSampling: false,
		HiddenLayers:     []int{64, 64},
		ActivationFunc:   "relu",
		Optimizer:        "adam",
		LossFunction:     "mse",
		L2Regularization: 0.001,
		DropoutRate:      0.0,
		TargetNetworkUpdate: 1000,
		GradientClipping: 1.0,
		NoiseScale:       0.1,
		ParameterNoise:   false,
		NumAgents:        1,
		CommunicationRange: 1.0,
		CooperativeReward: false,
		CompetitiveReward: false,
		LogInterval:      100,
		SaveInterval:     1000,
		EvaluationEpisodes: 10,
		EnableLogging:    true,
		EnableVisualizer: false,
		Seed:             time.Now().UnixNano(),
		NumCPUs:          runtime.NumCPU(),
		UseGPU:           false,
		MemoryLimit:      1024 * 1024 * 1024, // 1GB
	}
}

// NewParallelAgent creates a new parallel RL agent
func NewParallelAgent(config TrainingConfig, env Environment, policy Policy, valueFunc ValueFunction) (*ParallelAgent, error) {
	if env == nil {
		return nil, errors.New("environment cannot be nil")
	}
	
	if policy == nil {
		return nil, errors.New("policy cannot be nil")
	}
	
	// Set random seed for reproducibility
	if config.Seed != 0 {
		rand.Seed(config.Seed)
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	pa := &ParallelAgent{
		config:       config,
		environment:  env,
		policy:       policy,
		valueFunc:    valueFunc,
		ctx:          ctx,
		cancel:       cancel,
		experienceChan: make(chan *Experience, config.BatchSize*config.NumActors),
		updateChan:   make(chan *ParameterUpdate, config.NumLearners),
		resultChan:   make(chan *EpisodeResult, config.NumActors),
		statistics:   &TrainingStatistics{
			StartTime:    time.Now(),
			ActorStats:   make(map[int]*ActorStats),
			LearnerStats: make(map[int]*LearnerStats),
		},
		episodeLog:   make([]*Episode, 0),
	}
	
	// Initialize replay buffer
	if config.ReplayBufferSize > 0 {
		pa.replayBuffer = NewReplayBuffer(config.ReplayBufferSize, config.PriorityReplay)
	}
	
	// Initialize actors
	pa.actors = make([]*Actor, config.NumActors)
	for i := 0; i < config.NumActors; i++ {
		actor, err := pa.createActor(i)
		if err != nil {
			return nil, fmt.Errorf("failed to create actor %d: %v", i, err)
		}
		pa.actors[i] = actor
		pa.statistics.ActorStats[i] = &ActorStats{
			ActionsSelected: make(map[string]int64),
		}
	}
	
	// Initialize learners
	pa.learners = make([]*Learner, config.NumLearners)
	for i := 0; i < config.NumLearners; i++ {
		learner, err := pa.createLearner(i)
		if err != nil {
			return nil, fmt.Errorf("failed to create learner %d: %v", i, err)
		}
		pa.learners[i] = learner
		pa.statistics.LearnerStats[i] = &LearnerStats{}
	}
	
	// Initialize coordinator
	pa.coordinator = &Coordinator{
		Actors:     pa.actors,
		Learners:   pa.learners,
		Statistics: &CoordinatorStats{},
	}
	
	return pa, nil
}

// NewReplayBuffer creates a new replay buffer
func NewReplayBuffer(capacity int, prioritized bool) *ReplayBuffer {
	rb := &ReplayBuffer{
		experiences: make([]*Experience, capacity),
		capacity:    capacity,
		size:        0,
		position:    0,
		alpha:       0.6, // Priority exponent
		beta:        0.4, // Importance sampling exponent
		epsilon:     1e-6, // Small constant
	}
	
	if prioritized {
		rb.priorities = make([]float64, capacity)
	}
	
	return rb
}

// Train starts the parallel training process
func (pa *ParallelAgent) Train() error {
	if atomic.LoadInt32(&pa.running) == 1 {
		return errors.New("training is already running")
	}
	
	atomic.StoreInt32(&pa.running, 1)
	defer atomic.StoreInt32(&pa.running, 0)
	
	pa.startTime = time.Now()
	
	// Start coordinator
	go pa.runCoordinator()
	
	// Start learners
	for _, learner := range pa.learners {
		go pa.runLearner(learner)
	}
	
	// Start actors
	for _, actor := range pa.actors {
		go pa.runActor(actor)
	}
	
	// Main training loop
	return pa.runTrainingLoop()
}

// runTrainingLoop executes the main training loop
func (pa *ParallelAgent) runTrainingLoop() error {
	ticker := time.NewTicker(time.Duration(pa.config.LogInterval) * time.Second)
	defer ticker.Stop()
	
	evaluationTicker := time.NewTicker(time.Duration(pa.config.SaveInterval) * time.Second)
	defer evaluationTicker.Stop()
	
	for {
		select {
		case <-pa.ctx.Done():
			return pa.ctx.Err()
			
		case result := <-pa.resultChan:
			pa.processEpisodeResult(result)
			
			// Check termination conditions
			if pa.shouldTerminate() {
				pa.Stop()
				return nil
			}
			
		case <-ticker.C:
			if pa.config.EnableLogging {
				pa.logProgress()
			}
			
		case <-evaluationTicker.C:
			pa.evaluateAgent()
			
		case <-time.After(pa.config.TimeLimit):
			pa.Stop()
			return errors.New("training time limit exceeded")
		}
	}
}

// runActor executes an actor's training loop
func (pa *ParallelAgent) runActor(actor *Actor) {
	defer func() {
		actor.mutex.Lock()
		actor.Running = false
		actor.mutex.Unlock()
	}()
	
	actor.mutex.Lock()
	actor.Running = true
	actor.mutex.Unlock()
	
	for {
		select {
		case <-pa.ctx.Done():
			return
			
		case update := <-actor.UpdateChan:
			pa.applyParameterUpdate(actor, update)
			
		default:
			episode := pa.runEpisode(actor)
			if episode != nil {
				result := &EpisodeResult{
					Episode:     episode,
					ActorID:     actor.ID,
					Performance: float64(episode.TotalReward),
					Metrics:     pa.calculateEpisodeMetrics(episode),
				}
				
				select {
				case pa.resultChan <- result:
				case <-pa.ctx.Done():
					return
				}
			}
		}
	}
}

// runLearner executes a learner's training loop
func (pa *ParallelAgent) runLearner(learner *Learner) {
	defer func() {
		learner.mutex.Lock()
		learner.Running = false
		learner.mutex.Unlock()
	}()
	
	learner.mutex.Lock()
	learner.Running = true
	learner.mutex.Unlock()
	
	batch := make([]*Experience, 0, pa.config.ReplayBatchSize)
	
	for {
		select {
		case <-pa.ctx.Done():
			return
			
		case experience := <-learner.ExperienceChan:
			if pa.replayBuffer != nil {
				pa.replayBuffer.Add(experience)
			}
			
			batch = append(batch, experience)
			
			if len(batch) >= pa.config.ReplayBatchSize {
				pa.processLearningBatch(learner, batch)
				batch = batch[:0] // Clear batch
			}
			
		default:
			// Sample from replay buffer if available
			if pa.replayBuffer != nil && pa.replayBuffer.Size() >= pa.config.ReplayStartSize {
				samples := pa.replayBuffer.Sample(pa.config.ReplayBatchSize)
				if len(samples) > 0 {
					pa.processLearningBatch(learner, samples)
				}
			}
			
			time.Sleep(time.Millisecond) // Small delay to prevent busy waiting
		}
	}
}

// runCoordinator executes the coordinator's management loop
func (pa *ParallelAgent) runCoordinator() {
	defer func() {
		pa.coordinator.mutex.Lock()
		pa.coordinator.Running = false
		pa.coordinator.mutex.Unlock()
	}()
	
	pa.coordinator.mutex.Lock()
	pa.coordinator.Running = true
	pa.coordinator.mutex.Unlock()
	
	syncTicker := time.NewTicker(time.Duration(pa.config.UpdateFrequency) * time.Second)
	defer syncTicker.Stop()
	
	for {
		select {
		case <-pa.ctx.Done():
			return
			
		case <-syncTicker.C:
			pa.synchronizeParameters()
			pa.updateStatistics()
		}
	}
}

// runEpisode executes a single episode with an actor
func (pa *ParallelAgent) runEpisode(actor *Actor) *Episode {
	env := actor.Environment.Clone()
	state := env.Reset()
	
	episodeID := fmt.Sprintf("actor-%d-episode-%d", actor.ID, actor.Statistics.EpisodesCompleted)
	episode := &Episode{
		ID:          episodeID,
		Experiences: make([]*Experience, 0),
		TotalReward: 0,
		Steps:       0,
		Success:     false,
		Metadata:    make(map[string]interface{}),
	}
	
	startTime := time.Now()
	
	for step := 0; step < pa.config.MaxStepsPerEpisode; step++ {
		// Select action using current policy
		action := actor.Policy.SelectAction(state)
		
		// Take action in environment
		nextState, reward, done, info := env.Step(action)
		
		// Create experience
		experience := &Experience{
			State:     state,
			Action:    action,
			Reward:    reward,
			NextState: nextState,
			Done:      done,
			Timestamp: time.Now(),
			Metadata:  info,
		}
		
		episode.Experiences = append(episode.Experiences, experience)
		episode.TotalReward += reward
		episode.Steps++
		
		// Send experience to learners
		select {
		case pa.experienceChan <- experience:
		default:
			// Channel full, skip this experience
		}
		
		// Update statistics
		actor.Statistics.StepsCompleted++
		actor.Statistics.ExperiencesGenerated++
		
		if actionStr := action.String(); actionStr != "" {
			actor.Statistics.ActionsSelected[actionStr]++
		}
		
		state = nextState
		
		if done || env.IsTerminal() {
			episode.Success = !env.IsTerminal() || reward > 0
			break
		}
	}
	
	episode.Duration = time.Since(startTime)
	
	// Update actor statistics
	actor.Statistics.EpisodesCompleted++
	actor.Statistics.LastEpisodeTime = time.Now()
	
	if actor.Statistics.EpisodesCompleted == 1 {
		actor.Statistics.AverageReward = float64(episode.TotalReward)
		actor.Statistics.AverageSteps = float64(episode.Steps)
	} else {
		// Running average
		n := float64(actor.Statistics.EpisodesCompleted)
		actor.Statistics.AverageReward = (actor.Statistics.AverageReward*(n-1) + float64(episode.TotalReward)) / n
		actor.Statistics.AverageSteps = (actor.Statistics.AverageSteps*(n-1) + float64(episode.Steps)) / n
	}
	
	atomic.AddInt64(&pa.totalEpisodes, 1)
	atomic.AddInt64(&pa.totalSteps, int64(episode.Steps))
	
	return episode
}

// processLearningBatch processes a batch of experiences for learning
func (pa *ParallelAgent) processLearningBatch(learner *Learner, batch []*Experience) {
	if len(batch) == 0 {
		return
	}
	
	startTime := time.Now()
	
	// Update agent with batch
	if err := learner.Agent.Update(batch); err != nil {
		// Handle learning error
		return
	}
	
	// Update learner statistics
	learner.Statistics.UpdatesPerformed++
	learner.Statistics.ExperiencesProcessed += int64(len(batch))
	learner.Statistics.LastUpdateTime = time.Now()
	
	// Calculate processing time
	processingTime := time.Since(startTime)
	
	// Create parameter update for actors
	if policy := learner.Agent.GetPolicy(); policy != nil {
		params := policy.GetParameters()
		update := &ParameterUpdate{
			Parameters: params,
			Timestamp:  time.Now(),
			Metadata: map[string]interface{}{
				"learner_id":       learner.ID,
				"batch_size":       len(batch),
				"processing_time":  processingTime,
			},
		}
		
		// Broadcast update to actors
		select {
		case pa.updateChan <- update:
		default:
			// Channel full, skip this update
		}
	}
}

// Helper methods and implementations

func (pa *ParallelAgent) createActor(id int) (*Actor, error) {
	// Clone environment for this actor
	env := pa.environment.Clone()
	
	// Clone policy for this actor
	policy := pa.policy.Clone()
	
	// Create agent for this actor (simplified - would depend on agent type)
	var agent Agent
	
	actor := &Actor{
		ID:             id,
		Agent:          agent,
		Environment:    env,
		Policy:         policy,
		Statistics:     &ActorStats{ActionsSelected: make(map[string]int64)},
		ExperienceChan: pa.experienceChan,
		UpdateChan:     pa.updateChan,
	}
	
	return actor, nil
}

func (pa *ParallelAgent) createLearner(id int) (*Learner, error) {
	// Create agent for this learner (simplified - would depend on agent type)
	var agent Agent
	
	learner := &Learner{
		ID:             id,
		Agent:          agent,
		ReplayBuffer:   pa.replayBuffer,
		Statistics:     &LearnerStats{},
		ExperienceChan: pa.experienceChan,
		UpdateChan:     pa.updateChan,
	}
	
	return learner, nil
}

func (pa *ParallelAgent) processEpisodeResult(result *EpisodeResult) {
	pa.mutex.Lock()
	defer pa.mutex.Unlock()
	
	pa.episodeLog = append(pa.episodeLog, result.Episode)
	
	// Update global statistics
	pa.statistics.EpisodesCompleted++
	pa.statistics.StepsCompleted += int64(result.Episode.Steps)
	
	// Update reward statistics
	reward := float64(result.Episode.TotalReward)
	if pa.statistics.EpisodesCompleted == 1 {
		pa.statistics.AverageReward = reward
		pa.statistics.BestEpisodeReward = reward
		pa.statistics.WorstEpisodeReward = reward
	} else {
		// Running average
		n := float64(pa.statistics.EpisodesCompleted)
		pa.statistics.AverageReward = (pa.statistics.AverageReward*(n-1) + reward) / n
		
		if reward > pa.statistics.BestEpisodeReward {
			pa.statistics.BestEpisodeReward = reward
		}
		if reward < pa.statistics.WorstEpisodeReward {
			pa.statistics.WorstEpisodeReward = reward
		}
	}
	
	// Update trends
	pa.statistics.RewardTrend = append(pa.statistics.RewardTrend, reward)
	if len(pa.statistics.RewardTrend) > 1000 {
		pa.statistics.RewardTrend = pa.statistics.RewardTrend[1:]
	}
}

func (pa *ParallelAgent) shouldTerminate() bool {
	if pa.statistics.EpisodesCompleted >= int64(pa.config.MaxEpisodes) {
		return true
	}
	
	if pa.statistics.AverageReward >= pa.config.TargetReward {
		return true
	}
	
	return false
}

func (pa *ParallelAgent) applyParameterUpdate(actor *Actor, update *ParameterUpdate) {
	if actor.Policy != nil {
		actor.Policy.SetParameters(update.Parameters)
	}
}

func (pa *ParallelAgent) synchronizeParameters() {
	// Implement parameter synchronization between actors and learners
	// This would depend on the specific parallel strategy
}

func (pa *ParallelAgent) updateStatistics() {
	pa.statistics.mutex.Lock()
	defer pa.statistics.mutex.Unlock()
	
	pa.statistics.TotalTrainingTime = time.Since(pa.startTime)
	
	// Calculate throughput
	if pa.statistics.TotalTrainingTime > 0 {
		pa.statistics.ThroughputEPS = float64(pa.statistics.EpisodesCompleted) / pa.statistics.TotalTrainingTime.Seconds()
		pa.statistics.ThroughputSPS = float64(pa.statistics.StepsCompleted) / pa.statistics.TotalTrainingTime.Seconds()
	}
	
	// Calculate parallel efficiency
	activeActors := 0
	for _, actor := range pa.actors {
		actor.mutex.RLock()
		if actor.Running {
			activeActors++
		}
		actor.mutex.RUnlock()
	}
	
	activeLearners := 0
	for _, learner := range pa.learners {
		learner.mutex.RLock()
		if learner.Running {
			activeLearners++
		}
		learner.mutex.RUnlock()
	}
	
	totalWorkers := float64(activeActors + activeLearners)
	idealWorkers := float64(pa.config.NumActors + pa.config.NumLearners)
	if idealWorkers > 0 {
		pa.statistics.ParallelEfficiency = totalWorkers / idealWorkers
	}
}

func (pa *ParallelAgent) logProgress() {
	stats := pa.GetStatistics()
	fmt.Printf("Episode %d: Avg Reward = %.2f, Best = %.2f, Throughput = %.2f eps\n",
		stats.EpisodesCompleted,
		stats.AverageReward,
		stats.BestEpisodeReward,
		stats.ThroughputEPS)
}

func (pa *ParallelAgent) evaluateAgent() {
	// Run evaluation episodes without exploration
	// This would create a separate evaluation environment and run episodes
}

func (pa *ParallelAgent) calculateEpisodeMetrics(episode *Episode) map[string]float64 {
	metrics := make(map[string]float64)
	
	metrics["total_reward"] = float64(episode.TotalReward)
	metrics["episode_length"] = float64(episode.Steps)
	metrics["success"] = 0
	if episode.Success {
		metrics["success"] = 1
	}
	
	// Calculate reward variance
	if len(episode.Experiences) > 1 {
		rewards := make([]float64, len(episode.Experiences))
		for i, exp := range episode.Experiences {
			rewards[i] = float64(exp.Reward)
		}
		metrics["reward_variance"] = calculateVariance(rewards)
	}
	
	return metrics
}

// ReplayBuffer methods

func (rb *ReplayBuffer) Add(experience *Experience) {
	rb.mutex.Lock()
	defer rb.mutex.Unlock()
	
	rb.experiences[rb.position] = experience
	
	// Initialize priority for new experience
	if rb.priorities != nil {
		maxPriority := 1.0
		if rb.size > 0 {
			maxPriority = rb.getMaxPriority()
		}
		rb.priorities[rb.position] = maxPriority
	}
	
	rb.position = (rb.position + 1) % rb.capacity
	
	if rb.size < rb.capacity {
		rb.size++
	}
}

func (rb *ReplayBuffer) Sample(batchSize int) []*Experience {
	rb.mutex.RLock()
	defer rb.mutex.RUnlock()
	
	if rb.size == 0 {
		return nil
	}
	
	sampleSize := min(batchSize, rb.size)
	samples := make([]*Experience, sampleSize)
	
	if rb.priorities != nil {
		// Prioritized sampling
		indices := rb.sampleProportional(sampleSize)
		for i, idx := range indices {
			samples[i] = rb.experiences[idx]
		}
	} else {
		// Uniform sampling
		for i := 0; i < sampleSize; i++ {
			idx := rand.Intn(rb.size)
			samples[i] = rb.experiences[idx]
		}
	}
	
	return samples
}

func (rb *ReplayBuffer) Size() int {
	rb.mutex.RLock()
	defer rb.mutex.RUnlock()
	return rb.size
}

func (rb *ReplayBuffer) getMaxPriority() float64 {
	maxPriority := 0.0
	for i := 0; i < rb.size; i++ {
		if rb.priorities[i] > maxPriority {
			maxPriority = rb.priorities[i]
		}
	}
	return maxPriority
}

func (rb *ReplayBuffer) sampleProportional(batchSize int) []int {
	// Simplified proportional sampling
	indices := make([]int, batchSize)
	
	// Create cumulative probability distribution
	totalPriority := 0.0
	for i := 0; i < rb.size; i++ {
		totalPriority += math.Pow(rb.priorities[i], rb.alpha)
	}
	
	for i := 0; i < batchSize; i++ {
		// Sample from cumulative distribution
		target := rand.Float64() * totalPriority
		cumulative := 0.0
		
		for j := 0; j < rb.size; j++ {
			cumulative += math.Pow(rb.priorities[j], rb.alpha)
			if cumulative >= target {
				indices[i] = j
				break
			}
		}
	}
	
	return indices
}

// Public interface methods

func (pa *ParallelAgent) Stop() {
	if atomic.LoadInt32(&pa.running) == 0 {
		return
	}
	
	pa.cancel()
	
	// Wait for all goroutines to finish
	time.Sleep(100 * time.Millisecond)
	
	atomic.StoreInt32(&pa.running, 0)
}

func (pa *ParallelAgent) IsRunning() bool {
	return atomic.LoadInt32(&pa.running) == 1
}

func (pa *ParallelAgent) GetStatistics() *TrainingStatistics {
	pa.statistics.mutex.RLock()
	defer pa.statistics.mutex.RUnlock()
	
	// Return a copy to avoid race conditions
	stats := *pa.statistics
	return &stats
}

func (pa *ParallelAgent) GetEpisodeLog() []*Episode {
	pa.mutex.RLock()
	defer pa.mutex.RUnlock()
	
	// Return a copy
	log := make([]*Episode, len(pa.episodeLog))
	copy(log, pa.episodeLog)
	return log
}

func (pa *ParallelAgent) GetPolicy() Policy {
	return pa.policy
}

func (pa *ParallelAgent) GetValueFunction() ValueFunction {
	return pa.valueFunc
}

// Utility functions

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func calculateVariance(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))
	
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))
	
	return variance
}

// Simple implementations for basic RL components

// SimpleState implements State interface
type SimpleState struct {
	Values []float64 `json:"values"`
}

func (s *SimpleState) ToVector() []float64 {
	return s.Values
}

func (s *SimpleState) Equals(other State) bool {
	if otherSimple, ok := other.(*SimpleState); ok {
		if len(s.Values) != len(otherSimple.Values) {
			return false
		}
		for i, v := range s.Values {
			if math.Abs(v-otherSimple.Values[i]) > 1e-9 {
				return false
			}
		}
		return true
	}
	return false
}

func (s *SimpleState) Hash() string {
	return fmt.Sprintf("%v", s.Values)
}

func (s *SimpleState) Dimension() int {
	return len(s.Values)
}

func (s *SimpleState) IsValid() bool {
	return len(s.Values) > 0
}

// SimpleAction implements Action interface
type SimpleAction struct {
	ID     int     `json:"id"`
	Values []float64 `json:"values,omitempty"`
}

func (a *SimpleAction) ToVector() []float64 {
	if len(a.Values) > 0 {
		return a.Values
	}
	return []float64{float64(a.ID)}
}

func (a *SimpleAction) Equals(other Action) bool {
	if otherSimple, ok := other.(*SimpleAction); ok {
		return a.ID == otherSimple.ID
	}
	return false
}

func (a *SimpleAction) IsValid() bool {
	return a.ID >= 0
}

func (a *SimpleAction) String() string {
	return fmt.Sprintf("Action_%d", a.ID)
}

// SimplePolicy implements Policy interface
type SimplePolicy struct {
	QTable         map[string]map[int]float64 `json:"q_table"`
	ExplorationRate float64                  `json:"exploration_rate"`
	Actions        []Action                   `json:"actions"`
	mutex          sync.RWMutex
}

func NewSimplePolicy(actions []Action, explorationRate float64) *SimplePolicy {
	return &SimplePolicy{
		QTable:          make(map[string]map[int]float64),
		ExplorationRate: explorationRate,
		Actions:         actions,
	}
}

func (p *SimplePolicy) SelectAction(state State) Action {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	
	// Epsilon-greedy action selection
	if rand.Float64() < p.ExplorationRate {
		// Random action
		return p.Actions[rand.Intn(len(p.Actions))]
	}
	
	// Greedy action
	stateKey := state.Hash()
	if qValues, exists := p.QTable[stateKey]; exists {
		bestAction := 0
		bestValue := math.Inf(-1)
		
		for actionID, value := range qValues {
			if value > bestValue {
				bestValue = value
				bestAction = actionID
			}
		}
		
		if bestAction < len(p.Actions) {
			return p.Actions[bestAction]
		}
	}
	
	// Default to random action if no Q-values exist
	return p.Actions[rand.Intn(len(p.Actions))]
}

func (p *SimplePolicy) GetActionProbabilities(state State) map[Action]float64 {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	
	probs := make(map[Action]float64)
	
	// Initialize uniform probabilities
	uniformProb := 1.0 / float64(len(p.Actions))
	for _, action := range p.Actions {
		probs[action] = uniformProb
	}
	
	// Apply epsilon-greedy probabilities
	stateKey := state.Hash()
	if qValues, exists := p.QTable[stateKey]; exists {
		bestAction := 0
		bestValue := math.Inf(-1)
		
		for actionID, value := range qValues {
			if value > bestValue {
				bestValue = value
				bestAction = actionID
			}
		}
		
		// Set probabilities
		exploreProb := p.ExplorationRate / float64(len(p.Actions))
		greedyProb := 1.0 - p.ExplorationRate + exploreProb
		
		for i, action := range p.Actions {
			if i == bestAction {
				probs[action] = greedyProb
			} else {
				probs[action] = exploreProb
			}
		}
	}
	
	return probs
}

func (p *SimplePolicy) Update(experience *Experience) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	// Simple Q-learning update
	stateKey := experience.State.Hash()
	
	if p.QTable[stateKey] == nil {
		p.QTable[stateKey] = make(map[int]float64)
	}
	
	actionID := 0
	if simpleAction, ok := experience.Action.(*SimpleAction); ok {
		actionID = simpleAction.ID
	}
	
	// Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
	currentQ := p.QTable[stateKey][actionID]
	
	nextStateKey := experience.NextState.Hash()
	maxNextQ := 0.0
	
	if nextQValues, exists := p.QTable[nextStateKey]; exists {
		maxNextQ = math.Inf(-1)
		for _, q := range nextQValues {
			if q > maxNextQ {
				maxNextQ = q
			}
		}
		if maxNextQ == math.Inf(-1) {
			maxNextQ = 0.0
		}
	}
	
	learningRate := 0.1
	discountFactor := 0.95
	
	if experience.Done {
		maxNextQ = 0.0
	}
	
	newQ := currentQ + learningRate*(float64(experience.Reward)+discountFactor*maxNextQ-currentQ)
	p.QTable[stateKey][actionID] = newQ
	
	return nil
}

func (p *SimplePolicy) GetParameters() []float64 {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	
	params := make([]float64, 0)
	
	// Flatten Q-table into parameter vector
	stateKeys := make([]string, 0, len(p.QTable))
	for key := range p.QTable {
		stateKeys = append(stateKeys, key)
	}
	sort.Strings(stateKeys)
	
	for _, stateKey := range stateKeys {
		qValues := p.QTable[stateKey]
		actionIDs := make([]int, 0, len(qValues))
		for actionID := range qValues {
			actionIDs = append(actionIDs, actionID)
		}
		sort.Ints(actionIDs)
		
		for _, actionID := range actionIDs {
			params = append(params, qValues[actionID])
		}
	}
	
	return params
}

func (p *SimplePolicy) SetParameters(params []float64) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	// This is a simplified implementation
	// In practice, you'd need to maintain a consistent parameter structure
	return nil
}

func (p *SimplePolicy) Clone() Policy {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	
	clone := &SimplePolicy{
		QTable:          make(map[string]map[int]float64),
		ExplorationRate: p.ExplorationRate,
		Actions:         make([]Action, len(p.Actions)),
	}
	
	copy(clone.Actions, p.Actions)
	
	for stateKey, qValues := range p.QTable {
		clone.QTable[stateKey] = make(map[int]float64)
		for actionID, value := range qValues {
			clone.QTable[stateKey][actionID] = value
		}
	}
	
	return clone
}

// Example simple environment for testing
type GridWorldEnvironment struct {
	Width    int           `json:"width"`
	Height   int           `json:"height"`
	PlayerX  int           `json:"player_x"`
	PlayerY  int           `json:"player_y"`
	GoalX    int           `json:"goal_x"`
	GoalY    int           `json:"goal_y"`
	Obstacles map[string]bool `json:"obstacles"`
	Done     bool          `json:"done"`
	mutex    sync.RWMutex
}

func NewGridWorldEnvironment(width, height int) *GridWorldEnvironment {
	return &GridWorldEnvironment{
		Width:     width,
		Height:    height,
		PlayerX:   0,
		PlayerY:   0,
		GoalX:     width - 1,
		GoalY:     height - 1,
		Obstacles: make(map[string]bool),
		Done:      false,
	}
}

func (env *GridWorldEnvironment) Reset() State {
	env.mutex.Lock()
	defer env.mutex.Unlock()
	
	env.PlayerX = 0
	env.PlayerY = 0
	env.Done = false
	
	return &SimpleState{
		Values: []float64{float64(env.PlayerX), float64(env.PlayerY)},
	}
}

func (env *GridWorldEnvironment) Step(action Action) (State, Reward, bool, map[string]interface{}) {
	env.mutex.Lock()
	defer env.mutex.Unlock()
	
	if env.Done {
		return env.GetCurrentState(), 0, true, nil
	}
	
	// Parse action
	actionID := 0
	if simpleAction, ok := action.(*SimpleAction); ok {
		actionID = simpleAction.ID
	}
	
	// Apply action (0: up, 1: right, 2: down, 3: left)
	newX, newY := env.PlayerX, env.PlayerY
	
	switch actionID {
	case 0: // up
		newY = max(0, env.PlayerY-1)
	case 1: // right
		newX = min(env.Width-1, env.PlayerX+1)
	case 2: // down
		newY = min(env.Height-1, env.PlayerY+1)
	case 3: // left
		newX = max(0, env.PlayerX-1)
	}
	
	// Check for obstacles
	obstacleKey := fmt.Sprintf("%d,%d", newX, newY)
	if !env.Obstacles[obstacleKey] {
		env.PlayerX = newX
		env.PlayerY = newY
	}
	
	// Calculate reward
	var reward Reward = -0.1 // Small negative reward for each step
	
	// Check if goal reached
	if env.PlayerX == env.GoalX && env.PlayerY == env.GoalY {
		reward = 10.0
		env.Done = true
	}
	
	state := &SimpleState{
		Values: []float64{float64(env.PlayerX), float64(env.PlayerY)},
	}
	
	info := map[string]interface{}{
		"player_position": []int{env.PlayerX, env.PlayerY},
		"goal_position":   []int{env.GoalX, env.GoalY},
	}
	
	return state, reward, env.Done, info
}

func (env *GridWorldEnvironment) GetActionSpace() []Action {
	return []Action{
		&SimpleAction{ID: 0}, // up
		&SimpleAction{ID: 1}, // right
		&SimpleAction{ID: 2}, // down
		&SimpleAction{ID: 3}, // left
	}
}

func (env *GridWorldEnvironment) GetStateSpace() StateSpace {
	return StateSpace{
		Dimensions: []int{env.Width, env.Height},
		Continuous: false,
		Bounds:     [][]float64{{0, float64(env.Width)}, {0, float64(env.Height)}},
	}
}

func (env *GridWorldEnvironment) GetCurrentState() State {
	env.mutex.RLock()
	defer env.mutex.RUnlock()
	
	return &SimpleState{
		Values: []float64{float64(env.PlayerX), float64(env.PlayerY)},
	}
}

func (env *GridWorldEnvironment) IsTerminal() bool {
	env.mutex.RLock()
	defer env.mutex.RUnlock()
	return env.Done
}

func (env *GridWorldEnvironment) Clone() Environment {
	env.mutex.RLock()
	defer env.mutex.RUnlock()
	
	clone := &GridWorldEnvironment{
		Width:     env.Width,
		Height:    env.Height,
		PlayerX:   env.PlayerX,
		PlayerY:   env.PlayerY,
		GoalX:     env.GoalX,
		GoalY:     env.GoalY,
		Obstacles: make(map[string]bool),
		Done:      env.Done,
	}
	
	for key, value := range env.Obstacles {
		clone.Obstacles[key] = value
	}
	
	return clone
}

func (env *GridWorldEnvironment) Render() string {
	env.mutex.RLock()
	defer env.mutex.RUnlock()
	
	result := ""
	for y := 0; y < env.Height; y++ {
		for x := 0; x < env.Width; x++ {
			if x == env.PlayerX && y == env.PlayerY {
				result += "P "
			} else if x == env.GoalX && y == env.GoalY {
				result += "G "
			} else if env.Obstacles[fmt.Sprintf("%d,%d", x, y)] {
				result += "X "
			} else {
				result += ". "
			}
		}
		result += "\n"
	}
	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}