# Parallel Reinforcement Learning Agent

A comprehensive, high-performance parallel reinforcement learning system implemented in Go, featuring multiple RL algorithms, advanced parallel training strategies, and sophisticated concurrent architectures for solving complex sequential decision-making problems with distributed learning capabilities.

## Features

### Core Reinforcement Learning Algorithms
- **Q-Learning**: Tabular and function approximation variants with exploration strategies
- **Deep Q-Networks (DQN)**: Neural network-based value function approximation
- **Actor-Critic Methods**: Policy gradient methods with value function baselines
- **Policy Gradient**: Direct policy optimization with various estimators
- **Advantage Actor-Critic (A3C)**: Asynchronous advantage actor-critic learning
- **Proximal Policy Optimization (PPO)**: Stable policy gradient optimization
- **SARSA**: On-policy temporal difference learning
- **Monte Carlo Tree Search**: Planning-based methods for discrete action spaces
- **Evolutionary Strategies**: Population-based optimization for RL
- **Multi-Agent RL**: Cooperative and competitive multi-agent learning

### Advanced Parallel Training Strategies
- **Asynchronous Actor-Critic**: Multiple actors collecting experiences asynchronously
- **Distributed Training**: Parameter server architecture for large-scale learning
- **Population-Based Training**: Multiple agents with different hyperparameters
- **Experience Replay**: Parallel experience collection and sampling
- **Model Parallelism**: Distributed neural network computation
- **Data Parallelism**: Parallel batch processing and gradient computation
- **Gradient Sharing**: Efficient gradient aggregation across workers
- **Parameter Server**: Centralized parameter management for distributed learning

### Sophisticated Environment Management
- **Environment Abstraction**: Generic interface supporting various RL environments
- **Concurrent Environment Instances**: Parallel environment simulation
- **State Space Management**: Support for discrete and continuous state spaces
- **Action Space Handling**: Discrete and continuous action spaces with constraints
- **Reward Engineering**: Flexible reward function design and shaping
- **Environment Cloning**: Efficient environment replication for parallel actors
- **Custom Environment Support**: Extensible framework for domain-specific environments

### Advanced Experience Management
- **Experience Replay Buffer**: Efficient storage and sampling of experience tuples
- **Prioritized Experience Replay**: Priority-based sampling for improved learning
- **Importance Sampling**: Bias correction for off-policy learning
- **Experience Streaming**: Real-time experience processing and distribution
- **Memory-Efficient Storage**: Optimized data structures for large-scale replay
- **Concurrent Access Control**: Thread-safe operations on shared experience buffers

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Parallel Reinforcement Learning System      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Agent Management Layer           │   │
│  │                                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │ │   Policy    │ │    Value    │ │ Environment │    │   │
│  │ │ Management  │ │  Function   │ │ Management  │    │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Parallel Training Orchestration       │   │
│  │                                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │ │    Actor    │ │   Learner   │ │Coordinator  │    │   │
│  │ │  Management │ │ Management  │ │  Manager    │    │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  │                                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │ │ Experience  │ │ Parameter   │ │   Model     │    │   │
│  │ │   Replay    │ │Synchronizer │ │ Parallelism │    │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Actor Pool                        │   │
│  │                                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │ │   Actor 1   │ │   Actor 2   │ │   Actor N   │    │   │
│  │ │             │ │             │ │             │    │   │
│  │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │    │   │
│  │ │ │ Policy  │ │ │ │ Policy  │ │ │ │ Policy  │ │    │   │
│  │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │    │   │
│  │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │    │   │
│  │ │ │  Env    │ │ │ │  Env    │ │ │ │  Env    │ │    │   │
│  │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │    │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Learner Pool                      │   │
│  │                                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │ │  Learner 1  │ │  Learner 2  │ │  Learner M  │    │   │
│  │ │             │ │             │ │             │    │   │
│  │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │    │   │
│  │ │ │ Neural  │ │ │ │ Neural  │ │ │ │ Neural  │ │    │   │
│  │ │ │Network  │ │ │ │Network  │ │ │ │Network  │ │    │   │
│  │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │    │   │
│  │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │    │   │
│  │ │ │Optimizer│ │ │ │Optimizer│ │ │ │Optimizer│ │    │   │
│  │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │    │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Experience Management System           │   │
│  │                                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │ │  Experience │ │  Priority   │ │ Importance  │    │   │
│  │ │   Buffer    │ │  Sampling   │ │  Sampling   │    │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                  Monitoring and Analytics                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Performance Tracking and Analysis         │   │
│  │                                                     │   │
│  │ • Learning Progress Monitoring  • Parallel Efficiency│   │
│  │ • Episode Reward Tracking       • Resource Utilization│   │
│  │ • Policy Convergence Analysis   • Throughput Metrics │   │
│  │ • Experience Replay Statistics  • Actor Performance  │   │
│  │ • Gradient Flow Visualization   • Memory Usage       │   │
│  │ • Hyperparameter Optimization   • Training Stability │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Actor-Learner Architecture

The system implements a sophisticated actor-learner architecture:

#### Actor Components
- **Experience Collection**: Parallel actors interact with environment instances
- **Policy Execution**: Actors execute policies to generate trajectories
- **Exploration Management**: Dynamic exploration strategies per actor
- **State Processing**: Efficient state representation and preprocessing
- **Action Selection**: Optimized action sampling and deterministic policies

#### Learner Components
- **Batch Processing**: Efficient batched learning from experience replay
- **Gradient Computation**: Parallel gradient calculation and aggregation
- **Parameter Updates**: Optimized parameter synchronization across learners
- **Loss Monitoring**: Real-time loss tracking and convergence analysis
- **Model Checkpointing**: Periodic model saving and restoration

#### Coordination Mechanisms
- **Parameter Broadcasting**: Efficient parameter distribution to actors
- **Experience Streaming**: High-throughput experience data pipeline
- **Synchronization Barriers**: Coordinated updates across distributed components
- **Load Balancing**: Dynamic workload distribution optimization
- **Fault Tolerance**: Robust handling of component failures and recovery

## Usage Examples

### Basic Q-Learning Example

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/yourusername/concurrency-in-golang/parallelrl"
)

func main() {
    // Create a simple grid world environment
    env := parallelrl.NewGridWorldEnvironment(5, 5)
    
    // Define action space
    actions := env.GetActionSpace()
    
    // Create a simple Q-learning policy
    policy := parallelrl.NewSimplePolicy(actions, 0.1) // 10% exploration
    
    // Configure training
    config := parallelrl.DefaultTrainingConfig()
    config.AgentType = parallelrl.QLearning
    config.Algorithm = parallelrl.TabularQLearning
    config.MaxEpisodes = 1000
    config.NumActors = 2
    config.NumLearners = 1
    config.LearningRate = 0.1
    config.DiscountFactor = 0.95
    config.ExplorationRate = 0.1
    
    // Create parallel agent
    agent, err := parallelrl.NewParallelAgent(config, env, policy, nil)
    if err != nil {
        log.Fatalf("Failed to create agent: %v", err)
    }
    
    // Start training
    fmt.Println("Starting Q-Learning training...")
    err = agent.Train()
    if err != nil {
        log.Fatalf("Training failed: %v", err)
    }
    
    // Get final statistics
    stats := agent.GetStatistics()
    fmt.Printf("Training completed!\n")
    fmt.Printf("Episodes: %d\n", stats.EpisodesCompleted)
    fmt.Printf("Average Reward: %.2f\n", stats.AverageReward)
    fmt.Printf("Best Episode Reward: %.2f\n", stats.BestEpisodeReward)
    fmt.Printf("Training Time: %v\n", stats.TotalTrainingTime)
    fmt.Printf("Throughput: %.2f episodes/second\n", stats.ThroughputEPS)
}
```

### Advanced Deep Q-Network (DQN) Example

```go
func advancedDQNExample() {
    // Create a more complex environment
    env := parallelrl.NewGridWorldEnvironment(10, 10)
    
    // Add obstacles to make the environment more challenging
    gridEnv := env.(*parallelrl.GridWorldEnvironment)
    gridEnv.Obstacles["3,3"] = true
    gridEnv.Obstacles["4,4"] = true
    gridEnv.Obstacles["5,5"] = true
    
    actions := env.GetActionSpace()
    policy := parallelrl.NewSimplePolicy(actions, 0.2)
    
    // Advanced DQN configuration
    config := parallelrl.TrainingConfig{
        AgentType:        parallelrl.DeepQNetwork,
        Algorithm:        parallelrl.NeuralNetwork,
        ParallelStrategy: parallelrl.ExperienceReplay,
        
        // Learning parameters
        LearningRate:     0.001,
        DiscountFactor:   0.99,
        ExplorationRate:  1.0,
        ExplorationDecay: 0.995,
        MinExploration:   0.01,
        
        // Training settings
        MaxEpisodes:      5000,
        MaxStepsPerEpisode: 200,
        TimeLimit:        10 * time.Minute,
        TargetReward:     9.5, // Close to optimal reward of 10
        
        // Parallel settings
        NumWorkers:  8,
        NumActors:   6,
        NumLearners: 2,
        BatchSize:   64,
        UpdateFrequency: 4,
        
        // Experience replay
        ReplayBufferSize: 50000,
        ReplayBatchSize:  32,
        ReplayStartSize:  1000,
        PriorityReplay:   true,
        ImportanceSampling: true,
        
        // Neural network architecture
        HiddenLayers:     []int{128, 128, 64},
        ActivationFunc:   "relu",
        Optimizer:        "adam",
        LossFunction:     "mse",
        L2Regularization: 0.0001,
        DropoutRate:      0.1,
        
        // Advanced settings
        TargetNetworkUpdate: 1000,
        GradientClipping:    1.0,
        NoiseScale:          0.1,
        
        // Monitoring
        LogInterval:       100,
        SaveInterval:      1000,
        EvaluationEpisodes: 20,
        EnableLogging:     true,
        Seed:              42,
    }
    
    agent, err := parallelrl.NewParallelAgent(config, env, policy, nil)
    if err != nil {
        log.Fatalf("Failed to create DQN agent: %v", err)
    }
    
    // Monitor training progress
    go monitorTraining(agent)
    
    fmt.Println("Starting DQN training with experience replay...")
    err = agent.Train()
    if err != nil {
        log.Fatalf("DQN training failed: %v", err)
    }
    
    // Evaluate trained agent
    evaluateAgent(agent, env, 50)
}

func monitorTraining(agent *parallelrl.ParallelAgent) {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()
    
    fmt.Println("=== Training Progress Monitor ===")
    fmt.Println("Time\t\tEpisodes\tAvg Reward\tThroughput")
    
    startTime := time.Now()
    
    for range ticker.C {
        if !agent.IsRunning() {
            break
        }
        
        stats := agent.GetStatistics()
        elapsed := time.Since(startTime)
        
        fmt.Printf("%v\t%d\t\t%.2f\t\t%.2f eps\n",
            elapsed.Round(time.Second),
            stats.EpisodesCompleted,
            stats.AverageReward,
            stats.ThroughputEPS)
    }
}

func evaluateAgent(agent *parallelrl.ParallelAgent, env parallelrl.Environment, episodes int) {
    fmt.Printf("\n=== Agent Evaluation (%d episodes) ===\n", episodes)
    
    policy := agent.GetPolicy()
    rewards := make([]float64, episodes)
    
    for i := 0; i < episodes; i++ {
        evalEnv := env.Clone()
        state := evalEnv.Reset()
        totalReward := 0.0
        
        for step := 0; step < 200; step++ {
            action := policy.SelectAction(state)
            nextState, reward, done, _ := evalEnv.Step(action)
            
            totalReward += float64(reward)
            state = nextState
            
            if done {
                break
            }
        }
        
        rewards[i] = totalReward
    }
    
    // Calculate statistics
    avgReward := calculateMean(rewards)
    stdReward := calculateStdDev(rewards)
    minReward := calculateMin(rewards)
    maxReward := calculateMax(rewards)
    
    fmt.Printf("Average Reward: %.2f ± %.2f\n", avgReward, stdReward)
    fmt.Printf("Min Reward: %.2f\n", minReward)
    fmt.Printf("Max Reward: %.2f\n", maxReward)
    fmt.Printf("Success Rate: %.1f%%\n", calculateSuccessRate(rewards, 8.0))
}
```

### Multi-Agent Reinforcement Learning Example

```go
func multiAgentExample() {
    // Create environment supporting multiple agents
    env := parallelrl.NewGridWorldEnvironment(8, 8)
    
    // Configure multi-agent training
    config := parallelrl.DefaultTrainingConfig()
    config.AgentType = parallelrl.MultiAgent
    config.ParallelStrategy = parallelrl.PopulationBased
    config.NumAgents = 4
    config.NumActors = 8  // 2 actors per agent
    config.NumLearners = 4 // 1 learner per agent
    config.CooperativeReward = true
    config.CommunicationRange = 2.0
    config.MaxEpisodes = 3000
    
    // Create multiple policies for different agents
    actions := env.GetActionSpace()
    policies := make([]parallelrl.Policy, config.NumAgents)
    for i := 0; i < config.NumAgents; i++ {
        // Each agent starts with different exploration rates
        explorationRate := 0.1 + float64(i)*0.05
        policies[i] = parallelrl.NewSimplePolicy(actions, explorationRate)
    }
    
    // Train multiple agents
    agents := make([]*parallelrl.ParallelAgent, config.NumAgents)
    var wg sync.WaitGroup
    
    for i := 0; i < config.NumAgents; i++ {
        wg.Add(1)
        go func(agentID int) {
            defer wg.Done()
            
            agentConfig := config
            agentConfig.Seed = config.Seed + int64(agentID)
            
            agent, err := parallelrl.NewParallelAgent(agentConfig, env, policies[agentID], nil)
            if err != nil {
                log.Printf("Agent %d creation failed: %v", agentID, err)
                return
            }
            
            agents[agentID] = agent
            
            fmt.Printf("Starting training for Agent %d\n", agentID)
            err = agent.Train()
            if err != nil {
                log.Printf("Agent %d training failed: %v", agentID, err)
            } else {
                stats := agent.GetStatistics()
                fmt.Printf("Agent %d completed: %.2f avg reward\n", agentID, stats.AverageReward)
            }
        }(i)
    }
    
    wg.Wait()
    
    // Compare agent performance
    fmt.Println("\n=== Multi-Agent Performance Comparison ===")
    for i, agent := range agents {
        if agent != nil {
            stats := agent.GetStatistics()
            fmt.Printf("Agent %d: Episodes=%d, Avg Reward=%.2f, Efficiency=%.1f%%\n",
                i, stats.EpisodesCompleted, stats.AverageReward, stats.ParallelEfficiency*100)
        }
    }
}
```

### Asynchronous Advantage Actor-Critic (A3C) Example

```go
func a3cExample() {
    env := parallelrl.NewGridWorldEnvironment(6, 6)
    actions := env.GetActionSpace()
    
    // A3C configuration
    config := parallelrl.TrainingConfig{
        AgentType:        parallelrl.AdvantageActorCritic,
        Algorithm:        parallelrl.NeuralNetwork,
        ParallelStrategy: parallelrl.AsyncActorCritic,
        
        LearningRate:     0.0001,
        DiscountFactor:   0.99,
        ExplorationRate:  0.1,
        
        MaxEpisodes:      2000,
        NumActors:        8,  // Asynchronous actors
        NumLearners:      2,  // Shared learners
        BatchSize:        20, // Smaller batches for A3C
        UpdateFrequency:  5,  // Frequent updates
        
        HiddenLayers:     []int{64, 32},
        ActivationFunc:   "relu",
        Optimizer:        "rmsprop",
        LearningRate:     0.0007,
        
        GradientClipping: 0.5,
        EnableLogging:    true,
        LogInterval:      50,
        Seed:             123,
    }
    
    // Create actor and critic networks (simplified for example)
    policy := parallelrl.NewSimplePolicy(actions, config.ExplorationRate)
    
    agent, err := parallelrl.NewParallelAgent(config, env, policy, nil)
    if err != nil {
        log.Fatalf("Failed to create A3C agent: %v", err)
    }
    
    // Monitor learning curves
    go plotLearningCurves(agent)
    
    fmt.Println("Starting A3C training...")
    err = agent.Train()
    if err != nil {
        log.Fatalf("A3C training failed: %v", err)
    }
    
    // Analyze convergence
    analyzeConvergence(agent)
}

func plotLearningCurves(agent *parallelrl.ParallelAgent) {
    ticker := time.NewTicker(2 * time.Second)
    defer ticker.Stop()
    
    rewardHistory := make([]float64, 0)
    
    for range ticker.C {
        if !agent.IsRunning() {
            break
        }
        
        stats := agent.GetStatistics()
        rewardHistory = append(rewardHistory, stats.AverageReward)
        
        // Simple console plotting (in real implementation, use proper plotting library)
        if len(rewardHistory) > 1 {
            trend := "→"
            if rewardHistory[len(rewardHistory)-1] > rewardHistory[len(rewardHistory)-2] {
                trend = "↗"
            } else if rewardHistory[len(rewardHistory)-1] < rewardHistory[len(rewardHistory)-2] {
                trend = "↘"
            }
            
            fmt.Printf("Reward: %.2f %s (Episode %d)\n", 
                stats.AverageReward, trend, stats.EpisodesCompleted)
        }
    }
}

func analyzeConvergence(agent *parallelrl.ParallelAgent) {
    stats := agent.GetStatistics()
    
    fmt.Println("\n=== Convergence Analysis ===")
    fmt.Printf("Total Episodes: %d\n", stats.EpisodesCompleted)
    fmt.Printf("Training Time: %v\n", stats.TotalTrainingTime)
    fmt.Printf("Final Average Reward: %.2f\n", stats.AverageReward)
    fmt.Printf("Best Episode Reward: %.2f\n", stats.BestEpisodeReward)
    fmt.Printf("Parallel Efficiency: %.1f%%\n", stats.ParallelEfficiency*100)
    
    // Analyze reward trend
    if len(stats.RewardTrend) > 100 {
        recent := stats.RewardTrend[len(stats.RewardTrend)-100:]
        variance := calculateVariance(recent)
        
        if variance < 0.1 {
            fmt.Println("Status: CONVERGED (low variance)")
        } else {
            fmt.Println("Status: LEARNING (high variance)")
        }
        
        fmt.Printf("Recent Reward Variance: %.3f\n", variance)
    }
}
```

### Custom Environment Integration

```go
// Custom trading environment example
type TradingEnvironment struct {
    prices      []float64
    position    int
    cash        float64
    portfolio   float64
    currentStep int
    maxSteps    int
    mutex       sync.RWMutex
}

func NewTradingEnvironment(prices []float64, initialCash float64) *TradingEnvironment {
    return &TradingEnvironment{
        prices:   prices,
        cash:     initialCash,
        maxSteps: len(prices) - 1,
    }
}

func (env *TradingEnvironment) Reset() parallelrl.State {
    env.mutex.Lock()
    defer env.mutex.Unlock()
    
    env.position = 0
    env.cash = 1000.0 // Reset to initial cash
    env.portfolio = env.cash
    env.currentStep = 0
    
    return &parallelrl.SimpleState{
        Values: []float64{
            env.prices[env.currentStep],
            float64(env.position),
            env.cash,
            env.portfolio,
        },
    }
}

func (env *TradingEnvironment) Step(action parallelrl.Action) (parallelrl.State, parallelrl.Reward, bool, map[string]interface{}) {
    env.mutex.Lock()
    defer env.mutex.Unlock()
    
    if env.currentStep >= env.maxSteps {
        return env.getCurrentState(), 0, true, nil
    }
    
    actionID := 0
    if simpleAction, ok := action.(*parallelrl.SimpleAction); ok {
        actionID = simpleAction.ID
    }
    
    currentPrice := env.prices[env.currentStep]
    
    // Actions: 0=Hold, 1=Buy, 2=Sell
    reward := 0.0
    
    switch actionID {
    case 1: // Buy
        if env.cash >= currentPrice {
            env.position++
            env.cash -= currentPrice
        }
    case 2: // Sell
        if env.position > 0 {
            env.position--
            env.cash += currentPrice
        }
    }
    
    env.currentStep++
    
    // Calculate portfolio value
    if env.currentStep < len(env.prices) {
        nextPrice := env.prices[env.currentStep]
        env.portfolio = env.cash + float64(env.position)*nextPrice
        
        // Reward based on portfolio performance
        if env.currentStep > 0 {
            prevPortfolio := 1000.0 // Baseline
            reward = (env.portfolio - prevPortfolio) / prevPortfolio
        }
    }
    
    done := env.currentStep >= env.maxSteps
    
    info := map[string]interface{}{
        "portfolio_value": env.portfolio,
        "position":        env.position,
        "cash":           env.cash,
        "current_price":  currentPrice,
    }
    
    return env.getCurrentState(), parallelrl.Reward(reward), done, info
}

func (env *TradingEnvironment) GetActionSpace() []parallelrl.Action {
    return []parallelrl.Action{
        &parallelrl.SimpleAction{ID: 0}, // Hold
        &parallelrl.SimpleAction{ID: 1}, // Buy
        &parallelrl.SimpleAction{ID: 2}, // Sell
    }
}

func (env *TradingEnvironment) GetStateSpace() parallelrl.StateSpace {
    return parallelrl.StateSpace{
        Dimensions: []int{4}, // price, position, cash, portfolio
        Continuous: true,
        Bounds: [][]float64{
            {0, 1000},      // price range
            {-100, 100},    // position range
            {0, 10000},     // cash range
            {0, 20000},     // portfolio range
        },
    }
}

func (env *TradingEnvironment) getCurrentState() parallelrl.State {
    if env.currentStep >= len(env.prices) {
        return &parallelrl.SimpleState{Values: []float64{0, 0, 0, 0}}
    }
    
    return &parallelrl.SimpleState{
        Values: []float64{
            env.prices[env.currentStep],
            float64(env.position),
            env.cash,
            env.portfolio,
        },
    }
}

func (env *TradingEnvironment) GetCurrentState() parallelrl.State {
    env.mutex.RLock()
    defer env.mutex.RUnlock()
    return env.getCurrentState()
}

func (env *TradingEnvironment) IsTerminal() bool {
    env.mutex.RLock()
    defer env.mutex.RUnlock()
    return env.currentStep >= env.maxSteps
}

func (env *TradingEnvironment) Clone() parallelrl.Environment {
    env.mutex.RLock()
    defer env.mutex.RUnlock()
    
    clone := &TradingEnvironment{
        prices:      make([]float64, len(env.prices)),
        position:    env.position,
        cash:        env.cash,
        portfolio:   env.portfolio,
        currentStep: env.currentStep,
        maxSteps:    env.maxSteps,
    }
    
    copy(clone.prices, env.prices)
    return clone
}

func (env *TradingEnvironment) Render() string {
    env.mutex.RLock()
    defer env.mutex.RUnlock()
    
    return fmt.Sprintf("Step: %d, Price: %.2f, Position: %d, Cash: %.2f, Portfolio: %.2f",
        env.currentStep, env.prices[env.currentStep], env.position, env.cash, env.portfolio)
}

func tradingExample() {
    // Generate sample price data
    prices := generatePriceData(1000, 100.0, 0.02)
    
    env := NewTradingEnvironment(prices, 1000.0)
    actions := env.GetActionSpace()
    policy := parallelrl.NewSimplePolicy(actions, 0.1)
    
    config := parallelrl.DefaultTrainingConfig()
    config.MaxEpisodes = 500
    config.NumActors = 4
    config.NumLearners = 2
    config.LearningRate = 0.001
    config.ReplayBufferSize = 10000
    
    agent, err := parallelrl.NewParallelAgent(config, env, policy, nil)
    if err != nil {
        log.Fatalf("Failed to create trading agent: %v", err)
    }
    
    fmt.Println("Training trading agent...")
    err = agent.Train()
    if err != nil {
        log.Fatalf("Trading agent training failed: %v", err)
    }
    
    // Test the trained agent
    testTradingAgent(agent, env)
}

func generatePriceData(length int, initialPrice, volatility float64) []float64 {
    prices := make([]float64, length)
    prices[0] = initialPrice
    
    for i := 1; i < length; i++ {
        change := (rand.Float64() - 0.5) * volatility * prices[i-1]
        prices[i] = prices[i-1] + change
        
        // Ensure price doesn't go negative
        if prices[i] < 1.0 {
            prices[i] = 1.0
        }
    }
    
    return prices
}

func testTradingAgent(agent *parallelrl.ParallelAgent, env parallelrl.Environment) {
    fmt.Println("\n=== Trading Agent Test ===")
    
    policy := agent.GetPolicy()
    testEnv := env.Clone()
    state := testEnv.Reset()
    
    totalReward := 0.0
    actions := []string{"Hold", "Buy", "Sell"}
    
    for step := 0; step < 100; step++ {
        action := policy.SelectAction(state)
        nextState, reward, done, info := testEnv.Step(action)
        
        totalReward += float64(reward)
        
        actionID := 0
        if simpleAction, ok := action.(*parallelrl.SimpleAction); ok {
            actionID = simpleAction.ID
        }
        
        if step%10 == 0 {
            fmt.Printf("Step %d: Action=%s, Reward=%.4f, Portfolio=%.2f\n",
                step, actions[actionID], float64(reward), info["portfolio_value"])
        }
        
        if done {
            break
        }
        
        state = nextState
    }
    
    fmt.Printf("Total Reward: %.4f\n", totalReward)
}
```

### Hyperparameter Optimization Example

```go
func hyperparameterOptimization() {
    env := parallelrl.NewGridWorldEnvironment(5, 5)
    actions := env.GetActionSpace()
    
    // Define hyperparameter search space
    hyperparams := []struct {
        learningRate     float64
        explorationRate  float64
        discountFactor   float64
        numActors        int
    }{
        {0.1, 0.1, 0.9, 2},
        {0.1, 0.2, 0.95, 4},
        {0.01, 0.1, 0.99, 2},
        {0.05, 0.15, 0.95, 4},
        {0.1, 0.05, 0.9, 6},
    }
    
    results := make([]struct {
        config parallelrl.TrainingConfig
        score  float64
    }, len(hyperparams))
    
    var wg sync.WaitGroup
    
    for i, params := range hyperparams {
        wg.Add(1)
        go func(idx int, hp struct {
            learningRate     float64
            explorationRate  float64
            discountFactor   float64
            numActors        int
        }) {
            defer wg.Done()
            
            policy := parallelrl.NewSimplePolicy(actions, hp.explorationRate)
            
            config := parallelrl.DefaultTrainingConfig()
            config.LearningRate = hp.learningRate
            config.ExplorationRate = hp.explorationRate
            config.DiscountFactor = hp.discountFactor
            config.NumActors = hp.numActors
            config.MaxEpisodes = 500
            config.EnableLogging = false
            config.Seed = int64(idx) // Different seed for each config
            
            agent, err := parallelrl.NewParallelAgent(config, env, policy, nil)
            if err != nil {
                log.Printf("Config %d failed to create agent: %v", idx, err)
                return
            }
            
            err = agent.Train()
            if err != nil {
                log.Printf("Config %d training failed: %v", idx, err)
                return
            }
            
            stats := agent.GetStatistics()
            score := stats.AverageReward
            
            results[idx] = struct {
                config parallelrl.TrainingConfig
                score  float64
            }{config, score}
            
            fmt.Printf("Config %d: LR=%.3f, Exp=%.2f, Gamma=%.2f, Actors=%d, Score=%.2f\n",
                idx, hp.learningRate, hp.explorationRate, hp.discountFactor, hp.numActors, score)
                
        }(i, params)
    }
    
    wg.Wait()
    
    // Find best configuration
    bestIdx := 0
    bestScore := results[0].score
    
    for i, result := range results {
        if result.score > bestScore {
            bestScore = result.score
            bestIdx = i
        }
    }
    
    fmt.Printf("\n=== Best Configuration ===\n")
    best := results[bestIdx]
    fmt.Printf("Score: %.2f\n", best.score)
    fmt.Printf("Learning Rate: %.3f\n", best.config.LearningRate)
    fmt.Printf("Exploration Rate: %.2f\n", best.config.ExplorationRate)
    fmt.Printf("Discount Factor: %.2f\n", best.config.DiscountFactor)
    fmt.Printf("Number of Actors: %d\n", best.config.NumActors)
}
```

## Configuration Options

### TrainingConfig Fields

#### Agent Configuration
- **AgentType**: Type of RL agent (Q-Learning, DQN, Actor-Critic, etc.)
- **Algorithm**: Core learning algorithm (Tabular, Neural Network, etc.)
- **ParallelStrategy**: Parallel training approach (Async, Distributed, etc.)

#### Learning Parameters
- **LearningRate**: Step size for parameter updates
- **DiscountFactor**: Future reward discount factor (γ)
- **ExplorationRate**: Initial exploration probability (ε)
- **ExplorationDecay**: Rate of exploration decay over time
- **MinExploration**: Minimum exploration rate to maintain

#### Training Settings
- **MaxEpisodes**: Maximum number of training episodes
- **MaxStepsPerEpisode**: Maximum steps per episode
- **TimeLimit**: Maximum training time
- **TargetReward**: Target reward threshold for early stopping

#### Parallel Settings
- **NumWorkers**: Total number of worker threads
- **NumActors**: Number of parallel experience collectors
- **NumLearners**: Number of parallel learning processes
- **BatchSize**: Batch size for learning updates
- **UpdateFrequency**: Frequency of parameter synchronization

#### Experience Replay
- **ReplayBufferSize**: Maximum size of experience replay buffer
- **ReplayBatchSize**: Batch size for replay sampling
- **ReplayStartSize**: Minimum buffer size before learning starts
- **PriorityReplay**: Enable prioritized experience replay
- **ImportanceSampling**: Enable importance sampling for off-policy correction

#### Neural Network Architecture
- **HiddenLayers**: Layer sizes for neural networks
- **ActivationFunc**: Activation function (relu, tanh, sigmoid)
- **Optimizer**: Optimization algorithm (adam, sgd, rmsprop)
- **LossFunction**: Loss function for training
- **L2Regularization**: L2 regularization strength
- **DropoutRate**: Dropout probability for regularization

#### Advanced Settings
- **TargetNetworkUpdate**: Frequency of target network updates
- **GradientClipping**: Maximum gradient norm for clipping
- **NoiseScale**: Scale for parameter or action noise
- **ParameterNoise**: Enable parameter space noise

#### Multi-Agent Settings
- **NumAgents**: Number of agents in multi-agent scenarios
- **CommunicationRange**: Range for agent communication
- **CooperativeReward**: Enable cooperative reward shaping
- **CompetitiveReward**: Enable competitive reward shaping

#### Monitoring and Logging
- **LogInterval**: Episode interval for logging progress
- **SaveInterval**: Episode interval for model checkpoints
- **EvaluationEpisodes**: Number of episodes for evaluation
- **EnableLogging**: Enable detailed logging
- **EnableVisualizer**: Enable real-time visualization

#### Performance Tuning
- **Seed**: Random seed for reproducibility
- **NumCPUs**: Number of CPU cores to utilize
- **UseGPU**: Enable GPU acceleration (if available)
- **MemoryLimit**: Memory usage limit in bytes

### Default Configuration

```go
config := parallelrl.DefaultTrainingConfig()
// Customize for your specific needs
config.AgentType = parallelrl.DeepQNetwork
config.Algorithm = parallelrl.NeuralNetwork
config.ParallelStrategy = parallelrl.ExperienceReplay
config.MaxEpisodes = 5000
config.NumActors = 8
config.NumLearners = 2
config.ReplayBufferSize = 100000
config.HiddenLayers = []int{256, 256, 128}
config.LearningRate = 0.0001
```

## Parallel Training Strategies Deep Dive

### Asynchronous Actor-Critic (A3C)

A3C implements multiple actors that interact with separate environment instances and asynchronously update a shared policy and value function:

#### Architecture
- **Global Networks**: Shared policy and value networks
- **Worker Threads**: Independent actors with local network copies
- **Asynchronous Updates**: Workers update global networks without synchronization
- **Experience Streaming**: Direct learning from experience without replay buffer

#### Advantages
- **Sample Efficiency**: On-policy learning with immediate updates
- **Stability**: Reduced correlation between updates
- **Scalability**: Linear scaling with number of actors
- **Memory Efficiency**: No need for large replay buffers

#### Implementation Details
```go
type A3CWorker struct {
    GlobalPolicy    Policy
    GlobalValue     ValueFunction
    LocalPolicy     Policy
    LocalValue      ValueFunction
    Environment     Environment
    ExperienceBuffer []*Experience
    UpdateFrequency int
}

func (worker *A3CWorker) RunEpisode() {
    // Collect experience with local networks
    for step := 0; step < worker.UpdateFrequency; step++ {
        // Interact with environment
        // Store experience
    }
    
    // Compute advantages and policy gradients
    // Update global networks
    // Sync local networks with global
}
```

### Distributed Training with Parameter Server

Parameter server architecture separates parameter storage from computation:

#### Components
- **Parameter Server**: Centralized parameter storage and updates
- **Worker Nodes**: Distributed computation of gradients
- **Communication Layer**: Efficient parameter and gradient exchange
- **Load Balancer**: Dynamic workload distribution

#### Synchronization Modes
- **Synchronous**: Wait for all workers before updates
- **Asynchronous**: Update immediately when gradients arrive
- **Bounded Staleness**: Limit staleness of parameter updates
- **Local SGD**: Periodic synchronization with local updates

### Experience Replay Variants

#### Uniform Random Sampling
```go
func (buffer *ReplayBuffer) SampleUniform(batchSize int) []*Experience {
    indices := make([]int, batchSize)
    for i := range indices {
        indices[i] = rand.Intn(buffer.Size())
    }
    return buffer.GetExperiences(indices)
}
```

#### Prioritized Experience Replay
```go
func (buffer *PrioritizedReplayBuffer) SamplePrioritized(batchSize int) ([]*Experience, []float64) {
    // Sample based on TD-error priorities
    priorities := buffer.GetPriorities()
    indices := buffer.ProportionalSample(priorities, batchSize)
    
    // Compute importance sampling weights
    weights := buffer.ComputeImportanceWeights(indices)
    
    return buffer.GetExperiences(indices), weights
}
```

#### Hindsight Experience Replay (HER)
```go
func (buffer *HERBuffer) AddHERExperiences(episode *Episode) {
    for i, exp := range episode.Experiences {
        // Add original experience
        buffer.Add(exp)
        
        // Add hindsight experiences with different goals
        for _, futureState := range episode.Experiences[i+1:] {
            hindsightExp := &Experience{
                State:     exp.State,
                Action:    exp.Action,
                Reward:    buffer.ComputeHindsightReward(exp.NextState, futureState.State),
                NextState: exp.NextState,
                Done:      buffer.IsGoalAchieved(exp.NextState, futureState.State),
            }
            buffer.Add(hindsightExp)
        }
    }
}
```

## Performance Characteristics

### Computational Complexity

#### Training Complexity
- **Actor Overhead**: O(A × E × S) where A = actors, E = episodes, S = steps per episode
- **Learner Complexity**: O(L × B × N) where L = learners, B = batch size, N = network complexity
- **Experience Replay**: O(R × log R) for prioritized sampling where R = replay buffer size
- **Parameter Synchronization**: O(P × W) where P = parameters, W = workers

#### Memory Complexity
- **Experience Storage**: O(R × (S + A)) where R = buffer size, S = state size, A = action size
- **Network Parameters**: O(∑ᵢ(Lᵢ × Lᵢ₊₁)) for fully connected layers
- **Actor Memory**: O(A × (S + P)) where A = actors, S = state size, P = parameters
- **Coordination Overhead**: O(W × M) where W = workers, M = message size

### Scalability Metrics

| Parallel Strategy | Ideal Speedup | Communication Overhead | Memory Overhead | Scalability Limit |
|------------------|---------------|------------------------|-----------------|-------------------|
| Independent Actors | Linear | Minimal | Low | Environment Bound |
| A3C | Near-Linear | Low | Low | Diminishing Returns |
| Parameter Server | Sub-Linear | Medium | Medium | Network Bandwidth |
| Experience Replay | Variable | High | High | Memory Bound |
| Population-Based | Linear | Low | High | Computation Bound |

### Performance Optimization Techniques

#### Memory Optimization
- **Experience Compression**: Compress state representations
- **Lazy Loading**: Load experiences on-demand
- **Memory Pooling**: Reuse allocated memory structures
- **Garbage Collection**: Optimize Go GC for RL workloads

#### Computational Optimization
- **Vectorized Operations**: Batch processing where possible
- **SIMD Instructions**: Utilize hardware acceleration
- **Graph Optimization**: Optimize neural network computation graphs
- **Mixed Precision**: Use lower precision for non-critical computations

#### Communication Optimization
- **Gradient Compression**: Reduce communication volume
- **Asynchronous Updates**: Overlap computation and communication
- **Locality-Aware Scheduling**: Minimize data movement
- **Adaptive Batch Sizes**: Dynamic batch sizing based on performance

## Advanced Features

### Curriculum Learning

Gradually increase problem difficulty during training:

```go
type CurriculumManager struct {
    stages []TrainingStage
    currentStage int
    progressThreshold float64
}

type TrainingStage struct {
    name string
    environmentConfig EnvironmentConfig
    maxEpisodes int
    targetPerformance float64
}

func (cm *CurriculumManager) ShouldProgress(performance float64) bool {
    return performance >= cm.stages[cm.currentStage].targetPerformance
}

func (cm *CurriculumManager) NextStage() EnvironmentConfig {
    if cm.currentStage < len(cm.stages)-1 {
        cm.currentStage++
    }
    return cm.stages[cm.currentStage].environmentConfig
}
```

### Meta-Learning

Learn to learn across multiple tasks:

```go
type MetaLearner struct {
    baseAgent Agent
    taskDistribution []Task
    adaptationSteps int
    metaLearningRate float64
}

func (ml *MetaLearner) MetaTrain(episodes int) {
    for episode := 0; episode < episodes; episode++ {
        // Sample task from distribution
        task := ml.taskDistribution[rand.Intn(len(ml.taskDistribution))]
        
        // Fast adaptation on task
        adaptedAgent := ml.baseAgent.Clone()
        adaptedAgent.FastAdapt(task, ml.adaptationSteps)
        
        // Meta-update based on adaptation performance
        ml.MetaUpdate(adaptedAgent, task)
    }
}
```

### Transfer Learning

Transfer knowledge between related tasks:

```go
type TransferLearner struct {
    sourceAgent Agent
    targetAgent Agent
    transferLayers []string
    finetuneRate float64
}

func (tl *TransferLearner) TransferKnowledge() {
    sourceParams := tl.sourceAgent.GetParameters()
    
    // Transfer specific layers
    for _, layer := range tl.transferLayers {
        tl.targetAgent.SetLayerParameters(layer, sourceParams[layer])
    }
    
    // Freeze transferred layers initially
    tl.targetAgent.FreezeParameters(tl.transferLayers)
}
```

### Hierarchical Reinforcement Learning

Learn policies at multiple temporal scales:

```go
type HierarchicalAgent struct {
    metaPolicy Policy
    subPolicies []Policy
    goalSpace []Goal
    subgoalHorizon int
}

func (ha *HierarchicalAgent) SelectAction(state State) Action {
    // Meta-policy selects subgoal
    subgoal := ha.metaPolicy.SelectSubgoal(state)
    
    // Sub-policy selects action to achieve subgoal
    return ha.subPolicies[subgoal.ID].SelectAction(state, subgoal)
}
```

## Best Practices

### Algorithm Selection
1. **Tabular Methods**: Use for small, discrete state spaces with exact solutions needed
2. **Function Approximation**: Use for large or continuous state spaces
3. **Policy Gradient**: Use when policy parameterization is more natural than value functions
4. **Actor-Critic**: Combine benefits of policy gradient and value-based methods
5. **Model-Based**: Use when environment models are available or learnable

### Hyperparameter Tuning
1. **Learning Rate**: Start with 0.001 for neural networks, 0.1 for tabular methods
2. **Exploration**: Begin with high exploration (0.1-1.0), decay gradually
3. **Discount Factor**: Use 0.95-0.99 for most problems
4. **Batch Size**: 32-256 for most deep RL applications
5. **Network Architecture**: Start simple, increase complexity as needed

### Parallel Training
1. **Actor Count**: Use 2-8x number of CPU cores for good utilization
2. **Learner Count**: Usually 1-2 learners per 4-8 actors
3. **Update Frequency**: Balance between sample efficiency and synchronization overhead
4. **Experience Replay**: Size 10-100x batch size, start learning when 10% full
5. **Target Networks**: Update every 1000-10000 steps for stability

### Debugging and Monitoring
1. **Reward Tracking**: Monitor episode rewards, moving averages, and variance
2. **Loss Monitoring**: Track policy loss, value loss, and gradient norms
3. **Exploration Tracking**: Monitor action entropy and exploration rates
4. **Performance Profiling**: Identify bottlenecks in training pipeline
5. **Convergence Analysis**: Check for oscillations, plateaus, and instabilities

### Environment Design
1. **Reward Shaping**: Design rewards to guide learning without over-constraining
2. **State Representation**: Ensure states capture relevant information
3. **Action Spaces**: Design intuitive and learnable action spaces
4. **Episode Structure**: Balance episode length with learning signal
5. **Stochasticity**: Add appropriate randomness for robustness

## Common Use Cases

### Game Playing
- **Board Games**: Chess, Go, checkers with perfect information
- **Video Games**: Real-time strategy, first-person shooters, racing games
- **Card Games**: Poker, bridge with imperfect information
- **Arcade Games**: Classic Atari games, modern mobile games

### Robotics and Control
- **Robot Navigation**: Path planning, obstacle avoidance, SLAM
- **Manipulation**: Grasping, assembly, tool use
- **Locomotion**: Walking, running, climbing for legged robots
- **Autonomous Vehicles**: Driving, parking, traffic navigation

### Finance and Trading
- **Algorithmic Trading**: High-frequency trading, portfolio optimization
- **Risk Management**: Dynamic hedging, credit risk assessment
- **Market Making**: Spread optimization, inventory management
- **Cryptocurrency**: Trading, mining strategy optimization

### Industrial Optimization
- **Manufacturing**: Production scheduling, quality control
- **Supply Chain**: Inventory management, logistics optimization
- **Energy**: Smart grid management, renewable energy optimization
- **Telecommunications**: Network routing, resource allocation

### Healthcare and Medicine
- **Treatment Planning**: Personalized therapy recommendations
- **Drug Discovery**: Molecular design, compound optimization
- **Medical Imaging**: Automated diagnosis, treatment planning
- **Hospital Operations**: Scheduling, resource allocation

### Natural Language Processing
- **Dialogue Systems**: Chatbots, virtual assistants
- **Machine Translation**: Neural machine translation
- **Text Generation**: Creative writing, content generation
- **Information Extraction**: Knowledge graph construction

## Limitations and Considerations

### Algorithm Limitations
1. **Sample Complexity**: RL often requires many samples for complex problems
2. **Exploration Challenge**: Balancing exploration and exploitation
3. **Partial Observability**: Handling incomplete state information
4. **Non-Stationarity**: Adapting to changing environments
5. **Reward Engineering**: Designing appropriate reward functions

### Parallel Training Challenges
1. **Synchronization Overhead**: Communication costs in distributed training
2. **Load Balancing**: Ensuring equal workload distribution
3. **Fault Tolerance**: Handling worker failures gracefully
4. **Reproducibility**: Ensuring consistent results with parallel training
5. **Resource Contention**: Managing shared resources effectively

### Implementation Considerations
1. **Memory Management**: Go's garbage collector impact on performance
2. **Numerical Stability**: Floating-point precision in neural networks
3. **Concurrent Access**: Thread-safe operations on shared data structures
4. **Platform Dependencies**: Performance differences across architectures
5. **Debugging Complexity**: Debugging parallel and distributed systems

### Future Enhancements

Planned improvements for future versions:

- **GPU Acceleration**: CUDA/OpenCL support for neural network training
- **Distributed Computing**: Multi-node training across clusters
- **Advanced Algorithms**: PPO, SAC, TD3, and other state-of-the-art methods
- **Environment Integrations**: OpenAI Gym, Unity ML-Agents compatibility
- **Visualization Tools**: Real-time training visualization and analysis
- **Model Compression**: Quantization and pruning for deployment
- **Transfer Learning**: Advanced transfer and meta-learning capabilities
- **Multi-Task Learning**: Simultaneous learning across multiple tasks
- **Continual Learning**: Learning new tasks without forgetting old ones
- **Explainable AI**: Interpretability tools for RL policies

## Contributing and Extension

### Plugin Architecture
The system supports extensible components:

```go
type RLPlugin interface {
    Initialize(agent *ParallelAgent) error
    OnEpisodeStart(episode *Episode) error
    OnEpisodeEnd(episode *Episode) error
    OnUpdate(update *ParameterUpdate) error
    Shutdown() error
}

// Register plugins
agent.RegisterPlugin(&CustomRewardShapingPlugin{})
agent.RegisterPlugin(&AdvancedVisualizationPlugin{})
agent.RegisterPlugin(&ExperimentTrackingPlugin{})
```

### Custom Implementations
- **Custom Environments**: Implement the Environment interface for domain-specific problems
- **Custom Policies**: Implement the Policy interface for specialized decision-making
- **Custom Algorithms**: Extend the Agent interface for novel RL algorithms
- **Custom Networks**: Implement neural network architectures for specific needs

### Testing and Development
- **Unit Tests**: Comprehensive test coverage for all components
- **Integration Tests**: End-to-end testing of complete training pipelines
- **Benchmark Suite**: Performance benchmarks for scalability analysis
- **Example Environments**: Rich set of example environments for learning

### Research Applications
The framework is designed to support:
- **Algorithm Development**: Rapid prototyping of new RL algorithms
- **Scalability Research**: Investigation of parallel training techniques
- **Transfer Learning**: Cross-domain knowledge transfer experiments
- **Multi-Agent Systems**: Complex multi-agent interaction studies
- **Curriculum Learning**: Automated curriculum design and evaluation