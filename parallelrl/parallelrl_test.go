package parallelrl

import (
	"fmt"
	"math"
	"sync"
	"testing"
	"time"
)

func TestDefaultTrainingConfig(t *testing.T) {
	config := DefaultTrainingConfig()
	
	if config.AgentType != QLearning {
		t.Errorf("Expected default agent type QLearning, got %v", config.AgentType)
	}
	
	if config.Algorithm != TabularQLearning {
		t.Errorf("Expected default algorithm TabularQLearning, got %v", config.Algorithm)
	}
	
	if config.ParallelStrategy != AsyncActorCritic {
		t.Errorf("Expected default parallel strategy AsyncActorCritic, got %v", config.ParallelStrategy)
	}
	
	if config.LearningRate != 0.1 {
		t.Errorf("Expected default learning rate 0.1, got %f", config.LearningRate)
	}
	
	if config.DiscountFactor != 0.95 {
		t.Errorf("Expected default discount factor 0.95, got %f", config.DiscountFactor)
	}
	
	if config.ExplorationRate != 1.0 {
		t.Errorf("Expected default exploration rate 1.0, got %f", config.ExplorationRate)
	}
	
	if config.MaxEpisodes != 10000 {
		t.Errorf("Expected default max episodes 10000, got %d", config.MaxEpisodes)
	}
	
	if config.NumActors != 4 {
		t.Errorf("Expected default num actors 4, got %d", config.NumActors)
	}
	
	if config.NumLearners != 2 {
		t.Errorf("Expected default num learners 2, got %d", config.NumLearners)
	}
	
	if config.ReplayBufferSize != 100000 {
		t.Errorf("Expected default replay buffer size 100000, got %d", config.ReplayBufferSize)
	}
	
	if config.EnableLogging != true {
		t.Error("Expected logging to be enabled by default")
	}
}

func TestSimpleState(t *testing.T) {
	state1 := &SimpleState{Values: []float64{1.0, 2.0, 3.0}}
	state2 := &SimpleState{Values: []float64{1.0, 2.0, 3.0}}
	state3 := &SimpleState{Values: []float64{1.0, 2.0, 4.0}}
	
	// Test ToVector
	vector := state1.ToVector()
	if len(vector) != 3 {
		t.Errorf("Expected vector length 3, got %d", len(vector))
	}
	
	for i, v := range []float64{1.0, 2.0, 3.0} {
		if vector[i] != v {
			t.Errorf("Expected vector[%d] = %f, got %f", i, v, vector[i])
		}
	}
	
	// Test Equals
	if !state1.Equals(state2) {
		t.Error("Expected state1 to equal state2")
	}
	
	if state1.Equals(state3) {
		t.Error("Expected state1 not to equal state3")
	}
	
	// Test Hash
	hash1 := state1.Hash()
	hash2 := state2.Hash()
	hash3 := state3.Hash()
	
	if hash1 != hash2 {
		t.Error("Expected equal states to have same hash")
	}
	
	if hash1 == hash3 {
		t.Error("Expected different states to have different hashes")
	}
	
	// Test Dimension
	if state1.Dimension() != 3 {
		t.Errorf("Expected dimension 3, got %d", state1.Dimension())
	}
	
	// Test IsValid
	if !state1.IsValid() {
		t.Error("Expected state to be valid")
	}
	
	emptyState := &SimpleState{Values: []float64{}}
	if emptyState.IsValid() {
		t.Error("Expected empty state to be invalid")
	}
}

func TestSimpleAction(t *testing.T) {
	action1 := &SimpleAction{ID: 1, Values: []float64{0.5}}
	action2 := &SimpleAction{ID: 1, Values: []float64{0.7}}
	action3 := &SimpleAction{ID: 2, Values: []float64{0.5}}
	
	// Test ToVector
	vector1 := action1.ToVector()
	if len(vector1) != 1 || vector1[0] != 0.5 {
		t.Errorf("Expected vector [0.5], got %v", vector1)
	}
	
	// Test ToVector without values (uses ID)
	actionNoValues := &SimpleAction{ID: 5}
	vectorNoValues := actionNoValues.ToVector()
	if len(vectorNoValues) != 1 || vectorNoValues[0] != 5.0 {
		t.Errorf("Expected vector [5.0], got %v", vectorNoValues)
	}
	
	// Test Equals
	if !action1.Equals(action2) {
		t.Error("Expected actions with same ID to be equal")
	}
	
	if action1.Equals(action3) {
		t.Error("Expected actions with different IDs not to be equal")
	}
	
	// Test IsValid
	if !action1.IsValid() {
		t.Error("Expected action to be valid")
	}
	
	invalidAction := &SimpleAction{ID: -1}
	if invalidAction.IsValid() {
		t.Error("Expected action with negative ID to be invalid")
	}
	
	// Test String
	actionStr := action1.String()
	expectedStr := "Action_1"
	if actionStr != expectedStr {
		t.Errorf("Expected string '%s', got '%s'", expectedStr, actionStr)
	}
}

func TestSimplePolicy(t *testing.T) {
	actions := []Action{
		&SimpleAction{ID: 0},
		&SimpleAction{ID: 1},
		&SimpleAction{ID: 2},
		&SimpleAction{ID: 3},
	}
	
	policy := NewSimplePolicy(actions, 0.1)
	
	if policy == nil {
		t.Fatal("Policy should not be nil")
	}
	
	if policy.ExplorationRate != 0.1 {
		t.Errorf("Expected exploration rate 0.1, got %f", policy.ExplorationRate)
	}
	
	if len(policy.Actions) != 4 {
		t.Errorf("Expected 4 actions, got %d", len(policy.Actions))
	}
	
	// Test action selection
	state := &SimpleState{Values: []float64{1.0, 1.0}}
	action := policy.SelectAction(state)
	
	if action == nil {
		t.Error("Selected action should not be nil")
	}
	
	// Test action probabilities
	probs := policy.GetActionProbabilities(state)
	if len(probs) != 4 {
		t.Errorf("Expected 4 action probabilities, got %d", len(probs))
	}
	
	// Test that probabilities sum to 1
	total := 0.0
	for _, prob := range probs {
		total += prob
	}
	if math.Abs(total-1.0) > 1e-9 {
		t.Errorf("Expected probabilities to sum to 1.0, got %f", total)
	}
	
	// Test policy update
	experience := &Experience{
		State:     state,
		Action:    action,
		Reward:    1.0,
		NextState: &SimpleState{Values: []float64{2.0, 2.0}},
		Done:      false,
	}
	
	err := policy.Update(experience)
	if err != nil {
		t.Errorf("Policy update failed: %v", err)
	}
	
	// Test parameter extraction
	params := policy.GetParameters()
	if params == nil {
		t.Error("Parameters should not be nil")
	}
	
	// Test policy cloning
	clonedPolicy := policy.Clone()
	if clonedPolicy == nil {
		t.Error("Cloned policy should not be nil")
	}
	
	if clonedPolicy == policy {
		t.Error("Cloned policy should be a different instance")
	}
}

func TestGridWorldEnvironment(t *testing.T) {
	env := NewGridWorldEnvironment(5, 5)
	
	if env == nil {
		t.Fatal("Environment should not be nil")
	}
	
	if env.Width != 5 || env.Height != 5 {
		t.Errorf("Expected dimensions 5x5, got %dx%d", env.Width, env.Height)
	}
	
	// Test reset
	state := env.Reset()
	if state == nil {
		t.Error("Reset state should not be nil")
	}
	
	stateVector := state.ToVector()
	if len(stateVector) != 2 {
		t.Errorf("Expected state vector length 2, got %d", len(stateVector))
	}
	
	if stateVector[0] != 0 || stateVector[1] != 0 {
		t.Errorf("Expected initial position [0, 0], got [%f, %f]", stateVector[0], stateVector[1])
	}
	
	// Test action space
	actions := env.GetActionSpace()
	if len(actions) != 4 {
		t.Errorf("Expected 4 actions, got %d", len(actions))
	}
	
	// Test state space
	stateSpace := env.GetStateSpace()
	if len(stateSpace.Dimensions) != 2 {
		t.Errorf("Expected 2 state dimensions, got %d", len(stateSpace.Dimensions))
	}
	
	if stateSpace.Continuous {
		t.Error("Expected discrete state space")
	}
	
	// Test step function
	rightAction := &SimpleAction{ID: 1} // right
	nextState, reward, done, info := env.Step(rightAction)
	
	if nextState == nil {
		t.Error("Next state should not be nil")
	}
	
	if reward >= 0 {
		t.Errorf("Expected negative step reward, got %f", reward)
	}
	
	if done {
		t.Error("Environment should not be done after one step")
	}
	
	if info == nil {
		t.Error("Info should not be nil")
	}
	
	// Test that player moved right
	nextStateVector := nextState.ToVector()
	if nextStateVector[0] != 1 || nextStateVector[1] != 0 {
		t.Errorf("Expected position [1, 0] after moving right, got [%f, %f]", 
			nextStateVector[0], nextStateVector[1])
	}
	
	// Test terminal state
	if env.IsTerminal() {
		t.Error("Environment should not be terminal")
	}
	
	// Test current state
	currentState := env.GetCurrentState()
	if !currentState.Equals(nextState) {
		t.Error("Current state should equal next state")
	}
	
	// Test environment cloning
	clonedEnv := env.Clone()
	if clonedEnv == nil {
		t.Error("Cloned environment should not be nil")
	}
	
	if clonedEnv == env {
		t.Error("Cloned environment should be a different instance")
	}
	
	// Test render
	renderOutput := env.Render()
	if renderOutput == "" {
		t.Error("Render output should not be empty")
	}
}

func TestReplayBuffer(t *testing.T) {
	capacity := 10
	buffer := NewReplayBuffer(capacity, false)
	
	if buffer == nil {
		t.Fatal("Replay buffer should not be nil")
	}
	
	if buffer.capacity != capacity {
		t.Errorf("Expected capacity %d, got %d", capacity, buffer.capacity)
	}
	
	if buffer.Size() != 0 {
		t.Errorf("Expected initial size 0, got %d", buffer.Size())
	}
	
	// Test adding experiences
	for i := 0; i < 5; i++ {
		experience := &Experience{
			State:     &SimpleState{Values: []float64{float64(i)}},
			Action:    &SimpleAction{ID: i % 4},
			Reward:    Reward(i),
			NextState: &SimpleState{Values: []float64{float64(i + 1)}},
			Done:      false,
			Timestamp: time.Now(),
		}
		
		buffer.Add(experience)
	}
	
	if buffer.Size() != 5 {
		t.Errorf("Expected size 5 after adding 5 experiences, got %d", buffer.Size())
	}
	
	// Test sampling
	samples := buffer.Sample(3)
	if len(samples) != 3 {
		t.Errorf("Expected 3 samples, got %d", len(samples))
	}
	
	for _, sample := range samples {
		if sample == nil {
			t.Error("Sample should not be nil")
		}
	}
	
	// Test sampling more than available
	samples = buffer.Sample(10)
	if len(samples) != 5 {
		t.Errorf("Expected 5 samples when sampling more than available, got %d", len(samples))
	}
	
	// Test buffer overflow
	for i := 5; i < 15; i++ {
		experience := &Experience{
			State:     &SimpleState{Values: []float64{float64(i)}},
			Action:    &SimpleAction{ID: i % 4},
			Reward:    Reward(i),
			NextState: &SimpleState{Values: []float64{float64(i + 1)}},
			Done:      false,
			Timestamp: time.Now(),
		}
		
		buffer.Add(experience)
	}
	
	if buffer.Size() != capacity {
		t.Errorf("Expected size %d after overflow, got %d", capacity, buffer.Size())
	}
}

func TestPrioritizedReplayBuffer(t *testing.T) {
	capacity := 10
	buffer := NewReplayBuffer(capacity, true)
	
	if buffer.priorities == nil {
		t.Error("Prioritized buffer should have priorities array")
	}
	
	// Add experiences
	for i := 0; i < 5; i++ {
		experience := &Experience{
			State:     &SimpleState{Values: []float64{float64(i)}},
			Action:    &SimpleAction{ID: i % 4},
			Reward:    Reward(i),
			NextState: &SimpleState{Values: []float64{float64(i + 1)}},
			Done:      false,
			Timestamp: time.Now(),
		}
		
		buffer.Add(experience)
	}
	
	// Test that priorities are set
	for i := 0; i < buffer.Size(); i++ {
		if buffer.priorities[i] <= 0 {
			t.Errorf("Priority %d should be positive, got %f", i, buffer.priorities[i])
		}
	}
	
	// Test sampling with priorities
	samples := buffer.Sample(3)
	if len(samples) != 3 {
		t.Errorf("Expected 3 samples, got %d", len(samples))
	}
}

func TestExperience(t *testing.T) {
	state := &SimpleState{Values: []float64{1.0, 2.0}}
	action := &SimpleAction{ID: 1}
	reward := Reward(5.0)
	nextState := &SimpleState{Values: []float64{2.0, 3.0}}
	done := false
	timestamp := time.Now()
	metadata := map[string]interface{}{"test": "value"}
	
	experience := &Experience{
		State:     state,
		Action:    action,
		Reward:    reward,
		NextState: nextState,
		Done:      done,
		Timestamp: timestamp,
		Metadata:  metadata,
	}
	
	if experience.State != state {
		t.Error("Experience state mismatch")
	}
	
	if experience.Action != action {
		t.Error("Experience action mismatch")
	}
	
	if experience.Reward != reward {
		t.Error("Experience reward mismatch")
	}
	
	if experience.NextState != nextState {
		t.Error("Experience next state mismatch")
	}
	
	if experience.Done != done {
		t.Error("Experience done flag mismatch")
	}
	
	if experience.Timestamp != timestamp {
		t.Error("Experience timestamp mismatch")
	}
	
	if experience.Metadata["test"] != "value" {
		t.Error("Experience metadata mismatch")
	}
}

func TestEpisode(t *testing.T) {
	episode := &Episode{
		ID:          "test_episode",
		Experiences: make([]*Experience, 0),
		TotalReward: 0,
		Duration:    time.Minute,
		Steps:       10,
		Success:     true,
		Metadata:    map[string]interface{}{"test": "episode"},
	}
	
	if episode.ID != "test_episode" {
		t.Error("Episode ID mismatch")
	}
	
	if episode.TotalReward != 0 {
		t.Error("Episode total reward mismatch")
	}
	
	if episode.Duration != time.Minute {
		t.Error("Episode duration mismatch")
	}
	
	if episode.Steps != 10 {
		t.Error("Episode steps mismatch")
	}
	
	if !episode.Success {
		t.Error("Episode success flag mismatch")
	}
	
	if episode.Metadata["test"] != "episode" {
		t.Error("Episode metadata mismatch")
	}
}

func TestNewParallelAgent(t *testing.T) {
	config := DefaultTrainingConfig()
	config.NumActors = 2
	config.NumLearners = 1
	config.MaxEpisodes = 10
	
	env := NewGridWorldEnvironment(3, 3)
	actions := env.GetActionSpace()
	policy := NewSimplePolicy(actions, 0.1)
	
	agent, err := NewParallelAgent(config, env, policy, nil)
	if err != nil {
		t.Fatalf("Failed to create parallel agent: %v", err)
	}
	
	if agent == nil {
		t.Fatal("Agent should not be nil")
	}
	
	if len(agent.actors) != config.NumActors {
		t.Errorf("Expected %d actors, got %d", config.NumActors, len(agent.actors))
	}
	
	if len(agent.learners) != config.NumLearners {
		t.Errorf("Expected %d learners, got %d", config.NumLearners, len(agent.learners))
	}
	
	if agent.coordinator == nil {
		t.Error("Coordinator should not be nil")
	}
	
	if agent.statistics == nil {
		t.Error("Statistics should not be nil")
	}
	
	if agent.replayBuffer == nil {
		t.Error("Replay buffer should not be nil")
	}
	
	// Test invalid configurations
	_, err = NewParallelAgent(config, nil, policy, nil)
	if err == nil {
		t.Error("Expected error with nil environment")
	}
	
	_, err = NewParallelAgent(config, env, nil, nil)
	if err == nil {
		t.Error("Expected error with nil policy")
	}
}

func TestParallelAgentBasicOperations(t *testing.T) {
	config := DefaultTrainingConfig()
	config.NumActors = 1
	config.NumLearners = 1
	config.MaxEpisodes = 5
	config.MaxStepsPerEpisode = 10
	config.LogInterval = 1
	
	env := NewGridWorldEnvironment(3, 3)
	actions := env.GetActionSpace()
	policy := NewSimplePolicy(actions, 0.5)
	
	agent, err := NewParallelAgent(config, env, policy, nil)
	if err != nil {
		t.Fatalf("Failed to create parallel agent: %v", err)
	}
	
	// Test initial state
	if agent.IsRunning() {
		t.Error("Agent should not be running initially")
	}
	
	stats := agent.GetStatistics()
	if stats == nil {
		t.Error("Statistics should not be nil")
	}
	
	if stats.EpisodesCompleted != 0 {
		t.Error("Initial episodes completed should be 0")
	}
	
	// Test getting policy and value function
	agentPolicy := agent.GetPolicy()
	if agentPolicy != policy {
		t.Error("Agent policy should match provided policy")
	}
	
	agentValueFunc := agent.GetValueFunction()
	// Value function can be nil in this test setup
	_ = agentValueFunc
	
	// Test episode log
	log := agent.GetEpisodeLog()
	if len(log) != 0 {
		t.Error("Initial episode log should be empty")
	}
}

func TestConcurrentOperations(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping concurrent operations test in short mode")
	}
	
	config := DefaultTrainingConfig()
	config.NumActors = 3
	config.NumLearners = 2
	config.MaxEpisodes = 20
	config.MaxStepsPerEpisode = 50
	config.TimeLimit = 10 * time.Second
	config.EnableLogging = false // Disable logging for test
	
	env := NewGridWorldEnvironment(4, 4)
	actions := env.GetActionSpace()
	policy := NewSimplePolicy(actions, 0.3)
	
	agent, err := NewParallelAgent(config, env, policy, nil)
	if err != nil {
		t.Fatalf("Failed to create parallel agent: %v", err)
	}
	
	// Start training in background
	done := make(chan error, 1)
	go func() {
		done <- agent.Train()
	}()
	
	// Monitor for a short time
	timeout := time.After(2 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case err := <-done:
			// Training completed
			if err != nil && err.Error() != "context canceled" {
				t.Errorf("Training failed: %v", err)
			}
			
			// Check final statistics
			stats := agent.GetStatistics()
			if stats.EpisodesCompleted == 0 {
				t.Error("No episodes completed")
			}
			
			t.Logf("Completed %d episodes in %v", stats.EpisodesCompleted, stats.TotalTrainingTime)
			return
			
		case <-timeout:
			// Stop training after timeout
			agent.Stop()
			
			// Wait for completion
			select {
			case err := <-done:
				if err != nil && err.Error() != "context canceled" {
					t.Errorf("Training failed after stop: %v", err)
				}
			case <-time.After(1 * time.Second):
				t.Error("Training did not stop within timeout")
			}
			
			stats := agent.GetStatistics()
			t.Logf("Stopped after %d episodes in %v", stats.EpisodesCompleted, stats.TotalTrainingTime)
			return
			
		case <-ticker.C:
			// Check that agent is running
			if !agent.IsRunning() {
				t.Error("Agent should be running")
				return
			}
			
			// Check statistics
			stats := agent.GetStatistics()
			if stats.TotalTrainingTime == 0 {
				t.Error("Training time should be positive")
			}
		}
	}
}

func TestStatisticsCollection(t *testing.T) {
	config := DefaultTrainingConfig()
	config.NumActors = 2
	config.NumLearners = 1
	config.MaxEpisodes = 5
	config.EnableLogging = false
	
	env := NewGridWorldEnvironment(3, 3)
	actions := env.GetActionSpace()
	policy := NewSimplePolicy(actions, 0.2)
	
	agent, err := NewParallelAgent(config, env, policy, nil)
	if err != nil {
		t.Fatalf("Failed to create parallel agent: %v", err)
	}
	
	// Run training for a short time
	done := make(chan error, 1)
	go func() {
		done <- agent.Train()
	}()
	
	// Stop after short time
	time.Sleep(500 * time.Millisecond)
	agent.Stop()
	
	// Wait for completion
	select {
	case <-done:
	case <-time.After(1 * time.Second):
		t.Error("Training did not complete within timeout")
	}
	
	// Check statistics
	stats := agent.GetStatistics()
	
	if stats.TotalTrainingTime <= 0 {
		t.Error("Total training time should be positive")
	}
	
	if stats.EpisodesCompleted < 0 {
		t.Error("Episodes completed should be non-negative")
	}
	
	if len(stats.ActorStats) != config.NumActors {
		t.Errorf("Expected %d actor stats, got %d", config.NumActors, len(stats.ActorStats))
	}
	
	if len(stats.LearnerStats) != config.NumLearners {
		t.Errorf("Expected %d learner stats, got %d", config.NumLearners, len(stats.LearnerStats))
	}
	
	// Check actor statistics
	for i := 0; i < config.NumActors; i++ {
		actorStats, exists := stats.ActorStats[i]
		if !exists {
			t.Errorf("Actor %d statistics should exist", i)
			continue
		}
		
		if actorStats.EpisodesCompleted < 0 {
			t.Errorf("Actor %d episodes completed should be non-negative", i)
		}
		
		if actorStats.StepsCompleted < 0 {
			t.Errorf("Actor %d steps completed should be non-negative", i)
		}
		
		if actorStats.ActionsSelected == nil {
			t.Errorf("Actor %d actions selected should not be nil", i)
		}
	}
	
	// Check learner statistics
	for i := 0; i < config.NumLearners; i++ {
		learnerStats, exists := stats.LearnerStats[i]
		if !exists {
			t.Errorf("Learner %d statistics should exist", i)
			continue
		}
		
		if learnerStats.UpdatesPerformed < 0 {
			t.Errorf("Learner %d updates performed should be non-negative", i)
		}
		
		if learnerStats.ExperiencesProcessed < 0 {
			t.Errorf("Learner %d experiences processed should be non-negative", i)
		}
	}
}

func TestVarianceCalculation(t *testing.T) {
	// Test empty slice
	variance := calculateVariance([]float64{})
	if variance != 0 {
		t.Errorf("Expected variance 0 for empty slice, got %f", variance)
	}
	
	// Test single value
	variance = calculateVariance([]float64{5.0})
	if variance != 0 {
		t.Errorf("Expected variance 0 for single value, got %f", variance)
	}
	
	// Test known values
	values := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	variance = calculateVariance(values)
	expected := 2.0 // Variance of 1,2,3,4,5 is 2.0
	if math.Abs(variance-expected) > 1e-9 {
		t.Errorf("Expected variance %f, got %f", expected, variance)
	}
	
	// Test identical values
	variance = calculateVariance([]float64{3.0, 3.0, 3.0, 3.0})
	if variance != 0 {
		t.Errorf("Expected variance 0 for identical values, got %f", variance)
	}
}

func TestUtilityFunctions(t *testing.T) {
	// Test min function
	if min(5, 3) != 3 {
		t.Error("min(5, 3) should return 3")
	}
	
	if min(2, 7) != 2 {
		t.Error("min(2, 7) should return 2")
	}
	
	if min(4, 4) != 4 {
		t.Error("min(4, 4) should return 4")
	}
	
	// Test max function
	if max(5, 3) != 5 {
		t.Error("max(5, 3) should return 5")
	}
	
	if max(2, 7) != 7 {
		t.Error("max(2, 7) should return 7")
	}
	
	if max(4, 4) != 4 {
		t.Error("max(4, 4) should return 4")
	}
}

func TestConcurrentPolicyAccess(t *testing.T) {
	actions := []Action{
		&SimpleAction{ID: 0},
		&SimpleAction{ID: 1},
		&SimpleAction{ID: 2},
		&SimpleAction{ID: 3},
	}
	
	policy := NewSimplePolicy(actions, 0.1)
	state := &SimpleState{Values: []float64{1.0, 1.0}}
	
	// Test concurrent access to policy
	var wg sync.WaitGroup
	numGoroutines := 10
	
	// Concurrent action selection
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			for j := 0; j < 10; j++ {
				action := policy.SelectAction(state)
				if action == nil {
					t.Error("Action should not be nil")
				}
				
				probs := policy.GetActionProbabilities(state)
				if len(probs) != 4 {
					t.Error("Should have 4 action probabilities")
				}
			}
		}()
	}
	
	// Concurrent policy updates
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for j := 0; j < 5; j++ {
				experience := &Experience{
					State:     state,
					Action:    actions[j%len(actions)],
					Reward:    Reward(float64(workerID + j)),
					NextState: &SimpleState{Values: []float64{2.0, 2.0}},
					Done:      false,
				}
				
				err := policy.Update(experience)
				if err != nil {
					t.Errorf("Policy update failed: %v", err)
				}
			}
		}(i)
	}
	
	wg.Wait()
}

func TestConcurrentEnvironmentAccess(t *testing.T) {
	baseEnv := NewGridWorldEnvironment(4, 4)
	
	var wg sync.WaitGroup
	numGoroutines := 5
	
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			// Each worker uses a cloned environment
			env := baseEnv.Clone()
			state := env.Reset()
			
			for step := 0; step < 10; step++ {
				actions := env.GetActionSpace()
				action := actions[step%len(actions)]
				
				nextState, reward, done, info := env.Step(action)
				
				if nextState == nil {
					t.Error("Next state should not be nil")
				}
				
				if info == nil {
					t.Error("Info should not be nil")
				}
				
				// Use reward to avoid unused variable warning
				_ = reward
				
				if done {
					state = env.Reset()
				} else {
					state = nextState
				}
			}
		}(i)
	}
	
	wg.Wait()
}

func TestConcurrentReplayBuffer(t *testing.T) {
	buffer := NewReplayBuffer(100, false)
	
	var wg sync.WaitGroup
	numWriters := 5
	numReaders := 3
	
	// Concurrent writers
	for i := 0; i < numWriters; i++ {
		wg.Add(1)
		go func(writerID int) {
			defer wg.Done()
			
			for j := 0; j < 20; j++ {
				experience := &Experience{
					State:     &SimpleState{Values: []float64{float64(writerID), float64(j)}},
					Action:    &SimpleAction{ID: j % 4},
					Reward:    Reward(float64(writerID + j)),
					NextState: &SimpleState{Values: []float64{float64(writerID), float64(j + 1)}},
					Done:      false,
					Timestamp: time.Now(),
				}
				
				buffer.Add(experience)
			}
		}(i)
	}
	
	// Concurrent readers
	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			for j := 0; j < 10; j++ {
				samples := buffer.Sample(5)
				
				// Samples can be empty if buffer is empty
				for _, sample := range samples {
					if sample == nil {
						t.Error("Sample should not be nil")
					}
				}
				
				size := buffer.Size()
				if size < 0 {
					t.Error("Buffer size should be non-negative")
				}
				
				time.Sleep(time.Millisecond) // Small delay
			}
		}()
	}
	
	wg.Wait()
	
	// Final check
	if buffer.Size() > buffer.capacity {
		t.Errorf("Buffer size %d should not exceed capacity %d", buffer.Size(), buffer.capacity)
	}
}

// Benchmark tests

func BenchmarkPolicyActionSelection(b *testing.B) {
	actions := []Action{
		&SimpleAction{ID: 0},
		&SimpleAction{ID: 1},
		&SimpleAction{ID: 2},
		&SimpleAction{ID: 3},
	}
	
	policy := NewSimplePolicy(actions, 0.1)
	state := &SimpleState{Values: []float64{1.0, 2.0, 3.0}}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		policy.SelectAction(state)
	}
}

func BenchmarkPolicyUpdate(b *testing.B) {
	actions := []Action{
		&SimpleAction{ID: 0},
		&SimpleAction{ID: 1},
		&SimpleAction{ID: 2},
		&SimpleAction{ID: 3},
	}
	
	policy := NewSimplePolicy(actions, 0.1)
	
	experiences := make([]*Experience, b.N)
	for i := 0; i < b.N; i++ {
		experiences[i] = &Experience{
			State:     &SimpleState{Values: []float64{float64(i % 10)}},
			Action:    actions[i%len(actions)],
			Reward:    Reward(float64(i % 20)),
			NextState: &SimpleState{Values: []float64{float64((i + 1) % 10)}},
			Done:      false,
		}
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		policy.Update(experiences[i])
	}
}

func BenchmarkEnvironmentStep(b *testing.B) {
	env := NewGridWorldEnvironment(10, 10)
	actions := env.GetActionSpace()
	
	env.Reset()
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		action := actions[i%len(actions)]
		env.Step(action)
		
		if env.IsTerminal() {
			env.Reset()
		}
	}
}

func BenchmarkReplayBufferOperations(b *testing.B) {
	buffer := NewReplayBuffer(10000, false)
	
	// Pre-fill buffer
	for i := 0; i < 5000; i++ {
		experience := &Experience{
			State:     &SimpleState{Values: []float64{float64(i)}},
			Action:    &SimpleAction{ID: i % 4},
			Reward:    Reward(float64(i)),
			NextState: &SimpleState{Values: []float64{float64(i + 1)}},
			Done:      false,
		}
		buffer.Add(experience)
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		if i%2 == 0 {
			// Add operation
			experience := &Experience{
				State:     &SimpleState{Values: []float64{float64(i)}},
				Action:    &SimpleAction{ID: i % 4},
				Reward:    Reward(float64(i)),
				NextState: &SimpleState{Values: []float64{float64(i + 1)}},
				Done:      false,
			}
			buffer.Add(experience)
		} else {
			// Sample operation
			buffer.Sample(32)
		}
	}
}

func BenchmarkConcurrentPolicyAccess(b *testing.B) {
	actions := []Action{
		&SimpleAction{ID: 0},
		&SimpleAction{ID: 1},
		&SimpleAction{ID: 2},
		&SimpleAction{ID: 3},
	}
	
	policy := NewSimplePolicy(actions, 0.1)
	state := &SimpleState{Values: []float64{1.0, 2.0}}
	
	b.ResetTimer()
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			policy.SelectAction(state)
		}
	})
}