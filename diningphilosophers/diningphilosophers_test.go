package diningphilosophers

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBasicDining(t *testing.T) {
	strategies := []struct {
		name     string
		strategy DiningStrategy
	}{
		{"Ordered", &OrderedStrategy{}},
		{"Arbitrator", &ArbitratorStrategy{}},
		{"Limited", NewLimitedStrategy(4)},
		{"TryLock", &TryLockStrategy{}},
	}
	
	for _, s := range strategies {
		t.Run(s.name, func(t *testing.T) {
			table := NewDiningTable(5, s.strategy)
			table.Start(1 * time.Second)
			
			stats := table.GetStatistics()
			
			if stats.TotalMeals == 0 {
				t.Errorf("%s: No meals eaten", s.name)
			}
			
			// Check that all philosophers ate at least once
			starvation := false
			for _, p := range table.philosophers {
				if atomic.LoadInt32(&p.mealsEaten) == 0 {
					starvation = true
					break
				}
			}
			
			if starvation && s.name != "TryLock" {
				// TryLock might have starvation due to its non-blocking nature
				t.Errorf("%s: Some philosophers didn't eat", s.name)
			}
		})
	}
}

func TestDeadlockFreedom(t *testing.T) {
	// Test that ordered strategy prevents deadlock
	table := NewDiningTable(5, &OrderedStrategy{})
	
	// Create a scenario that would deadlock with naive strategy
	done := make(chan bool)
	timeout := time.After(2 * time.Second)
	
	go func() {
		table.Start(1 * time.Second)
		done <- true
	}()
	
	select {
	case <-done:
		// Good, no deadlock
		stats := table.GetStatistics()
		if stats.TotalMeals == 0 {
			t.Error("No meals eaten, possible deadlock")
		}
	case <-timeout:
		t.Error("Timeout: possible deadlock")
	}
}

func TestFairness(t *testing.T) {
	table := NewDiningTable(5, &OrderedStrategy{})
	table.Start(2 * time.Second)
	
	// Check fairness - no philosopher should eat significantly more than others
	meals := make([]int32, len(table.philosophers))
	totalMeals := int32(0)
	
	for i, p := range table.philosophers {
		meals[i] = atomic.LoadInt32(&p.mealsEaten)
		totalMeals += meals[i]
	}
	
	if totalMeals == 0 {
		t.Fatal("No meals eaten")
	}
	
	avgMeals := float64(totalMeals) / float64(len(table.philosophers))
	
	// Check that no philosopher ate more than 2x the average
	for i, m := range meals {
		if float64(m) > 2*avgMeals {
			t.Errorf("Philosopher %d ate %d times (avg: %.1f) - unfair distribution", 
				i, m, avgMeals)
		}
	}
}

func TestConcurrentDining(t *testing.T) {
	// Test multiple dining tables running concurrently
	numTables := 5
	var wg sync.WaitGroup
	
	for i := 0; i < numTables; i++ {
		wg.Add(1)
		go func(tableNum int) {
			defer wg.Done()
			
			table := NewDiningTable(5, &OrderedStrategy{})
			table.Start(1 * time.Second)
			
			stats := table.GetStatistics()
			if stats.TotalMeals == 0 {
				t.Errorf("Table %d: No meals eaten", tableNum)
			}
		}(i)
	}
	
	wg.Wait()
}

func TestLimitedStrategy(t *testing.T) {
	// Test that limited strategy actually limits concurrent diners
	maxDiners := 3
	strategy := NewLimitedStrategy(maxDiners)
	
	// Create a custom test that can track concurrent diners
	var concurrentDiners int32
	var maxConcurrent int32
	
	// Override the strategy to track concurrent diners
	originalTryToEat := strategy.TryToEat
	strategy.TryToEat = func(philosopher *Philosopher) bool {
		strategy.semaphore <- struct{}{}
		
		current := atomic.AddInt32(&concurrentDiners, 1)
		for {
			max := atomic.LoadInt32(&maxConcurrent)
			if current <= max || atomic.CompareAndSwapInt32(&maxConcurrent, max, current) {
				break
			}
		}
		
		philosopher.leftFork.mutex.Lock()
		philosopher.rightFork.mutex.Lock()
		
		// Simulate eating
		time.Sleep(10 * time.Millisecond)
		
		philosopher.leftFork.mutex.Unlock()
		philosopher.rightFork.mutex.Unlock()
		
		atomic.AddInt32(&concurrentDiners, -1)
		<-strategy.semaphore
		
		return true
	}
	
	table := NewDiningTable(5, strategy)
	table.Start(500 * time.Millisecond)
	
	if atomic.LoadInt32(&maxConcurrent) > int32(maxDiners) {
		t.Errorf("Max concurrent diners %d exceeded limit %d", 
			maxConcurrent, maxDiners)
	}
}

func TestVisualization(t *testing.T) {
	table := NewDiningTable(5, &OrderedStrategy{})
	
	// Get initial visualization
	vis := table.GetVisualization()
	
	if len(vis.philosophers) != 5 {
		t.Errorf("Expected 5 philosophers, got %d", len(vis.philosophers))
	}
	
	if len(vis.forks) != 5 {
		t.Errorf("Expected 5 forks, got %d", len(vis.forks))
	}
	
	// Run for a bit
	go table.Start(100 * time.Millisecond)
	time.Sleep(50 * time.Millisecond)
	
	// Get visualization during dining
	vis = table.GetVisualization()
	
	// At least some philosophers should have eaten
	mealsEaten := false
	for _, p := range vis.philosophers {
		if p.MealsEaten > 0 {
			mealsEaten = true
			break
		}
	}
	
	if !mealsEaten {
		t.Error("No meals eaten during visualization test")
	}
}

func BenchmarkDiningStrategies(b *testing.B) {
	strategies := []struct {
		name     string
		strategy DiningStrategy
	}{
		{"Ordered", &OrderedStrategy{}},
		{"Arbitrator", &ArbitratorStrategy{}},
		{"Limited", NewLimitedStrategy(4)},
		{"TryLock", &TryLockStrategy{}},
	}
	
	for _, s := range strategies {
		b.Run(s.name, func(b *testing.B) {
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				table := NewDiningTable(5, s.strategy)
				table.Start(100 * time.Millisecond)
			}
		})
	}
}

func BenchmarkLargeDiningTable(b *testing.B) {
	sizes := []int{5, 10, 20, 50}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				table := NewDiningTable(size, &OrderedStrategy{})
				table.Start(100 * time.Millisecond)
			}
		})
	}
}

func TestRaceConditions(t *testing.T) {
	// Run with -race flag to detect race conditions
	table := NewDiningTable(10, &OrderedStrategy{})
	
	// Start dining
	go table.Start(500 * time.Millisecond)
	
	// Concurrently access statistics
	done := make(chan bool)
	for i := 0; i < 5; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				stats := table.GetStatistics()
				_ = stats.TotalMeals
				time.Sleep(time.Millisecond)
			}
			done <- true
		}()
	}
	
	// Wait for all goroutines
	for i := 0; i < 5; i++ {
		<-done
	}
	
	// Final check
	stats := table.GetStatistics()
	if stats.TotalMeals == 0 {
		t.Error("No meals eaten in race condition test")
	}
}