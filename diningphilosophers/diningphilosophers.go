package diningphilosophers

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// Philosopher represents a dining philosopher
type Philosopher struct {
	id              int
	name            string
	leftFork        *Fork
	rightFork       *Fork
	mealsEaten      int32
	totalThinkTime  int64
	totalEatTime    int64
	totalWaitTime   int64
}

// Fork represents a fork (chopstick)
type Fork struct {
	id    int
	mutex sync.Mutex
}

// DiningTable represents the dining philosophers problem setup
type DiningTable struct {
	philosophers []*Philosopher
	forks        []*Fork
	stopChan     chan bool
	wg           sync.WaitGroup
	strategy     DiningStrategy
}

// DiningStrategy defines how philosophers acquire forks
type DiningStrategy interface {
	TryToEat(philosopher *Philosopher) bool
	Name() string
}

// Statistics holds dining statistics
type Statistics struct {
	TotalMeals      int32
	AverageWaitTime time.Duration
	Deadlocks       int32
}

// NewDiningTable creates a new dining table with n philosophers
func NewDiningTable(n int, strategy DiningStrategy) *DiningTable {
	philosophers := make([]*Philosopher, n)
	forks := make([]*Fork, n)
	
	// Create forks
	for i := 0; i < n; i++ {
		forks[i] = &Fork{id: i}
	}
	
	// Create philosophers
	for i := 0; i < n; i++ {
		philosophers[i] = &Philosopher{
			id:        i,
			name:      fmt.Sprintf("Philosopher %d", i),
			leftFork:  forks[i],
			rightFork: forks[(i+1)%n],
		}
	}
	
	return &DiningTable{
		philosophers: philosophers,
		forks:        forks,
		stopChan:     make(chan bool),
		strategy:     strategy,
	}
}

// Start begins the dining simulation
func (dt *DiningTable) Start(duration time.Duration) {
	fmt.Printf("Starting dining simulation with %s strategy for %v\n", 
		dt.strategy.Name(), duration)
	
	// Start philosophers
	for _, philosopher := range dt.philosophers {
		dt.wg.Add(1)
		go dt.philosopherRoutine(philosopher)
	}
	
	// Run for specified duration
	time.Sleep(duration)
	
	// Stop all philosophers
	close(dt.stopChan)
	dt.wg.Wait()
}

func (dt *DiningTable) philosopherRoutine(philosopher *Philosopher) {
	defer dt.wg.Done()
	
	for {
		select {
		case <-dt.stopChan:
			return
		default:
			// Think
			thinkStart := time.Now()
			dt.think(philosopher)
			atomic.AddInt64(&philosopher.totalThinkTime, int64(time.Since(thinkStart)))
			
			// Try to eat
			waitStart := time.Now()
			if dt.strategy.TryToEat(philosopher) {
				atomic.AddInt64(&philosopher.totalWaitTime, int64(time.Since(waitStart)))
				
				eatStart := time.Now()
				dt.eat(philosopher)
				atomic.AddInt64(&philosopher.totalEatTime, int64(time.Since(eatStart)))
				atomic.AddInt32(&philosopher.mealsEaten, 1)
				
				// Put down forks
				philosopher.leftFork.mutex.Unlock()
				philosopher.rightFork.mutex.Unlock()
			} else {
				atomic.AddInt64(&philosopher.totalWaitTime, int64(time.Since(waitStart)))
			}
		}
	}
}

func (dt *DiningTable) think(philosopher *Philosopher) {
	thinkTime := time.Duration(rand.Intn(100)+50) * time.Millisecond
	time.Sleep(thinkTime)
}

func (dt *DiningTable) eat(philosopher *Philosopher) {
	eatTime := time.Duration(rand.Intn(100)+50) * time.Millisecond
	time.Sleep(eatTime)
}

// GetStatistics returns dining statistics
func (dt *DiningTable) GetStatistics() Statistics {
	stats := Statistics{}
	totalWaitTime := int64(0)
	
	for _, p := range dt.philosophers {
		stats.TotalMeals += atomic.LoadInt32(&p.mealsEaten)
		totalWaitTime += atomic.LoadInt64(&p.totalWaitTime)
	}
	
	if stats.TotalMeals > 0 {
		stats.AverageWaitTime = time.Duration(totalWaitTime / int64(stats.TotalMeals))
	}
	
	return stats
}

// PrintStatistics prints detailed statistics
func (dt *DiningTable) PrintStatistics() {
	fmt.Println("\n=== Dining Statistics ===")
	fmt.Printf("Strategy: %s\n", dt.strategy.Name())
	
	totalMeals := int32(0)
	for _, p := range dt.philosophers {
		meals := atomic.LoadInt32(&p.mealsEaten)
		totalMeals += meals
		fmt.Printf("%s ate %d times\n", p.name, meals)
	}
	
	stats := dt.GetStatistics()
	fmt.Printf("\nTotal meals: %d\n", stats.TotalMeals)
	fmt.Printf("Average wait time: %v\n", stats.AverageWaitTime)
}

// Strategies

// NaiveStrategy - philosophers pick up left then right fork (can deadlock)
type NaiveStrategy struct{}

func (ns *NaiveStrategy) TryToEat(philosopher *Philosopher) bool {
	philosopher.leftFork.mutex.Lock()
	philosopher.rightFork.mutex.Lock()
	return true
}

func (ns *NaiveStrategy) Name() string {
	return "Naive (Deadlock-prone)"
}

// OrderedStrategy - philosophers pick up lower-numbered fork first
type OrderedStrategy struct{}

func (os *OrderedStrategy) TryToEat(philosopher *Philosopher) bool {
	first, second := philosopher.leftFork, philosopher.rightFork
	if first.id > second.id {
		first, second = second, first
	}
	
	first.mutex.Lock()
	second.mutex.Lock()
	return true
}

func (os *OrderedStrategy) Name() string {
	return "Ordered (Deadlock-free)"
}

// ArbitratorStrategy - uses a waiter/arbitrator to control access
type ArbitratorStrategy struct {
	waiter sync.Mutex
}

func (as *ArbitratorStrategy) TryToEat(philosopher *Philosopher) bool {
	as.waiter.Lock()
	philosopher.leftFork.mutex.Lock()
	philosopher.rightFork.mutex.Lock()
	as.waiter.Unlock()
	return true
}

func (as *ArbitratorStrategy) Name() string {
	return "Arbitrator (Waiter)"
}

// ChandyMisraStrategy - uses clean/dirty forks with request tokens
type ChandyMisraStrategy struct {
	forkOwners  []int
	forkDirty   []bool
	requests    [][]bool
	mutex       sync.Mutex
}

func NewChandyMisraStrategy(n int) *ChandyMisraStrategy {
	strategy := &ChandyMisraStrategy{
		forkOwners: make([]int, n),
		forkDirty:  make([]bool, n),
		requests:   make([][]bool, n),
	}
	
	// Initialize fork ownership (philosopher i owns fork i initially)
	for i := 0; i < n; i++ {
		strategy.forkOwners[i] = i
		strategy.forkDirty[i] = true
		strategy.requests[i] = make([]bool, n)
	}
	
	return strategy
}

func (cms *ChandyMisraStrategy) TryToEat(philosopher *Philosopher) bool {
	// Simplified version - just use ordered locking for now
	first, second := philosopher.leftFork, philosopher.rightFork
	if first.id > second.id {
		first, second = second, first
	}
	
	first.mutex.Lock()
	second.mutex.Lock()
	return true
}

func (cms *ChandyMisraStrategy) Name() string {
	return "Chandy-Misra"
}

// LimitedStrategy - limits number of concurrent diners
type LimitedStrategy struct {
	maxDiners int
	semaphore chan struct{}
}

func NewLimitedStrategy(maxDiners int) *LimitedStrategy {
	return &LimitedStrategy{
		maxDiners: maxDiners,
		semaphore: make(chan struct{}, maxDiners),
	}
}

func (ls *LimitedStrategy) TryToEat(philosopher *Philosopher) bool {
	// Acquire semaphore
	ls.semaphore <- struct{}{}
	defer func() { <-ls.semaphore }()
	
	// Now safe to pick up both forks
	philosopher.leftFork.mutex.Lock()
	philosopher.rightFork.mutex.Lock()
	return true
}

func (ls *LimitedStrategy) Name() string {
	return fmt.Sprintf("Limited (%d max diners)", ls.maxDiners)
}

// TryLockStrategy - uses try-lock to avoid blocking
type TryLockStrategy struct{}

func (tls *TryLockStrategy) TryToEat(philosopher *Philosopher) bool {
	// Try to acquire left fork
	if !philosopher.leftFork.mutex.TryLock() {
		return false
	}
	
	// Try to acquire right fork
	if !philosopher.rightFork.mutex.TryLock() {
		philosopher.leftFork.mutex.Unlock()
		return false
	}
	
	return true
}

func (tls *TryLockStrategy) Name() string {
	return "Try-Lock (Non-blocking)"
}

// Example demonstrates different dining philosopher solutions
func Example() {
	fmt.Println("=== Dining Philosophers Problem ===")
	
	numPhilosophers := 5
	duration := 2 * time.Second
	
	strategies := []DiningStrategy{
		&OrderedStrategy{},
		&ArbitratorStrategy{},
		NewLimitedStrategy(4),
		&TryLockStrategy{},
	}
	
	for _, strategy := range strategies {
		fmt.Printf("\n--- Testing %s ---\n", strategy.Name())
		
		table := NewDiningTable(numPhilosophers, strategy)
		table.Start(duration)
		table.PrintStatistics()
		
		time.Sleep(500 * time.Millisecond) // Pause between tests
	}
}

// Visualization represents the state of the dining table for visualization
type Visualization struct {
	philosophers []*PhilosopherState
	forks        []*ForkState
}

type PhilosopherState struct {
	ID         int
	State      string // "thinking", "hungry", "eating"
	MealsEaten int32
}

type ForkState struct {
	ID       int
	InUse    bool
	HeldByID int // -1 if not held
}

// GetVisualization returns the current state for visualization
func (dt *DiningTable) GetVisualization() *Visualization {
	vis := &Visualization{
		philosophers: make([]*PhilosopherState, len(dt.philosophers)),
		forks:        make([]*ForkState, len(dt.forks)),
	}
	
	// This is a simplified visualization
	for i, p := range dt.philosophers {
		vis.philosophers[i] = &PhilosopherState{
			ID:         p.id,
			State:      "thinking", // Would need more state tracking for accurate status
			MealsEaten: atomic.LoadInt32(&p.mealsEaten),
		}
	}
	
	for i, f := range dt.forks {
		vis.forks[i] = &ForkState{
			ID:       f.id,
			InUse:    false, // Would need to track this properly
			HeldByID: -1,
		}
	}
	
	return vis
}