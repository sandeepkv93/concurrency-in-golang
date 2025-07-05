package workstealingscheduler

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// Task represents a unit of work
type Task interface {
	Execute()
}

// SimpleTask is a basic implementation of Task
type SimpleTask struct {
	ID   int
	Work func()
}

func (t *SimpleTask) Execute() {
	if t.Work != nil {
		t.Work()
	}
}

// WorkQueue is a thread-safe queue for tasks
type WorkQueue struct {
	tasks []Task
	mutex sync.Mutex
}

func NewWorkQueue() *WorkQueue {
	return &WorkQueue{
		tasks: make([]Task, 0),
	}
}

func (q *WorkQueue) Push(task Task) {
	q.mutex.Lock()
	defer q.mutex.Unlock()
	q.tasks = append(q.tasks, task)
}

func (q *WorkQueue) Pop() (Task, bool) {
	q.mutex.Lock()
	defer q.mutex.Unlock()
	
	if len(q.tasks) == 0 {
		return nil, false
	}
	
	task := q.tasks[len(q.tasks)-1]
	q.tasks = q.tasks[:len(q.tasks)-1]
	return task, true
}

func (q *WorkQueue) PopFront() (Task, bool) {
	q.mutex.Lock()
	defer q.mutex.Unlock()
	
	if len(q.tasks) == 0 {
		return nil, false
	}
	
	task := q.tasks[0]
	q.tasks = q.tasks[1:]
	return task, true
}

func (q *WorkQueue) Size() int {
	q.mutex.Lock()
	defer q.mutex.Unlock()
	return len(q.tasks)
}

// Worker represents a worker thread with its own task queue
type Worker struct {
	id           int
	queue        *WorkQueue
	scheduler    *WorkStealingScheduler
	tasksExecuted int32
}

func (w *Worker) run() {
	for {
		// Try to get task from own queue
		task, ok := w.queue.Pop()
		if !ok {
			// Own queue is empty, try to steal from others
			task = w.steal()
			if task == nil {
				// No work available, check if should terminate
				if atomic.LoadInt32(&w.scheduler.done) == 1 {
					return
				}
				// Sleep briefly before trying again
				time.Sleep(time.Millisecond)
				continue
			}
		}
		
		// Execute the task
		task.Execute()
		atomic.AddInt32(&w.tasksExecuted, 1)
		atomic.AddInt32(&w.scheduler.completedTasks, 1)
	}
}

func (w *Worker) steal() Task {
	numWorkers := len(w.scheduler.workers)
	if numWorkers <= 1 {
		return nil
	}
	
	// Start from a random worker to avoid always stealing from the same one
	start := rand.Intn(numWorkers)
	
	for i := 0; i < numWorkers; i++ {
		victimID := (start + i) % numWorkers
		if victimID == w.id {
			continue // Don't steal from self
		}
		
		victim := w.scheduler.workers[victimID]
		if task, ok := victim.queue.PopFront(); ok {
			atomic.AddInt32(&w.scheduler.stealCount, 1)
			return task
		}
	}
	
	return nil
}

// WorkStealingScheduler manages multiple workers with work stealing
type WorkStealingScheduler struct {
	workers        []*Worker
	completedTasks int32
	stealCount     int32
	done           int32
	wg             sync.WaitGroup
}

// NewWorkStealingScheduler creates a new scheduler with the specified number of workers
func NewWorkStealingScheduler(numWorkers int) *WorkStealingScheduler {
	scheduler := &WorkStealingScheduler{
		workers: make([]*Worker, numWorkers),
	}
	
	// Create workers
	for i := 0; i < numWorkers; i++ {
		scheduler.workers[i] = &Worker{
			id:        i,
			queue:     NewWorkQueue(),
			scheduler: scheduler,
		}
	}
	
	return scheduler
}

// Submit adds a task to the scheduler
func (s *WorkStealingScheduler) Submit(task Task) {
	// Simple round-robin distribution
	workerID := rand.Intn(len(s.workers))
	s.workers[workerID].queue.Push(task)
}

// SubmitToWorker adds a task to a specific worker's queue
func (s *WorkStealingScheduler) SubmitToWorker(workerID int, task Task) {
	if workerID >= 0 && workerID < len(s.workers) {
		s.workers[workerID].queue.Push(task)
	}
}

// Start begins executing tasks
func (s *WorkStealingScheduler) Start() {
	atomic.StoreInt32(&s.done, 0)
	
	for _, worker := range s.workers {
		s.wg.Add(1)
		go func(w *Worker) {
			defer s.wg.Done()
			w.run()
		}(worker)
	}
}

// Stop signals workers to stop and waits for them to finish
func (s *WorkStealingScheduler) Stop() {
	atomic.StoreInt32(&s.done, 1)
	s.wg.Wait()
}

// GetStats returns statistics about the scheduler
func (s *WorkStealingScheduler) GetStats() (completedTasks, stealCount int32, workerStats []int32) {
	completedTasks = atomic.LoadInt32(&s.completedTasks)
	stealCount = atomic.LoadInt32(&s.stealCount)
	
	workerStats = make([]int32, len(s.workers))
	for i, worker := range s.workers {
		workerStats[i] = atomic.LoadInt32(&worker.tasksExecuted)
	}
	
	return
}

// Example demonstrates the work stealing scheduler
func Example() {
	numWorkers := 4
	numTasks := 100
	
	scheduler := NewWorkStealingScheduler(numWorkers)
	scheduler.Start()
	
	// Create tasks with uneven distribution
	for i := 0; i < numTasks; i++ {
		taskID := i
		task := &SimpleTask{
			ID: taskID,
			Work: func() {
				// Simulate work
				time.Sleep(time.Duration(rand.Intn(10)) * time.Millisecond)
				fmt.Printf("Task %d completed\n", taskID)
			},
		}
		
		// Unevenly distribute tasks to demonstrate work stealing
		if i < numTasks/2 {
			scheduler.SubmitToWorker(0, task) // First worker gets half the tasks
		} else {
			scheduler.Submit(task) // Rest are distributed randomly
		}
	}
	
	// Wait for all tasks to complete
	for {
		completed, _, _ := scheduler.GetStats()
		if int(completed) >= numTasks {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	
	scheduler.Stop()
	
	// Print statistics
	completed, steals, workerStats := scheduler.GetStats()
	fmt.Printf("\nStatistics:\n")
	fmt.Printf("Total tasks completed: %d\n", completed)
	fmt.Printf("Total steals: %d\n", steals)
	for i, count := range workerStats {
		fmt.Printf("Worker %d executed: %d tasks\n", i, count)
	}
}