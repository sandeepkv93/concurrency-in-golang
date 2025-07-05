package concurrentjobscheduler

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// JobScheduler manages concurrent job execution
type JobScheduler struct {
	workers        int
	jobQueue       chan Job
	workerPool     chan chan Job
	quit           chan bool
	wg             sync.WaitGroup
	metrics        *SchedulerMetrics
	config         SchedulerConfig
	jobStore       JobStore
	priorityQueue  *PriorityQueue
	cronScheduler  *CronScheduler
	retryManager   *RetryManager
	mu             sync.RWMutex
	running        bool
}

// SchedulerConfig holds scheduler configuration
type SchedulerConfig struct {
	MaxWorkers      int
	QueueSize       int
	EnablePriority  bool
	EnableCron      bool
	EnableRetry     bool
	RetryAttempts   int
	RetryDelay      time.Duration
	JobTimeout      time.Duration
	MetricsInterval time.Duration
}

// Job represents a schedulable job
type Job struct {
	ID          string
	Name        string
	Type        JobType
	Priority    Priority
	Payload     interface{}
	Handler     JobHandler
	Schedule    Schedule
	Timeout     time.Duration
	RetryPolicy RetryPolicy
	Dependencies []string
	CreatedAt   time.Time
	ScheduledAt time.Time
	StartedAt   time.Time
	CompletedAt time.Time
	Status      JobStatus
	Result      interface{}
	Error       error
	Attempts    int
	Context     context.Context
}

// JobType defines job categories
type JobType int

const (
	JobTypeImmediate JobType = iota
	JobTypeScheduled
	JobTypeCron
	JobTypeRecurring
	JobTypeBatch
)

// Priority defines job priority levels
type Priority int

const (
	PriorityLow Priority = iota
	PriorityNormal
	PriorityHigh
	PriorityCritical
)

// JobStatus represents job execution status
type JobStatus int

const (
	StatusPending JobStatus = iota
	StatusQueued
	StatusRunning
	StatusCompleted
	StatusFailed
	StatusCancelled
	StatusRetrying
)

// JobHandler function signature for job execution
type JobHandler func(ctx context.Context, payload interface{}) (interface{}, error)

// Schedule defines when a job should run
type Schedule struct {
	Type      ScheduleType
	Interval  time.Duration
	CronExpr  string
	StartTime time.Time
	EndTime   time.Time
	MaxRuns   int
	RunCount  int
}

// ScheduleType defines scheduling strategies
type ScheduleType int

const (
	ScheduleOnce ScheduleType = iota
	ScheduleInterval
	ScheduleCron
	ScheduleDelay
)

// RetryPolicy defines retry behavior
type RetryPolicy struct {
	MaxAttempts   int
	InitialDelay  time.Duration
	MaxDelay      time.Duration
	BackoffFactor float64
	RetryOn       []error
}

// Worker represents a job worker
type Worker struct {
	ID       int
	JobChan  chan Job
	QuitChan chan bool
	Scheduler *JobScheduler
}

// SchedulerMetrics tracks scheduler performance
type SchedulerMetrics struct {
	TotalJobs       int64
	CompletedJobs   int64
	FailedJobs      int64
	RetryJobs       int64
	AverageExecTime time.Duration
	QueueLength     int64
	ActiveWorkers   int64
	StartTime       time.Time
	mu              sync.RWMutex
}

// JobStore interface for job persistence
type JobStore interface {
	Save(job *Job) error
	Load(id string) (*Job, error)
	Delete(id string) error
	List(status JobStatus) ([]*Job, error)
	Update(job *Job) error
}

// PriorityQueue manages job priority ordering
type PriorityQueue struct {
	jobs []*Job
	mu   sync.RWMutex
}

// CronScheduler handles cron-based scheduling
type CronScheduler struct {
	entries map[string]*CronEntry
	ticker  *time.Ticker
	quit    chan bool
	mu      sync.RWMutex
}

// CronEntry represents a cron job entry
type CronEntry struct {
	Job      *Job
	NextRun  time.Time
	Pattern  string
	LastRun  time.Time
}

// RetryManager handles job retry logic
type RetryManager struct {
	retryQueue chan *Job
	quit       chan bool
	scheduler  *JobScheduler
}

// NewJobScheduler creates a new job scheduler
func NewJobScheduler(config SchedulerConfig) *JobScheduler {
	if config.MaxWorkers <= 0 {
		config.MaxWorkers = runtime.NumCPU()
	}
	if config.QueueSize <= 0 {
		config.QueueSize = 1000
	}
	if config.RetryAttempts <= 0 {
		config.RetryAttempts = 3
	}
	if config.RetryDelay <= 0 {
		config.RetryDelay = time.Second
	}
	if config.JobTimeout <= 0 {
		config.JobTimeout = 30 * time.Minute
	}
	if config.MetricsInterval <= 0 {
		config.MetricsInterval = time.Minute
	}

	scheduler := &JobScheduler{
		workers:       config.MaxWorkers,
		jobQueue:      make(chan Job, config.QueueSize),
		workerPool:    make(chan chan Job, config.MaxWorkers),
		quit:          make(chan bool),
		config:        config,
		metrics:       &SchedulerMetrics{StartTime: time.Now()},
		jobStore:      NewMemoryJobStore(),
		priorityQueue: NewPriorityQueue(),
	}

	if config.EnableCron {
		scheduler.cronScheduler = NewCronScheduler(scheduler)
	}

	if config.EnableRetry {
		scheduler.retryManager = NewRetryManager(scheduler)
	}

	return scheduler
}

// Start starts the job scheduler
func (js *JobScheduler) Start() error {
	js.mu.Lock()
	defer js.mu.Unlock()

	if js.running {
		return fmt.Errorf("scheduler already running")
	}

	// Start workers
	for i := 0; i < js.workers; i++ {
		worker := NewWorker(i, js)
		js.wg.Add(1)
		go worker.Start()
	}

	// Start job dispatcher
	js.wg.Add(1)
	go js.dispatch()

	// Start cron scheduler if enabled
	if js.cronScheduler != nil {
		js.wg.Add(1)
		go js.cronScheduler.Start()
	}

	// Start retry manager if enabled
	if js.retryManager != nil {
		js.wg.Add(1)
		go js.retryManager.Start()
	}

	// Start metrics collector
	js.wg.Add(1)
	go js.collectMetrics()

	js.running = true
	return nil
}

// Stop stops the job scheduler
func (js *JobScheduler) Stop() {
	js.mu.Lock()
	defer js.mu.Unlock()

	if !js.running {
		return
	}

	close(js.quit)
	js.wg.Wait()
	js.running = false
}

// SubmitJob submits a job for execution
func (js *JobScheduler) SubmitJob(job *Job) error {
	if job.ID == "" {
		job.ID = generateJobID()
	}

	job.CreatedAt = time.Now()
	job.Status = StatusPending

	// Set default timeout
	if job.Timeout == 0 {
		job.Timeout = js.config.JobTimeout
	}

	// Set default context
	if job.Context == nil {
		job.Context = context.Background()
	}

	// Save job
	if err := js.jobStore.Save(job); err != nil {
		return fmt.Errorf("failed to save job: %w", err)
	}

	// Handle different job types
	switch job.Type {
	case JobTypeImmediate:
		return js.queueJob(job)
	case JobTypeScheduled:
		return js.scheduleJob(job)
	case JobTypeCron:
		if js.cronScheduler != nil {
			return js.cronScheduler.AddJob(job)
		}
		return fmt.Errorf("cron scheduling not enabled")
	default:
		return js.queueJob(job)
	}
}

func (js *JobScheduler) queueJob(job *Job) error {
	atomic.AddInt64(&js.metrics.TotalJobs, 1)
	
	if js.config.EnablePriority {
		js.priorityQueue.Push(job)
		return nil
	}

	job.Status = StatusQueued
	job.ScheduledAt = time.Now()

	select {
	case js.jobQueue <- *job:
		return nil
	default:
		return fmt.Errorf("job queue is full")
	}
}

func (js *JobScheduler) scheduleJob(job *Job) error {
	delay := time.Until(job.Schedule.StartTime)
	if delay <= 0 {
		return js.queueJob(job)
	}

	go func() {
		timer := time.NewTimer(delay)
		defer timer.Stop()

		select {
		case <-timer.C:
			js.queueJob(job)
		case <-js.quit:
			return
		}
	}()

	return nil
}

func (js *JobScheduler) dispatch() {
	defer js.wg.Done()

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case job := <-js.jobQueue:
			js.dispatchJob(job)
		case <-ticker.C:
			if js.config.EnablePriority {
				if job := js.priorityQueue.Pop(); job != nil {
					js.dispatchJob(*job)
				}
			}
		case <-js.quit:
			return
		}
	}
}

func (js *JobScheduler) dispatchJob(job Job) {
	select {
	case jobChannel := <-js.workerPool:
		jobChannel <- job
	default:
		// No workers available, put back in queue
		go func() {
			time.Sleep(10 * time.Millisecond)
			select {
			case js.jobQueue <- job:
			case <-js.quit:
			}
		}()
	}
}

func (js *JobScheduler) collectMetrics() {
	defer js.wg.Done()

	ticker := time.NewTicker(js.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			js.updateMetrics()
		case <-js.quit:
			return
		}
	}
}

func (js *JobScheduler) updateMetrics() {
	js.metrics.mu.Lock()
	defer js.metrics.mu.Unlock()

	js.metrics.QueueLength = int64(len(js.jobQueue))
	if js.config.EnablePriority {
		js.metrics.QueueLength += js.priorityQueue.Size()
	}
}

// Worker implementation

func NewWorker(id int, scheduler *JobScheduler) *Worker {
	return &Worker{
		ID:        id,
		JobChan:   make(chan Job),
		QuitChan:  make(chan bool),
		Scheduler: scheduler,
	}
}

func (w *Worker) Start() {
	defer w.Scheduler.wg.Done()

	for {
		// Register worker in pool
		w.Scheduler.workerPool <- w.JobChan

		select {
		case job := <-w.JobChan:
			w.executeJob(job)
		case <-w.QuitChan:
			return
		case <-w.Scheduler.quit:
			return
		}
	}
}

func (w *Worker) executeJob(job Job) {
	atomic.AddInt64(&w.Scheduler.metrics.ActiveWorkers, 1)
	defer atomic.AddInt64(&w.Scheduler.metrics.ActiveWorkers, -1)

	start := time.Now()
	job.StartedAt = start
	job.Status = StatusRunning
	job.Attempts++

	// Update job in store
	w.Scheduler.jobStore.Update(&job)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(job.Context, job.Timeout)
	defer cancel()

	// Execute job
	result, err := w.executeWithTimeout(ctx, job)

	// Update job status
	job.CompletedAt = time.Now()
	duration := job.CompletedAt.Sub(start)

	if err != nil {
		job.Status = StatusFailed
		job.Error = err
		atomic.AddInt64(&w.Scheduler.metrics.FailedJobs, 1)

		// Check if retry is needed
		if w.Scheduler.shouldRetry(&job) {
			job.Status = StatusRetrying
			if w.Scheduler.retryManager != nil {
				w.Scheduler.retryManager.ScheduleRetry(&job)
			}
		}
	} else {
		job.Status = StatusCompleted
		job.Result = result
		atomic.AddInt64(&w.Scheduler.metrics.CompletedJobs, 1)
	}

	// Update metrics
	w.updateExecutionMetrics(duration)

	// Save final job state
	w.Scheduler.jobStore.Update(&job)
}

func (w *Worker) executeWithTimeout(ctx context.Context, job Job) (interface{}, error) {
	resultChan := make(chan interface{}, 1)
	errorChan := make(chan error, 1)

	go func() {
		defer func() {
			if r := recover(); r != nil {
				errorChan <- fmt.Errorf("job panicked: %v", r)
			}
		}()

		result, err := job.Handler(ctx, job.Payload)
		if err != nil {
			errorChan <- err
		} else {
			resultChan <- result
		}
	}()

	select {
	case result := <-resultChan:
		return result, nil
	case err := <-errorChan:
		return nil, err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (w *Worker) updateExecutionMetrics(duration time.Duration) {
	w.Scheduler.metrics.mu.Lock()
	defer w.Scheduler.metrics.mu.Unlock()

	// Update average execution time
	totalJobs := atomic.LoadInt64(&w.Scheduler.metrics.CompletedJobs) + atomic.LoadInt64(&w.Scheduler.metrics.FailedJobs)
	if totalJobs > 0 {
		currentAvg := w.Scheduler.metrics.AverageExecTime
		newAvg := currentAvg + (duration-currentAvg)/time.Duration(totalJobs)
		w.Scheduler.metrics.AverageExecTime = newAvg
	}
}

func (js *JobScheduler) shouldRetry(job *Job) bool {
	if !js.config.EnableRetry {
		return false
	}

	if job.Attempts >= job.RetryPolicy.MaxAttempts {
		return false
	}

	if job.Attempts >= js.config.RetryAttempts {
		return false
	}

	// Check if error type should be retried
	if len(job.RetryPolicy.RetryOn) > 0 {
		shouldRetry := false
		for _, retryErr := range job.RetryPolicy.RetryOn {
			if job.Error == retryErr {
				shouldRetry = true
				break
			}
		}
		if !shouldRetry {
			return false
		}
	}

	return true
}

// PriorityQueue implementation

func NewPriorityQueue() *PriorityQueue {
	return &PriorityQueue{
		jobs: make([]*Job, 0),
	}
}

func (pq *PriorityQueue) Push(job *Job) {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	pq.jobs = append(pq.jobs, job)
	pq.heapifyUp(len(pq.jobs) - 1)
}

func (pq *PriorityQueue) Pop() *Job {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if len(pq.jobs) == 0 {
		return nil
	}

	job := pq.jobs[0]
	last := len(pq.jobs) - 1
	pq.jobs[0] = pq.jobs[last]
	pq.jobs = pq.jobs[:last]

	if len(pq.jobs) > 0 {
		pq.heapifyDown(0)
	}

	return job
}

func (pq *PriorityQueue) Size() int64 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	return int64(len(pq.jobs))
}

func (pq *PriorityQueue) heapifyUp(index int) {
	if index == 0 {
		return
	}

	parentIndex := (index - 1) / 2
	if pq.jobs[parentIndex].Priority < pq.jobs[index].Priority {
		pq.jobs[parentIndex], pq.jobs[index] = pq.jobs[index], pq.jobs[parentIndex]
		pq.heapifyUp(parentIndex)
	}
}

func (pq *PriorityQueue) heapifyDown(index int) {
	leftChild := 2*index + 1
	rightChild := 2*index + 2
	largest := index

	if leftChild < len(pq.jobs) && pq.jobs[leftChild].Priority > pq.jobs[largest].Priority {
		largest = leftChild
	}

	if rightChild < len(pq.jobs) && pq.jobs[rightChild].Priority > pq.jobs[largest].Priority {
		largest = rightChild
	}

	if largest != index {
		pq.jobs[index], pq.jobs[largest] = pq.jobs[largest], pq.jobs[index]
		pq.heapifyDown(largest)
	}
}

// CronScheduler implementation

func NewCronScheduler(scheduler *JobScheduler) *CronScheduler {
	return &CronScheduler{
		entries: make(map[string]*CronEntry),
		ticker:  time.NewTicker(time.Minute),
		quit:    make(chan bool),
	}
}

func (cs *CronScheduler) Start() {
	defer cs.ticker.Stop()

	for {
		select {
		case <-cs.ticker.C:
			cs.checkCronJobs()
		case <-cs.quit:
			return
		}
	}
}

func (cs *CronScheduler) AddJob(job *Job) error {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	nextRun, err := cs.parseNextRun(job.Schedule.CronExpr)
	if err != nil {
		return err
	}

	entry := &CronEntry{
		Job:     job,
		NextRun: nextRun,
		Pattern: job.Schedule.CronExpr,
	}

	cs.entries[job.ID] = entry
	return nil
}

func (cs *CronScheduler) checkCronJobs() {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	now := time.Now()
	for _, entry := range cs.entries {
		if now.After(entry.NextRun) {
			// Schedule job execution
			entry.Job.ScheduledAt = now
			// Implementation would queue the job for execution
			
			// Calculate next run time
			if nextRun, err := cs.parseNextRun(entry.Pattern); err == nil {
				entry.NextRun = nextRun
				entry.LastRun = now
			}
		}
	}
}

func (cs *CronScheduler) parseNextRun(cronExpr string) (time.Time, error) {
	// Simplified cron parsing - in real implementation use a proper cron library
	// This is just a placeholder
	return time.Now().Add(time.Hour), nil
}

// RetryManager implementation

func NewRetryManager(scheduler *JobScheduler) *RetryManager {
	return &RetryManager{
		retryQueue: make(chan *Job, 100),
		quit:       make(chan bool),
		scheduler:  scheduler,
	}
}

func (rm *RetryManager) Start() {
	for {
		select {
		case job := <-rm.retryQueue:
			rm.handleRetry(job)
		case <-rm.quit:
			return
		}
	}
}

func (rm *RetryManager) ScheduleRetry(job *Job) {
	select {
	case rm.retryQueue <- job:
		atomic.AddInt64(&rm.scheduler.metrics.RetryJobs, 1)
	default:
		// Retry queue full, mark job as failed
		job.Status = StatusFailed
		rm.scheduler.jobStore.Update(job)
	}
}

func (rm *RetryManager) handleRetry(job *Job) {
	delay := rm.calculateRetryDelay(job)
	
	time.Sleep(delay)
	
	// Reset job for retry
	job.Status = StatusPending
	job.Error = nil
	job.StartedAt = time.Time{}
	job.CompletedAt = time.Time{}
	
	rm.scheduler.queueJob(job)
}

func (rm *RetryManager) calculateRetryDelay(job *Job) time.Duration {
	policy := job.RetryPolicy
	if policy.InitialDelay == 0 {
		policy.InitialDelay = rm.scheduler.config.RetryDelay
	}
	if policy.BackoffFactor == 0 {
		policy.BackoffFactor = 2.0
	}

	delay := policy.InitialDelay
	for i := 1; i < job.Attempts; i++ {
		delay = time.Duration(float64(delay) * policy.BackoffFactor)
		if policy.MaxDelay > 0 && delay > policy.MaxDelay {
			delay = policy.MaxDelay
			break
		}
	}

	return delay
}

// MemoryJobStore implementation

type MemoryJobStore struct {
	jobs map[string]*Job
	mu   sync.RWMutex
}

func NewMemoryJobStore() *MemoryJobStore {
	return &MemoryJobStore{
		jobs: make(map[string]*Job),
	}
}

func (mjs *MemoryJobStore) Save(job *Job) error {
	mjs.mu.Lock()
	defer mjs.mu.Unlock()
	mjs.jobs[job.ID] = job
	return nil
}

func (mjs *MemoryJobStore) Load(id string) (*Job, error) {
	mjs.mu.RLock()
	defer mjs.mu.RUnlock()
	
	job, exists := mjs.jobs[id]
	if !exists {
		return nil, fmt.Errorf("job not found")
	}
	return job, nil
}

func (mjs *MemoryJobStore) Delete(id string) error {
	mjs.mu.Lock()
	defer mjs.mu.Unlock()
	delete(mjs.jobs, id)
	return nil
}

func (mjs *MemoryJobStore) List(status JobStatus) ([]*Job, error) {
	mjs.mu.RLock()
	defer mjs.mu.RUnlock()
	
	var jobs []*Job
	for _, job := range mjs.jobs {
		if job.Status == status {
			jobs = append(jobs, job)
		}
	}
	return jobs, nil
}

func (mjs *MemoryJobStore) Update(job *Job) error {
	mjs.mu.Lock()
	defer mjs.mu.Unlock()
	mjs.jobs[job.ID] = job
	return nil
}

// Utility functions

func generateJobID() string {
	return fmt.Sprintf("job_%d", time.Now().UnixNano())
}

// GetMetrics returns current scheduler metrics
func (js *JobScheduler) GetMetrics() SchedulerMetrics {
	js.metrics.mu.RLock()
	defer js.metrics.mu.RUnlock()
	return *js.metrics
}

// GetJobStatus returns the status of a specific job
func (js *JobScheduler) GetJobStatus(jobID string) (*Job, error) {
	return js.jobStore.Load(jobID)
}

// CancelJob cancels a pending or running job
func (js *JobScheduler) CancelJob(jobID string) error {
	job, err := js.jobStore.Load(jobID)
	if err != nil {
		return err
	}

	if job.Status == StatusRunning {
		// Cancel the job's context
		if cancel, ok := job.Context.(context.CancelFunc); ok {
			cancel()
		}
	}

	job.Status = StatusCancelled
	return js.jobStore.Update(job)
}

// ListJobs returns jobs with specific status
func (js *JobScheduler) ListJobs(status JobStatus) ([]*Job, error) {
	jobs, err := js.jobStore.List(status)
	if err != nil {
		return nil, err
	}

	// Sort by creation time
	sort.Slice(jobs, func(i, j int) bool {
		return jobs[i].CreatedAt.Before(jobs[j].CreatedAt)
	})

	return jobs, nil
}

// Example demonstrates the concurrent job scheduler
func Example() {
	fmt.Println("=== Concurrent Job Scheduler Example ===")

	// Create scheduler configuration
	config := SchedulerConfig{
		MaxWorkers:      4,
		QueueSize:       100,
		EnablePriority:  true,
		EnableRetry:     true,
		RetryAttempts:   3,
		RetryDelay:      time.Second,
		JobTimeout:      30 * time.Second,
		MetricsInterval: 10 * time.Second,
	}

	// Create and start scheduler
	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		fmt.Printf("Failed to start scheduler: %v\n", err)
		return
	}
	defer scheduler.Stop()

	// Example job handlers
	simpleHandler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		data := payload.(string)
		time.Sleep(100 * time.Millisecond) // Simulate work
		return fmt.Sprintf("Processed: %s", data), nil
	}

	errorHandler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		return nil, fmt.Errorf("simulated error")
	}

	// Submit immediate jobs
	for i := 0; i < 10; i++ {
		job := &Job{
			Name:    fmt.Sprintf("job_%d", i),
			Type:    JobTypeImmediate,
			Priority: Priority(i % 4),
			Payload: fmt.Sprintf("data_%d", i),
			Handler: simpleHandler,
		}

		if err := scheduler.SubmitJob(job); err != nil {
			fmt.Printf("Failed to submit job: %v\n", err)
		}
	}

	// Submit a job that will fail and retry
	retryJob := &Job{
		Name:    "retry_job",
		Type:    JobTypeImmediate,
		Priority: PriorityHigh,
		Payload: "error_data",
		Handler: errorHandler,
		RetryPolicy: RetryPolicy{
			MaxAttempts:   3,
			InitialDelay:  time.Second,
			BackoffFactor: 2.0,
		},
	}

	if err := scheduler.SubmitJob(retryJob); err != nil {
		fmt.Printf("Failed to submit retry job: %v\n", err)
	}

	// Submit scheduled job
	scheduledJob := &Job{
		Name:    "scheduled_job",
		Type:    JobTypeScheduled,
		Priority: PriorityNormal,
		Payload: "scheduled_data",
		Handler: simpleHandler,
		Schedule: Schedule{
			Type:      ScheduleOnce,
			StartTime: time.Now().Add(2 * time.Second),
		},
	}

	if err := scheduler.SubmitJob(scheduledJob); err != nil {
		fmt.Printf("Failed to submit scheduled job: %v\n", err)
	}

	// Wait for jobs to complete
	time.Sleep(5 * time.Second)

	// Display metrics
	metrics := scheduler.GetMetrics()
	fmt.Printf("\nScheduler Metrics:\n")
	fmt.Printf("  Total Jobs: %d\n", metrics.TotalJobs)
	fmt.Printf("  Completed: %d\n", metrics.CompletedJobs)
	fmt.Printf("  Failed: %d\n", metrics.FailedJobs)
	fmt.Printf("  Retries: %d\n", metrics.RetryJobs)
	fmt.Printf("  Average Execution Time: %v\n", metrics.AverageExecTime)
	fmt.Printf("  Queue Length: %d\n", metrics.QueueLength)
	fmt.Printf("  Active Workers: %d\n", metrics.ActiveWorkers)

	// List completed jobs
	completedJobs, _ := scheduler.ListJobs(StatusCompleted)
	fmt.Printf("\nCompleted Jobs: %d\n", len(completedJobs))
	for _, job := range completedJobs {
		fmt.Printf("  Job %s: %v\n", job.Name, job.Result)
	}

	// List failed jobs
	failedJobs, _ := scheduler.ListJobs(StatusFailed)
	fmt.Printf("\nFailed Jobs: %d\n", len(failedJobs))
	for _, job := range failedJobs {
		fmt.Printf("  Job %s: %v (attempts: %d)\n", job.Name, job.Error, job.Attempts)
	}
}