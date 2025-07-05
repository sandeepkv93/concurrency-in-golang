package concurrentjobscheduler

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestJobScheduler(t *testing.T) {
	config := SchedulerConfig{
		MaxWorkers:    2,
		QueueSize:     10,
		EnablePriority: false,
		EnableRetry:   false,
		JobTimeout:    10 * time.Second,
	}

	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		t.Fatalf("Failed to start scheduler: %v", err)
	}
	defer scheduler.Stop()

	// Test basic job execution
	executed := int32(0)
	handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		atomic.AddInt32(&executed, 1)
		return payload, nil
	}

	job := &Job{
		Name:    "test_job",
		Type:    JobTypeImmediate,
		Payload: "test_data",
		Handler: handler,
	}

	err := scheduler.SubmitJob(job)
	if err != nil {
		t.Fatalf("Failed to submit job: %v", err)
	}

	// Wait for job execution
	time.Sleep(500 * time.Millisecond)

	if atomic.LoadInt32(&executed) != 1 {
		t.Errorf("Expected 1 job executed, got %d", executed)
	}

	// Check job status
	jobStatus, err := scheduler.GetJobStatus(job.ID)
	if err != nil {
		t.Fatalf("Failed to get job status: %v", err)
	}

	if jobStatus.Status != StatusCompleted {
		t.Errorf("Expected job status %v, got %v", StatusCompleted, jobStatus.Status)
	}

	if jobStatus.Result != "test_data" {
		t.Errorf("Expected result 'test_data', got %v", jobStatus.Result)
	}
}

func TestJobSchedulerMultipleJobs(t *testing.T) {
	config := SchedulerConfig{
		MaxWorkers: 3,
		QueueSize:  50,
	}

	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		t.Fatalf("Failed to start scheduler: %v", err)
	}
	defer scheduler.Stop()

	numJobs := 20
	executed := int32(0)
	
	handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		atomic.AddInt32(&executed, 1)
		time.Sleep(50 * time.Millisecond) // Simulate work
		return payload, nil
	}

	// Submit multiple jobs
	for i := 0; i < numJobs; i++ {
		job := &Job{
			Name:    fmt.Sprintf("job_%d", i),
			Type:    JobTypeImmediate,
			Payload: i,
			Handler: handler,
		}

		if err := scheduler.SubmitJob(job); err != nil {
			t.Errorf("Failed to submit job %d: %v", i, err)
		}
	}

	// Wait for all jobs to complete
	timeout := time.After(10 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			t.Fatalf("Timeout waiting for jobs to complete. Executed: %d/%d", 
				atomic.LoadInt32(&executed), numJobs)
		case <-ticker.C:
			if atomic.LoadInt32(&executed) == int32(numJobs) {
				goto done
			}
		}
	}

done:
	metrics := scheduler.GetMetrics()
	if metrics.CompletedJobs != int64(numJobs) {
		t.Errorf("Expected %d completed jobs, got %d", numJobs, metrics.CompletedJobs)
	}
}

func TestJobPriority(t *testing.T) {
	config := SchedulerConfig{
		MaxWorkers:    1, // Single worker to ensure priority order
		QueueSize:     10,
		EnablePriority: true,
	}

	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		t.Fatalf("Failed to start scheduler: %v", err)
	}
	defer scheduler.Stop()

	var executionOrder []string
	var mu sync.Mutex

	handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		mu.Lock()
		executionOrder = append(executionOrder, payload.(string))
		mu.Unlock()
		time.Sleep(100 * time.Millisecond)
		return payload, nil
	}

	// Submit jobs with different priorities
	jobs := []*Job{
		{Name: "low1", Type: JobTypeImmediate, Priority: PriorityLow, Payload: "low1", Handler: handler},
		{Name: "high1", Type: JobTypeImmediate, Priority: PriorityHigh, Payload: "high1", Handler: handler},
		{Name: "critical1", Type: JobTypeImmediate, Priority: PriorityCritical, Payload: "critical1", Handler: handler},
		{Name: "normal1", Type: JobTypeImmediate, Priority: PriorityNormal, Payload: "normal1", Handler: handler},
		{Name: "low2", Type: JobTypeImmediate, Priority: PriorityLow, Payload: "low2", Handler: handler},
	}

	// Submit all jobs quickly
	for _, job := range jobs {
		if err := scheduler.SubmitJob(job); err != nil {
			t.Errorf("Failed to submit job %s: %v", job.Name, err)
		}
	}

	// Wait for all jobs to complete
	time.Sleep(2 * time.Second)

	// Check execution order (critical should come first, then high, normal, low)
	if len(executionOrder) != 5 {
		t.Fatalf("Expected 5 jobs executed, got %d", len(executionOrder))
	}

	// First job should be critical priority
	if executionOrder[0] != "critical1" {
		t.Errorf("Expected first job to be critical1, got %s", executionOrder[0])
	}
}

func TestJobRetry(t *testing.T) {
	config := SchedulerConfig{
		MaxWorkers:    1,
		QueueSize:     10,
		EnableRetry:   true,
		RetryAttempts: 3,
		RetryDelay:    100 * time.Millisecond,
	}

	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		t.Fatalf("Failed to start scheduler: %v", err)
	}
	defer scheduler.Stop()

	attempts := int32(0)
	handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		attempt := atomic.AddInt32(&attempts, 1)
		if attempt < 3 {
			return nil, fmt.Errorf("simulated failure %d", attempt)
		}
		return "success", nil
	}

	job := &Job{
		Name:    "retry_job",
		Type:    JobTypeImmediate,
		Payload: "test",
		Handler: handler,
		RetryPolicy: RetryPolicy{
			MaxAttempts:  3,
			InitialDelay: 100 * time.Millisecond,
		},
	}

	if err := scheduler.SubmitJob(job); err != nil {
		t.Fatalf("Failed to submit job: %v", err)
	}

	// Wait for retries to complete
	time.Sleep(2 * time.Second)

	if atomic.LoadInt32(&attempts) != 3 {
		t.Errorf("Expected 3 attempts, got %d", attempts)
	}

	// Check final job status
	jobStatus, err := scheduler.GetJobStatus(job.ID)
	if err != nil {
		t.Fatalf("Failed to get job status: %v", err)
	}

	if jobStatus.Status != StatusCompleted {
		t.Errorf("Expected job to complete after retries, got status %v", jobStatus.Status)
	}

	if jobStatus.Result != "success" {
		t.Errorf("Expected result 'success', got %v", jobStatus.Result)
	}
}

func TestJobFailure(t *testing.T) {
	config := SchedulerConfig{
		MaxWorkers:  1,
		QueueSize:   10,
		EnableRetry: false,
	}

	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		t.Fatalf("Failed to start scheduler: %v", err)
	}
	defer scheduler.Stop()

	expectedError := fmt.Errorf("test error")
	handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		return nil, expectedError
	}

	job := &Job{
		Name:    "failing_job",
		Type:    JobTypeImmediate,
		Payload: "test",
		Handler: handler,
	}

	if err := scheduler.SubmitJob(job); err != nil {
		t.Fatalf("Failed to submit job: %v", err)
	}

	// Wait for job execution
	time.Sleep(500 * time.Millisecond)

	jobStatus, err := scheduler.GetJobStatus(job.ID)
	if err != nil {
		t.Fatalf("Failed to get job status: %v", err)
	}

	if jobStatus.Status != StatusFailed {
		t.Errorf("Expected job status %v, got %v", StatusFailed, jobStatus.Status)
	}

	if jobStatus.Error.Error() != expectedError.Error() {
		t.Errorf("Expected error '%v', got '%v'", expectedError, jobStatus.Error)
	}

	metrics := scheduler.GetMetrics()
	if metrics.FailedJobs != 1 {
		t.Errorf("Expected 1 failed job, got %d", metrics.FailedJobs)
	}
}

func TestJobTimeout(t *testing.T) {
	config := SchedulerConfig{
		MaxWorkers: 1,
		QueueSize:  10,
		JobTimeout: 200 * time.Millisecond,
	}

	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		t.Fatalf("Failed to start scheduler: %v", err)
	}
	defer scheduler.Stop()

	handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		// Sleep longer than timeout
		time.Sleep(500 * time.Millisecond)
		return "should not reach here", nil
	}

	job := &Job{
		Name:    "timeout_job",
		Type:    JobTypeImmediate,
		Payload: "test",
		Handler: handler,
		Timeout: 200 * time.Millisecond,
	}

	if err := scheduler.SubmitJob(job); err != nil {
		t.Fatalf("Failed to submit job: %v", err)
	}

	// Wait for timeout
	time.Sleep(1 * time.Second)

	jobStatus, err := scheduler.GetJobStatus(job.ID)
	if err != nil {
		t.Fatalf("Failed to get job status: %v", err)
	}

	if jobStatus.Status != StatusFailed {
		t.Errorf("Expected job status %v, got %v", StatusFailed, jobStatus.Status)
	}

	if jobStatus.Error != context.DeadlineExceeded {
		t.Errorf("Expected timeout error, got %v", jobStatus.Error)
	}
}

func TestScheduledJob(t *testing.T) {
	config := SchedulerConfig{
		MaxWorkers: 1,
		QueueSize:  10,
	}

	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		t.Fatalf("Failed to start scheduler: %v", err)
	}
	defer scheduler.Stop()

	executed := int32(0)
	handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		atomic.AddInt32(&executed, 1)
		return payload, nil
	}

	// Schedule job to run in 500ms
	job := &Job{
		Name:    "scheduled_job",
		Type:    JobTypeScheduled,
		Payload: "scheduled_data",
		Handler: handler,
		Schedule: Schedule{
			Type:      ScheduleOnce,
			StartTime: time.Now().Add(500 * time.Millisecond),
		},
	}

	if err := scheduler.SubmitJob(job); err != nil {
		t.Fatalf("Failed to submit scheduled job: %v", err)
	}

	// Check that job hasn't executed yet
	time.Sleep(200 * time.Millisecond)
	if atomic.LoadInt32(&executed) != 0 {
		t.Error("Job executed too early")
	}

	// Wait for scheduled execution
	time.Sleep(500 * time.Millisecond)
	if atomic.LoadInt32(&executed) != 1 {
		t.Error("Scheduled job did not execute")
	}
}

func TestJobCancellation(t *testing.T) {
	config := SchedulerConfig{
		MaxWorkers: 1,
		QueueSize:  10,
	}

	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		t.Fatalf("Failed to start scheduler: %v", err)
	}
	defer scheduler.Stop()

	executed := int32(0)
	handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
			atomic.AddInt32(&executed, 1)
			return payload, nil
		}
	}

	// Create a context that we can cancel
	ctx, cancel := context.WithCancel(context.Background())

	job := &Job{
		Name:    "cancelable_job",
		Type:    JobTypeImmediate,
		Payload: "test",
		Handler: handler,
		Context: ctx,
	}

	if err := scheduler.SubmitJob(job); err != nil {
		t.Fatalf("Failed to submit job: %v", err)
	}

	// Wait a bit then cancel
	time.Sleep(100 * time.Millisecond)
	if err := scheduler.CancelJob(job.ID); err != nil {
		t.Errorf("Failed to cancel job: %v", err)
	}

	// Cancel the context
	cancel()

	// Wait and verify job was cancelled
	time.Sleep(500 * time.Millisecond)

	if atomic.LoadInt32(&executed) != 0 {
		t.Error("Job should have been cancelled before execution")
	}

	jobStatus, err := scheduler.GetJobStatus(job.ID)
	if err != nil {
		t.Fatalf("Failed to get job status: %v", err)
	}

	if jobStatus.Status != StatusCancelled {
		t.Errorf("Expected job status %v, got %v", StatusCancelled, jobStatus.Status)
	}
}

func TestPriorityQueue(t *testing.T) {
	pq := NewPriorityQueue()

	jobs := []*Job{
		{ID: "1", Priority: PriorityLow},
		{ID: "2", Priority: PriorityHigh},
		{ID: "3", Priority: PriorityCritical},
		{ID: "4", Priority: PriorityNormal},
	}

	// Push jobs
	for _, job := range jobs {
		pq.Push(job)
	}

	if pq.Size() != 4 {
		t.Errorf("Expected queue size 4, got %d", pq.Size())
	}

	// Pop jobs - should come out in priority order
	expectedOrder := []string{"3", "2", "4", "1"} // Critical, High, Normal, Low

	for i, expectedID := range expectedOrder {
		job := pq.Pop()
		if job == nil {
			t.Fatalf("Expected job at position %d, got nil", i)
		}
		if job.ID != expectedID {
			t.Errorf("Position %d: expected job %s, got %s", i, expectedID, job.ID)
		}
	}

	if pq.Size() != 0 {
		t.Errorf("Expected empty queue, got size %d", pq.Size())
	}

	if pq.Pop() != nil {
		t.Error("Pop from empty queue should return nil")
	}
}

func TestMemoryJobStore(t *testing.T) {
	store := NewMemoryJobStore()

	job := &Job{
		ID:     "test_job",
		Name:   "Test Job",
		Status: StatusPending,
	}

	// Test Save
	if err := store.Save(job); err != nil {
		t.Errorf("Failed to save job: %v", err)
	}

	// Test Load
	loadedJob, err := store.Load("test_job")
	if err != nil {
		t.Errorf("Failed to load job: %v", err)
	}

	if loadedJob.ID != job.ID || loadedJob.Name != job.Name {
		t.Error("Loaded job does not match saved job")
	}

	// Test Update
	job.Status = StatusCompleted
	if err := store.Update(job); err != nil {
		t.Errorf("Failed to update job: %v", err)
	}

	updatedJob, _ := store.Load("test_job")
	if updatedJob.Status != StatusCompleted {
		t.Error("Job status was not updated")
	}

	// Test List
	jobs, err := store.List(StatusCompleted)
	if err != nil {
		t.Errorf("Failed to list jobs: %v", err)
	}

	if len(jobs) != 1 || jobs[0].ID != "test_job" {
		t.Error("List did not return expected jobs")
	}

	// Test Delete
	if err := store.Delete("test_job"); err != nil {
		t.Errorf("Failed to delete job: %v", err)
	}

	_, err = store.Load("test_job")
	if err == nil {
		t.Error("Job should have been deleted")
	}
}

func TestRetryManager(t *testing.T) {
	config := SchedulerConfig{
		MaxWorkers:    1,
		EnableRetry:   true,
		RetryAttempts: 2,
		RetryDelay:    100 * time.Millisecond,
	}

	scheduler := NewJobScheduler(config)
	rm := NewRetryManager(scheduler)

	job := &Job{
		ID:       "retry_test",
		Attempts: 1,
		RetryPolicy: RetryPolicy{
			MaxAttempts:   3,
			InitialDelay:  50 * time.Millisecond,
			BackoffFactor: 2.0,
		},
	}

	// Test retry delay calculation
	delay := rm.calculateRetryDelay(job)
	expectedDelay := 100 * time.Millisecond // 50ms * 2^1
	if delay != expectedDelay {
		t.Errorf("Expected delay %v, got %v", expectedDelay, delay)
	}

	// Test with max delay
	job.RetryPolicy.MaxDelay = 80 * time.Millisecond
	delay = rm.calculateRetryDelay(job)
	if delay != job.RetryPolicy.MaxDelay {
		t.Errorf("Expected delay capped at %v, got %v", job.RetryPolicy.MaxDelay, delay)
	}
}

func TestSchedulerMetrics(t *testing.T) {
	config := SchedulerConfig{
		MaxWorkers:      2,
		QueueSize:       10,
		MetricsInterval: 100 * time.Millisecond,
	}

	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		t.Fatalf("Failed to start scheduler: %v", err)
	}
	defer scheduler.Stop()

	handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		time.Sleep(50 * time.Millisecond)
		return payload, nil
	}

	// Submit several jobs
	for i := 0; i < 5; i++ {
		job := &Job{
			Name:    fmt.Sprintf("metrics_job_%d", i),
			Type:    JobTypeImmediate,
			Payload: i,
			Handler: handler,
		}

		if err := scheduler.SubmitJob(job); err != nil {
			t.Errorf("Failed to submit job: %v", err)
		}
	}

	// Wait for jobs to complete
	time.Sleep(1 * time.Second)

	metrics := scheduler.GetMetrics()

	if metrics.TotalJobs != 5 {
		t.Errorf("Expected 5 total jobs, got %d", metrics.TotalJobs)
	}

	if metrics.CompletedJobs != 5 {
		t.Errorf("Expected 5 completed jobs, got %d", metrics.CompletedJobs)
	}

	if metrics.AverageExecTime <= 0 {
		t.Error("Average execution time should be positive")
	}

	if metrics.StartTime.IsZero() {
		t.Error("Start time should be set")
	}
}

func TestConcurrentJobSubmission(t *testing.T) {
	config := SchedulerConfig{
		MaxWorkers: 4,
		QueueSize:  100,
	}

	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		t.Fatalf("Failed to start scheduler: %v", err)
	}
	defer scheduler.Stop()

	numWorkers := 10
	jobsPerWorker := 20
	totalJobs := numWorkers * jobsPerWorker

	executed := int32(0)
	handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		atomic.AddInt32(&executed, 1)
		return payload, nil
	}

	var wg sync.WaitGroup

	// Submit jobs concurrently
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			for j := 0; j < jobsPerWorker; j++ {
				job := &Job{
					Name:    fmt.Sprintf("worker_%d_job_%d", workerID, j),
					Type:    JobTypeImmediate,
					Payload: fmt.Sprintf("%d_%d", workerID, j),
					Handler: handler,
				}

				if err := scheduler.SubmitJob(job); err != nil {
					t.Errorf("Failed to submit job: %v", err)
				}
			}
		}(w)
	}

	wg.Wait()

	// Wait for all jobs to complete
	timeout := time.After(10 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			t.Fatalf("Timeout waiting for jobs. Executed: %d/%d", 
				atomic.LoadInt32(&executed), totalJobs)
		case <-ticker.C:
			if atomic.LoadInt32(&executed) == int32(totalJobs) {
				goto done
			}
		}
	}

done:
	metrics := scheduler.GetMetrics()
	if metrics.TotalJobs != int64(totalJobs) {
		t.Errorf("Expected %d total jobs, got %d", totalJobs, metrics.TotalJobs)
	}

	if metrics.CompletedJobs != int64(totalJobs) {
		t.Errorf("Expected %d completed jobs, got %d", totalJobs, metrics.CompletedJobs)
	}
}

func BenchmarkJobExecution(b *testing.B) {
	config := SchedulerConfig{
		MaxWorkers: 4,
		QueueSize:  1000,
	}

	scheduler := NewJobScheduler(config)
	if err := scheduler.Start(); err != nil {
		b.Fatalf("Failed to start scheduler: %v", err)
	}
	defer scheduler.Stop()

	handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
		return payload, nil
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		job := &Job{
			Name:    fmt.Sprintf("bench_job_%d", i),
			Type:    JobTypeImmediate,
			Payload: i,
			Handler: handler,
		}

		if err := scheduler.SubmitJob(job); err != nil {
			b.Errorf("Failed to submit job: %v", err)
		}
	}

	// Wait for all jobs to complete
	for {
		metrics := scheduler.GetMetrics()
		if metrics.CompletedJobs+metrics.FailedJobs >= int64(b.N) {
			break
		}
		time.Sleep(1 * time.Millisecond)
	}
}

func BenchmarkPriorityQueue(b *testing.B) {
	pq := NewPriorityQueue()

	b.ResetTimer()

	// Benchmark push operations
	b.Run("Push", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			job := &Job{
				ID:       fmt.Sprintf("job_%d", i),
				Priority: Priority(i % 4),
			}
			pq.Push(job)
		}
	})

	// Benchmark pop operations
	b.Run("Pop", func(b *testing.B) {
		for i := 0; i < b.N && pq.Size() > 0; i++ {
			pq.Pop()
		}
	})
}