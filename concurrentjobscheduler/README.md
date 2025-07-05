# Concurrent Job Scheduler

A comprehensive, thread-safe job scheduler implementation in Go that demonstrates advanced concurrency patterns for managing and executing jobs in parallel with features like priority queuing, retry mechanisms, cron scheduling, and metrics collection.

## Problem Description

Modern applications often need to execute tasks asynchronously, manage job priorities, handle failures gracefully, and schedule recurring tasks. A robust job scheduler must handle:

- **Concurrent Execution**: Multiple jobs running simultaneously without blocking
- **Priority Management**: High-priority jobs should execute before lower-priority ones
- **Failure Handling**: Jobs may fail and need retry mechanisms with backoff strategies
- **Scheduling**: Jobs may need to run immediately, at specific times, or on recurring schedules
- **Resource Management**: Limiting concurrent workers to prevent resource exhaustion
- **Monitoring**: Tracking job execution metrics and performance

## Solution Approach

This implementation provides a feature-rich concurrent job scheduler using Go's concurrency primitives:

1. **Worker Pool Pattern**: Fixed number of workers processing jobs from a shared queue
2. **Priority Queue**: Heap-based priority queue for job ordering
3. **Retry Manager**: Handles job retries with exponential backoff
4. **Cron Scheduler**: Manages recurring jobs with cron expressions
5. **Metrics Collection**: Real-time performance and status monitoring
6. **Job Persistence**: In-memory job store for tracking job states

## Key Components

### Core Structures

- **JobScheduler**: Main scheduler managing workers and job distribution
- **Job**: Represents a schedulable task with metadata and execution logic
- **Worker**: Goroutine that executes jobs from the queue
- **PriorityQueue**: Thread-safe heap for priority-based job ordering
- **CronScheduler**: Handles cron-based recurring jobs
- **RetryManager**: Manages job retry logic with backoff strategies

### Job Types

- **Immediate**: Execute as soon as possible
- **Scheduled**: Execute at a specific time
- **Cron**: Execute on recurring schedule
- **Recurring**: Execute repeatedly with intervals
- **Batch**: Group of related jobs

### Priority Levels

- **Critical**: Highest priority (3)
- **High**: High priority (2)
- **Normal**: Standard priority (1)
- **Low**: Lowest priority (0)

## Technical Features

### Concurrency Patterns

1. **Worker Pool**: Fixed number of goroutines processing jobs
2. **Channel-based Communication**: Workers communicate via channels
3. **Atomic Operations**: Thread-safe metrics updates
4. **Mutex Protection**: Protecting shared data structures
5. **Context Propagation**: Timeout and cancellation support

### Advanced Features

- **Priority Scheduling**: Jobs executed based on priority levels
- **Retry Mechanisms**: Exponential backoff with configurable policies
- **Timeout Handling**: Job execution timeouts with context cancellation
- **Cron Scheduling**: Recurring jobs with cron expressions
- **Metrics Collection**: Real-time performance monitoring
- **Job Persistence**: State management and recovery
- **Graceful Shutdown**: Clean termination of all workers

## Usage Examples

### Basic Job Scheduling

```go
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
scheduler.Start()
defer scheduler.Stop()

// Define job handler
handler := func(ctx context.Context, payload interface{}) (interface{}, error) {
    data := payload.(string)
    time.Sleep(100 * time.Millisecond) // Simulate work
    return fmt.Sprintf("Processed: %s", data), nil
}

// Submit immediate job
job := &Job{
    Name:     "process_data",
    Type:     JobTypeImmediate,
    Priority: PriorityHigh,
    Payload:  "user_data",
    Handler:  handler,
}

scheduler.SubmitJob(job)
```

### Scheduled Jobs

```go
// Submit job to run in 5 minutes
scheduledJob := &Job{
    Name:    "maintenance_task",
    Type:    JobTypeScheduled,
    Priority: PriorityNormal,
    Payload: "maintenance_data",
    Handler: handler,
    Schedule: Schedule{
        Type:      ScheduleOnce,
        StartTime: time.Now().Add(5 * time.Minute),
    },
}

scheduler.SubmitJob(scheduledJob)
```

### Jobs with Retry Policy

```go
// Job with custom retry configuration
retryJob := &Job{
    Name:    "flaky_task",
    Type:    JobTypeImmediate,
    Priority: PriorityHigh,
    Payload: "sensitive_data",
    Handler: flakyHandler,
    RetryPolicy: RetryPolicy{
        MaxAttempts:   5,
        InitialDelay:  time.Second,
        MaxDelay:      30 * time.Second,
        BackoffFactor: 2.0,
    },
}

scheduler.SubmitJob(retryJob)
```

### Cron Jobs

```go
// Enable cron scheduling
config.EnableCron = true
scheduler := NewJobScheduler(config)

// Submit recurring job
cronJob := &Job{
    Name:    "daily_report",
    Type:    JobTypeCron,
    Priority: PriorityNormal,
    Payload: "report_data",
    Handler: reportHandler,
    Schedule: Schedule{
        Type:     ScheduleCron,
        CronExpr: "0 0 * * *", // Daily at midnight
    },
}

scheduler.SubmitJob(cronJob)
```

### Monitoring and Metrics

```go
// Get scheduler metrics
metrics := scheduler.GetMetrics()
fmt.Printf("Total Jobs: %d\n", metrics.TotalJobs)
fmt.Printf("Completed: %d\n", metrics.CompletedJobs)
fmt.Printf("Failed: %d\n", metrics.FailedJobs)
fmt.Printf("Average Execution Time: %v\n", metrics.AverageExecTime)

// List jobs by status
completedJobs, _ := scheduler.ListJobs(StatusCompleted)
failedJobs, _ := scheduler.ListJobs(StatusFailed)

// Get specific job status
job, _ := scheduler.GetJobStatus("job_123")
fmt.Printf("Job Status: %v\n", job.Status)
```

## Implementation Details

### Worker Pool Management

The scheduler maintains a fixed pool of workers that continuously process jobs:

```go
// Worker lifecycle
func (w *Worker) Start() {
    for {
        w.Scheduler.workerPool <- w.JobChan  // Register in pool
        
        select {
        case job := <-w.JobChan:
            w.executeJob(job)  // Process job
        case <-w.QuitChan:
            return
        }
    }
}
```

### Priority Queue Implementation

Jobs are ordered using a binary heap based on priority:

```go
func (pq *PriorityQueue) Push(job *Job) {
    pq.jobs = append(pq.jobs, job)
    pq.heapifyUp(len(pq.jobs) - 1)
}

func (pq *PriorityQueue) Pop() *Job {
    if len(pq.jobs) == 0 {
        return nil
    }
    
    job := pq.jobs[0]
    pq.jobs[0] = pq.jobs[len(pq.jobs)-1]
    pq.jobs = pq.jobs[:len(pq.jobs)-1]
    pq.heapifyDown(0)
    return job
}
```

### Retry Logic with Exponential Backoff

The retry manager calculates delays using exponential backoff:

```go
func (rm *RetryManager) calculateRetryDelay(job *Job) time.Duration {
    delay := job.RetryPolicy.InitialDelay
    
    for i := 1; i < job.Attempts; i++ {
        delay = time.Duration(float64(delay) * job.RetryPolicy.BackoffFactor)
        if delay > job.RetryPolicy.MaxDelay {
            delay = job.RetryPolicy.MaxDelay
            break
        }
    }
    
    return delay
}
```

### Context-based Timeout Handling

Jobs execute with context-based timeouts:

```go
func (w *Worker) executeWithTimeout(ctx context.Context, job Job) (interface{}, error) {
    resultChan := make(chan interface{}, 1)
    errorChan := make(chan error, 1)
    
    go func() {
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
```

## Testing

The package includes comprehensive tests covering:

- **Basic job execution**: Immediate job processing
- **Priority scheduling**: Higher priority jobs execute first
- **Retry mechanisms**: Failed jobs retry with backoff
- **Timeout handling**: Jobs respect execution timeouts
- **Concurrent execution**: Multiple jobs execute simultaneously
- **Metrics collection**: Performance metrics accuracy
- **Graceful shutdown**: Clean termination

Run the tests:

```bash
go test -v ./concurrentjobscheduler
```

## Performance Considerations

1. **Worker Pool Size**: Balance between concurrency and resource usage
2. **Queue Size**: Prevent memory exhaustion with bounded queues
3. **Priority Queue**: O(log n) operations for job ordering
4. **Atomic Operations**: Minimize lock contention for metrics
5. **Context Propagation**: Efficient cancellation and timeout handling
6. **Memory Management**: Proper cleanup of completed jobs

## Real-World Applications

This job scheduler pattern is commonly used in:

- **Web Applications**: Background task processing
- **Data Processing**: ETL pipelines and batch jobs
- **Monitoring Systems**: Periodic health checks and alerts
- **Message Processing**: Queue-based message handling
- **Scheduled Tasks**: Cron-like job execution
- **Microservices**: Distributed task coordination

The implementation demonstrates advanced Go concurrency patterns and provides a robust foundation for building scalable job scheduling systems.