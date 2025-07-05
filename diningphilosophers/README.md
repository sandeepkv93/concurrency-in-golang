# Dining Philosophers Problem

## Problem Description

The Dining Philosophers Problem is a classic concurrency problem that illustrates synchronization challenges in concurrent systems. Five philosophers sit around a table with five forks, and each philosopher needs two forks to eat. The challenges are:

1. **Deadlock**: All philosophers pick up their left fork simultaneously
2. **Starvation**: Some philosophers never get to eat
3. **Resource Contention**: Multiple philosophers competing for shared forks
4. **Fairness**: Ensuring all philosophers get equal opportunity to eat

## Solution Approach

This implementation provides multiple strategies to solve the dining philosophers problem:

### Key Components

1. **Philosophers**: Individual goroutines representing each philosopher
2. **Forks**: Shared resources protected by mutexes
3. **Dining Table**: Coordinates the overall simulation
4. **Strategy Pattern**: Different approaches to deadlock prevention

### Deadlock Prevention Strategies

#### 1. Ordered Fork Acquisition
- **Approach**: Philosophers always pick up forks in a specific order (lower ID first)
- **Advantage**: Prevents circular wait conditions
- **Implementation**: Global ordering of resources

#### 2. Arbitrator Strategy
- **Approach**: Central arbitrator controls fork access
- **Advantage**: Prevents deadlock through centralized control
- **Implementation**: Semaphore-based resource management

#### 3. Limited Diners Strategy
- **Approach**: Only allow N-1 philosophers to attempt eating simultaneously
- **Advantage**: Guarantees at least one philosopher can eat
- **Implementation**: Counting semaphore with capacity N-1

#### 4. Try-Lock Strategy
- **Approach**: Philosophers attempt to acquire both forks atomically
- **Advantage**: Non-blocking with retry mechanism
- **Implementation**: Timeout-based lock acquisition

## Usage Example

```go
// Create dining table with 5 philosophers
table := NewDiningTable(5, &OrderedStrategy{})

// Start simulation
table.StartDining(10 * time.Second)

// Get statistics
stats := table.GetStatistics()
fmt.Printf("Total meals: %d\n", stats.TotalMeals)
fmt.Printf("Average wait time: %v\n", stats.AverageWaitTime)
```

## Technical Features

- **Multiple Strategies**: Different deadlock prevention approaches
- **Performance Metrics**: Detailed timing and statistics collection
- **Configurable Parameters**: Adjustable thinking and eating times
- **Graceful Shutdown**: Clean termination of all goroutines
- **Statistics Collection**: Comprehensive performance monitoring

## Implementation Details

### Philosopher Lifecycle
1. **Thinking**: Philosopher thinks for a random duration
2. **Hungry**: Attempts to acquire forks using selected strategy
3. **Eating**: Eats for a random duration while holding both forks
4. **Finished**: Releases forks and returns to thinking

### Synchronization Mechanisms
- **Mutexes**: Protect individual forks from concurrent access
- **Channels**: Coordinate between philosophers and table
- **Atomic Operations**: Thread-safe statistics updates
- **Wait Groups**: Synchronize goroutine lifecycle

## Strategy Comparison

| Strategy | Deadlock Prevention | Starvation Risk | Performance | Fairness |
|----------|-------------------|----------------|-------------|----------|
| Ordered | ✅ Guaranteed | Low | High | Medium |
| Arbitrator | ✅ Guaranteed | Very Low | Medium | High |
| Limited | ✅ Guaranteed | Low | High | Medium |
| Try-Lock | ✅ Timeout-based | Medium | Variable | Low |

## Advanced Features

### Statistics Collection
- **Meal Counting**: Track total meals per philosopher
- **Timing Analysis**: Measure thinking, waiting, and eating times
- **Deadlock Detection**: Monitor for potential deadlock situations
- **Performance Metrics**: Throughput and utilization statistics

### Configurable Behavior
- **Timing Parameters**: Adjustable thinking and eating durations
- **Strategy Selection**: Runtime strategy switching
- **Philosopher Count**: Variable number of philosophers
- **Simulation Duration**: Configurable simulation time

## Performance Characteristics

- **Throughput**: Depends on strategy and timing parameters
- **Fairness**: Varies by strategy implementation
- **Resource Utilization**: Efficient fork usage
- **Scalability**: Supports variable number of philosophers

## Real-World Applications

This problem and its solutions apply to various real-world scenarios:

- **Database Connection Pooling**: Multiple processes competing for connections
- **Resource Management**: Shared resource allocation in distributed systems
- **Traffic Control**: Intersection management with limited resources
- **Manufacturing**: Assembly line resource coordination
- **Operating Systems**: Process scheduling and resource allocation

## Testing

The implementation includes comprehensive tests covering:
- Deadlock prevention verification
- Starvation testing under different conditions
- Performance benchmarking for each strategy
- Correctness of statistics collection
- Race condition detection
- Stress testing with varying philosopher counts
- Strategy comparison and analysis
- Long-running simulation stability