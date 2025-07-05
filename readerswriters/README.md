# Readers-Writers Problem

A comprehensive implementation of the classic readers-writers synchronization problem demonstrating different concurrency strategies, performance trade-offs, and fairness policies in multi-threaded environments.

## Problem Description

The readers-writers problem is a fundamental synchronization challenge in concurrent programming:

- **Multiple Readers**: Multiple threads can read shared data simultaneously
- **Exclusive Writers**: Only one writer can modify data at a time
- **Mutual Exclusion**: Writers must have exclusive access (no concurrent readers or writers)
- **Starvation Prevention**: Ensuring both readers and writers get fair access
- **Performance Optimization**: Maximizing concurrent read access while maintaining data integrity

This problem appears in many real-world scenarios like database systems, file systems, caches, and any shared data structure where reads greatly outnumber writes.

## Solution Approach

The implementation provides four different synchronization strategies:

1. **Readers Preference**: Prioritizes readers, potential writer starvation
2. **Writers Preference**: Prioritizes writers, potential reader starvation  
3. **Fair Access**: Queue-based approach ensuring fairness
4. **Go RWMutex**: Leverages Go's built-in reader-writer mutex

Each strategy demonstrates different trade-offs between performance, fairness, and implementation complexity.

## Key Components

### Database Interface

```go
type Database struct {
    data            map[string]interface{}
    strategy        RWStrategy
    readCount       int32
    writeCount      int32
    totalReads      int64
    totalWrites     int64
    readWaitTime    int64
    writeWaitTime   int64
}
```

### Strategy Interface

```go
type RWStrategy interface {
    StartRead()
    EndRead()
    StartWrite()
    EndWrite()
    Name() string
}
```

### Simulation Framework

```go
type Simulation struct {
    db          *Database
    numReaders  int
    numWriters  int
    duration    time.Duration
    stopChan    chan bool
}
```

## Usage Examples

### Basic Readers-Writers Usage

```go
// Create database with readers preference strategy
strategy := &ReadersPreferenceStrategy{}
db := NewDatabase(strategy)

// Perform read operations
value, exists := db.Read("key1")
if exists {
    fmt.Printf("Read value: %v\n", value)
}

// Perform write operations
db.Write("key1", "new_value")
db.Write("key2", 42)

// Get access statistics
stats := db.GetStatistics()
fmt.Printf("Total reads: %d, Total writes: %d\n", 
    stats.TotalReads, stats.TotalWrites)
```

### Strategy Comparison

```go
// Test different strategies
strategies := []RWStrategy{
    &ReadersPreferenceStrategy{},
    &WritersPreferenceStrategy{},
    NewFairStrategy(),
    &RWMutexStrategy{},
}

for _, strategy := range strategies {
    fmt.Printf("\n=== Testing %s ===\n", strategy.Name())
    
    db := NewDatabase(strategy)
    sim := NewSimulation(db, 10, 3, 5*time.Second)
    stats := sim.Run()
    
    fmt.Printf("Results:\n")
    fmt.Printf("  Total reads: %d\n", stats.TotalReads)
    fmt.Printf("  Total writes: %d\n", stats.TotalWrites)
    fmt.Printf("  Read/Write ratio: %.2f\n", 
        float64(stats.TotalReads)/float64(stats.TotalWrites))
    fmt.Printf("  Avg read wait time: %v\n", stats.AvgReadWaitTime)
    fmt.Printf("  Avg write wait time: %v\n", stats.AvgWriteWaitTime)
}
```

### Real-time Monitoring

```go
// Create monitor for real-time access tracking
monitor := NewMonitor(db, 500*time.Millisecond)
monitor.Start()

// Run simulation
sim := NewSimulation(db, 8, 2, 10*time.Second)
stats := sim.Run()

monitor.Stop()

fmt.Printf("Final statistics: %+v\n", stats)
```

### Performance Analysis

```go
// Analyze performance under different loads
readerCounts := []int{1, 5, 10, 20, 50}
writerCounts := []int{1, 2, 5, 10}

for _, readers := range readerCounts {
    for _, writers := range writerCounts {
        db := NewDatabase(&RWMutexStrategy{})
        sim := NewSimulation(db, readers, writers, 3*time.Second)
        stats := sim.Run()
        
        throughput := float64(stats.TotalReads+stats.TotalWrites) / 3.0
        fmt.Printf("R:%d W:%d -> %.1f ops/sec\n", 
            readers, writers, throughput)
    }
}
```

## Technical Features

### Readers Preference Strategy

Gives priority to readers - once a reader is active, new readers can join immediately:

```go
type ReadersPreferenceStrategy struct {
    mutex      sync.Mutex    // Protects readCount
    readCount  int           // Number of active readers
    writeMutex sync.Mutex    // Blocks writers when readers active
}

func (rp *ReadersPreferenceStrategy) StartRead() {
    rp.mutex.Lock()
    rp.readCount++
    if rp.readCount == 1 {
        rp.writeMutex.Lock() // First reader blocks writers
    }
    rp.mutex.Unlock()
}

func (rp *ReadersPreferenceStrategy) EndRead() {
    rp.mutex.Lock()
    rp.readCount--
    if rp.readCount == 0 {
        rp.writeMutex.Unlock() // Last reader unblocks writers
    }
    rp.mutex.Unlock()
}
```

### Writers Preference Strategy

Ensures writers get priority access - when a writer arrives, new readers are blocked:

```go
type WritersPreferenceStrategy struct {
    readMutex    sync.Mutex  // Protects readCount
    writeMutex   sync.Mutex  // Protects writeCount  
    readTry      sync.Mutex  // Blocks new readers when writers waiting
    resource     sync.Mutex  // Actual resource access control
    readCount    int
    writeCount   int
}

func (wp *WritersPreferenceStrategy) StartWrite() {
    wp.writeMutex.Lock()
    wp.writeCount++
    if wp.writeCount == 1 {
        wp.readTry.Lock() // First writer blocks new readers
    }
    wp.writeMutex.Unlock()
    
    wp.resource.Lock() // Acquire exclusive access
}
```

### Fair Strategy with Queue

Implements FIFO queue to ensure fairness between readers and writers:

```go
type FairStrategy struct {
    queue      chan request
    resource   sync.RWMutex
    activeReads int32
}

type request struct {
    isWrite bool
    ready   chan bool
    done    chan bool
}

func (fs *FairStrategy) scheduler() {
    for req := range fs.queue {
        if req.isWrite {
            // Wait for all readers to finish
            for atomic.LoadInt32(&fs.activeReads) > 0 {
                time.Sleep(time.Millisecond)
            }
            fs.resource.Lock()
            req.ready <- true
            <-req.done
            fs.resource.Unlock()
        } else {
            atomic.AddInt32(&fs.activeReads, 1)
            fs.resource.RLock()
            req.ready <- true
            <-req.done
            fs.resource.RUnlock()
            atomic.AddInt32(&fs.activeReads, -1)
        }
    }
}
```

### Go RWMutex Strategy

Leverages Go's built-in reader-writer mutex for comparison:

```go
type RWMutexStrategy struct {
    rwMutex sync.RWMutex
}

func (rw *RWMutexStrategy) StartRead() {
    rw.rwMutex.RLock()
}

func (rw *RWMutexStrategy) EndRead() {
    rw.rwMutex.RUnlock()
}

func (rw *RWMutexStrategy) StartWrite() {
    rw.rwMutex.Lock()
}

func (rw *RWMutexStrategy) EndWrite() {
    rw.rwMutex.Unlock()
}
```

## Implementation Details

### Database Operations with Statistics

```go
func (db *Database) Read(key string) (interface{}, bool) {
    start := time.Now()
    db.strategy.StartRead()
    atomic.AddInt64(&db.readWaitTime, int64(time.Since(start)))
    
    atomic.AddInt32(&db.readCount, 1)
    defer func() {
        atomic.AddInt32(&db.readCount, -1)
        db.strategy.EndRead()
        atomic.AddInt64(&db.totalReads, 1)
    }()
    
    // Simulate read operation
    time.Sleep(time.Duration(10+time.Now().UnixNano()%10) * time.Millisecond)
    
    value, exists := db.data[key]
    return value, exists
}

func (db *Database) Write(key string, value interface{}) {
    start := time.Now()
    db.strategy.StartWrite()
    atomic.AddInt64(&db.writeWaitTime, int64(time.Since(start)))
    
    atomic.AddInt32(&db.writeCount, 1)
    defer func() {
        atomic.AddInt32(&db.writeCount, -1)
        db.strategy.EndWrite()
        atomic.AddInt64(&db.totalWrites, 1)
    }()
    
    // Simulate write operation
    time.Sleep(time.Duration(20+time.Now().UnixNano()%10) * time.Millisecond)
    
    db.data[key] = value
}
```

### Simulation Framework

```go
func (s *Simulation) Run() Statistics {
    fmt.Printf("Running %s strategy with %d readers and %d writers\n",
        s.db.strategy.Name(), s.numReaders, s.numWriters)
    
    // Start readers
    for i := 0; i < s.numReaders; i++ {
        s.wg.Add(1)
        go s.reader(i)
    }
    
    // Start writers
    for i := 0; i < s.numWriters; i++ {
        s.wg.Add(1)
        go s.writer(i)
    }
    
    // Run for specified duration
    time.Sleep(s.duration)
    close(s.stopChan)
    
    s.wg.Wait()
    return s.db.GetStatistics()
}

func (s *Simulation) reader(id int) {
    defer s.wg.Done()
    keys := []string{"key1", "key2", "key3", "key4", "key5"}
    
    for {
        select {
        case <-s.stopChan:
            return
        default:
            key := keys[id%len(keys)]
            s.db.Read(key)
            time.Sleep(time.Duration(5+id%5) * time.Millisecond)
        }
    }
}
```

### Real-time Monitoring

```go
type Monitor struct {
    db       *Database
    interval time.Duration
    stopChan chan bool
}

func (m *Monitor) Start() {
    go func() {
        ticker := time.NewTicker(m.interval)
        defer ticker.Stop()
        
        for {
            select {
            case <-ticker.C:
                readCount := atomic.LoadInt32(&m.db.readCount)
                writeCount := atomic.LoadInt32(&m.db.writeCount)
                fmt.Printf("Active: %d readers, %d writers\n", readCount, writeCount)
            case <-m.stopChan:
                return
            }
        }
    }()
}
```

## Performance Characteristics

### Strategy Comparison

Under typical workload (80% reads, 20% writes):

```
Strategy             Throughput    Read Latency    Write Latency    Fairness
Readers Preference   High          Low            High             Poor (writers starve)
Writers Preference   Medium        High           Low              Poor (readers starve)  
Fair Queue           Medium        Medium         Medium           Excellent
Go RWMutex          High          Low            Medium           Good
```

### Scalability Properties

- **Read-Heavy Workloads**: Readers preference and RWMutex perform best
- **Write-Heavy Workloads**: Writers preference minimizes write latency
- **Mixed Workloads**: Fair strategy provides predictable performance
- **High Contention**: All strategies experience increased latency

### Memory and CPU Usage

```
Strategy             Memory Overhead    CPU Overhead    Lock Contention
Readers Preference   Low               Low             Medium
Writers Preference   Low               Medium          High
Fair Queue          Medium            Medium          Low
Go RWMutex          Very Low          Very Low        Low
```

## Advanced Features

### Deadlock Prevention

All strategies are designed to prevent deadlock:
- Consistent lock ordering
- No nested lock acquisition in critical sections
- Timeout mechanisms where applicable

### Starvation Analysis

```go
func analyzeStarvation(stats Statistics, duration time.Duration) {
    expectedReads := duration.Seconds() * 10  // Estimate
    expectedWrites := duration.Seconds() * 2
    
    readStarvation := (expectedReads - float64(stats.TotalReads)) / expectedReads
    writeStarvation := (expectedWrites - float64(stats.TotalWrites)) / expectedWrites
    
    fmt.Printf("Read starvation: %.2f%%\n", readStarvation*100)
    fmt.Printf("Write starvation: %.2f%%\n", writeStarvation*100)
}
```

### Dynamic Strategy Switching

```go
func adaptiveStrategy(currentLoad LoadMetrics) RWStrategy {
    readRatio := float64(currentLoad.Reads) / float64(currentLoad.Total)
    
    if readRatio > 0.9 {
        return &ReadersPreferenceStrategy{} // Read-heavy
    } else if readRatio < 0.3 {
        return &WritersPreferenceStrategy{} // Write-heavy
    } else {
        return NewFairStrategy() // Balanced
    }
}
```

## Configuration and Tuning

### Performance Tuning Guidelines

1. **Read-Heavy Systems** (>80% reads): Use readers preference or RWMutex
2. **Write-Heavy Systems** (>50% writes): Use writers preference
3. **Balanced Systems**: Use fair strategy or RWMutex
4. **Real-Time Systems**: Use fair strategy for predictable latency

### Monitoring Recommendations

- Track read/write ratios over time
- Monitor wait times and detect starvation
- Measure throughput under different loads
- Profile lock contention and CPU usage

### Common Anti-Patterns

- **Reader Starvation**: Using writers preference with many writers
- **Writer Starvation**: Using readers preference with continuous readers
- **Over-Engineering**: Using complex fair strategy when RWMutex suffices
- **Under-Engineering**: Using simple mutex for read-heavy workloads

This readers-writers implementation demonstrates the fundamental trade-offs in concurrent programming between performance, fairness, and complexity, providing a solid foundation for understanding synchronization patterns in multi-threaded systems.