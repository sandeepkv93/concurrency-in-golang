# Concurrent Hash Map

## Problem Description

Standard hash maps are not thread-safe and can lead to race conditions when accessed concurrently. The challenge is to create a high-performance concurrent hash map that:

1. Supports concurrent read and write operations
2. Maintains high performance under concurrent access
3. Provides thread-safe operations without excessive locking
4. Scales well with the number of goroutines
5. Minimizes lock contention through smart partitioning

## Solution Approach

This implementation uses a **sharding strategy** to minimize lock contention and maximize concurrent access:

### Key Components

1. **Shard-based Architecture**: The map is divided into multiple shards, each protected by its own lock
2. **Hash-based Partitioning**: Keys are distributed across shards using a hash function
3. **Read-Write Locks**: Each shard uses RWMutex for concurrent read operations
4. **FNV Hash Function**: Fast hash function for consistent key distribution

### Concurrency Model

- **Reduced Lock Contention**: Multiple shards allow concurrent operations on different keys
- **Reader-Writer Locks**: Multiple readers can access the same shard concurrently
- **Hash-based Distribution**: Even distribution of keys across shards
- **Lock-free Operations**: Some operations can be performed without locks

### Implementation Details

- **Configurable Shard Count**: Adjustable number of shards for different use cases
- **Hash Function**: FNV-1a hash for fast and uniform distribution
- **Memory Efficiency**: Each shard maintains its own map for cache locality
- **LRU Variant**: Optional LRU eviction policy for memory management

## Usage Example

```go
// Create a concurrent hash map with 32 shards
hashMap := NewConcurrentHashMap(32)

// Concurrent operations
var wg sync.WaitGroup

// Writers
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()
        for j := 0; j < 1000; j++ {
            key := fmt.Sprintf("key_%d_%d", id, j)
            hashMap.Set(key, j)
        }
    }(i)
}

// Readers
for i := 0; i < 5; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()
        for j := 0; j < 1000; j++ {
            key := fmt.Sprintf("key_%d_%d", id, j)
            value, exists := hashMap.Get(key)
            if exists {
                fmt.Printf("Found: %v\n", value)
            }
        }
    }(i)
}

wg.Wait()
```

## Technical Features

- **Sharded Architecture**: Minimizes lock contention through partitioning
- **Concurrent Read Access**: Multiple goroutines can read simultaneously
- **Fast Hash Distribution**: FNV hash function for uniform key distribution
- **Read-Write Locks**: Optimized for read-heavy workloads
- **Memory Efficient**: Cache-friendly shard-based storage
- **Configurable Sharding**: Adjustable shard count for different scenarios
- **Thread-Safe Operations**: All operations are safe for concurrent use

## Advanced Features

### LRU Concurrent Hash Map
- **Eviction Policy**: Least Recently Used eviction when capacity is exceeded
- **TTL Support**: Time-to-live for automatic key expiration
- **Size Limiting**: Maximum number of entries per shard
- **Access Tracking**: Efficient LRU order maintenance

### Performance Optimizations
- **Lock-free Reads**: Some read operations can be performed without locks
- **Batch Operations**: Support for bulk operations
- **Memory Pooling**: Reuse of internal data structures
- **Cache Line Optimization**: Struct layout optimized for CPU cache

## Performance Characteristics

- **Read Performance**: O(1) average case, highly concurrent
- **Write Performance**: O(1) average case, reduced contention
- **Memory Usage**: Constant overhead per shard
- **Scalability**: Near-linear scaling with number of shards
- **Lock Contention**: Minimal contention due to sharding

## Use Cases

- **Caching**: High-performance concurrent cache implementations
- **Session Storage**: Web session management with concurrent access
- **Configuration Management**: Thread-safe configuration storage
- **Metrics Collection**: Concurrent metrics aggregation
- **Database Connection Pooling**: Connection state management

## Testing

The implementation includes comprehensive tests covering:
- Concurrent read and write operations
- Hash distribution uniformity
- Lock contention measurement
- Performance benchmarks
- Memory usage optimization
- LRU eviction correctness
- Race condition detection
- Stress testing under high load