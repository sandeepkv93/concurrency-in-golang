# Concurrent LRU Cache

A high-performance, thread-safe Least Recently Used (LRU) cache implementation in Go that demonstrates advanced concurrency patterns including sharding, TTL support, size-based eviction, and automatic loading mechanisms.

## Problem Description

Caching is essential for high-performance applications, but traditional cache implementations face several challenges:

- **Thread Safety**: Multiple goroutines accessing the cache simultaneously can cause data races
- **Lock Contention**: Single mutex protection can become a bottleneck under high concurrency
- **Memory Management**: Caches need to limit memory usage and evict old entries
- **Expiration**: Some cached data should expire after a time period
- **Loading**: Missing data should be loaded automatically from the source
- **Size Limits**: Caches should respect memory constraints based on data size

## Solution Approach

This implementation provides a comprehensive concurrent LRU cache system using several advanced techniques:

1. **Sharding**: Divides the cache into multiple shards to reduce lock contention
2. **LRU Eviction**: Maintains access order using doubly-linked lists
3. **Hash-based Distribution**: Uses FNV hash to distribute keys across shards
4. **Atomic Operations**: Thread-safe statistics without lock overhead
5. **TTL Support**: Time-to-live functionality with background cleanup
6. **Size-based Eviction**: Memory-aware cache with size limits
7. **Loading Cache**: Automatic data loading for cache misses

## Key Components

### Core Cache Types

- **LRUCache**: Basic sharded LRU cache with thread-safe operations
- **TTLCache**: LRU cache with time-to-live support and expiration
- **SizeLimitedCache**: Memory-aware cache with size-based eviction
- **LoadingCache**: Automatic data loading for cache misses

### Data Structures

- **cacheShard**: Individual cache shard with its own mutex and LRU list
- **entry**: Cache entry containing key, value, size, and timestamp
- **CacheStats**: Thread-safe statistics collection

## Technical Features

### Concurrency Patterns

1. **Sharding**: Reduces lock contention by distributing data across multiple shards
2. **Fine-grained Locking**: Each shard has its own mutex for independent access
3. **Atomic Operations**: Lock-free statistics updates using atomic operations
4. **Sync.Map**: Concurrent map for TTL and loading state management
5. **Background Cleanup**: Goroutine-based expired entry cleanup

### Advanced Features

- **FNV Hashing**: Fast, consistent hash function for key distribution
- **Doubly-linked List**: Efficient LRU order maintenance
- **Load Deduplication**: Prevents duplicate loading of the same key
- **Configurable Sharding**: Customizable number of shards for optimal performance
- **Memory Awareness**: Size-based eviction with configurable limits

## Usage Examples

### Basic LRU Cache

```go
// Create cache with capacity 1000 and 16 shards
cache := NewLRUCache(1000, 16)

// Set values
cache.Set("user:123", userData)
cache.Set("config:app", configData)

// Get values
if value, found := cache.Get("user:123"); found {
    user := value.(UserData)
    // Use user data
}

// Set with custom size
cache.SetWithSize("large_data", data, 1024)

// Get statistics
stats := cache.GetStats()
fmt.Printf("Hits: %d, Misses: %d, Hit Rate: %.2f%%\n", 
    stats.Hits, stats.Misses, 
    float64(stats.Hits)/float64(stats.Hits+stats.Misses)*100)
```

### TTL Cache with Expiration

```go
// Create TTL cache with 1-hour default TTL
ttlCache := NewTTLCache(500, 8, 1*time.Hour)

// Set with default TTL
ttlCache.Set("session:abc123", sessionData)

// Set with custom TTL
ttlCache.SetWithTTL("temp_token", token, 5*time.Minute)

// Start background cleanup every 30 seconds
stopCleanup := ttlCache.StartCleanup(30 * time.Second)
defer close(stopCleanup)

// Get with automatic expiration check
if value, found := ttlCache.Get("session:abc123"); found {
    session := value.(SessionData)
    // Use session data
}
```

### Size-Limited Cache

```go
// Create cache with 1GB size limit
sizeCache := NewSizeLimitedCache(1024*1024*1024, 16)

// Set with size information
success := sizeCache.SetWithSize("image:123", imageData, int64(len(imageData)))
if !success {
    fmt.Println("Item too large for cache")
}

// Monitor current size
currentSize := sizeCache.GetCurrentSize()
fmt.Printf("Cache using %d bytes\n", currentSize)
```

### Loading Cache

```go
// Create loading cache with automatic data loading
loader := func(key string) (interface{}, error) {
    // Load data from database, API, etc.
    return database.LoadUser(key)
}

loadingCache := NewLoadingCache(1000, 16, loader)

// Get with automatic loading
user, err := loadingCache.Get("user:123")
if err != nil {
    log.Printf("Failed to load user: %v", err)
    return
}

// Subsequent calls return cached value
cachedUser, _ := loadingCache.Get("user:123") // No loading
```

### Cache-Aside Pattern

```go
// Implement cache-aside pattern with GetOrLoad
cache := NewLRUCache(1000, 16)

userData, err := cache.GetOrLoad("user:123", func() (interface{}, error) {
    // Load from database only if not in cache
    return database.LoadUser("123")
})

if err != nil {
    log.Printf("Failed to get user: %v", err)
    return
}

user := userData.(UserData)
```

## Implementation Details

### Sharding Strategy

The cache uses FNV hashing to distribute keys across shards:

```go
func (c *LRUCache) getShard(key string) *cacheShard {
    h := fnv.New32a()
    h.Write([]byte(key))
    return c.shards[h.Sum32()%c.shardCount]
}
```

### LRU Implementation

Each shard maintains LRU order using Go's container/list:

```go
type cacheShard struct {
    mutex    sync.Mutex
    capacity int
    items    map[string]*list.Element  // Key -> List element
    lruList  *list.List               // Doubly-linked list
}

func (c *LRUCache) Get(key string) (interface{}, bool) {
    shard := c.getShard(key)
    
    shard.mutex.Lock()
    defer shard.mutex.Unlock()
    
    if elem, found := shard.items[key]; found {
        shard.lruList.MoveToFront(elem)  // Update LRU order
        return elem.Value.(*entry).value, true
    }
    
    return nil, false
}
```

### TTL Implementation

TTL cache uses sync.Map for concurrent expiration tracking:

```go
func (tc *TTLCache) Get(key string) (interface{}, bool) {
    // Check expiration first
    if expiry, ok := tc.ttlMap.Load(key); ok {
        if time.Now().After(expiry.(time.Time)) {
            tc.Delete(key)
            tc.ttlMap.Delete(key)
            return nil, false
        }
    }
    
    return tc.LRUCache.Get(key)
}
```

### Load Deduplication

Loading cache prevents duplicate loads using sync.Map:

```go
func (lc *LoadingCache) Get(key string) (interface{}, error) {
    // Try cache first
    if val, found := lc.LRUCache.Get(key); found {
        return val, nil
    }
    
    // Check if already loading
    if loading, exists := lc.loadingMap.LoadOrStore(key, true); exists {
        // Wait for loading to complete
        for {
            time.Sleep(10 * time.Millisecond)
            if val, found := lc.LRUCache.Get(key); found {
                return val, nil
            }
        }
    }
    
    // Load value
    defer lc.loadingMap.Delete(key)
    val, err := lc.loader(key)
    if err != nil {
        return nil, err
    }
    
    lc.LRUCache.Set(key, val)
    return val, nil
}
```

### Statistics Collection

Thread-safe statistics using atomic operations:

```go
func (c *LRUCache) Get(key string) (interface{}, bool) {
    // ... cache logic ...
    
    if found {
        atomic.AddInt64(&c.stats.Hits, 1)
        return value, true
    }
    
    atomic.AddInt64(&c.stats.Misses, 1)
    return nil, false
}
```

## Testing

The package includes comprehensive tests covering:

- **Concurrent Access**: Multiple goroutines accessing the cache
- **LRU Behavior**: Proper eviction of least recently used items
- **TTL Functionality**: Automatic expiration of timed entries
- **Size Limits**: Memory-based eviction policies
- **Loading**: Automatic data loading and deduplication
- **Statistics**: Accurate metrics collection

Run the tests:

```bash
go test -v ./concurrentlrucache
go test -race ./concurrentlrucache  # Race condition detection
```

## Performance Considerations

1. **Shard Count**: More shards reduce contention but increase memory overhead
2. **Hash Function**: FNV provides good distribution with minimal overhead
3. **Atomic Operations**: Lock-free statistics updates improve performance
4. **Memory Layout**: Efficient use of doubly-linked lists and maps
5. **Lock Granularity**: Per-shard locking minimizes contention

### Performance Tuning

```go
// High-throughput scenario
cache := NewLRUCache(10000, 32)  // More shards

// Memory-constrained scenario
cache := NewLRUCache(1000, 4)    // Fewer shards

// Large object scenario
sizeCache := NewSizeLimitedCache(1024*1024*1024, 16)
```

## Real-World Applications

This concurrent LRU cache is suitable for:

- **Web Applications**: Session caching, user data caching
- **API Gateways**: Response caching, rate limiting data
- **Database Applications**: Query result caching
- **Microservices**: Service-to-service call caching
- **Content Delivery**: File and media caching
- **Configuration Systems**: Application settings caching

## Advanced Features

### Custom Eviction Policies

```go
type CustomCache struct {
    *LRUCache
    evictionPolicy func(key string, value interface{}) bool
}

func (cc *CustomCache) shouldEvict(key string, value interface{}) bool {
    return cc.evictionPolicy(key, value)
}
```

### Metrics and Monitoring

```go
func (c *LRUCache) GetDetailedStats() DetailedStats {
    stats := c.GetStats()
    return DetailedStats{
        CacheStats: stats,
        HitRate:    float64(stats.Hits) / float64(stats.Hits + stats.Misses),
        Size:       c.Len(),
        Shards:     len(c.shards),
    }
}
```

The implementation demonstrates advanced Go concurrency patterns and provides a robust foundation for building high-performance caching systems in distributed applications.