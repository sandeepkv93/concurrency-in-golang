package concurrentlrucache

import (
	"container/list"
	"fmt"
	"hash/fnv"
	"sync"
	"sync/atomic"
	"time"
)

// LRUCache represents a concurrent LRU cache
type LRUCache struct {
	capacity   int
	shards     []*cacheShard
	shardCount uint32
	stats      CacheStats
}

// cacheShard represents a shard of the cache
type cacheShard struct {
	mutex    sync.Mutex
	capacity int
	items    map[string]*list.Element
	lruList  *list.List
}

// entry represents a cache entry
type entry struct {
	key       string
	value     interface{}
	size      int
	timestamp time.Time
}

// CacheStats holds cache statistics
type CacheStats struct {
	Hits      int64
	Misses    int64
	Sets      int64
	Evictions int64
}

// NewLRUCache creates a new concurrent LRU cache
func NewLRUCache(capacity int, shardCount int) *LRUCache {
	if capacity <= 0 {
		capacity = 100
	}
	if shardCount <= 0 {
		shardCount = 16
	}
	
	cache := &LRUCache{
		capacity:   capacity,
		shards:     make([]*cacheShard, shardCount),
		shardCount: uint32(shardCount),
	}
	
	shardCapacity := capacity / shardCount
	if shardCapacity < 1 {
		shardCapacity = 1
	}
	
	for i := 0; i < shardCount; i++ {
		cache.shards[i] = &cacheShard{
			capacity: shardCapacity,
			items:    make(map[string]*list.Element),
			lruList:  list.New(),
		}
	}
	
	return cache
}

// getShard returns the shard for a given key
func (c *LRUCache) getShard(key string) *cacheShard {
	h := fnv.New32a()
	h.Write([]byte(key))
	return c.shards[h.Sum32()%c.shardCount]
}

// Get retrieves a value from the cache
func (c *LRUCache) Get(key string) (interface{}, bool) {
	shard := c.getShard(key)
	
	shard.mutex.Lock()
	defer shard.mutex.Unlock()
	
	if elem, found := shard.items[key]; found {
		// Move to front
		shard.lruList.MoveToFront(elem)
		atomic.AddInt64(&c.stats.Hits, 1)
		return elem.Value.(*entry).value, true
	}
	
	atomic.AddInt64(&c.stats.Misses, 1)
	return nil, false
}

// Set adds or updates a value in the cache
func (c *LRUCache) Set(key string, value interface{}) {
	c.SetWithSize(key, value, 1)
}

// SetWithSize adds or updates a value with a specific size
func (c *LRUCache) SetWithSize(key string, value interface{}, size int) {
	shard := c.getShard(key)
	
	shard.mutex.Lock()
	defer shard.mutex.Unlock()
	
	atomic.AddInt64(&c.stats.Sets, 1)
	
	// Check if key exists
	if elem, found := shard.items[key]; found {
		// Update existing entry
		shard.lruList.MoveToFront(elem)
		elem.Value.(*entry).value = value
		elem.Value.(*entry).size = size
		elem.Value.(*entry).timestamp = time.Now()
		return
	}
	
	// Add new entry
	ent := &entry{
		key:       key,
		value:     value,
		size:      size,
		timestamp: time.Now(),
	}
	elem := shard.lruList.PushFront(ent)
	shard.items[key] = elem
	
	// Evict if necessary
	for shard.lruList.Len() > shard.capacity {
		shard.evictOldest()
	}
}

// evictOldest removes the least recently used item
func (s *cacheShard) evictOldest() {
	elem := s.lruList.Back()
	if elem != nil {
		s.lruList.Remove(elem)
		ent := elem.Value.(*entry)
		delete(s.items, ent.key)
	}
}

// Delete removes a key from the cache
func (c *LRUCache) Delete(key string) bool {
	shard := c.getShard(key)
	
	shard.mutex.Lock()
	defer shard.mutex.Unlock()
	
	if elem, found := shard.items[key]; found {
		shard.lruList.Remove(elem)
		delete(shard.items, key)
		return true
	}
	
	return false
}

// Clear removes all items from the cache
func (c *LRUCache) Clear() {
	for _, shard := range c.shards {
		shard.mutex.Lock()
		shard.items = make(map[string]*list.Element)
		shard.lruList = list.New()
		shard.mutex.Unlock()
	}
}

// Len returns the current number of items in the cache
func (c *LRUCache) Len() int {
	count := 0
	for _, shard := range c.shards {
		shard.mutex.Lock()
		count += len(shard.items)
		shard.mutex.Unlock()
	}
	return count
}

// GetStats returns cache statistics
func (c *LRUCache) GetStats() CacheStats {
	return CacheStats{
		Hits:      atomic.LoadInt64(&c.stats.Hits),
		Misses:    atomic.LoadInt64(&c.stats.Misses),
		Sets:      atomic.LoadInt64(&c.stats.Sets),
		Evictions: atomic.LoadInt64(&c.stats.Evictions),
	}
}

// GetOrLoad retrieves a value or loads it if not present
func (c *LRUCache) GetOrLoad(key string, loader func() (interface{}, error)) (interface{}, error) {
	// Try to get from cache first
	if val, found := c.Get(key); found {
		return val, nil
	}
	
	// Load value
	val, err := loader()
	if err != nil {
		return nil, err
	}
	
	// Store in cache
	c.Set(key, val)
	return val, nil
}

// TTLCache is an LRU cache with time-to-live support
type TTLCache struct {
	*LRUCache
	defaultTTL time.Duration
	ttlMap     sync.Map
}

// NewTTLCache creates a new TTL cache
func NewTTLCache(capacity int, shardCount int, defaultTTL time.Duration) *TTLCache {
	return &TTLCache{
		LRUCache:   NewLRUCache(capacity, shardCount),
		defaultTTL: defaultTTL,
	}
}

// SetWithTTL adds a value with a specific TTL
func (tc *TTLCache) SetWithTTL(key string, value interface{}, ttl time.Duration) {
	tc.Set(key, value)
	expiry := time.Now().Add(ttl)
	tc.ttlMap.Store(key, expiry)
}

// Get retrieves a value, checking for expiration
func (tc *TTLCache) Get(key string) (interface{}, bool) {
	// Check if expired
	if expiry, ok := tc.ttlMap.Load(key); ok {
		if time.Now().After(expiry.(time.Time)) {
			tc.Delete(key)
			tc.ttlMap.Delete(key)
			return nil, false
		}
	}
	
	return tc.LRUCache.Get(key)
}

// Set adds a value with the default TTL
func (tc *TTLCache) Set(key string, value interface{}) {
	tc.LRUCache.Set(key, value)
	if tc.defaultTTL > 0 {
		expiry := time.Now().Add(tc.defaultTTL)
		tc.ttlMap.Store(key, expiry)
	}
}

// Delete removes a key from the cache
func (tc *TTLCache) Delete(key string) bool {
	tc.ttlMap.Delete(key)
	return tc.LRUCache.Delete(key)
}

// StartCleanup starts a background goroutine to clean expired entries
func (tc *TTLCache) StartCleanup(interval time.Duration) chan struct{} {
	stop := make(chan struct{})
	
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		
		for {
			select {
			case <-ticker.C:
				tc.cleanupExpired()
			case <-stop:
				return
			}
		}
	}()
	
	return stop
}

func (tc *TTLCache) cleanupExpired() {
	now := time.Now()
	
	tc.ttlMap.Range(func(key, value interface{}) bool {
		expiry := value.(time.Time)
		if now.After(expiry) {
			tc.Delete(key.(string))
		}
		return true
	})
}

// SizeLimitedCache is an LRU cache with size limits
type SizeLimitedCache struct {
	*LRUCache
	maxSize     int64
	currentSize int64
	mutex       sync.RWMutex
}

// NewSizeLimitedCache creates a new size-limited cache
func NewSizeLimitedCache(maxSize int64, shardCount int) *SizeLimitedCache {
	return &SizeLimitedCache{
		LRUCache: NewLRUCache(int(maxSize), shardCount),
		maxSize:  maxSize,
	}
}

// SetWithSize adds a value with a specific size
func (slc *SizeLimitedCache) SetWithSize(key string, value interface{}, size int64) bool {
	if size > slc.maxSize {
		return false // Item too large
	}
	
	// Check if we need to evict items
	slc.mutex.Lock()
	defer slc.mutex.Unlock()
	
	// Evict items until we have space
	for slc.currentSize+size > slc.maxSize && slc.Len() > 0 {
		// Find and evict oldest item
		var oldestKey string
		oldestTime := time.Now()
		
		for _, shard := range slc.shards {
			shard.mutex.Lock()
			if shard.lruList.Back() != nil {
				ent := shard.lruList.Back().Value.(*entry)
				if ent.timestamp.Before(oldestTime) {
					oldestTime = ent.timestamp
					oldestKey = ent.key
				}
			}
			shard.mutex.Unlock()
		}
		
		if oldestKey != "" {
			slc.Delete(oldestKey)
		} else {
			break
		}
	}
	
	// Add new item
	slc.LRUCache.SetWithSize(key, value, int(size))
	slc.currentSize += size
	
	return true
}

// GetCurrentSize returns the current total size
func (slc *SizeLimitedCache) GetCurrentSize() int64 {
	slc.mutex.RLock()
	defer slc.mutex.RUnlock()
	return slc.currentSize
}

// LoadingCache is a cache that automatically loads missing values
type LoadingCache struct {
	*LRUCache
	loader     func(key string) (interface{}, error)
	loadingMap sync.Map
}

// NewLoadingCache creates a new loading cache
func NewLoadingCache(capacity int, shardCount int, loader func(string) (interface{}, error)) *LoadingCache {
	return &LoadingCache{
		LRUCache: NewLRUCache(capacity, shardCount),
		loader:   loader,
	}
}

// Get retrieves a value, loading it if necessary
func (lc *LoadingCache) Get(key string) (interface{}, error) {
	// Try cache first
	if val, found := lc.LRUCache.Get(key); found {
		return val, nil
	}
	
	// Check if already loading
	if loading, exists := lc.loadingMap.LoadOrStore(key, true); exists && loading.(bool) {
		// Wait for loading to complete
		for {
			time.Sleep(10 * time.Millisecond)
			if val, found := lc.LRUCache.Get(key); found {
				return val, nil
			}
			if loading, exists := lc.loadingMap.Load(key); !exists || !loading.(bool) {
				break
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

// Example demonstrates various cache implementations
func Example() {
	fmt.Println("=== Concurrent LRU Cache Example ===")
	
	// Basic LRU Cache
	cache := NewLRUCache(100, 4)
	
	// Add some items
	for i := 0; i < 10; i++ {
		key := fmt.Sprintf("key%d", i)
		value := fmt.Sprintf("value%d", i)
		cache.Set(key, value)
	}
	
	// Retrieve items
	if val, found := cache.Get("key5"); found {
		fmt.Printf("Found: %v\n", val)
	}
	
	// Get statistics
	stats := cache.GetStats()
	fmt.Printf("Cache stats - Hits: %d, Misses: %d, Sets: %d\n", 
		stats.Hits, stats.Misses, stats.Sets)
	
	// TTL Cache example
	fmt.Println("\n=== TTL Cache Example ===")
	ttlCache := NewTTLCache(50, 4, 1*time.Second)
	
	ttlCache.Set("temp1", "expires in 1 second")
	ttlCache.SetWithTTL("temp2", "expires in 100ms", 100*time.Millisecond)
	
	fmt.Println("Immediately after setting:")
	if val, found := ttlCache.Get("temp1"); found {
		fmt.Printf("temp1: %v\n", val)
	}
	if val, found := ttlCache.Get("temp2"); found {
		fmt.Printf("temp2: %v\n", val)
	}
	
	time.Sleep(150 * time.Millisecond)
	fmt.Println("\nAfter 150ms:")
	if _, found := ttlCache.Get("temp1"); found {
		fmt.Println("temp1: still exists")
	}
	if _, found := ttlCache.Get("temp2"); !found {
		fmt.Println("temp2: expired")
	}
	
	// Loading Cache example
	fmt.Println("\n=== Loading Cache Example ===")
	loader := func(key string) (interface{}, error) {
		fmt.Printf("Loading %s...\n", key)
		time.Sleep(100 * time.Millisecond) // Simulate loading
		return fmt.Sprintf("loaded_%s", key), nil
	}
	
	loadingCache := NewLoadingCache(20, 4, loader)
	
	// First access loads the value
	val1, _ := loadingCache.Get("data1")
	fmt.Printf("First access: %v\n", val1)
	
	// Second access retrieves from cache
	val2, _ := loadingCache.Get("data1")
	fmt.Printf("Second access: %v (from cache)\n", val2)
}