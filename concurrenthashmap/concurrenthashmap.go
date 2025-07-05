package concurrenthashmap

import (
	"fmt"
	"hash/fnv"
	"sync"
	"sync/atomic"
)

// ConcurrentHashMap is a thread-safe hash map implementation
type ConcurrentHashMap struct {
	shards    []*shard
	shardCount uint32
}

// shard represents a portion of the map protected by its own lock
type shard struct {
	sync.RWMutex
	items map[string]interface{}
}

// NewConcurrentHashMap creates a new concurrent hash map with the specified number of shards
func NewConcurrentHashMap(shardCount int) *ConcurrentHashMap {
	if shardCount <= 0 {
		shardCount = 32 // Default shard count
	}
	
	m := &ConcurrentHashMap{
		shards:     make([]*shard, shardCount),
		shardCount: uint32(shardCount),
	}
	
	for i := 0; i < shardCount; i++ {
		m.shards[i] = &shard{
			items: make(map[string]interface{}),
		}
	}
	
	return m
}

// getShard returns the shard for a given key
func (m *ConcurrentHashMap) getShard(key string) *shard {
	hash := fnv.New32a()
	hash.Write([]byte(key))
	return m.shards[hash.Sum32()%m.shardCount]
}

// Set adds or updates a key-value pair
func (m *ConcurrentHashMap) Set(key string, value interface{}) {
	shard := m.getShard(key)
	shard.Lock()
	shard.items[key] = value
	shard.Unlock()
}

// Get retrieves a value by key
func (m *ConcurrentHashMap) Get(key string) (interface{}, bool) {
	shard := m.getShard(key)
	shard.RLock()
	value, exists := shard.items[key]
	shard.RUnlock()
	return value, exists
}

// Delete removes a key-value pair
func (m *ConcurrentHashMap) Delete(key string) bool {
	shard := m.getShard(key)
	shard.Lock()
	_, exists := shard.items[key]
	delete(shard.items, key)
	shard.Unlock()
	return exists
}

// Has checks if a key exists
func (m *ConcurrentHashMap) Has(key string) bool {
	shard := m.getShard(key)
	shard.RLock()
	_, exists := shard.items[key]
	shard.RUnlock()
	return exists
}

// Len returns the total number of items
func (m *ConcurrentHashMap) Len() int {
	count := 0
	for _, shard := range m.shards {
		shard.RLock()
		count += len(shard.items)
		shard.RUnlock()
	}
	return count
}

// Clear removes all items
func (m *ConcurrentHashMap) Clear() {
	for _, shard := range m.shards {
		shard.Lock()
		shard.items = make(map[string]interface{})
		shard.Unlock()
	}
}

// Keys returns all keys
func (m *ConcurrentHashMap) Keys() []string {
	keys := make([]string, 0)
	for _, shard := range m.shards {
		shard.RLock()
		for key := range shard.items {
			keys = append(keys, key)
		}
		shard.RUnlock()
	}
	return keys
}

// Items returns all key-value pairs
func (m *ConcurrentHashMap) Items() map[string]interface{} {
	items := make(map[string]interface{})
	for _, shard := range m.shards {
		shard.RLock()
		for key, value := range shard.items {
			items[key] = value
		}
		shard.RUnlock()
	}
	return items
}

// GetOrSet returns existing value or sets and returns new value
func (m *ConcurrentHashMap) GetOrSet(key string, value interface{}) (interface{}, bool) {
	shard := m.getShard(key)
	shard.Lock()
	defer shard.Unlock()
	
	if existing, exists := shard.items[key]; exists {
		return existing, true
	}
	
	shard.items[key] = value
	return value, false
}

// Update atomically updates a value
func (m *ConcurrentHashMap) Update(key string, updateFunc func(value interface{}, exists bool) interface{}) {
	shard := m.getShard(key)
	shard.Lock()
	defer shard.Unlock()
	
	value, exists := shard.items[key]
	shard.items[key] = updateFunc(value, exists)
}

// Range iterates over all key-value pairs
func (m *ConcurrentHashMap) Range(f func(key string, value interface{}) bool) {
	for _, shard := range m.shards {
		shard.RLock()
		for key, value := range shard.items {
			if !f(key, value) {
				shard.RUnlock()
				return
			}
		}
		shard.RUnlock()
	}
}

// AtomicCounter is a concurrent counter using the hash map
type AtomicCounter struct {
	m *ConcurrentHashMap
}

// NewAtomicCounter creates a new atomic counter
func NewAtomicCounter() *AtomicCounter {
	return &AtomicCounter{
		m: NewConcurrentHashMap(16),
	}
}

// Increment increments a counter
func (c *AtomicCounter) Increment(key string) int64 {
	var result int64
	c.m.Update(key, func(value interface{}, exists bool) interface{} {
		if exists {
			result = value.(int64) + 1
		} else {
			result = 1
		}
		return result
	})
	return result
}

// Get returns the current count
func (c *AtomicCounter) Get(key string) int64 {
	value, exists := c.m.Get(key)
	if !exists {
		return 0
	}
	return value.(int64)
}

// LRUConcurrentHashMap is a concurrent hash map with LRU eviction
type LRUConcurrentHashMap struct {
	capacity   int
	shards     []*lruShard
	shardCount uint32
}

type lruShard struct {
	sync.Mutex
	items    map[string]*lruItem
	head     *lruItem
	tail     *lruItem
	capacity int
}

type lruItem struct {
	key   string
	value interface{}
	prev  *lruItem
	next  *lruItem
}

// NewLRUConcurrentHashMap creates a new LRU concurrent hash map
func NewLRUConcurrentHashMap(capacity, shardCount int) *LRUConcurrentHashMap {
	if shardCount <= 0 {
		shardCount = 32
	}
	
	shardCapacity := capacity / shardCount
	if shardCapacity < 1 {
		shardCapacity = 1
	}
	
	m := &LRUConcurrentHashMap{
		capacity:   capacity,
		shards:     make([]*lruShard, shardCount),
		shardCount: uint32(shardCount),
	}
	
	for i := 0; i < shardCount; i++ {
		m.shards[i] = &lruShard{
			items:    make(map[string]*lruItem),
			capacity: shardCapacity,
		}
	}
	
	return m
}

func (m *LRUConcurrentHashMap) getShard(key string) *lruShard {
	hash := fnv.New32a()
	hash.Write([]byte(key))
	return m.shards[hash.Sum32()%m.shardCount]
}

// Set adds or updates a key-value pair with LRU eviction
func (m *LRUConcurrentHashMap) Set(key string, value interface{}) {
	shard := m.getShard(key)
	shard.Lock()
	defer shard.Unlock()
	
	// Check if key exists
	if item, exists := shard.items[key]; exists {
		item.value = value
		shard.moveToFront(item)
		return
	}
	
	// Add new item
	item := &lruItem{
		key:   key,
		value: value,
	}
	
	shard.items[key] = item
	shard.addToFront(item)
	
	// Evict if necessary
	if len(shard.items) > shard.capacity {
		shard.removeOldest()
	}
}

// Get retrieves a value and marks it as recently used
func (m *LRUConcurrentHashMap) Get(key string) (interface{}, bool) {
	shard := m.getShard(key)
	shard.Lock()
	defer shard.Unlock()
	
	item, exists := shard.items[key]
	if !exists {
		return nil, false
	}
	
	shard.moveToFront(item)
	return item.value, true
}

func (s *lruShard) moveToFront(item *lruItem) {
	if item == s.head {
		return
	}
	
	s.removeFromList(item)
	s.addToFront(item)
}

func (s *lruShard) addToFront(item *lruItem) {
	item.next = s.head
	item.prev = nil
	
	if s.head != nil {
		s.head.prev = item
	}
	s.head = item
	
	if s.tail == nil {
		s.tail = item
	}
}

func (s *lruShard) removeFromList(item *lruItem) {
	if item.prev != nil {
		item.prev.next = item.next
	} else {
		s.head = item.next
	}
	
	if item.next != nil {
		item.next.prev = item.prev
	} else {
		s.tail = item.prev
	}
}

func (s *lruShard) removeOldest() {
	if s.tail == nil {
		return
	}
	
	delete(s.items, s.tail.key)
	s.removeFromList(s.tail)
}

// Statistics tracks map operations
type Statistics struct {
	Gets    int64
	Sets    int64
	Deletes int64
	Hits    int64
	Misses  int64
}

// ConcurrentHashMapWithStats wraps a map with statistics
type ConcurrentHashMapWithStats struct {
	*ConcurrentHashMap
	stats Statistics
}

// NewConcurrentHashMapWithStats creates a map with statistics tracking
func NewConcurrentHashMapWithStats(shardCount int) *ConcurrentHashMapWithStats {
	return &ConcurrentHashMapWithStats{
		ConcurrentHashMap: NewConcurrentHashMap(shardCount),
	}
}

// Get retrieves a value and updates statistics
func (m *ConcurrentHashMapWithStats) Get(key string) (interface{}, bool) {
	atomic.AddInt64(&m.stats.Gets, 1)
	value, exists := m.ConcurrentHashMap.Get(key)
	if exists {
		atomic.AddInt64(&m.stats.Hits, 1)
	} else {
		atomic.AddInt64(&m.stats.Misses, 1)
	}
	return value, exists
}

// Set adds or updates a value and updates statistics
func (m *ConcurrentHashMapWithStats) Set(key string, value interface{}) {
	atomic.AddInt64(&m.stats.Sets, 1)
	m.ConcurrentHashMap.Set(key, value)
}

// Delete removes a value and updates statistics
func (m *ConcurrentHashMapWithStats) Delete(key string) bool {
	atomic.AddInt64(&m.stats.Deletes, 1)
	return m.ConcurrentHashMap.Delete(key)
}

// GetStats returns current statistics
func (m *ConcurrentHashMapWithStats) GetStats() Statistics {
	return Statistics{
		Gets:    atomic.LoadInt64(&m.stats.Gets),
		Sets:    atomic.LoadInt64(&m.stats.Sets),
		Deletes: atomic.LoadInt64(&m.stats.Deletes),
		Hits:    atomic.LoadInt64(&m.stats.Hits),
		Misses:  atomic.LoadInt64(&m.stats.Misses),
	}
}

// Example demonstrates concurrent hash map usage
func Example() {
	fmt.Println("=== Concurrent HashMap Example ===")
	
	// Basic usage
	m := NewConcurrentHashMap(16)
	
	// Concurrent writes
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				key := fmt.Sprintf("key_%d_%d", id, j)
				m.Set(key, fmt.Sprintf("value_%d_%d", id, j))
			}
		}(i)
	}
	
	wg.Wait()
	fmt.Printf("Total items after concurrent writes: %d\n", m.Len())
	
	// Concurrent reads
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			key := fmt.Sprintf("key_%d_50", id)
			if value, exists := m.Get(key); exists {
				fmt.Printf("Reader %d found: %v\n", id, value)
			}
		}(i)
	}
	
	wg.Wait()
	
	// Atomic counter example
	fmt.Println("\n=== Atomic Counter Example ===")
	counter := NewAtomicCounter()
	
	// Concurrent increments
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				counter.Increment("visits")
			}
		}()
	}
	
	wg.Wait()
	fmt.Printf("Total visits: %d\n", counter.Get("visits"))
	
	// LRU example
	fmt.Println("\n=== LRU Concurrent HashMap Example ===")
	lru := NewLRUConcurrentHashMap(10, 4)
	
	// Add items beyond capacity
	for i := 0; i < 15; i++ {
		key := fmt.Sprintf("item_%d", i)
		lru.Set(key, i)
	}
	
	// Check what remains
	fmt.Println("Items still in cache:")
	for i := 0; i < 15; i++ {
		key := fmt.Sprintf("item_%d", i)
		if _, exists := lru.Get(key); exists {
			fmt.Printf("%s ", key)
		}
	}
	fmt.Println()
}