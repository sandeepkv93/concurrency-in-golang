package concurrentlrucache

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBasicLRUOperations(t *testing.T) {
	cache := NewLRUCache(3, 1)
	
	// Test Set and Get
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	cache.Set("key3", "value3")
	
	// Test Get
	val, found := cache.Get("key1")
	if !found || val != "value1" {
		t.Errorf("Expected 'value1', got %v", val)
	}
	
	// Test capacity - adding 4th item should evict least recently used
	cache.Set("key4", "value4")
	
	// key2 should be evicted (least recently used)
	if _, found := cache.Get("key2"); found {
		t.Error("key2 should have been evicted")
	}
	
	// key1 should still exist (was accessed recently)
	if _, found := cache.Get("key1"); !found {
		t.Error("key1 should still exist")
	}
	
	// Test Delete
	if !cache.Delete("key3") {
		t.Error("Delete should return true for existing key")
	}
	
	if cache.Delete("key3") {
		t.Error("Delete should return false for non-existent key")
	}
}

func TestLRUEviction(t *testing.T) {
	cache := NewLRUCache(3, 1)
	
	// Fill cache
	cache.Set("a", 1)
	cache.Set("b", 2)
	cache.Set("c", 3)
	
	// Access 'a' to make it recently used
	cache.Get("a")
	
	// Add new item, 'b' should be evicted
	cache.Set("d", 4)
	
	if _, found := cache.Get("b"); found {
		t.Error("'b' should have been evicted")
	}
	
	// a, c, d should exist
	for _, key := range []string{"a", "c", "d"} {
		if _, found := cache.Get(key); !found {
			t.Errorf("'%s' should exist", key)
		}
	}
}

func TestConcurrentAccess(t *testing.T) {
	cache := NewLRUCache(1000, 16)
	numGoroutines := 100
	numOperations := 1000
	
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)
	
	// Concurrent writes
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			for j := 0; j < numOperations; j++ {
				key := fmt.Sprintf("key_%d_%d", id, j)
				value := fmt.Sprintf("value_%d_%d", id, j)
				cache.Set(key, value)
			}
		}(i)
	}
	
	// Concurrent reads
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			for j := 0; j < numOperations; j++ {
				key := fmt.Sprintf("key_%d_%d", id, j/2)
				cache.Get(key)
			}
		}(i)
	}
	
	wg.Wait()
	close(errors)
	
	// Check for errors
	for err := range errors {
		if err != nil {
			t.Errorf("Concurrent access error: %v", err)
		}
	}
	
	// Verify cache still works
	cache.Set("test", "value")
	if val, found := cache.Get("test"); !found || val != "value" {
		t.Error("Cache broken after concurrent access")
	}
}

func TestGetOrLoad(t *testing.T) {
	cache := NewLRUCache(10, 2)
	loadCount := 0
	
	loader := func() (interface{}, error) {
		loadCount++
		return "loaded_value", nil
	}
	
	// First call should load
	val1, err := cache.GetOrLoad("key1", loader)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if val1 != "loaded_value" {
		t.Errorf("Expected 'loaded_value', got %v", val1)
	}
	if loadCount != 1 {
		t.Errorf("Expected 1 load, got %d", loadCount)
	}
	
	// Second call should get from cache
	val2, err := cache.GetOrLoad("key1", loader)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if val2 != "loaded_value" {
		t.Errorf("Expected 'loaded_value', got %v", val2)
	}
	if loadCount != 1 {
		t.Errorf("Expected still 1 load, got %d", loadCount)
	}
}

func TestStatistics(t *testing.T) {
	cache := NewLRUCache(10, 2)
	
	// Perform operations
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	
	cache.Get("key1") // Hit
	cache.Get("key3") // Miss
	cache.Get("key2") // Hit
	cache.Get("key4") // Miss
	
	stats := cache.GetStats()
	
	if stats.Hits != 2 {
		t.Errorf("Expected 2 hits, got %d", stats.Hits)
	}
	
	if stats.Misses != 2 {
		t.Errorf("Expected 2 misses, got %d", stats.Misses)
	}
	
	if stats.Sets != 2 {
		t.Errorf("Expected 2 sets, got %d", stats.Sets)
	}
}

func TestTTLCache(t *testing.T) {
	cache := NewTTLCache(10, 2, 100*time.Millisecond)
	
	// Set with default TTL
	cache.Set("key1", "value1")
	
	// Set with custom TTL
	cache.SetWithTTL("key2", "value2", 50*time.Millisecond)
	
	// Both should exist immediately
	if _, found := cache.Get("key1"); !found {
		t.Error("key1 should exist")
	}
	if _, found := cache.Get("key2"); !found {
		t.Error("key2 should exist")
	}
	
	// Wait for key2 to expire
	time.Sleep(60 * time.Millisecond)
	
	if _, found := cache.Get("key1"); !found {
		t.Error("key1 should still exist")
	}
	if _, found := cache.Get("key2"); found {
		t.Error("key2 should have expired")
	}
	
	// Wait for key1 to expire
	time.Sleep(50 * time.Millisecond)
	
	if _, found := cache.Get("key1"); found {
		t.Error("key1 should have expired")
	}
}

func TestTTLCacheCleanup(t *testing.T) {
	cache := NewTTLCache(100, 4, 50*time.Millisecond)
	
	// Start cleanup
	stop := cache.StartCleanup(25 * time.Millisecond)
	defer close(stop)
	
	// Add items
	for i := 0; i < 10; i++ {
		cache.Set(fmt.Sprintf("key%d", i), i)
	}
	
	// Items should exist
	if cache.Len() != 10 {
		t.Errorf("Expected 10 items, got %d", cache.Len())
	}
	
	// Wait for expiration and cleanup
	time.Sleep(100 * time.Millisecond)
	
	// All items should be cleaned up
	if cache.Len() > 0 {
		t.Errorf("Expected 0 items after cleanup, got %d", cache.Len())
	}
}

func TestSizeLimitedCache(t *testing.T) {
	cache := NewSizeLimitedCache(100, 2)
	
	// Add items within size limit
	cache.SetWithSize("key1", "value1", 30)
	cache.SetWithSize("key2", "value2", 40)
	
	if cache.GetCurrentSize() != 70 {
		t.Errorf("Expected size 70, got %d", cache.GetCurrentSize())
	}
	
	// Add item that requires eviction
	cache.SetWithSize("key3", "value3", 40)
	
	// Should have evicted something to make room
	if cache.GetCurrentSize() > 100 {
		t.Errorf("Size %d exceeds limit 100", cache.GetCurrentSize())
	}
	
	// Try to add item larger than max size
	if cache.SetWithSize("huge", "data", 200) {
		t.Error("Should not accept item larger than max size")
	}
}

func TestLoadingCache(t *testing.T) {
	loadCount := make(map[string]int)
	var mu sync.Mutex
	
	loader := func(key string) (interface{}, error) {
		mu.Lock()
		loadCount[key]++
		mu.Unlock()
		
		time.Sleep(50 * time.Millisecond) // Simulate loading
		return fmt.Sprintf("loaded_%s", key), nil
	}
	
	cache := NewLoadingCache(10, 2, loader)
	
	// Concurrent access to same key
	var wg sync.WaitGroup
	results := make([]interface{}, 5)
	
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			val, err := cache.Get("key1")
			if err != nil {
				t.Errorf("Error loading: %v", err)
			}
			results[idx] = val
		}(i)
	}
	
	wg.Wait()
	
	// Should only load once
	mu.Lock()
	count := loadCount["key1"]
	mu.Unlock()
	
	if count != 1 {
		t.Errorf("Expected 1 load, got %d", count)
	}
	
	// All results should be the same
	for i, val := range results {
		if val != "loaded_key1" {
			t.Errorf("Result %d: expected 'loaded_key1', got %v", i, val)
		}
	}
}

func TestSharding(t *testing.T) {
	shardCounts := []int{1, 4, 16}
	
	for _, shardCount := range shardCounts {
		t.Run(fmt.Sprintf("Shards_%d", shardCount), func(t *testing.T) {
			cache := NewLRUCache(100, shardCount)
			
			// Add items
			for i := 0; i < 50; i++ {
				cache.Set(fmt.Sprintf("key%d", i), i)
			}
			
			// Verify all items
			for i := 0; i < 50; i++ {
				key := fmt.Sprintf("key%d", i)
				val, found := cache.Get(key)
				if !found {
					t.Errorf("Key %s not found", key)
				}
				if val != i {
					t.Errorf("Key %s: expected %d, got %v", key, i, val)
				}
			}
		})
	}
}

func TestRaceConditions(t *testing.T) {
	cache := NewLRUCache(100, 16)
	
	var wg sync.WaitGroup
	
	// Multiple goroutines accessing same keys
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			for j := 0; j < 100; j++ {
				key := fmt.Sprintf("shared_%d", j%10)
				
				switch j % 3 {
				case 0:
					cache.Set(key, id*1000+j)
				case 1:
					cache.Get(key)
				case 2:
					cache.Delete(key)
				}
			}
		}(i)
	}
	
	wg.Wait()
}

func BenchmarkLRUCache(b *testing.B) {
	cache := NewLRUCache(1000, 16)
	
	// Pre-populate
	for i := 0; i < 1000; i++ {
		cache.Set(fmt.Sprintf("key%d", i), i)
	}
	
	b.ResetTimer()
	
	b.Run("Get_Hit", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				cache.Get(fmt.Sprintf("key%d", i%1000))
				i++
			}
		})
	})
	
	b.Run("Get_Miss", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				cache.Get(fmt.Sprintf("missing%d", i))
				i++
			}
		})
	})
	
	b.Run("Set", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				cache.Set(fmt.Sprintf("key%d", i%2000), i)
				i++
			}
		})
	})
	
	b.Run("Mixed", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				key := fmt.Sprintf("key%d", i%1500)
				if i%2 == 0 {
					cache.Get(key)
				} else {
					cache.Set(key, i)
				}
				i++
			}
		})
	})
}

func BenchmarkCacheComparison(b *testing.B) {
	// Compare with simple map + mutex
	b.Run("LRUCache", func(b *testing.B) {
		cache := NewLRUCache(1000, 16)
		benchmarkCache(b, &lruAdapter{cache})
	})
	
	b.Run("SimpleMap", func(b *testing.B) {
		cache := &simpleMapCache{
			m: make(map[string]interface{}),
		}
		benchmarkCache(b, cache)
	})
	
	b.Run("SyncMap", func(b *testing.B) {
		cache := &syncMapCache{}
		benchmarkCache(b, cache)
	})
}

type cacheInterface interface {
	Get(key string) (interface{}, bool)
	Set(key string, value interface{})
}

type lruAdapter struct {
	*LRUCache
}

type simpleMapCache struct {
	sync.RWMutex
	m map[string]interface{}
}

func (c *simpleMapCache) Get(key string) (interface{}, bool) {
	c.RLock()
	v, ok := c.m[key]
	c.RUnlock()
	return v, ok
}

func (c *simpleMapCache) Set(key string, value interface{}) {
	c.Lock()
	c.m[key] = value
	c.Unlock()
}

type syncMapCache struct {
	sync.Map
}

func (c *syncMapCache) Get(key string) (interface{}, bool) {
	return c.Load(key)
}

func (c *syncMapCache) Set(key string, value interface{}) {
	c.Store(key, value)
}

func benchmarkCache(b *testing.B, cache cacheInterface) {
	// Pre-populate
	for i := 0; i < 1000; i++ {
		cache.Set(fmt.Sprintf("key%d", i), i)
	}
	
	b.ResetTimer()
	
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := fmt.Sprintf("key%d", i%1000)
			if i%3 == 0 {
				cache.Set(key, i)
			} else {
				cache.Get(key)
			}
			i++
		}
	})
}

func TestMemoryUsage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory test in short mode")
	}
	
	cache := NewLRUCache(10000, 16)
	
	var m1 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)
	
	// Add many items
	for i := 0; i < 10000; i++ {
		cache.Set(fmt.Sprintf("key_%d", i), fmt.Sprintf("value_%d", i))
	}
	
	var m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m2)
	
	bytesPerItem := float64(m2.Alloc-m1.Alloc) / 10000
	t.Logf("Memory per item: ~%.2f bytes", bytesPerItem)
	
	// Clear cache
	cache.Clear()
	runtime.GC()
	
	var m3 runtime.MemStats
	runtime.ReadMemStats(&m3)
	
	if m3.Alloc >= m2.Alloc {
		t.Error("Memory not released after clearing cache")
	}
}