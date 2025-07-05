package concurrenthashmap

import (
	"fmt"
	"math/rand"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBasicOperations(t *testing.T) {
	m := NewConcurrentHashMap(16)
	
	// Test Set and Get
	m.Set("key1", "value1")
	m.Set("key2", 42)
	m.Set("key3", true)
	
	// Test Get
	if val, exists := m.Get("key1"); !exists || val != "value1" {
		t.Errorf("Expected 'value1', got %v", val)
	}
	
	if val, exists := m.Get("key2"); !exists || val != 42 {
		t.Errorf("Expected 42, got %v", val)
	}
	
	if val, exists := m.Get("key3"); !exists || val != true {
		t.Errorf("Expected true, got %v", val)
	}
	
	// Test non-existent key
	if _, exists := m.Get("nonexistent"); exists {
		t.Error("Non-existent key should not exist")
	}
	
	// Test Has
	if !m.Has("key1") {
		t.Error("key1 should exist")
	}
	
	if m.Has("nonexistent") {
		t.Error("nonexistent should not exist")
	}
	
	// Test Delete
	if !m.Delete("key2") {
		t.Error("Delete should return true for existing key")
	}
	
	if m.Has("key2") {
		t.Error("key2 should not exist after deletion")
	}
	
	if m.Delete("key2") {
		t.Error("Delete should return false for non-existent key")
	}
	
	// Test Len
	if m.Len() != 2 {
		t.Errorf("Expected length 2, got %d", m.Len())
	}
	
	// Test Clear
	m.Clear()
	if m.Len() != 0 {
		t.Errorf("Expected length 0 after clear, got %d", m.Len())
	}
}

func TestConcurrentAccess(t *testing.T) {
	m := NewConcurrentHashMap(32)
	numGoroutines := 100
	numOperations := 1000
	
	var wg sync.WaitGroup
	
	// Concurrent writes
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				key := fmt.Sprintf("key_%d_%d", id, j)
				value := fmt.Sprintf("value_%d_%d", id, j)
				m.Set(key, value)
			}
		}(i)
	}
	
	wg.Wait()
	
	// Verify all writes
	expectedLen := numGoroutines * numOperations
	if m.Len() != expectedLen {
		t.Errorf("Expected %d items, got %d", expectedLen, m.Len())
	}
	
	// Concurrent reads
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				key := fmt.Sprintf("key_%d_%d", id, j)
				expectedValue := fmt.Sprintf("value_%d_%d", id, j)
				if val, exists := m.Get(key); !exists || val != expectedValue {
					t.Errorf("Expected %s, got %v", expectedValue, val)
				}
			}
		}(i)
	}
	
	wg.Wait()
}

func TestGetOrSet(t *testing.T) {
	m := NewConcurrentHashMap(16)
	
	// First call should set the value
	val1, existed1 := m.GetOrSet("key1", "value1")
	if existed1 {
		t.Error("Key should not have existed")
	}
	if val1 != "value1" {
		t.Errorf("Expected 'value1', got %v", val1)
	}
	
	// Second call should return existing value
	val2, existed2 := m.GetOrSet("key1", "value2")
	if !existed2 {
		t.Error("Key should have existed")
	}
	if val2 != "value1" {
		t.Errorf("Expected 'value1', got %v", val2)
	}
}

func TestUpdate(t *testing.T) {
	m := NewConcurrentHashMap(16)
	
	// Update non-existent key
	m.Update("counter", func(value interface{}, exists bool) interface{} {
		if !exists {
			return int64(1)
		}
		return value.(int64) + 1
	})
	
	val, _ := m.Get("counter")
	if val.(int64) != 1 {
		t.Errorf("Expected 1, got %v", val)
	}
	
	// Update existing key
	m.Update("counter", func(value interface{}, exists bool) interface{} {
		if !exists {
			return int64(1)
		}
		return value.(int64) + 1
	})
	
	val, _ = m.Get("counter")
	if val.(int64) != 2 {
		t.Errorf("Expected 2, got %v", val)
	}
}

func TestRange(t *testing.T) {
	m := NewConcurrentHashMap(16)
	
	// Add some items
	for i := 0; i < 10; i++ {
		m.Set(fmt.Sprintf("key%d", i), i)
	}
	
	// Test range
	count := 0
	sum := 0
	m.Range(func(key string, value interface{}) bool {
		count++
		sum += value.(int)
		return true
	})
	
	if count != 10 {
		t.Errorf("Expected 10 items, got %d", count)
	}
	
	expectedSum := 45 // 0+1+2+...+9
	if sum != expectedSum {
		t.Errorf("Expected sum %d, got %d", expectedSum, sum)
	}
	
	// Test early termination
	count = 0
	m.Range(func(key string, value interface{}) bool {
		count++
		return count < 5 // Stop after 5 items
	})
	
	if count != 5 {
		t.Errorf("Expected to stop at 5, got %d", count)
	}
}

func TestAtomicCounter(t *testing.T) {
	counter := NewAtomicCounter()
	numGoroutines := 100
	incrementsPerGoroutine := 1000
	
	var wg sync.WaitGroup
	
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < incrementsPerGoroutine; j++ {
				counter.Increment("visits")
			}
		}()
	}
	
	wg.Wait()
	
	expected := int64(numGoroutines * incrementsPerGoroutine)
	actual := counter.Get("visits")
	
	if actual != expected {
		t.Errorf("Expected %d, got %d", expected, actual)
	}
}

func TestLRUConcurrentHashMap(t *testing.T) {
	lru := NewLRUConcurrentHashMap(10, 4)
	
	// Add items up to capacity
	for i := 0; i < 10; i++ {
		lru.Set(fmt.Sprintf("key%d", i), i)
	}
	
	// Access some items to make them recently used
	lru.Get("key0")
	lru.Get("key1")
	lru.Get("key2")
	
	// Add more items to trigger eviction
	for i := 10; i < 15; i++ {
		lru.Set(fmt.Sprintf("key%d", i), i)
	}
	
	// Check that recently accessed items are still there
	if _, exists := lru.Get("key0"); !exists {
		t.Error("Recently accessed key0 should not be evicted")
	}
	
	if _, exists := lru.Get("key1"); !exists {
		t.Error("Recently accessed key1 should not be evicted")
	}
	
	// Check that some old items were evicted
	evicted := 0
	for i := 3; i < 10; i++ {
		if _, exists := lru.Get(fmt.Sprintf("key%d", i)); !exists {
			evicted++
		}
	}
	
	if evicted == 0 {
		t.Error("Expected some items to be evicted")
	}
}

func TestStatistics(t *testing.T) {
	m := NewConcurrentHashMapWithStats(16)
	
	// Perform operations
	m.Set("key1", "value1")
	m.Set("key2", "value2")
	
	m.Get("key1") // Hit
	m.Get("key3") // Miss
	m.Get("key2") // Hit
	m.Get("key4") // Miss
	
	m.Delete("key1")
	
	stats := m.GetStats()
	
	if stats.Sets != 2 {
		t.Errorf("Expected 2 sets, got %d", stats.Sets)
	}
	
	if stats.Gets != 4 {
		t.Errorf("Expected 4 gets, got %d", stats.Gets)
	}
	
	if stats.Hits != 2 {
		t.Errorf("Expected 2 hits, got %d", stats.Hits)
	}
	
	if stats.Misses != 2 {
		t.Errorf("Expected 2 misses, got %d", stats.Misses)
	}
	
	if stats.Deletes != 1 {
		t.Errorf("Expected 1 delete, got %d", stats.Deletes)
	}
}

func TestRaceConditions(t *testing.T) {
	// This test should be run with -race flag
	m := NewConcurrentHashMap(32)
	
	var wg sync.WaitGroup
	
	// Multiple goroutines accessing same keys
	keys := []string{"shared1", "shared2", "shared3"}
	
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				key := keys[j%len(keys)]
				
				// Mixed operations
				switch j % 4 {
				case 0:
					m.Set(key, id*1000+j)
				case 1:
					m.Get(key)
				case 2:
					m.Update(key, func(value interface{}, exists bool) interface{} {
						if exists {
							return value.(int) + 1
						}
						return 0
					})
				case 3:
					m.Has(key)
				}
			}
		}(i)
	}
	
	wg.Wait()
}

func BenchmarkConcurrentHashMap(b *testing.B) {
	shardCounts := []int{1, 16, 32, 64}
	
	for _, shardCount := range shardCounts {
		b.Run(fmt.Sprintf("Shards_%d", shardCount), func(b *testing.B) {
			m := NewConcurrentHashMap(shardCount)
			
			// Pre-populate
			for i := 0; i < 10000; i++ {
				m.Set(strconv.Itoa(i), i)
			}
			
			b.ResetTimer()
			
			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					key := strconv.Itoa(i % 10000)
					switch i % 3 {
					case 0:
						m.Get(key)
					case 1:
						m.Set(key, i)
					case 2:
						m.Has(key)
					}
					i++
				}
			})
		})
	}
}

func BenchmarkComparison(b *testing.B) {
	// Compare with standard map + RWMutex
	b.Run("ConcurrentHashMap", func(b *testing.B) {
		m := NewConcurrentHashMap(32)
		benchmarkMap(b, mapAdapter{m})
	})
	
	b.Run("MapWithRWMutex", func(b *testing.B) {
		m := &mapWithRWMutex{
			m: make(map[string]interface{}),
		}
		benchmarkMap(b, m)
	})
	
	b.Run("SyncMap", func(b *testing.B) {
		m := &sync.Map{}
		benchmarkMap(b, syncMapAdapter{m})
	})
}

type mapInterface interface {
	Set(key string, value interface{})
	Get(key string) (interface{}, bool)
}

type mapAdapter struct {
	*ConcurrentHashMap
}

type mapWithRWMutex struct {
	sync.RWMutex
	m map[string]interface{}
}

func (m *mapWithRWMutex) Set(key string, value interface{}) {
	m.Lock()
	m.m[key] = value
	m.Unlock()
}

func (m *mapWithRWMutex) Get(key string) (interface{}, bool) {
	m.RLock()
	v, ok := m.m[key]
	m.RUnlock()
	return v, ok
}

type syncMapAdapter struct {
	*sync.Map
}

func (m syncMapAdapter) Set(key string, value interface{}) {
	m.Store(key, value)
}

func (m syncMapAdapter) Get(key string) (interface{}, bool) {
	return m.Load(key)
}

func benchmarkMap(b *testing.B, m mapInterface) {
	// Pre-populate
	for i := 0; i < 1000; i++ {
		m.Set(strconv.Itoa(i), i)
	}
	
	b.ResetTimer()
	
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := strconv.Itoa(i % 1000)
			if i%2 == 0 {
				m.Get(key)
			} else {
				m.Set(key, i)
			}
			i++
		}
	})
}

func TestMemoryUsage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory usage test in short mode")
	}
	
	m := NewConcurrentHashMap(32)
	
	// Measure memory before
	var memBefore runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)
	
	// Add many items
	numItems := 100000
	for i := 0; i < numItems; i++ {
		m.Set(fmt.Sprintf("key_%d", i), fmt.Sprintf("value_%d", i))
	}
	
	// Measure memory after
	var memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memAfter)
	
	bytesPerItem := float64(memAfter.Alloc-memBefore.Alloc) / float64(numItems)
	t.Logf("Memory usage: ~%.2f bytes per item", bytesPerItem)
	
	// Clear and check memory is released
	m.Clear()
	runtime.GC()
	
	var memCleared runtime.MemStats
	runtime.ReadMemStats(&memCleared)
	
	if memCleared.Alloc >= memAfter.Alloc {
		t.Error("Memory was not released after clearing map")
	}
}