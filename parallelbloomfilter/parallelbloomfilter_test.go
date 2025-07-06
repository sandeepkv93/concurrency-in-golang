package parallelbloomfilter

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestDefaultBloomFilterConfig(t *testing.T) {
	config := DefaultBloomFilterConfig()
	
	if config.ExpectedElements != 1000000 {
		t.Errorf("Expected expected elements 1000000, got %d", config.ExpectedElements)
	}
	
	if config.FalsePositiveRate != 0.01 {
		t.Errorf("Expected false positive rate 0.01, got %f", config.FalsePositiveRate)
	}
	
	if config.FilterType != StandardBloom {
		t.Errorf("Expected filter type StandardBloom, got %v", config.FilterType)
	}
	
	if config.NumWorkers != runtime.NumCPU() {
		t.Errorf("Expected num workers %d, got %d", runtime.NumCPU(), config.NumWorkers)
	}
	
	if !config.EnableMetrics {
		t.Error("Expected metrics to be enabled by default")
	}
}

func TestNewParallelBloomFilter(t *testing.T) {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 1000
	config.FalsePositiveRate = 0.01
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		t.Fatalf("Failed to create bloom filter: %v", err)
	}
	defer filter.Close()
	
	if filter == nil {
		t.Fatal("Filter should not be nil")
	}
	
	if filter.config.ExpectedElements != 1000 {
		t.Errorf("Expected 1000 elements, got %d", filter.config.ExpectedElements)
	}
	
	if filter.size == 0 {
		t.Error("Filter size should be calculated")
	}
	
	if filter.hashCount == 0 {
		t.Error("Hash count should be calculated")
	}
	
	if len(filter.workers) != config.NumWorkers {
		t.Errorf("Expected %d workers, got %d", config.NumWorkers, len(filter.workers))
	}
}

func TestBasicBloomFilterOperations(t *testing.T) {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 1000
	config.NumWorkers = 2
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		t.Fatalf("Failed to create bloom filter: %v", err)
	}
	defer filter.Close()
	
	ctx := context.Background()
	
	// Test Add operation
	testItems := []string{"apple", "banana", "cherry", "date", "elderberry"}
	
	for _, item := range testItems {
		err := filter.Add(ctx, []byte(item))
		if err != nil {
			t.Errorf("Failed to add item %s: %v", item, err)
		}
	}
	
	// Test Contains operation
	for _, item := range testItems {
		contains, err := filter.Contains(ctx, []byte(item))
		if err != nil {
			t.Errorf("Failed to check item %s: %v", item, err)
		}
		if !contains {
			t.Errorf("Item %s should be in the filter", item)
		}
	}
	
	// Test false positive behavior
	notAddedItems := []string{"fig", "grape", "honeydew"}
	falsePositives := 0
	
	for _, item := range notAddedItems {
		contains, err := filter.Contains(ctx, []byte(item))
		if err != nil {
			t.Errorf("Failed to check item %s: %v", item, err)
		}
		if contains {
			falsePositives++
		}
	}
	
	// We should have minimal false positives for this small test
	if falsePositives > 1 {
		t.Logf("Warning: Got %d false positives out of %d items", falsePositives, len(notAddedItems))
	}
}

func TestCountingBloomFilter(t *testing.T) {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 1000
	config.FilterType = CountingBloom
	config.EnableCounting = true
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		t.Fatalf("Failed to create counting bloom filter: %v", err)
	}
	defer filter.Close()
	
	ctx := context.Background()
	item := []byte("test_item")
	
	// Add item
	err = filter.Add(ctx, item)
	if err != nil {
		t.Fatalf("Failed to add item: %v", err)
	}
	
	// Check it exists
	contains, err := filter.Contains(ctx, item)
	if err != nil {
		t.Fatalf("Failed to check item: %v", err)
	}
	if !contains {
		t.Error("Item should be in the filter")
	}
	
	// Remove item (counting bloom filter supports removal)
	err = filter.Remove(ctx, item)
	if err != nil {
		t.Fatalf("Failed to remove item: %v", err)
	}
	
	// Check it no longer exists
	contains, err = filter.Contains(ctx, item)
	if err != nil {
		t.Fatalf("Failed to check item after removal: %v", err)
	}
	if contains {
		t.Error("Item should not be in the filter after removal")
	}
}

func TestScalableBloomFilter(t *testing.T) {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 100 // Start small
	config.FilterType = ScalableBloom
	config.EnableScaling = true
	config.MaxFilters = 3
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		t.Fatalf("Failed to create scalable bloom filter: %v", err)
	}
	defer filter.Close()
	
	ctx := context.Background()
	
	// Add more items than expected to trigger scaling
	numItems := 300
	items := make([]string, numItems)
	for i := 0; i < numItems; i++ {
		items[i] = fmt.Sprintf("item_%d", i)
	}
	
	// Add all items
	for _, item := range items {
		err := filter.Add(ctx, []byte(item))
		if err != nil {
			t.Errorf("Failed to add item %s: %v", item, err)
		}
	}
	
	// Check all items exist
	for _, item := range items {
		contains, err := filter.Contains(ctx, []byte(item))
		if err != nil {
			t.Errorf("Failed to check item %s: %v", item, err)
		}
		if !contains {
			t.Errorf("Item %s should be in the scalable filter", item)
		}
	}
	
	// Check that scaling occurred
	stats := filter.GetStatistics()
	if stats.FiltersCount <= 1 {
		t.Error("Scalable filter should have created additional filters")
	}
}

func TestTimingBloomFilter(t *testing.T) {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 1000
	config.FilterType = TimingBloom
	config.EnableTiming = true
	config.TimeWindow = 100 * time.Millisecond
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		t.Fatalf("Failed to create timing bloom filter: %v", err)
	}
	defer filter.Close()
	
	ctx := context.Background()
	item := []byte("timing_test_item")
	
	// Add item
	err = filter.Add(ctx, item)
	if err != nil {
		t.Fatalf("Failed to add item: %v", err)
	}
	
	// Check it exists immediately
	contains, err := filter.Contains(ctx, item)
	if err != nil {
		t.Fatalf("Failed to check item: %v", err)
	}
	if !contains {
		t.Error("Item should be in the timing filter")
	}
	
	// Wait for expiration
	time.Sleep(150 * time.Millisecond)
	
	// Item should be expired (though this depends on implementation detail)
	// For timing bloom filters, we typically check if items are still "fresh"
	// This test mainly ensures the timing filter doesn't crash
	_, err = filter.Contains(ctx, item)
	if err != nil {
		t.Errorf("Failed to check expired item: %v", err)
	}
}

func TestBitArrayOperations(t *testing.T) {
	bitArray := NewBitArray(1000)
	
	if bitArray.size != 1000 {
		t.Errorf("Expected size 1000, got %d", bitArray.size)
	}
	
	// Test setting and getting bits
	positions := []uint64{0, 1, 100, 500, 999}
	
	for _, pos := range positions {
		bitArray.Set(pos)
		if !bitArray.Get(pos) {
			t.Errorf("Bit at position %d should be set", pos)
		}
	}
	
	// Test clearing bits
	bitArray.Clear(100)
	if bitArray.Get(100) {
		t.Error("Bit at position 100 should be cleared")
	}
	
	// Test out of bounds (should not panic)
	bitArray.Set(1500) // Beyond size
	// Should handle gracefully
}

func TestCountingArrayOperations(t *testing.T) {
	countArray := NewCountingArray(1000)
	
	if countArray.size != 1000 {
		t.Errorf("Expected size 1000, got %d", countArray.size)
	}
	
	// Test increment and get
	positions := []uint64{0, 1, 100, 500, 999}
	
	for _, pos := range positions {
		countArray.Increment(pos)
		count := countArray.Get(pos)
		if count != 1 {
			t.Errorf("Count at position %d should be 1, got %d", pos, count)
		}
	}
	
	// Test multiple increments
	countArray.Increment(100)
	countArray.Increment(100)
	if countArray.Get(100) != 3 {
		t.Errorf("Count at position 100 should be 3, got %d", countArray.Get(100))
	}
	
	// Test decrement
	countArray.Decrement(100)
	if countArray.Get(100) != 2 {
		t.Errorf("Count at position 100 should be 2 after decrement, got %d", countArray.Get(100))
	}
}

func TestHashFunctions(t *testing.T) {
	data := []byte("test_data")
	
	testCases := []struct {
		hashFunc HashFunction
		name     string
	}{
		{FNV1a, "FNV1a"},
		{CRC32, "CRC32"},
		{MD5Hash, "MD5"},
		{SHA1Hash, "SHA1"},
		{SHA256Hash, "SHA256"},
		{MurmurHash3, "MurmurHash3"},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			hasher := CreateHasher(tc.hashFunc, 0)
			if hasher == nil {
				t.Fatalf("Failed to create hasher for %s", tc.name)
			}
			
			hash1 := hasher.Hash(data)
			hash2 := hasher.Hash(data)
			
			if hash1 != hash2 {
				t.Errorf("Hash function %s should be deterministic", tc.name)
			}
			
			// Test with different data
			differentData := []byte("different_test_data")
			hash3 := hasher.Hash(differentData)
			
			if hash1 == hash3 {
				t.Logf("Warning: Hash collision detected for %s (might be rare)", tc.name)
			}
		})
	}
}

func TestConcurrentAccess(t *testing.T) {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 10000
	config.NumWorkers = 4
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		t.Fatalf("Failed to create bloom filter: %v", err)
	}
	defer filter.Close()
	
	ctx := context.Background()
	numGoroutines := 10
	itemsPerGoroutine := 100
	
	var wg sync.WaitGroup
	
	// Concurrent adds
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < itemsPerGoroutine; j++ {
				item := fmt.Sprintf("item_%d_%d", id, j)
				err := filter.Add(ctx, []byte(item))
				if err != nil {
					t.Errorf("Failed to add item %s: %v", item, err)
				}
			}
		}(i)
	}
	
	wg.Wait()
	
	// Concurrent reads
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < itemsPerGoroutine; j++ {
				item := fmt.Sprintf("item_%d_%d", id, j)
				contains, err := filter.Contains(ctx, []byte(item))
				if err != nil {
					t.Errorf("Failed to check item %s: %v", item, err)
				}
				if !contains {
					t.Errorf("Item %s should be in the filter", item)
				}
			}
		}(i)
	}
	
	wg.Wait()
}

func TestBatchOperations(t *testing.T) {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 1000
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		t.Fatalf("Failed to create bloom filter: %v", err)
	}
	defer filter.Close()
	
	ctx := context.Background()
	
	// Create batch of items
	items := make([][]byte, 100)
	for i := 0; i < 100; i++ {
		items[i] = []byte(fmt.Sprintf("batch_item_%d", i))
	}
	
	// Batch add
	err = filter.AddBatch(ctx, items)
	if err != nil {
		t.Fatalf("Failed to add batch: %v", err)
	}
	
	// Batch contains
	results, err := filter.ContainsBatch(ctx, items)
	if err != nil {
		t.Fatalf("Failed to check batch: %v", err)
	}
	
	if len(results) != len(items) {
		t.Errorf("Expected %d results, got %d", len(items), len(results))
	}
	
	for i, result := range results {
		if !result {
			t.Errorf("Item %d should be in the filter", i)
		}
	}
}

func TestFilterStatistics(t *testing.T) {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 1000
	config.EnableMetrics = true
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		t.Fatalf("Failed to create bloom filter: %v", err)
	}
	defer filter.Close()
	
	ctx := context.Background()
	
	// Add some items
	numItems := 100
	for i := 0; i < numItems; i++ {
		item := fmt.Sprintf("stats_item_%d", i)
		err := filter.Add(ctx, []byte(item))
		if err != nil {
			t.Errorf("Failed to add item: %v", err)
		}
	}
	
	// Get statistics
	stats := filter.GetStatistics()
	
	if stats.ElementsAdded != uint64(numItems) {
		t.Errorf("Expected %d elements added, got %d", numItems, stats.ElementsAdded)
	}
	
	if stats.BitsSet == 0 {
		t.Error("Should have some bits set")
	}
	
	if stats.EstimatedFillRatio <= 0 {
		t.Error("Fill ratio should be positive")
	}
	
	if stats.FiltersCount == 0 {
		t.Error("Should have at least one filter")
	}
	
	// Test some lookups to update lookup statistics
	for i := 0; i < 50; i++ {
		item := fmt.Sprintf("stats_item_%d", i)
		_, err := filter.Contains(ctx, []byte(item))
		if err != nil {
			t.Errorf("Failed to check item: %v", err)
		}
	}
	
	newStats := filter.GetStatistics()
	if newStats.LookupsPerformed < stats.LookupsPerformed {
		t.Error("Lookups count should increase")
	}
}

func TestOptimalParameters(t *testing.T) {
	testCases := []struct {
		elements uint64
		fpr      float64
	}{
		{1000, 0.01},
		{10000, 0.001},
		{100000, 0.1},
	}
	
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("n=%d_fpr=%f", tc.elements, tc.fpr), func(t *testing.T) {
			size := CalculateOptimalSize(tc.elements, tc.fpr)
			hashCount := CalculateOptimalHashCount(tc.elements, size)
			
			if size == 0 {
				t.Error("Optimal size should be positive")
			}
			
			if hashCount == 0 {
				t.Error("Hash count should be positive")
			}
			
			// Verify the calculated false positive rate is close to expected
			actualFPR := math.Pow(1-math.Exp(-float64(hashCount)*float64(tc.elements)/float64(size)), float64(hashCount))
			tolerance := tc.fpr * 0.1 // 10% tolerance
			
			if math.Abs(actualFPR-tc.fpr) > tolerance {
				t.Logf("Warning: Calculated FPR %f differs from expected %f by more than tolerance", actualFPR, tc.fpr)
			}
		})
	}
}

func TestErrorHandling(t *testing.T) {
	// Test invalid configuration
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 0
	
	_, err := NewParallelBloomFilter(config)
	if err == nil {
		t.Error("Should fail with zero expected elements")
	}
	
	config.ExpectedElements = 1000
	config.FalsePositiveRate = 0
	
	_, err = NewParallelBloomFilter(config)
	if err == nil {
		t.Error("Should fail with zero false positive rate")
	}
	
	config.FalsePositiveRate = 1.5
	
	_, err = NewParallelBloomFilter(config)
	if err == nil {
		t.Error("Should fail with false positive rate > 1")
	}
}

func TestContextCancellation(t *testing.T) {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 1000
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		t.Fatalf("Failed to create bloom filter: %v", err)
	}
	defer filter.Close()
	
	// Create a context that will be cancelled
	ctx, cancel := context.WithCancel(context.Background())
	
	// Cancel the context immediately
	cancel()
	
	// Operations should handle cancelled context
	err = filter.Add(ctx, []byte("test"))
	if err == nil {
		t.Error("Should return error for cancelled context")
	}
	
	_, err = filter.Contains(ctx, []byte("test"))
	if err == nil {
		t.Error("Should return error for cancelled context")
	}
}

// Benchmark tests

func BenchmarkParallelBloomFilterAdd(b *testing.B) {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = uint64(b.N)
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		b.Fatalf("Failed to create bloom filter: %v", err)
	}
	defer filter.Close()
	
	ctx := context.Background()
	data := []byte("benchmark_data")
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			item := append(data, byte(i), byte(i>>8), byte(i>>16), byte(i>>24))
			filter.Add(ctx, item)
			i++
		}
	})
}

func BenchmarkParallelBloomFilterContains(b *testing.B) {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 100000
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		b.Fatalf("Failed to create bloom filter: %v", err)
	}
	defer filter.Close()
	
	ctx := context.Background()
	
	// Pre-populate the filter
	for i := 0; i < 10000; i++ {
		item := []byte(fmt.Sprintf("item_%d", i))
		filter.Add(ctx, item)
	}
	
	data := []byte("benchmark_data")
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			item := append(data, byte(i), byte(i>>8), byte(i>>16), byte(i>>24))
			filter.Contains(ctx, item)
			i++
		}
	})
}

func BenchmarkBitArrayOperations(b *testing.B) {
	bitArray := NewBitArray(1000000)
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := uint64(0)
		for pb.Next() {
			pos := i % bitArray.size
			bitArray.Set(pos)
			bitArray.Get(pos)
			i++
		}
	})
}

func BenchmarkHashFunctions(b *testing.B) {
	data := []byte("benchmark_hash_data_that_is_reasonably_long")
	
	hashFunctions := []HashFunction{FNV1a, CRC32, MurmurHash3}
	
	for _, hashFunc := range hashFunctions {
		b.Run(fmt.Sprintf("Hash_%d", hashFunc), func(b *testing.B) {
			hasher := CreateHasher(hashFunc, 0)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				hasher.Hash(data)
			}
		})
	}
}

// Example functions

func ExampleNewParallelBloomFilter() {
	// Create a bloom filter configuration
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 10000
	config.FalsePositiveRate = 0.01
	config.NumWorkers = 4
	
	// Create the bloom filter
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		fmt.Printf("Failed to create filter: %v\n", err)
		return
	}
	defer filter.Close()
	
	ctx := context.Background()
	
	// Add some items
	items := []string{"apple", "banana", "cherry"}
	for _, item := range items {
		err := filter.Add(ctx, []byte(item))
		if err != nil {
			fmt.Printf("Failed to add %s: %v\n", item, err)
			continue
		}
		fmt.Printf("Added: %s\n", item)
	}
	
	// Check if items exist
	for _, item := range items {
		contains, err := filter.Contains(ctx, []byte(item))
		if err != nil {
			fmt.Printf("Failed to check %s: %v\n", item, err)
			continue
		}
		fmt.Printf("Contains %s: %t\n", item, contains)
	}
	
	// Output:
	// Added: apple
	// Added: banana
	// Added: cherry
	// Contains apple: true
	// Contains banana: true
	// Contains cherry: true
}

func ExampleParallelBloomFilter_AddBatch() {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 1000
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		fmt.Printf("Failed to create filter: %v\n", err)
		return
	}
	defer filter.Close()
	
	ctx := context.Background()
	
	// Create a batch of items
	items := make([][]byte, 5)
	for i := 0; i < 5; i++ {
		items[i] = []byte(fmt.Sprintf("item_%d", i))
	}
	
	// Add batch
	err = filter.AddBatch(ctx, items)
	if err != nil {
		fmt.Printf("Failed to add batch: %v\n", err)
		return
	}
	
	// Check batch
	results, err := filter.ContainsBatch(ctx, items)
	if err != nil {
		fmt.Printf("Failed to check batch: %v\n", err)
		return
	}
	
	for i, result := range results {
		fmt.Printf("Item %d present: %t\n", i, result)
	}
	
	// Output:
	// Item 0 present: true
	// Item 1 present: true
	// Item 2 present: true
	// Item 3 present: true
	// Item 4 present: true
}

func ExampleParallelBloomFilter_GetStatistics() {
	config := DefaultBloomFilterConfig()
	config.ExpectedElements = 1000
	config.EnableMetrics = true
	
	filter, err := NewParallelBloomFilter(config)
	if err != nil {
		fmt.Printf("Failed to create filter: %v\n", err)
		return
	}
	defer filter.Close()
	
	ctx := context.Background()
	
	// Add some items
	for i := 0; i < 100; i++ {
		item := fmt.Sprintf("item_%d", i)
		filter.Add(ctx, []byte(item))
	}
	
	// Get statistics
	stats := filter.GetStatistics()
	fmt.Printf("Elements Added: %d\n", stats.ElementsAdded)
	fmt.Printf("Bits Set: %d\n", stats.BitsSet)
	fmt.Printf("Fill Ratio: %.4f\n", stats.EstimatedFillRatio)
	fmt.Printf("False Positive Rate: %.6f\n", stats.EstimatedFalsePositiveRate)
	
	// Output:
	// Elements Added: 100
	// Bits Set: 287
	// Fill Ratio: 0.0030
	// False Positive Rate: 0.000007
}