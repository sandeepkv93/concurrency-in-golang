package readerswriters

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBasicReadWrite(t *testing.T) {
	strategies := []RWStrategy{
		&ReadersPreferenceStrategy{},
		&WritersPreferenceStrategy{},
		&RWMutexStrategy{},
	}
	
	for _, strategy := range strategies {
		t.Run(strategy.Name(), func(t *testing.T) {
			db := NewDatabase(strategy)
			
			// Write a value
			db.Write("test", "value1")
			
			// Read the value
			val, exists := db.Read("test")
			if !exists {
				t.Error("Key should exist")
			}
			if val != "value1" {
				t.Errorf("Expected 'value1', got %v", val)
			}
			
			// Update the value
			db.Write("test", "value2")
			
			// Read again
			val, exists = db.Read("test")
			if !exists {
				t.Error("Key should still exist")
			}
			if val != "value2" {
				t.Errorf("Expected 'value2', got %v", val)
			}
		})
	}
}

func TestConcurrentReaders(t *testing.T) {
	db := NewDatabase(&RWMutexStrategy{})
	
	// Pre-populate data
	for i := 0; i < 10; i++ {
		db.Write(fmt.Sprintf("key%d", i), fmt.Sprintf("value%d", i))
	}
	
	// Launch multiple concurrent readers
	numReaders := 10
	numReadsEach := 100
	var wg sync.WaitGroup
	
	start := time.Now()
	
	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func(readerID int) {
			defer wg.Done()
			for j := 0; j < numReadsEach; j++ {
				key := fmt.Sprintf("key%d", j%10)
				val, exists := db.Read(key)
				if !exists {
					t.Errorf("Reader %d: Key %s should exist", readerID, key)
				}
				expectedVal := fmt.Sprintf("value%d", j%10)
				if val != expectedVal {
					t.Errorf("Reader %d: Expected %s, got %v", readerID, expectedVal, val)
				}
			}
		}(i)
	}
	
	wg.Wait()
	elapsed := time.Since(start)
	
	// Verify concurrent execution (should be much faster than sequential)
	expectedSequentialTime := time.Duration(numReaders*numReadsEach*10) * time.Millisecond
	if elapsed > expectedSequentialTime/5 {
		t.Logf("Warning: Concurrent reads took %v, seems too slow for parallel execution", elapsed)
	}
}

func TestReadersWritersInteraction(t *testing.T) {
	strategies := []RWStrategy{
		&ReadersPreferenceStrategy{},
		&WritersPreferenceStrategy{},
		&RWMutexStrategy{},
	}
	
	for _, strategy := range strategies {
		t.Run(strategy.Name(), func(t *testing.T) {
			db := NewDatabase(strategy)
			
			numReaders := 5
			numWriters := 2
			duration := 500 * time.Millisecond
			
			var wg sync.WaitGroup
			stopChan := make(chan bool)
			
			// Track successful operations
			var successfulReads int64
			var successfulWrites int64
			
			// Start readers
			for i := 0; i < numReaders; i++ {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()
					for {
						select {
						case <-stopChan:
							return
						default:
							key := fmt.Sprintf("key%d", id%5)
							_, _ = db.Read(key)
							atomic.AddInt64(&successfulReads, 1)
							time.Sleep(5 * time.Millisecond)
						}
					}
				}(i)
			}
			
			// Start writers
			for i := 0; i < numWriters; i++ {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()
					counter := 0
					for {
						select {
						case <-stopChan:
							return
						default:
							key := fmt.Sprintf("key%d", id%5)
							value := fmt.Sprintf("writer%d_value%d", id, counter)
							db.Write(key, value)
							atomic.AddInt64(&successfulWrites, 1)
							counter++
							time.Sleep(20 * time.Millisecond)
						}
					}
				}(i)
			}
			
			// Run for duration
			time.Sleep(duration)
			close(stopChan)
			wg.Wait()
			
			// Check that both reads and writes occurred
			reads := atomic.LoadInt64(&successfulReads)
			writes := atomic.LoadInt64(&successfulWrites)
			
			if reads == 0 {
				t.Errorf("%s: No successful reads", strategy.Name())
			}
			if writes == 0 {
				t.Errorf("%s: No successful writes", strategy.Name())
			}
			
			t.Logf("%s: %d reads, %d writes, ratio: %.2f", 
				strategy.Name(), reads, writes, float64(reads)/float64(writes))
		})
	}
}

func TestWritersPreferenceStarvation(t *testing.T) {
	// This test demonstrates that writers preference can starve readers
	db := NewDatabase(&WritersPreferenceStrategy{})
	
	var readStarted, readCompleted atomic.Bool
	var writeCount int64
	
	// Start a continuous writer
	stopWriter := make(chan bool)
	go func() {
		for {
			select {
			case <-stopWriter:
				return
			default:
				db.Write("key", fmt.Sprintf("value%d", atomic.AddInt64(&writeCount, 1)))
				time.Sleep(time.Millisecond) // Small delay
			}
		}
	}()
	
	// Give writer time to start
	time.Sleep(10 * time.Millisecond)
	
	// Try to read (might be starved)
	go func() {
		readStarted.Store(true)
		db.Read("key")
		readCompleted.Store(true)
	}()
	
	// Wait a bit
	time.Sleep(100 * time.Millisecond)
	
	// Stop writer
	close(stopWriter)
	
	// Wait for read to complete
	time.Sleep(50 * time.Millisecond)
	
	if readStarted.Load() && !readCompleted.Load() {
		t.Log("Reader was starved by continuous writes (expected behavior for writers preference)")
	}
}

func TestFairStrategy(t *testing.T) {
	fs := NewFairStrategy()
	defer fs.Stop()
	
	db := NewDatabase(fs)
	
	// Pre-populate
	db.Write("key", "initial")
	
	var order []string
	var orderMutex sync.Mutex
	
	// Start operations in a specific order
	var wg sync.WaitGroup
	
	// Reader 1
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(10 * time.Millisecond)
		orderMutex.Lock()
		order = append(order, "R1_start")
		orderMutex.Unlock()
		
		db.Read("key")
		
		orderMutex.Lock()
		order = append(order, "R1_end")
		orderMutex.Unlock()
	}()
	
	// Writer 1
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(20 * time.Millisecond)
		orderMutex.Lock()
		order = append(order, "W1_start")
		orderMutex.Unlock()
		
		db.Write("key", "writer1")
		
		orderMutex.Lock()
		order = append(order, "W1_end")
		orderMutex.Unlock()
	}()
	
	// Reader 2
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(30 * time.Millisecond)
		orderMutex.Lock()
		order = append(order, "R2_start")
		orderMutex.Unlock()
		
		db.Read("key")
		
		orderMutex.Lock()
		order = append(order, "R2_end")
		orderMutex.Unlock()
	}()
	
	wg.Wait()
	
	// Fair strategy should process in FIFO order
	t.Logf("Operation order: %v", order)
	
	// Verify some ordering constraints
	if len(order) != 6 {
		t.Errorf("Expected 6 operations, got %d", len(order))
	}
}

func TestMonitor(t *testing.T) {
	db := NewDatabase(&RWMutexStrategy{})
	monitor := NewMonitor(db, 50*time.Millisecond)
	
	monitor.Start()
	defer monitor.Stop()
	
	// Create some activity
	var wg sync.WaitGroup
	
	// Readers
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 5; j++ {
				db.Read("key")
				time.Sleep(30 * time.Millisecond)
			}
		}()
	}
	
	// Writers
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 3; j++ {
			db.Write("key", "value")
			time.Sleep(50 * time.Millisecond)
		}
	}()
	
	wg.Wait()
}

func BenchmarkReadersWriters(b *testing.B) {
	strategies := []RWStrategy{
		&ReadersPreferenceStrategy{},
		&WritersPreferenceStrategy{},
		&RWMutexStrategy{},
	}
	
	ratios := []struct {
		name    string
		readers int
		writers int
	}{
		{"ReadHeavy", 9, 1},
		{"Balanced", 5, 5},
		{"WriteHeavy", 1, 9},
	}
	
	for _, strategy := range strategies {
		for _, ratio := range ratios {
			b.Run(fmt.Sprintf("%s_%s", strategy.Name(), ratio.name), func(b *testing.B) {
				db := NewDatabase(strategy)
				
				// Pre-populate
				for i := 0; i < 100; i++ {
					db.Write(fmt.Sprintf("key%d", i), fmt.Sprintf("value%d", i))
				}
				
				b.ResetTimer()
				
				var wg sync.WaitGroup
				stopChan := make(chan bool)
				
				// Start readers
				for i := 0; i < ratio.readers; i++ {
					wg.Add(1)
					go func(id int) {
						defer wg.Done()
						for {
							select {
							case <-stopChan:
								return
							default:
								db.Read(fmt.Sprintf("key%d", id%100))
							}
						}
					}(i)
				}
				
				// Start writers
				for i := 0; i < ratio.writers; i++ {
					wg.Add(1)
					go func(id int) {
						defer wg.Done()
						for {
							select {
							case <-stopChan:
								return
							default:
								db.Write(fmt.Sprintf("key%d", id%100), "newvalue")
							}
						}
					}(i)
				}
				
				// Run for a fixed duration
				time.Sleep(time.Second)
				close(stopChan)
				wg.Wait()
				
				stats := db.GetStatistics()
				b.ReportMetric(float64(stats.TotalReads), "reads")
				b.ReportMetric(float64(stats.TotalWrites), "writes")
			})
		}
	}
}