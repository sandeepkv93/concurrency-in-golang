package boundedbuffer

import (
	"sync"
	"testing"
	"time"
)

func TestBoundedBufferBasic(t *testing.T) {
	buffer := NewBoundedBuffer(3)

	// Test basic put and get
	buffer.Put(1)
	buffer.Put(2)
	buffer.Put(3)

	if buffer.Size() != 3 {
		t.Errorf("Expected size 3, got %d", buffer.Size())
	}

	val1 := buffer.Get().(int)
	val2 := buffer.Get().(int)
	val3 := buffer.Get().(int)

	if val1 != 1 || val2 != 2 || val3 != 3 {
		t.Errorf("Expected 1, 2, 3, got %d, %d, %d", val1, val2, val3)
	}

	if buffer.Size() != 0 {
		t.Errorf("Expected size 0, got %d", buffer.Size())
	}
}

func TestBoundedBufferConcurrent(t *testing.T) {
	buffer := NewBoundedBuffer(10)
	numProducers := 5
	numConsumers := 3
	itemsPerProducer := 20

	var wg sync.WaitGroup
	produced := make(map[int]int)
	consumed := make(map[int]int)
	var prodMutex, consMutex sync.Mutex

	// Start producers
	for p := 0; p < numProducers; p++ {
		wg.Add(1)
		go func(producerID int) {
			defer wg.Done()
			for i := 0; i < itemsPerProducer; i++ {
				item := producerID*1000 + i
				buffer.Put(item)
				prodMutex.Lock()
				produced[item]++
				prodMutex.Unlock()
			}
		}(p)
	}

	// Start consumers
	totalItems := numProducers * itemsPerProducer
	itemsPerConsumer := totalItems / numConsumers
	remainingItems := totalItems % numConsumers

	for c := 0; c < numConsumers; c++ {
		items := itemsPerConsumer
		if c < remainingItems {
			items++
		}
		wg.Add(1)
		go func(consumerID int, itemCount int) {
			defer wg.Done()
			for i := 0; i < itemCount; i++ {
				item := buffer.Get().(int)
				consMutex.Lock()
				consumed[item]++
				consMutex.Unlock()
			}
		}(c, items)
	}

	wg.Wait()

	// Verify all produced items were consumed exactly once
	if len(produced) != len(consumed) {
		t.Errorf("Produced %d unique items, consumed %d", len(produced), len(consumed))
	}

	for item, count := range produced {
		if count != 1 {
			t.Errorf("Item %d was produced %d times", item, count)
		}
		if consumed[item] != 1 {
			t.Errorf("Item %d was consumed %d times", item, consumed[item])
		}
	}
}

func TestBoundedBufferBlocking(t *testing.T) {
	buffer := NewBoundedBuffer(2)

	// Fill the buffer
	buffer.Put(1)
	buffer.Put(2)

	// Try to put another item in a goroutine (should block)
	blocked := make(chan bool, 1)
	go func() {
		blocked <- true
		buffer.Put(3) // This should block
		blocked <- false
	}()

	// Wait for the goroutine to signal it's about to block
	<-blocked

	// Give it some time to ensure it's blocked
	time.Sleep(100 * time.Millisecond)

	// Verify the goroutine is still blocked
	select {
	case <-blocked:
		t.Error("Put operation should have blocked")
	default:
		// Expected: goroutine is blocked
	}

	// Remove an item to unblock
	buffer.Get()

	// Now the goroutine should unblock
	unblocked := <-blocked
	if unblocked {
		t.Error("Goroutine should have unblocked")
	}
}

func BenchmarkBoundedBuffer(b *testing.B) {
	buffer := NewBoundedBuffer(100)
	
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i%2 == 0 {
				buffer.Put(i)
			} else {
				buffer.Get()
			}
			i++
		}
	})
}