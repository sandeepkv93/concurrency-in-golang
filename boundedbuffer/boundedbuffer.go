package boundedbuffer

import (
	"fmt"
	"sync"
	"time"
)

// BoundedBuffer represents a fixed-size buffer for producer-consumer pattern
type BoundedBuffer struct {
	buffer    []interface{}
	capacity  int
	size      int
	head      int
	tail      int
	mutex     sync.Mutex
	notFull   *sync.Cond
	notEmpty  *sync.Cond
}

// NewBoundedBuffer creates a new bounded buffer with given capacity
func NewBoundedBuffer(capacity int) *BoundedBuffer {
	bb := &BoundedBuffer{
		buffer:   make([]interface{}, capacity),
		capacity: capacity,
		size:     0,
		head:     0,
		tail:     0,
	}
	bb.notFull = sync.NewCond(&bb.mutex)
	bb.notEmpty = sync.NewCond(&bb.mutex)
	return bb
}

// Put adds an item to the buffer, blocking if buffer is full
func (bb *BoundedBuffer) Put(item interface{}) {
	bb.mutex.Lock()
	defer bb.mutex.Unlock()

	// Wait while buffer is full
	for bb.size == bb.capacity {
		bb.notFull.Wait()
	}

	// Add item to buffer
	bb.buffer[bb.tail] = item
	bb.tail = (bb.tail + 1) % bb.capacity
	bb.size++

	// Signal that buffer is not empty
	bb.notEmpty.Signal()
}

// Get removes and returns an item from the buffer, blocking if buffer is empty
func (bb *BoundedBuffer) Get() interface{} {
	bb.mutex.Lock()
	defer bb.mutex.Unlock()

	// Wait while buffer is empty
	for bb.size == 0 {
		bb.notEmpty.Wait()
	}

	// Remove item from buffer
	item := bb.buffer[bb.head]
	bb.head = (bb.head + 1) % bb.capacity
	bb.size--

	// Signal that buffer is not full
	bb.notFull.Signal()

	return item
}

// Size returns the current number of items in the buffer
func (bb *BoundedBuffer) Size() int {
	bb.mutex.Lock()
	defer bb.mutex.Unlock()
	return bb.size
}

// Example demonstrates the bounded buffer in action
func Example() {
	buffer := NewBoundedBuffer(5)
	var wg sync.WaitGroup

	// Producer
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			fmt.Printf("Producer: putting %d\n", i)
			buffer.Put(i)
			time.Sleep(100 * time.Millisecond)
		}
	}()

	// Consumer
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			item := buffer.Get()
			fmt.Printf("Consumer: got %v\n", item)
			time.Sleep(150 * time.Millisecond)
		}
	}()

	wg.Wait()
}