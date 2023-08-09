package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

func main() {
	var mu sync.Mutex
	cond := sync.NewCond(&mu)

	queue := make([]int, 0)

	const numConsumers = 5
	var wg sync.WaitGroup
	wg.Add(numConsumers + 1)

	// Create a context with cancel
	context, cancel := context.WithCancel(context.Background())

	// Multiple consumer goroutines
	for i := 0; i < numConsumers; i++ {
		go func(id int) {
			defer wg.Done()
			for {
				valueReceiverChannel := make(chan int, 1)
				go func() {
					mu.Lock()
					for len(queue) == 0 {
						cond.Wait()
					}
					item := queue[0]
					queue = queue[1:]
					valueReceiverChannel <- item
					mu.Unlock()
				}()

				select {
				case <-context.Done():
					fmt.Printf("Received cancellation signal in consumer %d\n", id)
					return
				case item := <-valueReceiverChannel:
					fmt.Printf("Consumer %d consumed %d\n", id, item)
				}

				time.Sleep(time.Millisecond * 100) // Simulating work
			}
		}(i)
	}

	// Producer goroutine
	go func() {
		defer wg.Done()
		for i := 0; i <= 100; i++ {
			mu.Lock()
			queue = append(queue, i)
			fmt.Printf("Produced %d\n", i)
			cond.Broadcast() // Notify all waiting consumers
			mu.Unlock()
			time.Sleep(time.Millisecond * 50) // Simulating work
		}
		cancel() // Cancel the context after producing 100 items
	}()

	wg.Wait()
}
