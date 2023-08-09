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
	var consumerWg sync.WaitGroup
	consumerWg.Add(numConsumers)

	// Create a context with cancel
	context, cancel := context.WithCancel(context.Background())

	// Multiple consumer goroutines
	for i := 1; i <= numConsumers; i++ {
		go func(id int) {
			defer consumerWg.Done()
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

	const numProducers = 4
	const itemsPerProducer = 25
	producerWg := sync.WaitGroup{}
	producerWg.Add(numProducers)
	// Multiple producer goroutines
	for i := 1; i <= numProducers; i++ {
		go func(id int) {
			defer producerWg.Done()
			start := (id-1)*itemsPerProducer + 1
			end := start + itemsPerProducer - 1
			for j := start; j <= end; j++ {
				mu.Lock()
				queue = append(queue, j)
				fmt.Printf("Producer %d produced %d\n", id, j)
				cond.Broadcast() // Notify all waiting consumers
				mu.Unlock()
				time.Sleep(time.Millisecond * 50) // Simulating work
			}
		}(i)
	}
	producerWg.Wait()

	// Cancel the context
	cancel()

	// Wait for consumers to finish
	consumerWg.Wait()
}
