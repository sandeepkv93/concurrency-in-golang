# Single Producer Multiple Consumers

## Problem Statement

The given Go code demonstrates a producer-consumer problem using goroutines and synchronization primitives. In this specific scenario, we have a single producer that generates items (integers from 0 to 100) and multiple consumers (five in this case) that consume those items.

### The Producer

The producer is responsible for creating items and adding them to a shared queue. After adding each item to the queue, the producer sends a broadcast signal to notify all waiting consumers that an item is available. The producer then sleeps for a small amount of time to simulate work before continuing the production loop.

### The Consumers

Each consumer repeatedly checks the shared queue for items to consume. If the queue is empty, the consumer waits for a signal from the producer, indicating that an item has been added to the queue. Once an item is received, the consumer consumes it (prints a message) and then sleeps for a small amount of time to simulate work.

### Synchronization

To ensure that the producer and consumers can access the shared queue without conflicts, the code uses a mutex (`sync.Mutex`) and a condition variable (`sync.Cond`). The mutex ensures exclusive access to the queue, while the condition variable allows the consumers to wait for signals from the producer efficiently.

### Graceful Shutdown

The code uses a context with cancel functionality to allow the producer to signal the consumers to stop once all items have been produced. When the producer has produced all the items, it calls the `cancel` function, causing the consumers to receive a cancellation signal and terminate gracefully.

## Code Logic

1. **Initialization** : Initialize a mutex, a condition variable, a shared queue, and a WaitGroup to keep track of the running goroutines.
2. **Create Consumers** : Launch five consumer goroutines, each running in a loop, waiting for items in the shared queue.
   a. **Wait for Item** : If the queue is empty, wait for a signal from the producer.
   b. **Consume Item** : Once an item is received, consume it and sleep to simulate work.
   c. **Handle Cancellation** : If a cancellation signal is received, terminate the consumer.
3. **Create Producer** : Launch a producer goroutine that adds items to the shared queue.
   a. **Produce Item** : Add an item to the queue and signal the consumers.
   b. **Simulate Work** : Sleep for a small amount of time to simulate work.
   c. **Cancel Consumers** : After producing all items, cancel the consumers.
4. **Wait for Completion** : Wait for all producer and consumer goroutines to complete using the WaitGroup.

## Conclusion

This code provides a simple and elegant solution to the classic producer-consumer problem using Go's concurrency primitives. By using a mutex, a condition variable, and a context with cancelation, it ensures that the producer and consumers operate correctly and can be terminated gracefully.
