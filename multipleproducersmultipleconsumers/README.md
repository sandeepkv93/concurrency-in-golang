# Multiple Producers Multiple Consumers

## Problem Statement

The given Go code illustrates a modified version of the producer-consumer problem with multiple producers and multiple consumers operating concurrently. In this particular scenario, there are four producers that generate items (integers) and five consumers that consume those items.

### The Producers

Unlike the single-producer version, this code has four producer goroutines, each responsible for producing a specific range of integers. Each producer:

1. **Produces an Item** : Adds an integer to the shared queue.
2. **Notifies Consumers** : Sends a broadcast signal to notify all waiting consumers that an item is available.
3. **Simulates Work** : Sleeps for a brief period to mimic work before continuing the production loop.

### The Consumers

Each of the five consumer goroutines repeatedly checks the shared queue for items to consume:

1. **Waits for an Item** : If the queue is empty, waits for a signal from a producer.
2. **Consumes an Item** : Once an item is received, consumes it and sleeps for a brief period to simulate work.
3. **Handles Cancellation** : If a cancellation signal is received, terminates the consumer.

### Synchronization

The code utilizes a mutex (`sync.Mutex`) and a condition variable (`sync.Cond`) to ensure that the producers and consumers can access the shared queue without conflicts. A separate WaitGroup is used for producers and consumers to wait for their completion.

### Graceful Shutdown

Once all producers have completed, the main function calls the `cancel` function, causing the consumers to receive a cancellation signal and terminate gracefully.

## Code Logic

1. **Initialization** : Initialize mutex, condition variable, shared queue, and separate WaitGroups for consumers and producers.
2. **Create Consumers** : Launch five consumer goroutines, each running in a loop, waiting for items in the shared queue.
3. **Create Producers** : Launch four producer goroutines, each responsible for producing a specific range of integers and adding them to the shared queue.
4. **Wait for Producers** : Wait for all producer goroutines to complete using the producer WaitGroup.
5. **Cancel Consumers** : Cancel the consumers using the context's cancel function.
6. **Wait for Consumers** : Wait for all consumer goroutines to complete using the consumer WaitGroup.

## Conclusion

This code is an extension of the classic producer-consumer problem, with the added complexity of multiple producers. It demonstrates how to use Go's concurrency primitives, including goroutines, mutexes, condition variables, contexts, and WaitGroups, to ensure proper synchronization and graceful termination in a multi-producer, multi-consumer scenario. It illustrates a powerful pattern that can be applied in various real-world applications where multiple producers and consumers interact concurrently.
