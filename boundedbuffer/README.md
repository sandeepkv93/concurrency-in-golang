# Bounded Buffer Problem

## Problem Description

The bounded buffer problem is a classic synchronization problem in concurrent programming, also known as the producer-consumer problem with a fixed-size buffer. Multiple producers generate data and place it into a shared buffer, while multiple consumers remove data from the buffer. The challenge is to ensure that:

1. Producers don't add data when the buffer is full
2. Consumers don't try to remove data when the buffer is empty
3. Access to the buffer is thread-safe

## Solution Approach

This implementation uses Go's synchronization primitives to solve the bounded buffer problem:

### Key Components

1. **Buffer Structure**: A fixed-size slice with head/tail pointers for efficient circular buffer operations
2. **Mutex**: Ensures thread-safe access to the buffer
3. **Condition Variables**: Used for blocking and signaling between producers and consumers
   - `notFull`: Signals when buffer has space for producers
   - `notEmpty`: Signals when buffer has data for consumers

### Synchronization Strategy

- **Producers**: Wait on `notFull` condition when buffer is full, signal `notEmpty` after adding data
- **Consumers**: Wait on `notEmpty` condition when buffer is empty, signal `notFull` after removing data
- **Mutual Exclusion**: All buffer operations are protected by a mutex

### Implementation Details

- **Circular Buffer**: Uses modulo arithmetic for efficient space utilization
- **Graceful Shutdown**: Supports closing the buffer to stop all operations
- **Thread Safety**: All operations are atomic and thread-safe
- **Deadlock Prevention**: Proper ordering of lock acquisition and condition signaling

## Usage Example

```go
buffer := NewBoundedBuffer(10) // Create buffer with capacity 10

// Producer
go func() {
    for i := 0; i < 100; i++ {
        buffer.Put(i)
    }
    buffer.Close()
}()

// Consumer
go func() {
    for {
        item, ok := buffer.Get()
        if !ok {
            break // Buffer is closed
        }
        fmt.Println("Consumed:", item)
    }
}()
```

## Technical Features

- **Capacity Management**: Fixed-size buffer with overflow protection
- **Blocking Operations**: Producers and consumers block when necessary
- **Clean Shutdown**: Graceful termination of all operations
- **High Concurrency**: Supports multiple producers and consumers simultaneously
- **Memory Efficient**: Uses circular buffer to avoid memory allocation overhead

## Testing

The implementation includes comprehensive tests covering:
- Basic put/get operations
- Multiple producers and consumers
- Buffer overflow and underflow scenarios
- Graceful shutdown behavior
- Race condition detection