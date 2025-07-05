# Rate Limiter

## Problem Description

Rate limiting is essential for controlling the rate of requests to prevent system overload and ensure fair resource usage. Common challenges include:

1. **Traffic Shaping**: Limiting request rates to prevent system overload
2. **Burst Handling**: Managing sudden spikes in traffic
3. **Fairness**: Ensuring equitable access to resources
4. **Distributed Systems**: Coordinating rate limits across multiple instances
5. **Different Traffic Patterns**: Handling various request patterns efficiently

## Solution Approach

This implementation provides three different rate limiting algorithms, each suitable for different use cases:

### Rate Limiting Algorithms

#### 1. Token Bucket Algorithm
- **Approach**: Tokens are added to a bucket at a fixed rate; requests consume tokens
- **Advantages**: Allows bursts up to bucket capacity, smooth traffic shaping
- **Use Case**: API rate limiting with burst allowance

#### 2. Leaky Bucket Algorithm
- **Approach**: Requests are processed at a fixed rate, excess requests are dropped or queued
- **Advantages**: Guarantees output rate, protects downstream systems
- **Use Case**: Traffic shaping for consistent output rates

#### 3. Sliding Window Algorithm
- **Approach**: Tracks request counts within a sliding time window
- **Advantages**: More accurate rate limiting, handles time-based quotas
- **Use Case**: Per-minute or per-hour rate limits

### Key Components

1. **Rate Limiter Interface**: Common interface for all algorithms
2. **Token Management**: Efficient token allocation and consumption
3. **Time Windows**: Sliding window implementation for time-based limits
4. **Concurrency Control**: Thread-safe operations with minimal locking
5. **Context Support**: Cancellation and timeout handling

## Usage Examples

### Token Bucket Limiter
```go
// Create limiter: 10 tokens capacity, refill 5 tokens per second
limiter := NewTokenBucketLimiter(10, 5)

// Check if request is allowed
if limiter.Allow() {
    // Process request
    fmt.Println("Request allowed")
} else {
    fmt.Println("Request denied")
}

// Wait for permission with context
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
defer cancel()

if err := limiter.Wait(ctx); err != nil {
    fmt.Printf("Request timed out: %v\n", err)
} else {
    fmt.Println("Request processed")
}
```

### Leaky Bucket Limiter
```go
// Create limiter: process 2 requests per second, queue up to 5 requests
limiter := NewLeakyBucketLimiter(2, 5)

// Try to add request to bucket
if limiter.Allow() {
    fmt.Println("Request accepted")
} else {
    fmt.Println("Bucket full, request dropped")
}
```

### Sliding Window Limiter
```go
// Create limiter: 100 requests per minute
limiter := NewSlidingWindowLimiter(100, time.Minute)

// Check multiple requests
if limiter.AllowN(5) {
    fmt.Println("Batch request allowed")
} else {
    fmt.Println("Rate limit exceeded")
}
```

## Technical Features

### Token Bucket Algorithm
- **Burst Capacity**: Allows controlled bursts of traffic
- **Smooth Refill**: Continuous token replenishment
- **Flexible Configuration**: Adjustable capacity and refill rate
- **Background Processing**: Automatic token refill via goroutine

### Leaky Bucket Algorithm
- **Consistent Output**: Guarantees steady processing rate
- **Queue Management**: Configurable queue size for request buffering
- **Overflow Protection**: Drops excess requests when queue is full
- **Backpressure**: Natural flow control mechanism

### Sliding Window Algorithm
- **Precise Counting**: Accurate request tracking within time windows
- **Memory Efficient**: Circular buffer for request timestamps
- **Configurable Windows**: Flexible time window sizes
- **Smooth Behavior**: Gradual limit enforcement

## Advanced Features

### Distributed Rate Limiting
- **Shared State**: Redis-backed rate limiter for distributed systems
- **Consensus**: Coordination across multiple instances
- **Fault Tolerance**: Graceful degradation when coordination fails

### Adaptive Rate Limiting
- **Dynamic Adjustment**: Rate limits adjust based on system load
- **Feedback Control**: Automatic rate adjustment based on response times
- **Circuit Breaker**: Integration with circuit breaker patterns

### Multi-tier Rate Limiting
- **Hierarchical Limits**: Different limits for different user tiers
- **Composite Limiters**: Multiple rate limiting strategies combined
- **Priority Queues**: Different treatment for different request types

## Performance Characteristics

| Algorithm | Memory Usage | CPU Usage | Burst Handling | Accuracy |
|-----------|-------------|-----------|----------------|----------|
| Token Bucket | Low | Low | Excellent | Good |
| Leaky Bucket | Medium | Low | Poor | Excellent |
| Sliding Window | Medium | Medium | Good | Excellent |

## Use Cases

### API Rate Limiting
- **User Quotas**: Different limits for different user types
- **Endpoint Protection**: Protect expensive operations
- **Abuse Prevention**: Prevent API abuse and DoS attacks

### Traffic Shaping
- **Bandwidth Control**: Limit network bandwidth usage
- **QoS**: Quality of Service guarantees
- **Fair Sharing**: Equitable resource distribution

### System Protection
- **Overload Prevention**: Protect downstream services
- **Resource Management**: CPU and memory usage control
- **Graceful Degradation**: Maintain service quality under load

## Integration Examples

### HTTP Middleware
```go
func RateLimitMiddleware(limiter RateLimiter) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            if !limiter.Allow() {
                http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
                return
            }
            next.ServeHTTP(w, r)
        })
    }
}
```

### gRPC Interceptor
```go
func RateLimitInterceptor(limiter RateLimiter) grpc.UnaryServerInterceptor {
    return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
        if err := limiter.Wait(ctx); err != nil {
            return nil, status.Errorf(codes.ResourceExhausted, "rate limit exceeded")
        }
        return handler(ctx, req)
    }
}
```

## Testing

The implementation includes comprehensive tests covering:
- Algorithm correctness and behavior
- Concurrent access and thread safety
- Performance benchmarking
- Burst handling capabilities
- Time-based behavior verification
- Edge cases and error conditions
- Integration with real systems
- Memory and CPU usage optimization
- Distributed coordination testing