package ratelimiter

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// RateLimiter interface defines the contract for rate limiters
type RateLimiter interface {
	Allow() bool
	AllowN(n int) bool
	Wait(ctx context.Context) error
	WaitN(ctx context.Context, n int) error
}

// TokenBucketLimiter implements token bucket algorithm
type TokenBucketLimiter struct {
	capacity     int
	refillRate   int           // tokens per second
	tokens       int
	lastRefill   time.Time
	mutex        sync.Mutex
	refillTicker *time.Ticker
	done         chan bool
}

// NewTokenBucketLimiter creates a new token bucket rate limiter
func NewTokenBucketLimiter(capacity, refillRate int) *TokenBucketLimiter {
	limiter := &TokenBucketLimiter{
		capacity:   capacity,
		refillRate: refillRate,
		tokens:     capacity,
		lastRefill: time.Now(),
		done:       make(chan bool),
	}
	
	// Start background refill goroutine
	limiter.refillTicker = time.NewTicker(time.Second / time.Duration(refillRate))
	go limiter.refillLoop()
	
	return limiter
}

func (l *TokenBucketLimiter) refillLoop() {
	for {
		select {
		case <-l.refillTicker.C:
			l.refill()
		case <-l.done:
			return
		}
	}
}

func (l *TokenBucketLimiter) refill() {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	
	if l.tokens < l.capacity {
		l.tokens++
	}
}

// Allow checks if a request can proceed
func (l *TokenBucketLimiter) Allow() bool {
	return l.AllowN(1)
}

// AllowN checks if n requests can proceed
func (l *TokenBucketLimiter) AllowN(n int) bool {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	
	if l.tokens >= n {
		l.tokens -= n
		return true
	}
	return false
}

// Wait blocks until a request can proceed
func (l *TokenBucketLimiter) Wait(ctx context.Context) error {
	return l.WaitN(ctx, 1)
}

// WaitN blocks until n requests can proceed
func (l *TokenBucketLimiter) WaitN(ctx context.Context, n int) error {
	if n > l.capacity {
		return fmt.Errorf("requested tokens %d exceeds capacity %d", n, l.capacity)
	}
	
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if l.AllowN(n) {
				return nil
			}
		}
	}
}

// Stop stops the background refill goroutine
func (l *TokenBucketLimiter) Stop() {
	l.refillTicker.Stop()
	close(l.done)
}

// LeakyBucketLimiter implements leaky bucket algorithm
type LeakyBucketLimiter struct {
	capacity   int
	leakRate   time.Duration // time between leaks
	queue      chan time.Time
	done       chan bool
	wg         sync.WaitGroup
}

// NewLeakyBucketLimiter creates a new leaky bucket rate limiter
func NewLeakyBucketLimiter(capacity int, leakRate time.Duration) *LeakyBucketLimiter {
	limiter := &LeakyBucketLimiter{
		capacity: capacity,
		leakRate: leakRate,
		queue:    make(chan time.Time, capacity),
		done:     make(chan bool),
	}
	
	// Start leak goroutine
	limiter.wg.Add(1)
	go limiter.leak()
	
	return limiter
}

func (l *LeakyBucketLimiter) leak() {
	defer l.wg.Done()
	ticker := time.NewTicker(l.leakRate)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			select {
			case <-l.queue:
				// Leaked one request
			default:
				// Queue is empty
			}
		case <-l.done:
			return
		}
	}
}

// Allow checks if a request can proceed
func (l *LeakyBucketLimiter) Allow() bool {
	select {
	case l.queue <- time.Now():
		return true
	default:
		return false
	}
}

// AllowN checks if n requests can proceed
func (l *LeakyBucketLimiter) AllowN(n int) bool {
	if n > l.capacity {
		return false
	}
	
	// Try to add n items atomically
	for i := 0; i < n; i++ {
		if !l.Allow() {
			// Rollback on failure
			for j := 0; j < i; j++ {
				<-l.queue
			}
			return false
		}
	}
	return true
}

// Wait blocks until a request can proceed
func (l *LeakyBucketLimiter) Wait(ctx context.Context) error {
	select {
	case l.queue <- time.Now():
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// WaitN blocks until n requests can proceed
func (l *LeakyBucketLimiter) WaitN(ctx context.Context, n int) error {
	if n > l.capacity {
		return fmt.Errorf("requested items %d exceeds capacity %d", n, l.capacity)
	}
	
	for i := 0; i < n; i++ {
		if err := l.Wait(ctx); err != nil {
			// Rollback on failure
			for j := 0; j < i; j++ {
				<-l.queue
			}
			return err
		}
	}
	return nil
}

// Stop stops the leak goroutine
func (l *LeakyBucketLimiter) Stop() {
	close(l.done)
	l.wg.Wait()
}

// SlidingWindowLimiter implements sliding window rate limiting
type SlidingWindowLimiter struct {
	windowSize time.Duration
	maxRequests int
	requests   []time.Time
	mutex      sync.Mutex
}

// NewSlidingWindowLimiter creates a new sliding window rate limiter
func NewSlidingWindowLimiter(windowSize time.Duration, maxRequests int) *SlidingWindowLimiter {
	return &SlidingWindowLimiter{
		windowSize:  windowSize,
		maxRequests: maxRequests,
		requests:    make([]time.Time, 0),
	}
}

func (l *SlidingWindowLimiter) cleanup() {
	cutoff := time.Now().Add(-l.windowSize)
	i := 0
	for i < len(l.requests) && l.requests[i].Before(cutoff) {
		i++
	}
	if i > 0 {
		l.requests = l.requests[i:]
	}
}

// Allow checks if a request can proceed
func (l *SlidingWindowLimiter) Allow() bool {
	return l.AllowN(1)
}

// AllowN checks if n requests can proceed
func (l *SlidingWindowLimiter) AllowN(n int) bool {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	
	l.cleanup()
	
	if len(l.requests)+n <= l.maxRequests {
		now := time.Now()
		for i := 0; i < n; i++ {
			l.requests = append(l.requests, now)
		}
		return true
	}
	return false
}

// Wait blocks until a request can proceed
func (l *SlidingWindowLimiter) Wait(ctx context.Context) error {
	return l.WaitN(ctx, 1)
}

// WaitN blocks until n requests can proceed
func (l *SlidingWindowLimiter) WaitN(ctx context.Context, n int) error {
	if n > l.maxRequests {
		return fmt.Errorf("requested items %d exceeds max requests %d", n, l.maxRequests)
	}
	
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if l.AllowN(n) {
				return nil
			}
		}
	}
}

// Example demonstrates different rate limiters
func Example() {
	fmt.Println("=== Token Bucket Rate Limiter ===")
	tokenBucket := NewTokenBucketLimiter(5, 2) // 5 tokens capacity, 2 tokens per second
	defer tokenBucket.Stop()
	
	// Burst of requests
	for i := 0; i < 7; i++ {
		if tokenBucket.Allow() {
			fmt.Printf("Request %d: Allowed\n", i+1)
		} else {
			fmt.Printf("Request %d: Denied\n", i+1)
		}
	}
	
	// Wait for refill
	time.Sleep(2 * time.Second)
	fmt.Println("\nAfter 2 seconds:")
	for i := 0; i < 3; i++ {
		if tokenBucket.Allow() {
			fmt.Printf("Request %d: Allowed\n", i+8)
		} else {
			fmt.Printf("Request %d: Denied\n", i+8)
		}
	}
	
	fmt.Println("\n=== Leaky Bucket Rate Limiter ===")
	leakyBucket := NewLeakyBucketLimiter(3, 100*time.Millisecond)
	defer leakyBucket.Stop()
	
	// Fill the bucket
	for i := 0; i < 5; i++ {
		if leakyBucket.Allow() {
			fmt.Printf("Request %d: Allowed\n", i+1)
		} else {
			fmt.Printf("Request %d: Denied\n", i+1)
		}
	}
	
	// Wait for leak
	time.Sleep(200 * time.Millisecond)
	fmt.Println("\nAfter 200ms:")
	if leakyBucket.Allow() {
		fmt.Println("Request 6: Allowed")
	} else {
		fmt.Println("Request 6: Denied")
	}
	
	fmt.Println("\n=== Sliding Window Rate Limiter ===")
	slidingWindow := NewSlidingWindowLimiter(1*time.Second, 3)
	
	// Requests within window
	for i := 0; i < 5; i++ {
		if slidingWindow.Allow() {
			fmt.Printf("Request %d: Allowed\n", i+1)
		} else {
			fmt.Printf("Request %d: Denied\n", i+1)
		}
		time.Sleep(200 * time.Millisecond)
	}
}