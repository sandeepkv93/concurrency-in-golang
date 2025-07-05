package ratelimiter

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestTokenBucketLimiterBasic(t *testing.T) {
	limiter := NewTokenBucketLimiter(5, 10)
	defer limiter.Stop()
	
	// Should allow 5 requests immediately (full bucket)
	for i := 0; i < 5; i++ {
		if !limiter.Allow() {
			t.Errorf("Request %d should have been allowed", i+1)
		}
	}
	
	// 6th request should be denied
	if limiter.Allow() {
		t.Error("6th request should have been denied")
	}
	
	// Wait for refill
	time.Sleep(150 * time.Millisecond)
	
	// Should allow at least one more request
	if !limiter.Allow() {
		t.Error("Request should have been allowed after refill")
	}
}

func TestTokenBucketLimiterWait(t *testing.T) {
	limiter := NewTokenBucketLimiter(2, 5)
	defer limiter.Stop()
	
	// Use up all tokens
	limiter.AllowN(2)
	
	// Wait should succeed
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	
	start := time.Now()
	err := limiter.Wait(ctx)
	elapsed := time.Since(start)
	
	if err != nil {
		t.Errorf("Wait failed: %v", err)
	}
	
	if elapsed < 100*time.Millisecond {
		t.Error("Wait returned too quickly")
	}
}

func TestLeakyBucketLimiterBasic(t *testing.T) {
	limiter := NewLeakyBucketLimiter(3, 50*time.Millisecond)
	defer limiter.Stop()
	
	// Should allow 3 requests immediately
	for i := 0; i < 3; i++ {
		if !limiter.Allow() {
			t.Errorf("Request %d should have been allowed", i+1)
		}
	}
	
	// 4th request should be denied
	if limiter.Allow() {
		t.Error("4th request should have been denied")
	}
	
	// Wait for leak
	time.Sleep(60 * time.Millisecond)
	
	// Should allow one more request
	if !limiter.Allow() {
		t.Error("Request should have been allowed after leak")
	}
}

func TestSlidingWindowLimiterBasic(t *testing.T) {
	limiter := NewSlidingWindowLimiter(100*time.Millisecond, 3)
	
	// Should allow 3 requests
	for i := 0; i < 3; i++ {
		if !limiter.Allow() {
			t.Errorf("Request %d should have been allowed", i+1)
		}
	}
	
	// 4th request should be denied
	if limiter.Allow() {
		t.Error("4th request should have been denied")
	}
	
	// Wait for window to slide
	time.Sleep(110 * time.Millisecond)
	
	// Should allow new requests
	if !limiter.Allow() {
		t.Error("Request should have been allowed after window slide")
	}
}

func TestRateLimiterConcurrency(t *testing.T) {
	tests := []struct {
		name    string
		limiter RateLimiter
		cleanup func()
	}{
		{
			name: "TokenBucket",
			limiter: func() RateLimiter {
				l := NewTokenBucketLimiter(10, 100)
				return l
			}(),
			cleanup: func() {
				if l, ok := tests[0].limiter.(*TokenBucketLimiter); ok {
					l.Stop()
				}
			},
		},
		{
			name: "LeakyBucket",
			limiter: func() RateLimiter {
				l := NewLeakyBucketLimiter(10, 10*time.Millisecond)
				return l
			}(),
			cleanup: func() {
				if l, ok := tests[1].limiter.(*LeakyBucketLimiter); ok {
					l.Stop()
				}
			},
		},
		{
			name:    "SlidingWindow",
			limiter: NewSlidingWindowLimiter(100*time.Millisecond, 10),
			cleanup: func() {},
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer tt.cleanup()
			
			var allowed int32
			var denied int32
			var wg sync.WaitGroup
			
			// Run concurrent requests
			for i := 0; i < 5; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for j := 0; j < 20; j++ {
						if tt.limiter.Allow() {
							atomic.AddInt32(&allowed, 1)
						} else {
							atomic.AddInt32(&denied, 1)
						}
						time.Sleep(5 * time.Millisecond)
					}
				}()
			}
			
			wg.Wait()
			
			total := atomic.LoadInt32(&allowed) + atomic.LoadInt32(&denied)
			if total != 100 {
				t.Errorf("Expected 100 total requests, got %d", total)
			}
			
			// Should have some denials due to rate limiting
			if atomic.LoadInt32(&denied) == 0 {
				t.Error("Expected some requests to be denied")
			}
		})
	}
}

func TestContextCancellation(t *testing.T) {
	limiter := NewTokenBucketLimiter(1, 1)
	defer limiter.Stop()
	
	// Use the token
	limiter.Allow()
	
	// Create a context that will be cancelled
	ctx, cancel := context.WithCancel(context.Background())
	
	// Start waiting in a goroutine
	errCh := make(chan error)
	go func() {
		errCh <- limiter.Wait(ctx)
	}()
	
	// Cancel the context
	time.Sleep(50 * time.Millisecond)
	cancel()
	
	// Should receive context error
	err := <-errCh
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
}

func BenchmarkTokenBucketLimiter(b *testing.B) {
	limiter := NewTokenBucketLimiter(1000, 10000)
	defer limiter.Stop()
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			limiter.Allow()
		}
	})
}

func BenchmarkSlidingWindowLimiter(b *testing.B) {
	limiter := NewSlidingWindowLimiter(time.Second, 10000)
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			limiter.Allow()
		}
	})
}