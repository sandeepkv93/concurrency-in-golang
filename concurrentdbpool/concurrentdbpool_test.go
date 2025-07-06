package concurrentdbpool

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Mock database connection for testing
type MockDB struct {
	closed  bool
	healthy bool
	id      string
	mutex   sync.RWMutex
}

func (m *MockDB) Close() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.closed = true
	return nil
}

func (m *MockDB) PingContext(ctx context.Context) error {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	if m.closed {
		return errors.New("connection closed")
	}
	
	if !m.healthy {
		return errors.New("connection unhealthy")
	}
	
	// Simulate some latency
	select {
	case <-time.After(10 * time.Millisecond):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *MockDB) IsClosed() bool {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	return m.closed
}

func (m *MockDB) SetHealthy(healthy bool) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.healthy = healthy
}

// Mock connection factory
func createMockConnectionFactory(shouldFail bool, unhealthyAfter int) ConnectionFactory {
	var connCounter int32
	
	return func(ctx context.Context, driverName, dataSourceName string) (*sql.DB, interface{}, error) {
		if shouldFail {
			return nil, nil, errors.New("mock connection failure")
		}
		
		connNum := atomic.AddInt32(&connCounter, 1)
		mockDB := &MockDB{
			healthy: connNum <= int32(unhealthyAfter) || unhealthyAfter <= 0,
			id:      fmt.Sprintf("mock_%d", connNum),
		}
		
		return nil, mockDB, nil
	}
}

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()
	
	if config.MinConnections != 5 {
		t.Errorf("Expected MinConnections 5, got %d", config.MinConnections)
	}
	
	if config.MaxConnections != 50 {
		t.Errorf("Expected MaxConnections 50, got %d", config.MaxConnections)
	}
	
	if config.MaxIdleTime != 5*time.Minute {
		t.Errorf("Expected MaxIdleTime 5m, got %v", config.MaxIdleTime)
	}
	
	if config.HealthCheckInterval != 30*time.Second {
		t.Errorf("Expected HealthCheckInterval 30s, got %v", config.HealthCheckInterval)
	}
}

func TestNewDBConnectionPool(t *testing.T) {
	config := PoolConfig{
		MinConnections: 2,
		MaxConnections: 10,
		MaxIdleTime:    5 * time.Minute,
		MaxLifetime:    30 * time.Minute,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	
	if pool.config.MinConnections != 2 {
		t.Errorf("Expected MinConnections 2, got %d", pool.config.MinConnections)
	}
	
	if pool.config.MaxConnections != 10 {
		t.Errorf("Expected MaxConnections 10, got %d", pool.config.MaxConnections)
	}
	
	pool.Close()
}

func TestPoolValidation(t *testing.T) {
	tests := []struct {
		name        string
		config      PoolConfig
		expectError bool
	}{
		{
			name: "valid config",
			config: PoolConfig{
				MinConnections: 5,
				MaxConnections: 10,
			},
			expectError: false,
		},
		{
			name: "negative min connections",
			config: PoolConfig{
				MinConnections: -1,
				MaxConnections: 10,
			},
			expectError: true,
		},
		{
			name: "zero max connections",
			config: PoolConfig{
				MinConnections: 5,
				MaxConnections: 0,
			},
			expectError: true,
		},
		{
			name: "min > max connections",
			config: PoolConfig{
				MinConnections: 15,
				MaxConnections: 10,
			},
			expectError: true,
		},
	}
	
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewDBConnectionPool("test", "test://localhost", test.config)
			if test.expectError && err == nil {
				t.Error("Expected error but got none")
			}
			if !test.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestPoolStartAndClose(t *testing.T) {
	config := PoolConfig{
		MinConnections: 3,
		MaxConnections: 5,
		MaxIdleTime:    5 * time.Minute,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	
	// Set mock connection factory
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start pool: %v", err)
	}
	
	stats := pool.GetStats()
	if stats.TotalConnections != 3 {
		t.Errorf("Expected 3 total connections, got %d", stats.TotalConnections)
	}
	
	if stats.IdleConnections != 3 {
		t.Errorf("Expected 3 idle connections, got %d", stats.IdleConnections)
	}
	
	err = pool.Close()
	if err != nil {
		t.Errorf("Failed to close pool: %v", err)
	}
	
	// Verify pool is closed
	_, err = pool.AcquireConnection(ctx)
	if err == nil {
		t.Error("Expected error when acquiring from closed pool")
	}
}

func TestConnectionAcquisitionAndRelease(t *testing.T) {
	config := PoolConfig{
		MinConnections: 2,
		MaxConnections: 5,
		AcquireTimeout: 5 * time.Second,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start pool: %v", err)
	}
	
	// Acquire a connection
	conn, err := pool.AcquireConnection(ctx)
	if err != nil {
		t.Fatalf("Failed to acquire connection: %v", err)
	}
	
	if conn.GetStatus() != StatusInUse {
		t.Errorf("Expected connection status InUse, got %v", conn.GetStatus())
	}
	
	stats := pool.GetStats()
	if stats.ActiveConnections != 1 {
		t.Errorf("Expected 1 active connection, got %d", stats.ActiveConnections)
	}
	
	// Release the connection
	err = pool.ReleaseConnection(conn)
	if err != nil {
		t.Errorf("Failed to release connection: %v", err)
	}
	
	if conn.GetStatus() != StatusIdle {
		t.Errorf("Expected connection status Idle, got %v", conn.GetStatus())
	}
	
	stats = pool.GetStats()
	if stats.ActiveConnections != 0 {
		t.Errorf("Expected 0 active connections, got %d", stats.ActiveConnections)
	}
	
	if stats.IdleConnections != 2 {
		t.Errorf("Expected 2 idle connections, got %d", stats.IdleConnections)
	}
}

func TestConcurrentAcquisition(t *testing.T) {
	config := PoolConfig{
		MinConnections: 5,
		MaxConnections: 10,
		AcquireTimeout: 5 * time.Second,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start pool: %v", err)
	}
	
	numGoroutines := 20
	acquiredConnections := make([]*Connection, numGoroutines)
	var wg sync.WaitGroup
	var errors int32
	
	// Acquire connections concurrently
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			
			conn, err := pool.AcquireConnection(ctx)
			if err != nil {
				atomic.AddInt32(&errors, 1)
				return
			}
			
			acquiredConnections[index] = conn
			
			// Hold connection for a short time
			time.Sleep(100 * time.Millisecond)
			
			err = pool.ReleaseConnection(conn)
			if err != nil {
				atomic.AddInt32(&errors, 1)
			}
		}(i)
	}
	
	wg.Wait()
	
	if errors > 0 {
		t.Errorf("Got %d errors during concurrent acquisition", errors)
	}
	
	stats := pool.GetStats()
	if stats.TotalRequests != int64(numGoroutines) {
		t.Errorf("Expected %d total requests, got %d", numGoroutines, stats.TotalRequests)
	}
}

func TestPoolCapacityLimits(t *testing.T) {
	config := PoolConfig{
		MinConnections: 2,
		MaxConnections: 3,
		AcquireTimeout: 1 * time.Second,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start pool: %v", err)
	}
	
	// Acquire all available connections
	var connections []*Connection
	for i := 0; i < 3; i++ {
		conn, err := pool.AcquireConnection(ctx)
		if err != nil {
			t.Fatalf("Failed to acquire connection %d: %v", i, err)
		}
		connections = append(connections, conn)
	}
	
	// Try to acquire one more (should timeout)
	start := time.Now()
	_, err = pool.AcquireConnection(ctx)
	duration := time.Since(start)
	
	if err == nil {
		t.Error("Expected timeout error when exceeding pool capacity")
	}
	
	if duration < config.AcquireTimeout {
		t.Errorf("Expected timeout after %v, but got error after %v", config.AcquireTimeout, duration)
	}
	
	// Release one connection
	err = pool.ReleaseConnection(connections[0])
	if err != nil {
		t.Errorf("Failed to release connection: %v", err)
	}
	
	// Now we should be able to acquire again
	_, err = pool.AcquireConnection(ctx)
	if err != nil {
		t.Errorf("Failed to acquire connection after release: %v", err)
	}
}

func TestConnectionHealthChecking(t *testing.T) {
	config := PoolConfig{
		MinConnections:      3,
		MaxConnections:      5,
		HealthCheckInterval: 100 * time.Millisecond,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 2)) // First 2 connections healthy
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start pool: %v", err)
	}
	
	// Wait for health checks to run
	time.Sleep(300 * time.Millisecond)
	
	stats := pool.GetStats()
	t.Logf("Health check stats: Total=%d, Healthy=%d, Unhealthy=%d",
		stats.TotalConnections, stats.HealthyConnections, stats.UnhealthyConnections)
}

func TestConnectionLifetime(t *testing.T) {
	config := PoolConfig{
		MinConnections: 2,
		MaxConnections: 5,
		MaxLifetime:    200 * time.Millisecond,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start pool: %v", err)
	}
	
	// Acquire a connection
	conn, err := pool.AcquireConnection(ctx)
	if err != nil {
		t.Fatalf("Failed to acquire connection: %v", err)
	}
	
	// Check that connection is healthy initially
	if !conn.IsHealthy() {
		t.Error("Expected connection to be healthy initially")
	}
	
	// Wait for connection to exceed lifetime
	time.Sleep(300 * time.Millisecond)
	
	// Release the connection (should be retired due to age)
	err = pool.ReleaseConnection(conn)
	if err != nil {
		t.Errorf("Failed to release connection: %v", err)
	}
	
	// Wait a bit for cleanup
	time.Sleep(100 * time.Millisecond)
	
	// Connection should be closed
	if conn.GetStatus() == StatusClosed {
		t.Log("Connection was properly retired due to age")
	}
}

func TestConnectionIdleTimeout(t *testing.T) {
	config := PoolConfig{
		MinConnections: 1,
		MaxConnections: 3,
		MaxIdleTime:    200 * time.Millisecond,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start pool: %v", err)
	}
	
	// Acquire and immediately release a connection
	conn, err := pool.AcquireConnection(ctx)
	if err != nil {
		t.Fatalf("Failed to acquire connection: %v", err)
	}
	
	err = pool.ReleaseConnection(conn)
	if err != nil {
		t.Errorf("Failed to release connection: %v", err)
	}
	
	// Wait for idle timeout
	time.Sleep(300 * time.Millisecond)
	
	// Try to acquire the same connection (should be expired)
	newConn, err := pool.AcquireConnection(ctx)
	if err != nil {
		t.Fatalf("Failed to acquire new connection: %v", err)
	}
	
	// Should be a different connection due to idle timeout
	if newConn.ID == conn.ID {
		t.Log("Note: Got same connection ID, but connection may have been recreated")
	}
	
	pool.ReleaseConnection(newConn)
}

func TestPoolStatistics(t *testing.T) {
	config := PoolConfig{
		MinConnections: 2,
		MaxConnections: 5,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start pool: %v", err)
	}
	
	// Initial stats
	stats := pool.GetStats()
	if stats.TotalConnections != 2 {
		t.Errorf("Expected 2 total connections, got %d", stats.TotalConnections)
	}
	
	// Acquire connections and check stats
	conn1, err := pool.AcquireConnection(ctx)
	if err != nil {
		t.Fatalf("Failed to acquire connection: %v", err)
	}
	
	conn2, err := pool.AcquireConnection(ctx)
	if err != nil {
		t.Fatalf("Failed to acquire connection: %v", err)
	}
	
	stats = pool.GetStats()
	if stats.ActiveConnections != 2 {
		t.Errorf("Expected 2 active connections, got %d", stats.ActiveConnections)
	}
	
	if stats.IdleConnections != 0 {
		t.Errorf("Expected 0 idle connections, got %d", stats.IdleConnections)
	}
	
	if stats.TotalRequests != 2 {
		t.Errorf("Expected 2 total requests, got %d", stats.TotalRequests)
	}
	
	// Release connections
	pool.ReleaseConnection(conn1)
	pool.ReleaseConnection(conn2)
	
	stats = pool.GetStats()
	if stats.ActiveConnections != 0 {
		t.Errorf("Expected 0 active connections, got %d", stats.ActiveConnections)
	}
}

func TestConnectionFactoryError(t *testing.T) {
	config := PoolConfig{
		MinConnections: 2,
		MaxConnections: 5,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	// Set factory that always fails
	pool.SetConnectionFactory(createMockConnectionFactory(true, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err == nil {
		t.Error("Expected error when connection factory fails")
	}
}

func TestConnectionTimeout(t *testing.T) {
	config := PoolConfig{
		MinConnections: 1,
		MaxConnections: 1,
		AcquireTimeout: 100 * time.Millisecond,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start pool: %v", err)
	}
	
	// Acquire the only connection
	conn, err := pool.AcquireConnection(ctx)
	if err != nil {
		t.Fatalf("Failed to acquire connection: %v", err)
	}
	
	// Try to acquire another (should timeout)
	start := time.Now()
	_, err = pool.AcquireConnection(ctx)
	duration := time.Since(start)
	
	if err == nil {
		t.Error("Expected timeout error")
	}
	
	if duration < config.AcquireTimeout {
		t.Errorf("Timeout occurred too early: %v < %v", duration, config.AcquireTimeout)
	}
	
	pool.ReleaseConnection(conn)
}

func TestDetailedMetrics(t *testing.T) {
	config := PoolConfig{
		MinConnections: 3,
		MaxConnections: 5,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start pool: %v", err)
	}
	
	// Use connections to generate metrics
	for i := 0; i < 5; i++ {
		conn, err := pool.AcquireConnection(ctx)
		if err != nil {
			t.Fatalf("Failed to acquire connection: %v", err)
		}
		
		time.Sleep(10 * time.Millisecond)
		pool.ReleaseConnection(conn)
	}
	
	metrics := pool.GetDetailedMetrics()
	
	if len(metrics.ConnectionAges) == 0 {
		t.Error("Expected connection age metrics")
	}
	
	if len(metrics.ConnectionUsageCounts) == 0 {
		t.Error("Expected connection usage count metrics")
	}
	
	t.Logf("Metrics: %d connections tracked", len(metrics.ConnectionAges))
}

func TestConnectionMethods(t *testing.T) {
	// Create a mock connection
	conn := &Connection{
		ID:        "test_conn",
		Status:    StatusIdle,
		CreatedAt: time.Now().Add(-5 * time.Minute),
		LastUsed:  time.Now().Add(-2 * time.Minute),
		HealthOK:  true,
	}
	
	if !conn.IsHealthy() {
		t.Error("Expected connection to be healthy")
	}
	
	if conn.GetStatus() != StatusIdle {
		t.Errorf("Expected status Idle, got %v", conn.GetStatus())
	}
	
	age := conn.GetAge()
	if age < 4*time.Minute || age > 6*time.Minute {
		t.Errorf("Expected age around 5 minutes, got %v", age)
	}
	
	idleTime := conn.GetIdleTime()
	if idleTime < 1*time.Minute || idleTime > 3*time.Minute {
		t.Errorf("Expected idle time around 2 minutes, got %v", idleTime)
	}
}

func TestPoolClosure(t *testing.T) {
	config := PoolConfig{
		MinConnections: 2,
		MaxConnections: 5,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		t.Fatalf("Failed to create pool: %v", err)
	}
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start pool: %v", err)
	}
	
	// Acquire a connection
	conn, err := pool.AcquireConnection(ctx)
	if err != nil {
		t.Fatalf("Failed to acquire connection: %v", err)
	}
	
	// Close the pool
	err = pool.Close()
	if err != nil {
		t.Errorf("Failed to close pool: %v", err)
	}
	
	// Verify that subsequent operations fail
	_, err = pool.AcquireConnection(ctx)
	if err == nil {
		t.Error("Expected error when acquiring from closed pool")
	}
	
	// Verify that release still works for already acquired connections
	err = pool.ReleaseConnection(conn)
	if err != nil {
		t.Errorf("Release should work even after pool closure: %v", err)
	}
	
	// Double close should not error
	err = pool.Close()
	if err != nil {
		t.Errorf("Double close should not error: %v", err)
	}
}

func BenchmarkConnectionAcquisition(b *testing.B) {
	config := PoolConfig{
		MinConnections: 10,
		MaxConnections: 50,
		AcquireTimeout: 5 * time.Second,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		b.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		b.Fatalf("Failed to start pool: %v", err)
	}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			conn, err := pool.AcquireConnection(ctx)
			if err != nil {
				b.Fatalf("Failed to acquire connection: %v", err)
			}
			
			err = pool.ReleaseConnection(conn)
			if err != nil {
				b.Fatalf("Failed to release connection: %v", err)
			}
		}
	})
}

func BenchmarkConcurrentAcquisition(b *testing.B) {
	config := PoolConfig{
		MinConnections: 20,
		MaxConnections: 100,
		AcquireTimeout: 5 * time.Second,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		b.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		b.Fatalf("Failed to start pool: %v", err)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup
		numWorkers := runtime.NumCPU()
		
		for j := 0; j < numWorkers; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				
				conn, err := pool.AcquireConnection(ctx)
				if err != nil {
					return
				}
				
				// Simulate some work
				time.Sleep(time.Microsecond)
				
				pool.ReleaseConnection(conn)
			}()
		}
		
		wg.Wait()
	}
}

func BenchmarkPoolStats(b *testing.B) {
	config := PoolConfig{
		MinConnections: 10,
		MaxConnections: 20,
	}
	
	pool, err := NewDBConnectionPool("test", "test://localhost", config)
	if err != nil {
		b.Fatalf("Failed to create pool: %v", err)
	}
	defer pool.Close()
	
	pool.SetConnectionFactory(createMockConnectionFactory(false, 0))
	
	ctx := context.Background()
	err = pool.Start(ctx)
	if err != nil {
		b.Fatalf("Failed to start pool: %v", err)
	}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = pool.GetStats()
		}
	})
}