package concurrentdbpool

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// PoolConfig contains configuration for the connection pool
type PoolConfig struct {
	MinConnections     int           // Minimum number of connections to maintain
	MaxConnections     int           // Maximum number of connections allowed
	MaxIdleTime        time.Duration // Maximum time a connection can be idle
	MaxLifetime        time.Duration // Maximum time a connection can exist
	HealthCheckInterval time.Duration // Interval for health checks
	ConnectTimeout     time.Duration // Timeout for establishing connections
	AcquireTimeout     time.Duration // Timeout for acquiring connections
	RetryInterval      time.Duration // Interval between connection retries
	MaxRetries         int           // Maximum connection retry attempts
}

// DefaultConfig returns a default configuration
func DefaultConfig() PoolConfig {
	return PoolConfig{
		MinConnections:      5,
		MaxConnections:      50,
		MaxIdleTime:         5 * time.Minute,
		MaxLifetime:         30 * time.Minute,
		HealthCheckInterval: 30 * time.Second,
		ConnectTimeout:      10 * time.Second,
		AcquireTimeout:      30 * time.Second,
		RetryInterval:       5 * time.Second,
		MaxRetries:          3,
	}
}

// ConnectionStatus represents the status of a connection
type ConnectionStatus int

const (
	StatusIdle ConnectionStatus = iota
	StatusInUse
	StatusHealthCheck
	StatusClosed
)

// Connection wraps a database connection with metadata
type Connection struct {
	ID          string
	DB          *sql.DB
	RawConn     interface{} // Raw connection for database-specific operations
	Status      ConnectionStatus
	CreatedAt   time.Time
	LastUsed    time.Time
	UsageCount  int64
	HealthOK    bool
	mutex       sync.RWMutex
}

// PoolStats contains statistics about the connection pool
type PoolStats struct {
	TotalConnections    int32
	ActiveConnections   int32
	IdleConnections     int32
	WaitingRequests     int32
	TotalRequests       int64
	SuccessfulRequests  int64
	FailedRequests      int64
	AverageAcquireTime  time.Duration
	HealthyConnections  int32
	UnhealthyConnections int32
}

// DBConnectionPool represents a concurrent database connection pool
type DBConnectionPool struct {
	config          PoolConfig
	driverName      string
	dataSourceName  string
	connections     map[string]*Connection
	idleConnections chan *Connection
	mutex           sync.RWMutex
	stats           PoolStats
	statsMutex      sync.RWMutex
	closed          bool
	closeChan       chan struct{}
	healthChecker   *HealthChecker
	connectionFactory ConnectionFactory
	wg              sync.WaitGroup
}

// ConnectionFactory is a function type for creating database connections
type ConnectionFactory func(ctx context.Context, driverName, dataSourceName string) (*sql.DB, interface{}, error)

// HealthChecker manages connection health checking
type HealthChecker struct {
	pool     *DBConnectionPool
	interval time.Duration
	running  bool
	stopChan chan struct{}
	wg       sync.WaitGroup
}

// PoolObserver is an interface for observing pool events
type PoolObserver interface {
	OnConnectionCreated(connID string)
	OnConnectionClosed(connID string)
	OnConnectionAcquired(connID string, waitTime time.Duration)
	OnConnectionReleased(connID string, usageDuration time.Duration)
	OnHealthCheckStarted(connID string)
	OnHealthCheckCompleted(connID string, healthy bool)
	OnPoolStatsUpdated(stats PoolStats)
}

// DefaultConnectionFactory creates standard SQL connections
func DefaultConnectionFactory(ctx context.Context, driverName, dataSourceName string) (*sql.DB, interface{}, error) {
	db, err := sql.Open(driverName, dataSourceName)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open database: %w", err)
	}
	
	// Verify connection
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, nil, fmt.Errorf("failed to ping database: %w", err)
	}
	
	return db, db, nil
}

// NewDBConnectionPool creates a new database connection pool
func NewDBConnectionPool(driverName, dataSourceName string, config PoolConfig) (*DBConnectionPool, error) {
	if config.MinConnections < 0 {
		return nil, errors.New("min connections cannot be negative")
	}
	
	if config.MaxConnections <= 0 {
		return nil, errors.New("max connections must be positive")
	}
	
	if config.MinConnections > config.MaxConnections {
		return nil, errors.New("min connections cannot exceed max connections")
	}
	
	pool := &DBConnectionPool{
		config:            config,
		driverName:        driverName,
		dataSourceName:    dataSourceName,
		connections:       make(map[string]*Connection),
		idleConnections:   make(chan *Connection, config.MaxConnections),
		closeChan:         make(chan struct{}),
		connectionFactory: DefaultConnectionFactory,
	}
	
	pool.healthChecker = &HealthChecker{
		pool:     pool,
		interval: config.HealthCheckInterval,
		stopChan: make(chan struct{}),
	}
	
	return pool, nil
}

// Start initializes the connection pool
func (p *DBConnectionPool) Start(ctx context.Context) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	if p.closed {
		return errors.New("pool is closed")
	}
	
	// Create minimum connections
	for i := 0; i < p.config.MinConnections; i++ {
		conn, err := p.createConnection(ctx)
		if err != nil {
			// Clean up any created connections
			p.closeAllConnections()
			return fmt.Errorf("failed to create initial connections: %w", err)
		}
		
		p.connections[conn.ID] = conn
		select {
		case p.idleConnections <- conn:
		default:
			// Channel is full, shouldn't happen with initial connections
		}
		
		atomic.AddInt32(&p.stats.TotalConnections, 1)
		atomic.AddInt32(&p.stats.IdleConnections, 1)
	}
	
	// Start health checker
	p.healthChecker.Start(ctx)
	
	// Start connection cleanup routine
	p.wg.Add(1)
	go p.connectionCleanupRoutine(ctx)
	
	return nil
}

// AcquireConnection gets a connection from the pool
func (p *DBConnectionPool) AcquireConnection(ctx context.Context) (*Connection, error) {
	startTime := time.Now()
	
	if p.closed {
		return nil, errors.New("pool is closed")
	}
	
	atomic.AddInt64(&p.stats.TotalRequests, 1)
	atomic.AddInt32(&p.stats.WaitingRequests, 1)
	defer atomic.AddInt32(&p.stats.WaitingRequests, -1)
	
	// Try to get an idle connection first
	select {
	case conn := <-p.idleConnections:
		if p.isConnectionHealthy(conn) {
			p.markConnectionInUse(conn)
			atomic.AddInt64(&p.stats.SuccessfulRequests, 1)
			p.updateAverageAcquireTime(time.Since(startTime))
			return conn, nil
		} else {
			// Connection is unhealthy, close it and try to create a new one
			p.closeConnection(conn)
		}
	default:
		// No idle connections available
	}
	
	// Try to create a new connection if under max limit
	if p.canCreateNewConnection() {
		conn, err := p.createConnection(ctx)
		if err == nil {
			p.mutex.Lock()
			p.connections[conn.ID] = conn
			p.mutex.Unlock()
			
			p.markConnectionInUse(conn)
			atomic.AddInt32(&p.stats.TotalConnections, 1)
			atomic.AddInt64(&p.stats.SuccessfulRequests, 1)
			p.updateAverageAcquireTime(time.Since(startTime))
			return conn, nil
		}
	}
	
	// Wait for an available connection with timeout
	timeoutCtx, cancel := context.WithTimeout(ctx, p.config.AcquireTimeout)
	defer cancel()
	
	for {
		select {
		case conn := <-p.idleConnections:
			if p.isConnectionHealthy(conn) {
				p.markConnectionInUse(conn)
				atomic.AddInt64(&p.stats.SuccessfulRequests, 1)
				p.updateAverageAcquireTime(time.Since(startTime))
				return conn, nil
			} else {
				p.closeConnection(conn)
			}
			
		case <-timeoutCtx.Done():
			atomic.AddInt64(&p.stats.FailedRequests, 1)
			return nil, fmt.Errorf("timeout acquiring connection: %w", timeoutCtx.Err())
			
		case <-p.closeChan:
			atomic.AddInt64(&p.stats.FailedRequests, 1)
			return nil, errors.New("pool is closed")
		}
	}
}

// ReleaseConnection returns a connection to the pool
func (p *DBConnectionPool) ReleaseConnection(conn *Connection) error {
	if conn == nil {
		return errors.New("connection is nil")
	}
	
	usageStart := conn.LastUsed
	usageDuration := time.Since(usageStart)
	
	conn.mutex.Lock()
	if conn.Status == StatusClosed {
		conn.mutex.Unlock()
		return errors.New("connection is already closed")
	}
	
	conn.Status = StatusIdle
	conn.LastUsed = time.Now()
	conn.UsageCount++
	conn.mutex.Unlock()
	
	// Check if connection should be retired
	if p.shouldRetireConnection(conn) {
		p.closeConnection(conn)
		return nil
	}
	
	// Return to idle pool
	select {
	case p.idleConnections <- conn:
		atomic.AddInt32(&p.stats.ActiveConnections, -1)
		atomic.AddInt32(&p.stats.IdleConnections, 1)
	default:
		// Pool is full, close this connection
		p.closeConnection(conn)
	}
	
	// Notify observers
	if observer := p.getObserver(); observer != nil {
		observer.OnConnectionReleased(conn.ID, usageDuration)
	}
	
	return nil
}

// Close shuts down the connection pool
func (p *DBConnectionPool) Close() error {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	
	if p.closed {
		return nil
	}
	
	p.closed = true
	close(p.closeChan)
	
	// Stop health checker
	p.healthChecker.Stop()
	
	// Close all connections
	p.closeAllConnections()
	
	// Wait for cleanup routines to finish
	p.wg.Wait()
	
	return nil
}

// GetStats returns current pool statistics
func (p *DBConnectionPool) GetStats() PoolStats {
	p.statsMutex.RLock()
	defer p.statsMutex.RUnlock()
	return p.stats
}

// SetConnectionFactory sets a custom connection factory
func (p *DBConnectionPool) SetConnectionFactory(factory ConnectionFactory) {
	p.connectionFactory = factory
}

// Private methods

func (p *DBConnectionPool) createConnection(ctx context.Context) (*Connection, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, p.config.ConnectTimeout)
	defer cancel()
	
	db, rawConn, err := p.connectionFactory(timeoutCtx, p.driverName, p.dataSourceName)
	if err != nil {
		return nil, err
	}
	
	conn := &Connection{
		ID:        p.generateConnectionID(),
		DB:        db,
		RawConn:   rawConn,
		Status:    StatusIdle,
		CreatedAt: time.Now(),
		LastUsed:  time.Now(),
		HealthOK:  true,
	}
	
	// Notify observers
	if observer := p.getObserver(); observer != nil {
		observer.OnConnectionCreated(conn.ID)
	}
	
	return conn, nil
}

func (p *DBConnectionPool) closeConnection(conn *Connection) {
	if conn == nil {
		return
	}
	
	conn.mutex.Lock()
	if conn.Status == StatusClosed {
		conn.mutex.Unlock()
		return
	}
	
	conn.Status = StatusClosed
	conn.mutex.Unlock()
	
	// Close the database connection
	if conn.DB != nil {
		conn.DB.Close()
	}
	
	// Remove from connections map
	p.mutex.Lock()
	delete(p.connections, conn.ID)
	p.mutex.Unlock()
	
	// Update stats
	atomic.AddInt32(&p.stats.TotalConnections, -1)
	
	// Determine which counter to decrement
	switch conn.Status {
	case StatusInUse:
		atomic.AddInt32(&p.stats.ActiveConnections, -1)
	case StatusIdle:
		atomic.AddInt32(&p.stats.IdleConnections, -1)
	}
	
	// Notify observers
	if observer := p.getObserver(); observer != nil {
		observer.OnConnectionClosed(conn.ID)
	}
}

func (p *DBConnectionPool) closeAllConnections() {
	// Close all idle connections
	for {
		select {
		case conn := <-p.idleConnections:
			p.closeConnection(conn)
		default:
			goto closeRemaining
		}
	}
	
closeRemaining:
	// Close remaining connections
	for _, conn := range p.connections {
		p.closeConnection(conn)
	}
}

func (p *DBConnectionPool) markConnectionInUse(conn *Connection) {
	conn.mutex.Lock()
	conn.Status = StatusInUse
	conn.LastUsed = time.Now()
	conn.mutex.Unlock()
	
	atomic.AddInt32(&p.stats.ActiveConnections, 1)
	atomic.AddInt32(&p.stats.IdleConnections, -1)
	
	// Notify observers
	if observer := p.getObserver(); observer != nil {
		observer.OnConnectionAcquired(conn.ID, 0) // TODO: track actual wait time
	}
}

func (p *DBConnectionPool) isConnectionHealthy(conn *Connection) bool {
	conn.mutex.RLock()
	defer conn.mutex.RUnlock()
	
	// Check if connection is expired
	if p.config.MaxLifetime > 0 && time.Since(conn.CreatedAt) > p.config.MaxLifetime {
		return false
	}
	
	// Check if connection has been idle too long
	if p.config.MaxIdleTime > 0 && time.Since(conn.LastUsed) > p.config.MaxIdleTime {
		return false
	}
	
	return conn.HealthOK && conn.Status != StatusClosed
}

func (p *DBConnectionPool) shouldRetireConnection(conn *Connection) bool {
	conn.mutex.RLock()
	defer conn.mutex.RUnlock()
	
	// Retire if exceeded lifetime
	if p.config.MaxLifetime > 0 && time.Since(conn.CreatedAt) > p.config.MaxLifetime {
		return true
	}
	
	// Retire if unhealthy
	if !conn.HealthOK {
		return true
	}
	
	return false
}

func (p *DBConnectionPool) canCreateNewConnection() bool {
	return atomic.LoadInt32(&p.stats.TotalConnections) < int32(p.config.MaxConnections)
}

func (p *DBConnectionPool) generateConnectionID() string {
	return fmt.Sprintf("conn_%d_%d", time.Now().UnixNano(), atomic.AddInt32(&p.stats.TotalConnections, 0))
}

func (p *DBConnectionPool) updateAverageAcquireTime(duration time.Duration) {
	// Simple moving average implementation
	p.statsMutex.Lock()
	defer p.statsMutex.Unlock()
	
	if p.stats.AverageAcquireTime == 0 {
		p.stats.AverageAcquireTime = duration
	} else {
		p.stats.AverageAcquireTime = (p.stats.AverageAcquireTime + duration) / 2
	}
}

func (p *DBConnectionPool) connectionCleanupRoutine(ctx context.Context) {
	defer p.wg.Done()
	
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			p.cleanupExpiredConnections()
			
		case <-ctx.Done():
			return
			
		case <-p.closeChan:
			return
		}
	}
}

func (p *DBConnectionPool) cleanupExpiredConnections() {
	p.mutex.RLock()
	var expiredConnections []*Connection
	
	for _, conn := range p.connections {
		if !p.isConnectionHealthy(conn) && conn.Status == StatusIdle {
			expiredConnections = append(expiredConnections, conn)
		}
	}
	p.mutex.RUnlock()
	
	for _, conn := range expiredConnections {
		p.closeConnection(conn)
	}
}

func (p *DBConnectionPool) getObserver() PoolObserver {
	// Implementation would depend on how observers are managed
	return nil
}

// Health Checker implementation

// Start begins the health checking routine
func (hc *HealthChecker) Start(ctx context.Context) {
	if hc.running {
		return
	}
	
	hc.running = true
	hc.wg.Add(1)
	
	go func() {
		defer hc.wg.Done()
		ticker := time.NewTicker(hc.interval)
		defer ticker.Stop()
		
		for {
			select {
			case <-ticker.C:
				hc.checkConnections(ctx)
				
			case <-ctx.Done():
				return
				
			case <-hc.stopChan:
				return
			}
		}
	}()
}

// Stop halts the health checking routine
func (hc *HealthChecker) Stop() {
	if !hc.running {
		return
	}
	
	hc.running = false
	close(hc.stopChan)
	hc.wg.Wait()
}

func (hc *HealthChecker) checkConnections(ctx context.Context) {
	hc.pool.mutex.RLock()
	connections := make([]*Connection, 0, len(hc.pool.connections))
	for _, conn := range hc.pool.connections {
		if conn.Status == StatusIdle {
			connections = append(connections, conn)
		}
	}
	hc.pool.mutex.RUnlock()
	
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 5) // Limit concurrent health checks
	
	for _, conn := range connections {
		wg.Add(1)
		go func(c *Connection) {
			defer wg.Done()
			
			semaphore <- struct{}{}
			defer func() { <-semaphore }()
			
			hc.checkSingleConnection(ctx, c)
		}(conn)
	}
	
	wg.Wait()
}

func (hc *HealthChecker) checkSingleConnection(ctx context.Context, conn *Connection) {
	conn.mutex.Lock()
	if conn.Status != StatusIdle {
		conn.mutex.Unlock()
		return
	}
	
	conn.Status = StatusHealthCheck
	conn.mutex.Unlock()
	
	// Notify observers
	if observer := hc.pool.getObserver(); observer != nil {
		observer.OnHealthCheckStarted(conn.ID)
	}
	
	// Perform health check
	healthy := true
	if conn.DB != nil {
		timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		err := conn.DB.PingContext(timeoutCtx)
		cancel()
		
		if err != nil {
			healthy = false
		}
	}
	
	// Update connection health status
	conn.mutex.Lock()
	conn.HealthOK = healthy
	if conn.Status == StatusHealthCheck {
		conn.Status = StatusIdle
	}
	conn.mutex.Unlock()
	
	// Update stats
	if healthy {
		atomic.AddInt32(&hc.pool.stats.HealthyConnections, 1)
	} else {
		atomic.AddInt32(&hc.pool.stats.UnhealthyConnections, 1)
	}
	
	// Notify observers
	if observer := hc.pool.getObserver(); observer != nil {
		observer.OnHealthCheckCompleted(conn.ID, healthy)
	}
}

// Connection methods

// IsHealthy returns whether the connection is healthy
func (c *Connection) IsHealthy() bool {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.HealthOK && c.Status != StatusClosed
}

// GetStatus returns the current status of the connection
func (c *Connection) GetStatus() ConnectionStatus {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.Status
}

// GetUsageCount returns the number of times this connection has been used
func (c *Connection) GetUsageCount() int64 {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.UsageCount
}

// GetAge returns how long the connection has existed
func (c *Connection) GetAge() time.Duration {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return time.Since(c.CreatedAt)
}

// GetIdleTime returns how long the connection has been idle
func (c *Connection) GetIdleTime() time.Duration {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	if c.Status == StatusIdle {
		return time.Since(c.LastUsed)
	}
	return 0
}

// Utility functions

// CreateTestPool creates a connection pool for testing
func CreateTestPool(config PoolConfig) *DBConnectionPool {
	pool := &DBConnectionPool{
		config:          config,
		driverName:      "test",
		dataSourceName:  "test",
		connections:     make(map[string]*Connection),
		idleConnections: make(chan *Connection, config.MaxConnections),
		closeChan:       make(chan struct{}),
	}
	
	// Set a mock connection factory for testing
	pool.connectionFactory = func(ctx context.Context, driverName, dataSourceName string) (*sql.DB, interface{}, error) {
		// Return a mock connection for testing
		return nil, "mock", nil
	}
	
	return pool
}

// PoolMetrics provides additional metrics
type PoolMetrics struct {
	ConnectionAges        []time.Duration
	ConnectionUsageCounts []int64
	IdleTimes            []time.Duration
	HealthCheckLatencies []time.Duration
}

// GetDetailedMetrics returns detailed metrics about the pool
func (p *DBConnectionPool) GetDetailedMetrics() PoolMetrics {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	
	metrics := PoolMetrics{
		ConnectionAges:        make([]time.Duration, 0, len(p.connections)),
		ConnectionUsageCounts: make([]int64, 0, len(p.connections)),
		IdleTimes:            make([]time.Duration, 0, len(p.connections)),
	}
	
	for _, conn := range p.connections {
		conn.mutex.RLock()
		metrics.ConnectionAges = append(metrics.ConnectionAges, time.Since(conn.CreatedAt))
		metrics.ConnectionUsageCounts = append(metrics.ConnectionUsageCounts, conn.UsageCount)
		if conn.Status == StatusIdle {
			metrics.IdleTimes = append(metrics.IdleTimes, time.Since(conn.LastUsed))
		}
		conn.mutex.RUnlock()
	}
	
	return metrics
}