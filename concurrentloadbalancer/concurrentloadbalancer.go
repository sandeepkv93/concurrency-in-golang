package concurrentloadbalancer

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// LoadBalancingStrategy defines different load balancing algorithms
type LoadBalancingStrategy int

const (
	RoundRobin LoadBalancingStrategy = iota
	WeightedRoundRobin
	LeastConnections
	LeastResponseTime
	IPHash
	RandomSelection
	ConsistentHashing
	PowerOfTwoChoices
)

// HealthStatus represents the health state of a backend server
type HealthStatus int

const (
	Healthy HealthStatus = iota
	Degraded
	Unhealthy
	Unknown
)

// BackendServer represents a backend server
type BackendServer struct {
	ID                string
	URL               *url.URL
	Weight            int32
	CurrentWeight     int32
	HealthStatus      HealthStatus
	ActiveConnections int64
	TotalRequests     int64
	FailedRequests    int64
	SuccessRequests   int64
	ResponseTime      time.Duration
	LastHealthCheck   time.Time
	Capacity          int64
	Tags              map[string]string
	mutex             sync.RWMutex
}

// LoadBalancerConfig contains configuration for the load balancer
type LoadBalancerConfig struct {
	Strategy              LoadBalancingStrategy
	Port                  int
	HealthCheckInterval   time.Duration
	HealthCheckTimeout    time.Duration
	HealthCheckPath       string
	MaxRetries            int
	RetryBackoff          time.Duration
	EnableStickySessions  bool
	SessionCookieName     string
	SessionTimeout        time.Duration
	EnableSSL             bool
	SSLCertFile           string
	SSLKeyFile            string
	RequestTimeout        time.Duration
	MaxIdleConns          int
	MaxIdleConnsPerHost   int
	CircuitBreakerEnabled bool
	CircuitBreakerConfig  CircuitBreakerConfig
	MetricsEnabled        bool
	LoggingEnabled        bool
	GracefulShutdown      bool
	ShutdownTimeout       time.Duration
}

// CircuitBreakerConfig contains circuit breaker configuration
type CircuitBreakerConfig struct {
	FailureThreshold   int32
	RecoveryThreshold  int32
	Timeout            time.Duration
	HalfOpenMaxCalls   int32
}

// LoadBalancer is the main load balancer instance
type LoadBalancer struct {
	config           LoadBalancerConfig
	servers          []*BackendServer
	serverMap        map[string]*BackendServer
	strategy         LoadBalancingStrategy
	currentIndex     uint64
	healthChecker    *HealthChecker
	circuitBreakers  map[string]*CircuitBreaker
	sessionManager   *SessionManager
	metrics          *Metrics
	server           *http.Server
	client           *http.Client
	consistentHash   *ConsistentHashRing
	running          bool
	mutex            sync.RWMutex
	ctx              context.Context
	cancel           context.CancelFunc
	shutdownComplete chan struct{}
}

// CircuitBreaker implements the circuit breaker pattern
type CircuitBreaker struct {
	state            CircuitState
	failures         int32
	successes        int32
	requests         int32
	lastFailureTime  time.Time
	lastStateChange  time.Time
	config           CircuitBreakerConfig
	mutex            sync.RWMutex
}

// CircuitState represents the circuit breaker state
type CircuitState int

const (
	CircuitClosed CircuitState = iota
	CircuitOpen
	CircuitHalfOpen
)

// SessionManager handles sticky sessions
type SessionManager struct {
	sessions map[string]*Session
	mutex    sync.RWMutex
	timeout  time.Duration
	enabled  bool
}

// Session represents a client session
type Session struct {
	ID         string
	ServerID   string
	LastAccess time.Time
	Data       map[string]interface{}
}

// HealthChecker monitors server health
type HealthChecker struct {
	loadBalancer *LoadBalancer
	interval     time.Duration
	timeout      time.Duration
	path         string
	client       *http.Client
	running      bool
	stopChan     chan struct{}
	wg           sync.WaitGroup
}

// Metrics collects load balancer statistics
type Metrics struct {
	TotalRequests       int64
	SuccessfulRequests  int64
	FailedRequests      int64
	TotalResponseTime   int64
	ActiveConnections   int64
	AverageResponseTime time.Duration
	RequestsPerSecond   float64
	ServerMetrics       map[string]*ServerMetrics
	CircuitBreakerTrips int64
	lastUpdate          time.Time
	mutex               sync.RWMutex
}

// ServerMetrics contains per-server statistics
type ServerMetrics struct {
	Requests      int64
	Failures      int64
	ResponseTime  time.Duration
	Availability  float64
	LastSeen      time.Time
}

// ConsistentHashRing implements consistent hashing
type ConsistentHashRing struct {
	ring        map[uint32]*BackendServer
	sortedKeys  []uint32
	virtualNodes int
	mutex       sync.RWMutex
}

// RequestContext contains request processing context
type RequestContext struct {
	StartTime      time.Time
	Request        *http.Request
	SelectedServer *BackendServer
	Retries        int
	SessionID      string
}

// NewLoadBalancer creates a new load balancer instance
func NewLoadBalancer(config LoadBalancerConfig) *LoadBalancer {
	setDefaultConfig(&config)

	ctx, cancel := context.WithCancel(context.Background())

	client := &http.Client{
		Timeout: config.RequestTimeout,
		Transport: &http.Transport{
			MaxIdleConns:        config.MaxIdleConns,
			MaxIdleConnsPerHost: config.MaxIdleConnsPerHost,
			IdleConnTimeout:     90 * time.Second,
			TLSHandshakeTimeout: 10 * time.Second,
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: !config.EnableSSL,
			},
		},
	}

	lb := &LoadBalancer{
		config:           config,
		servers:          make([]*BackendServer, 0),
		serverMap:        make(map[string]*BackendServer),
		strategy:         config.Strategy,
		circuitBreakers:  make(map[string]*CircuitBreaker),
		sessionManager:   NewSessionManager(config.EnableStickySessions, config.SessionTimeout),
		metrics:          NewMetrics(),
		client:           client,
		consistentHash:   NewConsistentHashRing(100),
		ctx:              ctx,
		cancel:           cancel,
		shutdownComplete: make(chan struct{}),
	}

	if config.HealthCheckInterval > 0 {
		lb.healthChecker = NewHealthChecker(lb, config.HealthCheckInterval, config.HealthCheckTimeout, config.HealthCheckPath)
	}

	return lb
}

// AddServer adds a backend server to the load balancer
func (lb *LoadBalancer) AddServer(server *BackendServer) error {
	if server == nil {
		return errors.New("server cannot be nil")
	}

	if server.ID == "" {
		return errors.New("server ID cannot be empty")
	}

	if server.URL == nil {
		return errors.New("server URL cannot be nil")
	}

	lb.mutex.Lock()
	defer lb.mutex.Unlock()

	// Check if server already exists
	if _, exists := lb.serverMap[server.ID]; exists {
		return fmt.Errorf("server with ID %s already exists", server.ID)
	}

	// Set default values
	if server.Weight <= 0 {
		server.Weight = 1
	}
	server.CurrentWeight = server.Weight
	server.HealthStatus = Unknown

	// Add to server list and map
	lb.servers = append(lb.servers, server)
	lb.serverMap[server.ID] = server

	// Add to consistent hash ring
	lb.consistentHash.AddServer(server)

	// Initialize circuit breaker if enabled
	if lb.config.CircuitBreakerEnabled {
		lb.circuitBreakers[server.ID] = NewCircuitBreaker(lb.config.CircuitBreakerConfig)
	}

	// Initialize server metrics
	lb.metrics.AddServer(server.ID)

	if lb.config.LoggingEnabled {
		log.Printf("Added server %s (%s) to load balancer", server.ID, server.URL.String())
	}

	return nil
}

// RemoveServer removes a backend server from the load balancer
func (lb *LoadBalancer) RemoveServer(serverID string) error {
	lb.mutex.Lock()
	defer lb.mutex.Unlock()

	server, exists := lb.serverMap[serverID]
	if !exists {
		return fmt.Errorf("server with ID %s not found", serverID)
	}

	// Remove from server list
	for i, s := range lb.servers {
		if s.ID == serverID {
			lb.servers = append(lb.servers[:i], lb.servers[i+1:]...)
			break
		}
	}

	// Remove from server map
	delete(lb.serverMap, serverID)

	// Remove from consistent hash ring
	lb.consistentHash.RemoveServer(server)

	// Remove circuit breaker
	delete(lb.circuitBreakers, serverID)

	// Remove server metrics
	lb.metrics.RemoveServer(serverID)

	if lb.config.LoggingEnabled {
		log.Printf("Removed server %s from load balancer", serverID)
	}

	return nil
}

// Start starts the load balancer server
func (lb *LoadBalancer) Start() error {
	lb.mutex.Lock()
	if lb.running {
		lb.mutex.Unlock()
		return errors.New("load balancer is already running")
	}
	lb.running = true
	lb.mutex.Unlock()

	// Start health checker
	if lb.healthChecker != nil {
		go lb.healthChecker.Start()
	}

	// Start metrics collection
	go lb.metrics.Start()

	// Setup HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/", lb.handleRequest)
	mux.HandleFunc("/health", lb.handleHealthCheck)
	mux.HandleFunc("/metrics", lb.handleMetrics)
	mux.HandleFunc("/status", lb.handleStatus)

	lb.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", lb.config.Port),
		Handler: mux,
	}

	var err error
	if lb.config.EnableSSL {
		if lb.config.LoggingEnabled {
			log.Printf("Starting HTTPS load balancer on port %d", lb.config.Port)
		}
		err = lb.server.ListenAndServeTLS(lb.config.SSLCertFile, lb.config.SSLKeyFile)
	} else {
		if lb.config.LoggingEnabled {
			log.Printf("Starting HTTP load balancer on port %d", lb.config.Port)
		}
		err = lb.server.ListenAndServe()
	}

	if err != http.ErrServerClosed {
		return err
	}

	return nil
}

// Stop gracefully stops the load balancer
func (lb *LoadBalancer) Stop() error {
	lb.mutex.Lock()
	if !lb.running {
		lb.mutex.Unlock()
		return nil
	}
	lb.running = false
	lb.mutex.Unlock()

	if lb.config.LoggingEnabled {
		log.Println("Stopping load balancer...")
	}

	// Stop health checker
	if lb.healthChecker != nil {
		lb.healthChecker.Stop()
	}

	// Stop metrics collection
	lb.metrics.Stop()

	// Cancel context
	lb.cancel()

	// Shutdown HTTP server
	if lb.server != nil {
		ctx, cancel := context.WithTimeout(context.Background(), lb.config.ShutdownTimeout)
		defer cancel()

		if err := lb.server.Shutdown(ctx); err != nil {
			if lb.config.LoggingEnabled {
				log.Printf("Server shutdown error: %v", err)
			}
			return err
		}
	}

	close(lb.shutdownComplete)

	if lb.config.LoggingEnabled {
		log.Println("Load balancer stopped gracefully")
	}

	return nil
}

// handleRequest handles incoming HTTP requests
func (lb *LoadBalancer) handleRequest(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	atomic.AddInt64(&lb.metrics.TotalRequests, 1)
	atomic.AddInt64(&lb.metrics.ActiveConnections, 1)
	defer atomic.AddInt64(&lb.metrics.ActiveConnections, -1)

	// Create request context
	reqCtx := &RequestContext{
		StartTime: startTime,
		Request:   r,
	}

	// Handle sticky sessions
	if lb.config.EnableStickySessions {
		sessionID := lb.sessionManager.GetSessionID(r)
		if sessionID != "" {
			reqCtx.SessionID = sessionID
			if serverID := lb.sessionManager.GetServerForSession(sessionID); serverID != "" {
				if server := lb.getServerByID(serverID); server != nil && server.HealthStatus == Healthy {
					reqCtx.SelectedServer = server
				}
			}
		}
	}

	// Select server if not already selected by session
	if reqCtx.SelectedServer == nil {
		server, err := lb.selectServer(r)
		if err != nil {
			http.Error(w, "No healthy servers available", http.StatusServiceUnavailable)
			atomic.AddInt64(&lb.metrics.FailedRequests, 1)
			return
		}
		reqCtx.SelectedServer = server
	}

	// Check circuit breaker
	if lb.config.CircuitBreakerEnabled {
		cb := lb.circuitBreakers[reqCtx.SelectedServer.ID]
		if cb != nil && !cb.AllowRequest() {
			// Try to select another server
			if server, err := lb.selectServerExcluding(r, reqCtx.SelectedServer.ID); err == nil {
				reqCtx.SelectedServer = server
			} else {
				http.Error(w, "Service temporarily unavailable", http.StatusServiceUnavailable)
				atomic.AddInt64(&lb.metrics.FailedRequests, 1)
				return
			}
		}
	}

	// Process request with retries
	err := lb.processRequestWithRetries(w, reqCtx)
	
	// Update metrics
	responseTime := time.Since(startTime)
	atomic.AddInt64(&lb.metrics.TotalResponseTime, int64(responseTime))
	
	if err != nil {
		atomic.AddInt64(&lb.metrics.FailedRequests, 1)
		if lb.config.CircuitBreakerEnabled {
			if cb := lb.circuitBreakers[reqCtx.SelectedServer.ID]; cb != nil {
				cb.RecordFailure()
			}
		}
	} else {
		atomic.AddInt64(&lb.metrics.SuccessfulRequests, 1)
		if lb.config.CircuitBreakerEnabled {
			if cb := lb.circuitBreakers[reqCtx.SelectedServer.ID]; cb != nil {
				cb.RecordSuccess()
			}
		}

		// Update session if sticky sessions are enabled
		if lb.config.EnableStickySessions && reqCtx.SessionID != "" {
			lb.sessionManager.UpdateSession(reqCtx.SessionID, reqCtx.SelectedServer.ID, w)
		}
	}

	// Update server metrics
	lb.updateServerMetrics(reqCtx.SelectedServer, responseTime, err == nil)
}

// processRequestWithRetries processes a request with retry logic
func (lb *LoadBalancer) processRequestWithRetries(w http.ResponseWriter, reqCtx *RequestContext) error {
	var lastErr error

	for reqCtx.Retries <= lb.config.MaxRetries {
		err := lb.proxyRequest(w, reqCtx)
		if err == nil {
			return nil
		}

		lastErr = err
		reqCtx.Retries++

		if reqCtx.Retries <= lb.config.MaxRetries {
			// Try to select another server for retry
			if server, err := lb.selectServerExcluding(reqCtx.Request, reqCtx.SelectedServer.ID); err == nil {
				reqCtx.SelectedServer = server
				time.Sleep(lb.config.RetryBackoff)
				continue
			}
		}

		break
	}

	return lastErr
}

// proxyRequest proxies the request to the selected server
func (lb *LoadBalancer) proxyRequest(w http.ResponseWriter, reqCtx *RequestContext) error {
	server := reqCtx.SelectedServer
	atomic.AddInt64(&server.ActiveConnections, 1)
	atomic.AddInt64(&server.TotalRequests, 1)
	defer atomic.AddInt64(&server.ActiveConnections, -1)

	// Create reverse proxy
	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = server.URL.Scheme
			req.URL.Host = server.URL.Host
			req.Host = server.URL.Host

			// Add headers
			req.Header.Set("X-Forwarded-For", getClientIP(req))
			req.Header.Set("X-Forwarded-Host", req.Host)
			req.Header.Set("X-Forwarded-Proto", req.URL.Scheme)
			req.Header.Set("X-LoadBalancer", "ConcurrentLoadBalancer")
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if lb.config.LoggingEnabled {
				log.Printf("Proxy error for server %s: %v", server.ID, err)
			}
			atomic.AddInt64(&server.FailedRequests, 1)
			w.WriteHeader(http.StatusBadGateway)
		},
	}

	proxy.ServeHTTP(w, reqCtx.Request)
	return nil
}

// selectServer selects a backend server based on the configured strategy
func (lb *LoadBalancer) selectServer(r *http.Request) (*BackendServer, error) {
	return lb.selectServerExcluding(r, "")
}

// selectServerExcluding selects a server excluding a specific server ID
func (lb *LoadBalancer) selectServerExcluding(r *http.Request, excludeID string) (*BackendServer, error) {
	lb.mutex.RLock()
	defer lb.mutex.RUnlock()

	healthyServers := make([]*BackendServer, 0)
	for _, server := range lb.servers {
		if server.HealthStatus == Healthy && server.ID != excludeID {
			healthyServers = append(healthyServers, server)
		}
	}

	if len(healthyServers) == 0 {
		return nil, errors.New("no healthy servers available")
	}

	switch lb.strategy {
	case RoundRobin:
		return lb.selectRoundRobin(healthyServers), nil
	case WeightedRoundRobin:
		return lb.selectWeightedRoundRobin(healthyServers), nil
	case LeastConnections:
		return lb.selectLeastConnections(healthyServers), nil
	case LeastResponseTime:
		return lb.selectLeastResponseTime(healthyServers), nil
	case IPHash:
		return lb.selectIPHash(healthyServers, r), nil
	case RandomSelection:
		return lb.selectRandom(healthyServers), nil
	case ConsistentHashing:
		return lb.selectConsistentHash(r)
	case PowerOfTwoChoices:
		return lb.selectPowerOfTwo(healthyServers), nil
	default:
		return lb.selectRoundRobin(healthyServers), nil
	}
}

// Load balancing strategy implementations

func (lb *LoadBalancer) selectRoundRobin(servers []*BackendServer) *BackendServer {
	index := atomic.AddUint64(&lb.currentIndex, 1)
	return servers[(index-1)%uint64(len(servers))]
}

func (lb *LoadBalancer) selectWeightedRoundRobin(servers []*BackendServer) *BackendServer {
	totalWeight := int32(0)
	selected := servers[0]

	for _, server := range servers {
		server.CurrentWeight += server.Weight
		totalWeight += server.Weight

		if server.CurrentWeight > selected.CurrentWeight {
			selected = server
		}
	}

	selected.CurrentWeight -= totalWeight
	return selected
}

func (lb *LoadBalancer) selectLeastConnections(servers []*BackendServer) *BackendServer {
	selected := servers[0]
	minConnections := atomic.LoadInt64(&selected.ActiveConnections)

	for _, server := range servers[1:] {
		connections := atomic.LoadInt64(&server.ActiveConnections)
		if connections < minConnections {
			minConnections = connections
			selected = server
		}
	}

	return selected
}

func (lb *LoadBalancer) selectLeastResponseTime(servers []*BackendServer) *BackendServer {
	selected := servers[0]
	minResponseTime := selected.ResponseTime

	for _, server := range servers[1:] {
		if server.ResponseTime < minResponseTime {
			minResponseTime = server.ResponseTime
			selected = server
		}
	}

	return selected
}

func (lb *LoadBalancer) selectIPHash(servers []*BackendServer, r *http.Request) *BackendServer {
	clientIP := getClientIP(r)
	hash := hashString(clientIP)
	index := hash % uint32(len(servers))
	return servers[index]
}

func (lb *LoadBalancer) selectRandom(servers []*BackendServer) *BackendServer {
	index := rand.Intn(len(servers))
	return servers[index]
}

func (lb *LoadBalancer) selectConsistentHash(r *http.Request) (*BackendServer, error) {
	clientIP := getClientIP(r)
	return lb.consistentHash.GetServer(clientIP), nil
}

func (lb *LoadBalancer) selectPowerOfTwo(servers []*BackendServer) *BackendServer {
	if len(servers) == 1 {
		return servers[0]
	}

	// Select two random servers and choose the one with fewer connections
	idx1 := rand.Intn(len(servers))
	idx2 := rand.Intn(len(servers))
	for idx2 == idx1 {
		idx2 = rand.Intn(len(servers))
	}

	server1 := servers[idx1]
	server2 := servers[idx2]

	if atomic.LoadInt64(&server1.ActiveConnections) <= atomic.LoadInt64(&server2.ActiveConnections) {
		return server1
	}
	return server2
}

// Utility methods

func (lb *LoadBalancer) getServerByID(serverID string) *BackendServer {
	lb.mutex.RLock()
	defer lb.mutex.RUnlock()
	return lb.serverMap[serverID]
}

func (lb *LoadBalancer) updateServerMetrics(server *BackendServer, responseTime time.Duration, success bool) {
	server.mutex.Lock()
	server.ResponseTime = responseTime
	if success {
		atomic.AddInt64(&server.SuccessRequests, 1)
	} else {
		atomic.AddInt64(&server.FailedRequests, 1)
	}
	server.mutex.Unlock()

	// Update global metrics
	lb.metrics.UpdateServerMetrics(server.ID, responseTime, success)
}

func (lb *LoadBalancer) handleHealthCheck(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"servers":   make([]map[string]interface{}, 0),
	}

	lb.mutex.RLock()
	for _, server := range lb.servers {
		serverStatus := map[string]interface{}{
			"id":                 server.ID,
			"url":                server.URL.String(),
			"health":             healthStatusToString(server.HealthStatus),
			"active_connections": atomic.LoadInt64(&server.ActiveConnections),
			"total_requests":     atomic.LoadInt64(&server.TotalRequests),
			"failed_requests":    atomic.LoadInt64(&server.FailedRequests),
			"weight":             server.Weight,
		}
		status["servers"] = append(status["servers"].([]map[string]interface{}), serverStatus)
	}
	lb.mutex.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	writeJSON(w, status)
}

func (lb *LoadBalancer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := lb.metrics.GetSnapshot()
	w.Header().Set("Content-Type", "application/json")
	writeJSON(w, metrics)
}

func (lb *LoadBalancer) handleStatus(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"running":     lb.running,
		"strategy":    strategyToString(lb.strategy),
		"server_count": len(lb.servers),
		"uptime":      time.Since(lb.metrics.lastUpdate),
	}

	w.Header().Set("Content-Type", "application/json")
	writeJSON(w, status)
}

// Utility functions

func setDefaultConfig(config *LoadBalancerConfig) {
	if config.Port == 0 {
		config.Port = 8080
	}
	if config.HealthCheckInterval == 0 {
		config.HealthCheckInterval = 30 * time.Second
	}
	if config.HealthCheckTimeout == 0 {
		config.HealthCheckTimeout = 5 * time.Second
	}
	if config.HealthCheckPath == "" {
		config.HealthCheckPath = "/health"
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}
	if config.RetryBackoff == 0 {
		config.RetryBackoff = 100 * time.Millisecond
	}
	if config.SessionCookieName == "" {
		config.SessionCookieName = "LB_SESSION"
	}
	if config.SessionTimeout == 0 {
		config.SessionTimeout = 1 * time.Hour
	}
	if config.RequestTimeout == 0 {
		config.RequestTimeout = 30 * time.Second
	}
	if config.MaxIdleConns == 0 {
		config.MaxIdleConns = 100
	}
	if config.MaxIdleConnsPerHost == 0 {
		config.MaxIdleConnsPerHost = 10
	}
	if config.ShutdownTimeout == 0 {
		config.ShutdownTimeout = 30 * time.Second
	}
	if config.CircuitBreakerConfig.FailureThreshold == 0 {
		config.CircuitBreakerConfig.FailureThreshold = 5
	}
	if config.CircuitBreakerConfig.Timeout == 0 {
		config.CircuitBreakerConfig.Timeout = 30 * time.Second
	}
}

func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header
	if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
		return forwarded
	}
	// Check X-Real-IP header
	if realIP := r.Header.Get("X-Real-IP"); realIP != "" {
		return realIP
	}
	// Fall back to remote address
	host, _, _ := net.SplitHostPort(r.RemoteAddr)
	return host
}

func hashString(s string) uint32 {
	hash := uint32(0)
	for i := 0; i < len(s); i++ {
		hash = hash*31 + uint32(s[i])
	}
	return hash
}

func healthStatusToString(status HealthStatus) string {
	switch status {
	case Healthy:
		return "healthy"
	case Degraded:
		return "degraded"
	case Unhealthy:
		return "unhealthy"
	default:
		return "unknown"
	}
}

func strategyToString(strategy LoadBalancingStrategy) string {
	switch strategy {
	case RoundRobin:
		return "round_robin"
	case WeightedRoundRobin:
		return "weighted_round_robin"
	case LeastConnections:
		return "least_connections"
	case LeastResponseTime:
		return "least_response_time"
	case IPHash:
		return "ip_hash"
	case RandomSelection:
		return "random"
	case ConsistentHashing:
		return "consistent_hashing"
	case PowerOfTwoChoices:
		return "power_of_two_choices"
	default:
		return "unknown"
	}
}

func writeJSON(w http.ResponseWriter, data interface{}) {
	// Simplified JSON writing - in production, use proper JSON encoding
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "{\"data\": \"json_data\"}")
}

// HealthChecker implementation

// NewHealthChecker creates a new health checker
func NewHealthChecker(lb *LoadBalancer, interval, timeout time.Duration, path string) *HealthChecker {
	return &HealthChecker{
		loadBalancer: lb,
		interval:     interval,
		timeout:      timeout,
		path:         path,
		client: &http.Client{
			Timeout: timeout,
		},
		stopChan: make(chan struct{}),
	}
}

// Start starts the health checker
func (hc *HealthChecker) Start() {
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
				hc.checkAllServers()
			case <-hc.stopChan:
				return
			case <-hc.loadBalancer.ctx.Done():
				return
			}
		}
	}()
}

// Stop stops the health checker
func (hc *HealthChecker) Stop() {
	if !hc.running {
		return
	}
	hc.running = false
	close(hc.stopChan)
	hc.wg.Wait()
}

// checkAllServers checks the health of all servers
func (hc *HealthChecker) checkAllServers() {
	hc.loadBalancer.mutex.RLock()
	servers := make([]*BackendServer, len(hc.loadBalancer.servers))
	copy(servers, hc.loadBalancer.servers)
	hc.loadBalancer.mutex.RUnlock()

	var wg sync.WaitGroup
	for _, server := range servers {
		wg.Add(1)
		go func(s *BackendServer) {
			defer wg.Done()
			hc.checkServerHealth(s)
		}(server)
	}
	wg.Wait()
}

// checkServerHealth checks the health of a single server
func (hc *HealthChecker) checkServerHealth(server *BackendServer) {
	url := fmt.Sprintf("%s%s", server.URL.String(), hc.path)
	
	start := time.Now()
	resp, err := hc.client.Get(url)
	responseTime := time.Since(start)

	server.mutex.Lock()
	server.LastHealthCheck = time.Now()
	server.ResponseTime = responseTime

	if err != nil || resp.StatusCode >= 400 {
		switch server.HealthStatus {
		case Healthy:
			server.HealthStatus = Degraded
		case Degraded:
			server.HealthStatus = Unhealthy
		default:
			server.HealthStatus = Unhealthy
		}
	} else {
		server.HealthStatus = Healthy
	}
	server.mutex.Unlock()

	if resp != nil {
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
}

// CircuitBreaker implementation

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(config CircuitBreakerConfig) *CircuitBreaker {
	return &CircuitBreaker{
		state:  CircuitClosed,
		config: config,
	}
}

// AllowRequest checks if a request should be allowed
func (cb *CircuitBreaker) AllowRequest() bool {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	switch cb.state {
	case CircuitClosed:
		return true
	case CircuitOpen:
		if time.Since(cb.lastStateChange) > cb.config.Timeout {
			cb.state = CircuitHalfOpen
			cb.requests = 0
			return true
		}
		return false
	case CircuitHalfOpen:
		return cb.requests < cb.config.HalfOpenMaxCalls
	}
	return false
}

// RecordSuccess records a successful request
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	atomic.AddInt32(&cb.successes, 1)
	atomic.AddInt32(&cb.requests, 1)

	if cb.state == CircuitHalfOpen {
		if cb.successes >= cb.config.RecoveryThreshold {
			cb.state = CircuitClosed
			cb.failures = 0
			cb.successes = 0
			cb.lastStateChange = time.Now()
		}
	}
}

// RecordFailure records a failed request
func (cb *CircuitBreaker) RecordFailure() {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	atomic.AddInt32(&cb.failures, 1)
	atomic.AddInt32(&cb.requests, 1)
	cb.lastFailureTime = time.Now()

	if cb.state == CircuitClosed && cb.failures >= cb.config.FailureThreshold {
		cb.state = CircuitOpen
		cb.lastStateChange = time.Now()
	} else if cb.state == CircuitHalfOpen {
		cb.state = CircuitOpen
		cb.lastStateChange = time.Now()
	}
}

// SessionManager implementation

// NewSessionManager creates a new session manager
func NewSessionManager(enabled bool, timeout time.Duration) *SessionManager {
	sm := &SessionManager{
		sessions: make(map[string]*Session),
		timeout:  timeout,
		enabled:  enabled,
	}

	if enabled {
		go sm.cleanupExpiredSessions()
	}

	return sm
}

// GetSessionID gets or creates a session ID for a request
func (sm *SessionManager) GetSessionID(r *http.Request) string {
	if !sm.enabled {
		return ""
	}

	cookie, err := r.Cookie("LB_SESSION")
	if err != nil {
		return ""
	}
	return cookie.Value
}

// GetServerForSession gets the server ID for a session
func (sm *SessionManager) GetServerForSession(sessionID string) string {
	if !sm.enabled {
		return ""
	}

	sm.mutex.RLock()
	defer sm.mutex.RUnlock()

	session, exists := sm.sessions[sessionID]
	if !exists || time.Since(session.LastAccess) > sm.timeout {
		return ""
	}

	return session.ServerID
}

// UpdateSession updates or creates a session
func (sm *SessionManager) UpdateSession(sessionID, serverID string, w http.ResponseWriter) {
	if !sm.enabled {
		return
	}

	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	if sessionID == "" {
		sessionID = generateSessionID()
		http.SetCookie(w, &http.Cookie{
			Name:     "LB_SESSION",
			Value:    sessionID,
			Path:     "/",
			HttpOnly: true,
			Secure:   false, // Set to true for HTTPS
			SameSite: http.SameSiteLaxMode,
		})
	}

	sm.sessions[sessionID] = &Session{
		ID:         sessionID,
		ServerID:   serverID,
		LastAccess: time.Now(),
		Data:       make(map[string]interface{}),
	}
}

// cleanupExpiredSessions removes expired sessions
func (sm *SessionManager) cleanupExpiredSessions() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		sm.mutex.Lock()
		now := time.Now()
		for sessionID, session := range sm.sessions {
			if now.Sub(session.LastAccess) > sm.timeout {
				delete(sm.sessions, sessionID)
			}
		}
		sm.mutex.Unlock()
	}
}

// generateSessionID generates a new session ID
func generateSessionID() string {
	return fmt.Sprintf("session_%d_%d", time.Now().UnixNano(), rand.Int63())
}

// Metrics implementation

// NewMetrics creates a new metrics collector
func NewMetrics() *Metrics {
	return &Metrics{
		ServerMetrics: make(map[string]*ServerMetrics),
		lastUpdate:    time.Now(),
	}
}

// Start starts metrics collection
func (m *Metrics) Start() {
	go m.calculateRequestsPerSecond()
}

// Stop stops metrics collection
func (m *Metrics) Stop() {
	// Metrics collection is passive, no explicit stop needed
}

// AddServer adds a server to metrics tracking
func (m *Metrics) AddServer(serverID string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.ServerMetrics[serverID] = &ServerMetrics{
		LastSeen: time.Now(),
	}
}

// RemoveServer removes a server from metrics tracking
func (m *Metrics) RemoveServer(serverID string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	delete(m.ServerMetrics, serverID)
}

// UpdateServerMetrics updates metrics for a specific server
func (m *Metrics) UpdateServerMetrics(serverID string, responseTime time.Duration, success bool) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	metrics, exists := m.ServerMetrics[serverID]
	if !exists {
		metrics = &ServerMetrics{}
		m.ServerMetrics[serverID] = metrics
	}

	atomic.AddInt64(&metrics.Requests, 1)
	metrics.ResponseTime = responseTime
	metrics.LastSeen = time.Now()

	if !success {
		atomic.AddInt64(&metrics.Failures, 1)
	}

	// Calculate availability
	if metrics.Requests > 0 {
		metrics.Availability = float64(metrics.Requests-metrics.Failures) / float64(metrics.Requests)
	}
}

// GetSnapshot returns a snapshot of current metrics
func (m *Metrics) GetSnapshot() map[string]interface{} {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	totalRequests := atomic.LoadInt64(&m.TotalRequests)
	successfulRequests := atomic.LoadInt64(&m.SuccessfulRequests)
	failedRequests := atomic.LoadInt64(&m.FailedRequests)
	totalResponseTime := atomic.LoadInt64(&m.TotalResponseTime)

	var avgResponseTime time.Duration
	if totalRequests > 0 {
		avgResponseTime = time.Duration(totalResponseTime / totalRequests)
	}

	snapshot := map[string]interface{}{
		"total_requests":       totalRequests,
		"successful_requests":  successfulRequests,
		"failed_requests":      failedRequests,
		"average_response_time": avgResponseTime.String(),
		"requests_per_second":  m.RequestsPerSecond,
		"active_connections":   atomic.LoadInt64(&m.ActiveConnections),
		"circuit_breaker_trips": atomic.LoadInt64(&m.CircuitBreakerTrips),
		"servers": make(map[string]interface{}),
	}

	// Add server metrics
	serverMetrics := make(map[string]interface{})
	for serverID, metrics := range m.ServerMetrics {
		serverMetrics[serverID] = map[string]interface{}{
			"requests":      atomic.LoadInt64(&metrics.Requests),
			"failures":      atomic.LoadInt64(&metrics.Failures),
			"response_time": metrics.ResponseTime.String(),
			"availability":  metrics.Availability,
			"last_seen":     metrics.LastSeen,
		}
	}
	snapshot["servers"] = serverMetrics

	return snapshot
}

// calculateRequestsPerSecond calculates requests per second
func (m *Metrics) calculateRequestsPerSecond() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	lastRequests := atomic.LoadInt64(&m.TotalRequests)
	lastTime := time.Now()

	for range ticker.C {
		currentRequests := atomic.LoadInt64(&m.TotalRequests)
		currentTime := time.Now()

		duration := currentTime.Sub(lastTime).Seconds()
		if duration > 0 {
			m.mutex.Lock()
			m.RequestsPerSecond = float64(currentRequests-lastRequests) / duration
			m.mutex.Unlock()
		}

		lastRequests = currentRequests
		lastTime = currentTime
	}
}

// ConsistentHashRing implementation

// NewConsistentHashRing creates a new consistent hash ring
func NewConsistentHashRing(virtualNodes int) *ConsistentHashRing {
	return &ConsistentHashRing{
		ring:         make(map[uint32]*BackendServer),
		virtualNodes: virtualNodes,
	}
}

// AddServer adds a server to the hash ring
func (chr *ConsistentHashRing) AddServer(server *BackendServer) {
	chr.mutex.Lock()
	defer chr.mutex.Unlock()

	for i := 0; i < chr.virtualNodes; i++ {
		virtualKey := fmt.Sprintf("%s:%d", server.ID, i)
		hash := hashString(virtualKey)
		chr.ring[hash] = server
		chr.sortedKeys = append(chr.sortedKeys, hash)
	}

	sort.Slice(chr.sortedKeys, func(i, j int) bool {
		return chr.sortedKeys[i] < chr.sortedKeys[j]
	})
}

// RemoveServer removes a server from the hash ring
func (chr *ConsistentHashRing) RemoveServer(server *BackendServer) {
	chr.mutex.Lock()
	defer chr.mutex.Unlock()

	for i := 0; i < chr.virtualNodes; i++ {
		virtualKey := fmt.Sprintf("%s:%d", server.ID, i)
		hash := hashString(virtualKey)
		delete(chr.ring, hash)

		// Remove from sorted keys
		for j, key := range chr.sortedKeys {
			if key == hash {
				chr.sortedKeys = append(chr.sortedKeys[:j], chr.sortedKeys[j+1:]...)
				break
			}
		}
	}
}

// GetServer gets the server for a given key
func (chr *ConsistentHashRing) GetServer(key string) *BackendServer {
	chr.mutex.RLock()
	defer chr.mutex.RUnlock()

	if len(chr.ring) == 0 {
		return nil
	}

	hash := hashString(key)

	// Find the first server with hash >= key hash
	idx := sort.Search(len(chr.sortedKeys), func(i int) bool {
		return chr.sortedKeys[i] >= hash
	})

	// Wrap around if necessary
	if idx == len(chr.sortedKeys) {
		idx = 0
	}

	return chr.ring[chr.sortedKeys[idx]]
}