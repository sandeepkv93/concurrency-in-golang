package concurrentapigateway

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// LoadBalancingStrategy defines different load balancing algorithms
type LoadBalancingStrategy int

const (
	RoundRobin LoadBalancingStrategy = iota
	LeastConnections
	WeightedRoundRobin
	IPHash
	Random
)

// BackendServer represents a backend service
type BackendServer struct {
	ID              string
	URL             *url.URL
	Weight          int
	Health          HealthStatus
	ActiveRequests  int32
	TotalRequests   int64
	FailedRequests  int64
	LastHealthCheck time.Time
	ResponseTime    time.Duration
	mutex           sync.RWMutex
}

// HealthStatus represents the health state of a backend server
type HealthStatus int

const (
	Healthy HealthStatus = iota
	Unhealthy
	Unknown
)

// Route represents an API route configuration
type Route struct {
	ID           string
	Pattern      string
	Methods      []string
	Backend      string
	StripPrefix  string
	AddHeaders   map[string]string
	Timeout      time.Duration
	RetryCount   int
	CacheEnabled bool
	CacheTTL     time.Duration
}

// GatewayConfig contains the configuration for the API gateway
type GatewayConfig struct {
	Port                    int
	ReadTimeout            time.Duration
	WriteTimeout           time.Duration
	IdleTimeout            time.Duration
	MaxConcurrentRequests  int
	HealthCheckInterval    time.Duration
	CircuitBreakerEnabled  bool
	RateLimitEnabled       bool
	AuthenticationEnabled  bool
	MetricsEnabled         bool
	CacheEnabled           bool
	LoggingEnabled         bool
}

// APIGateway represents the main gateway instance
type APIGateway struct {
	config           GatewayConfig
	backends         map[string]*Backend
	routes           map[string]*Route
	server           *http.Server
	loadBalancer     *LoadBalancer
	rateLimiter      *RateLimiter
	circuitBreaker   *CircuitBreaker
	cache            *Cache
	metrics          *Metrics
	middleware       []Middleware
	healthChecker    *HealthChecker
	requestSemaphore chan struct{}
	mutex            sync.RWMutex
	running          bool
}

// Backend represents a group of backend servers
type Backend struct {
	ID            string
	Servers       []*BackendServer
	Strategy      LoadBalancingStrategy
	HealthCheck   HealthCheckConfig
	currentIndex  uint32
	mutex         sync.RWMutex
}

// HealthCheckConfig defines health check parameters
type HealthCheckConfig struct {
	Enabled  bool
	Path     string
	Interval time.Duration
	Timeout  time.Duration
	Method   string
}

// LoadBalancer handles request distribution
type LoadBalancer struct {
	backends map[string]*Backend
	mutex    sync.RWMutex
}

// RateLimiter implements token bucket rate limiting
type RateLimiter struct {
	clients map[string]*TokenBucket
	mutex   sync.RWMutex
	enabled bool
}

// TokenBucket represents a token bucket for rate limiting
type TokenBucket struct {
	tokens    int
	capacity  int
	refillRate int
	lastRefill time.Time
	mutex     sync.Mutex
}

// CircuitBreaker implements circuit breaker pattern
type CircuitBreaker struct {
	backends map[string]*CircuitState
	mutex    sync.RWMutex
	enabled  bool
}

// CircuitState represents the state of a circuit breaker
type CircuitState struct {
	state        CircuitBreakerState
	failures     int32
	requests     int32
	lastFailure  time.Time
	nextAttempt  time.Time
	threshold    int
	timeout      time.Duration
	mutex        sync.RWMutex
}

// CircuitBreakerState represents circuit breaker states
type CircuitBreakerState int

const (
	Closed CircuitBreakerState = iota
	Open
	HalfOpen
)

// Cache provides response caching
type Cache struct {
	entries map[string]*CacheEntry
	mutex   sync.RWMutex
	enabled bool
}

// CacheEntry represents a cached response
type CacheEntry struct {
	Data      []byte
	Headers   http.Header
	Status    int
	ExpiresAt time.Time
}

// Metrics collects gateway statistics
type Metrics struct {
	TotalRequests    int64
	SuccessfulRequests int64
	FailedRequests   int64
	AverageResponseTime time.Duration
	ActiveConnections int32
	RequestsPerSecond float64
	mutex            sync.RWMutex
}

// Middleware represents a middleware function
type Middleware func(http.Handler) http.Handler

// HealthChecker monitors backend health
type HealthChecker struct {
	gateway  *APIGateway
	interval time.Duration
	running  bool
	stopChan chan struct{}
	wg       sync.WaitGroup
}

// AuthContext contains authentication information
type AuthContext struct {
	UserID   string
	Username string
	Roles    []string
	Claims   map[string]interface{}
}

// NewAPIGateway creates a new API gateway instance
func NewAPIGateway(config GatewayConfig) *APIGateway {
	if config.Port == 0 {
		config.Port = 8080
	}
	if config.ReadTimeout == 0 {
		config.ReadTimeout = 30 * time.Second
	}
	if config.WriteTimeout == 0 {
		config.WriteTimeout = 30 * time.Second
	}
	if config.MaxConcurrentRequests == 0 {
		config.MaxConcurrentRequests = 1000
	}
	if config.HealthCheckInterval == 0 {
		config.HealthCheckInterval = 30 * time.Second
	}

	gateway := &APIGateway{
		config:           config,
		backends:         make(map[string]*Backend),
		routes:           make(map[string]*Route),
		loadBalancer:     NewLoadBalancer(),
		rateLimiter:      NewRateLimiter(config.RateLimitEnabled),
		circuitBreaker:   NewCircuitBreaker(config.CircuitBreakerEnabled),
		cache:            NewCache(config.CacheEnabled),
		metrics:          NewMetrics(),
		middleware:       make([]Middleware, 0),
		requestSemaphore: make(chan struct{}, config.MaxConcurrentRequests),
	}

	gateway.healthChecker = NewHealthChecker(gateway, config.HealthCheckInterval)
	gateway.setupServer()

	return gateway
}

// AddBackend adds a backend service configuration
func (gw *APIGateway) AddBackend(backend *Backend) error {
	gw.mutex.Lock()
	defer gw.mutex.Unlock()

	if _, exists := gw.backends[backend.ID]; exists {
		return fmt.Errorf("backend %s already exists", backend.ID)
	}

	gw.backends[backend.ID] = backend
	gw.loadBalancer.AddBackend(backend)

	return nil
}

// AddRoute adds a new route configuration
func (gw *APIGateway) AddRoute(route *Route) error {
	gw.mutex.Lock()
	defer gw.mutex.Unlock()

	if _, exists := gw.routes[route.ID]; exists {
		return fmt.Errorf("route %s already exists", route.ID)
	}

	// Validate backend exists
	if _, exists := gw.backends[route.Backend]; !exists {
		return fmt.Errorf("backend %s does not exist", route.Backend)
	}

	gw.routes[route.ID] = route
	return nil
}

// AddMiddleware adds a middleware to the processing chain
func (gw *APIGateway) AddMiddleware(middleware Middleware) {
	gw.mutex.Lock()
	defer gw.mutex.Unlock()
	gw.middleware = append(gw.middleware, middleware)
}

// Start starts the API gateway server
func (gw *APIGateway) Start() error {
	gw.mutex.Lock()
	if gw.running {
		gw.mutex.Unlock()
		return fmt.Errorf("gateway is already running")
	}
	gw.running = true
	gw.mutex.Unlock()

	// Start health checker
	if gw.config.HealthCheckInterval > 0 {
		gw.healthChecker.Start()
	}

	log.Printf("Starting API Gateway on port %d", gw.config.Port)
	return gw.server.ListenAndServe()
}

// Stop gracefully stops the API gateway
func (gw *APIGateway) Stop(ctx context.Context) error {
	gw.mutex.Lock()
	if !gw.running {
		gw.mutex.Unlock()
		return nil
	}
	gw.running = false
	gw.mutex.Unlock()

	// Stop health checker
	gw.healthChecker.Stop()

	// Shutdown server
	return gw.server.Shutdown(ctx)
}

// GetMetrics returns current gateway metrics
func (gw *APIGateway) GetMetrics() *Metrics {
	return gw.metrics.Copy()
}

// Private methods

func (gw *APIGateway) setupServer() {
	mux := http.NewServeMux()
	mux.HandleFunc("/", gw.handleRequest)
	mux.HandleFunc("/health", gw.handleHealth)
	mux.HandleFunc("/metrics", gw.handleMetrics)

	// Apply middleware chain
	var handler http.Handler = mux
	for i := len(gw.middleware) - 1; i >= 0; i-- {
		handler = gw.middleware[i](handler)
	}

	gw.server = &http.Server{
		Addr:         fmt.Sprintf(":%d", gw.config.Port),
		Handler:      handler,
		ReadTimeout:  gw.config.ReadTimeout,
		WriteTimeout: gw.config.WriteTimeout,
		IdleTimeout:  gw.config.IdleTimeout,
	}
}

func (gw *APIGateway) handleRequest(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	atomic.AddInt64(&gw.metrics.TotalRequests, 1)
	atomic.AddInt32(&gw.metrics.ActiveConnections, 1)
	defer atomic.AddInt32(&gw.metrics.ActiveConnections, -1)

	// Acquire semaphore for concurrent request limiting
	select {
	case gw.requestSemaphore <- struct{}{}:
		defer func() { <-gw.requestSemaphore }()
	default:
		http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
		atomic.AddInt64(&gw.metrics.FailedRequests, 1)
		return
	}

	// Rate limiting
	if gw.config.RateLimitEnabled {
		clientIP := gw.getClientIP(r)
		if !gw.rateLimiter.Allow(clientIP) {
			http.Error(w, "Rate Limit Exceeded", http.StatusTooManyRequests)
			atomic.AddInt64(&gw.metrics.FailedRequests, 1)
			return
		}
	}

	// Find matching route
	route := gw.findRoute(r)
	if route == nil {
		http.Error(w, "Not Found", http.StatusNotFound)
		atomic.AddInt64(&gw.metrics.FailedRequests, 1)
		return
	}

	// Authentication
	if gw.config.AuthenticationEnabled {
		authCtx, err := gw.authenticate(r)
		if err != nil {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			atomic.AddInt64(&gw.metrics.FailedRequests, 1)
			return
		}
		r = r.WithContext(context.WithValue(r.Context(), "auth", authCtx))
	}

	// Check cache
	if route.CacheEnabled && gw.config.CacheEnabled {
		cacheKey := gw.generateCacheKey(r)
		if cached := gw.cache.Get(cacheKey); cached != nil {
			gw.writeCachedResponse(w, cached)
			atomic.AddInt64(&gw.metrics.SuccessfulRequests, 1)
			gw.updateResponseTime(time.Since(start))
			return
		}
	}

	// Circuit breaker check
	if gw.config.CircuitBreakerEnabled {
		if !gw.circuitBreaker.Allow(route.Backend) {
			http.Error(w, "Service Temporarily Unavailable", http.StatusServiceUnavailable)
			atomic.AddInt64(&gw.metrics.FailedRequests, 1)
			return
		}
	}

	// Proxy request
	err := gw.proxyRequest(w, r, route)
	if err != nil {
		log.Printf("Proxy error: %v", err)
		
		// Record circuit breaker failure
		if gw.config.CircuitBreakerEnabled {
			gw.circuitBreaker.RecordFailure(route.Backend)
		}
		
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		atomic.AddInt64(&gw.metrics.FailedRequests, 1)
		return
	}

	// Record success
	if gw.config.CircuitBreakerEnabled {
		gw.circuitBreaker.RecordSuccess(route.Backend)
	}
	
	atomic.AddInt64(&gw.metrics.SuccessfulRequests, 1)
	gw.updateResponseTime(time.Since(start))
}

func (gw *APIGateway) proxyRequest(w http.ResponseWriter, r *http.Request, route *Route) error {
	// Select backend server
	backend := gw.backends[route.Backend]
	server, err := gw.loadBalancer.SelectServer(backend)
	if err != nil {
		return err
	}

	// Create reverse proxy
	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = server.URL.Scheme
			req.URL.Host = server.URL.Host
			
			// Strip prefix if configured
			if route.StripPrefix != "" {
				req.URL.Path = strings.TrimPrefix(req.URL.Path, route.StripPrefix)
			}
			
			// Add custom headers
			for key, value := range route.AddHeaders {
				req.Header.Set(key, value)
			}
			
			// Add forwarded headers
			req.Header.Set("X-Forwarded-For", gw.getClientIP(r))
			req.Header.Set("X-Forwarded-Proto", r.Header.Get("X-Forwarded-Proto"))
			if req.Header.Get("X-Forwarded-Proto") == "" {
				if r.TLS != nil {
					req.Header.Set("X-Forwarded-Proto", "https")
				} else {
					req.Header.Set("X-Forwarded-Proto", "http")
				}
			}
		},
		ModifyResponse: func(resp *http.Response) error {
			// Cache response if enabled
			if route.CacheEnabled && gw.config.CacheEnabled {
				if resp.StatusCode == http.StatusOK {
					cacheKey := gw.generateCacheKey(r)
					gw.cacheResponse(cacheKey, resp, route.CacheTTL)
				}
			}
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("Proxy error for backend %s: %v", server.ID, err)
			atomic.AddInt32(&server.ActiveRequests, -1)
			atomic.AddInt64(&server.FailedRequests, 1)
		},
	}

	// Update server metrics
	atomic.AddInt32(&server.ActiveRequests, 1)
	atomic.AddInt64(&server.TotalRequests, 1)
	defer atomic.AddInt32(&server.ActiveRequests, -1)

	// Set timeout
	if route.Timeout > 0 {
		ctx, cancel := context.WithTimeout(r.Context(), route.Timeout)
		defer cancel()
		r = r.WithContext(ctx)
	}

	// Proxy the request
	proxy.ServeHTTP(w, r)
	return nil
}

func (gw *APIGateway) findRoute(r *http.Request) *Route {
	path := r.URL.Path
	method := r.Method

	for _, route := range gw.routes {
		// Simple pattern matching (could be enhanced with regex)
		if gw.matchPattern(path, route.Pattern) {
			// Check method
			if len(route.Methods) == 0 || gw.containsMethod(route.Methods, method) {
				return route
			}
		}
	}

	return nil
}

func (gw *APIGateway) matchPattern(path, pattern string) bool {
	// Simple prefix matching (could be enhanced)
	if pattern == "/" {
		return true
	}
	return strings.HasPrefix(path, pattern)
}

func (gw *APIGateway) containsMethod(methods []string, method string) bool {
	for _, m := range methods {
		if m == method {
			return true
		}
	}
	return false
}

func (gw *APIGateway) authenticate(r *http.Request) (*AuthContext, error) {
	// Simple token-based authentication (can be enhanced)
	token := r.Header.Get("Authorization")
	if token == "" {
		return nil, fmt.Errorf("missing authorization header")
	}

	// Remove "Bearer " prefix
	if strings.HasPrefix(token, "Bearer ") {
		token = strings.TrimPrefix(token, "Bearer ")
	}

	// Simple validation (in practice, validate JWT or lookup in database)
	if token == "valid-token" {
		return &AuthContext{
			UserID:   "user123",
			Username: "testuser",
			Roles:    []string{"user"},
			Claims:   make(map[string]interface{}),
		}, nil
	}

	return nil, fmt.Errorf("invalid token")
}

func (gw *APIGateway) getClientIP(r *http.Request) string {
	// Check forwarded headers
	if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
		return strings.Split(forwarded, ",")[0]
	}
	if realIP := r.Header.Get("X-Real-IP"); realIP != "" {
		return realIP
	}
	
	// Extract from remote address
	parts := strings.Split(r.RemoteAddr, ":")
	if len(parts) > 0 {
		return parts[0]
	}
	
	return r.RemoteAddr
}

func (gw *APIGateway) generateCacheKey(r *http.Request) string {
	key := fmt.Sprintf("%s:%s", r.Method, r.URL.Path)
	if r.URL.RawQuery != "" {
		key += "?" + r.URL.RawQuery
	}
	
	hash := sha256.Sum256([]byte(key))
	return hex.EncodeToString(hash[:])
}

func (gw *APIGateway) cacheResponse(key string, resp *http.Response, ttl time.Duration) {
	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return
	}
	resp.Body.Close()

	// Store in cache
	entry := &CacheEntry{
		Data:      body,
		Headers:   resp.Header,
		Status:    resp.StatusCode,
		ExpiresAt: time.Now().Add(ttl),
	}
	
	gw.cache.Set(key, entry)
}

func (gw *APIGateway) writeCachedResponse(w http.ResponseWriter, entry *CacheEntry) {
	// Copy headers
	for key, values := range entry.Headers {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	
	w.WriteHeader(entry.Status)
	w.Write(entry.Data)
}

func (gw *APIGateway) updateResponseTime(duration time.Duration) {
	gw.metrics.mutex.Lock()
	defer gw.metrics.mutex.Unlock()
	
	if gw.metrics.AverageResponseTime == 0 {
		gw.metrics.AverageResponseTime = duration
	} else {
		gw.metrics.AverageResponseTime = (gw.metrics.AverageResponseTime + duration) / 2
	}
}

func (gw *APIGateway) handleHealth(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"status": "healthy",
		"timestamp": time.Now(),
		"backends": make(map[string]interface{}),
	}

	gw.mutex.RLock()
	for id, backend := range gw.backends {
		backendStatus := map[string]interface{}{
			"servers": make([]map[string]interface{}, 0),
		}

		for _, server := range backend.Servers {
			serverStatus := map[string]interface{}{
				"id":              server.ID,
				"url":             server.URL.String(),
				"health":          gw.healthStatusToString(server.Health),
				"active_requests": atomic.LoadInt32(&server.ActiveRequests),
				"total_requests":  atomic.LoadInt64(&server.TotalRequests),
				"failed_requests": atomic.LoadInt64(&server.FailedRequests),
			}
			backendStatus["servers"] = append(backendStatus["servers"].([]map[string]interface{}), serverStatus)
		}

		status["backends"].(map[string]interface{})[id] = backendStatus
	}
	gw.mutex.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (gw *APIGateway) handleMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := gw.GetMetrics()
	
	response := map[string]interface{}{
		"total_requests":      metrics.TotalRequests,
		"successful_requests": metrics.SuccessfulRequests,
		"failed_requests":     metrics.FailedRequests,
		"active_connections":  metrics.ActiveConnections,
		"average_response_time": metrics.AverageResponseTime.String(),
		"requests_per_second": metrics.RequestsPerSecond,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (gw *APIGateway) healthStatusToString(status HealthStatus) string {
	switch status {
	case Healthy:
		return "healthy"
	case Unhealthy:
		return "unhealthy"
	default:
		return "unknown"
	}
}

// LoadBalancer implementation

func NewLoadBalancer() *LoadBalancer {
	return &LoadBalancer{
		backends: make(map[string]*Backend),
	}
}

func (lb *LoadBalancer) AddBackend(backend *Backend) {
	lb.mutex.Lock()
	defer lb.mutex.Unlock()
	lb.backends[backend.ID] = backend
}

func (lb *LoadBalancer) SelectServer(backend *Backend) (*BackendServer, error) {
	backend.mutex.RLock()
	defer backend.mutex.RUnlock()

	healthyServers := make([]*BackendServer, 0)
	for _, server := range backend.Servers {
		if server.Health == Healthy {
			healthyServers = append(healthyServers, server)
		}
	}

	if len(healthyServers) == 0 {
		return nil, fmt.Errorf("no healthy servers available")
	}

	switch backend.Strategy {
	case RoundRobin:
		return lb.selectRoundRobin(backend, healthyServers), nil
	case LeastConnections:
		return lb.selectLeastConnections(healthyServers), nil
	case WeightedRoundRobin:
		return lb.selectWeightedRoundRobin(backend, healthyServers), nil
	default:
		return healthyServers[0], nil
	}
}

func (lb *LoadBalancer) selectRoundRobin(backend *Backend, servers []*BackendServer) *BackendServer {
	index := atomic.AddUint32(&backend.currentIndex, 1)
	return servers[(index-1)%uint32(len(servers))]
}

func (lb *LoadBalancer) selectLeastConnections(servers []*BackendServer) *BackendServer {
	minConnections := atomic.LoadInt32(&servers[0].ActiveRequests)
	selected := servers[0]

	for _, server := range servers[1:] {
		connections := atomic.LoadInt32(&server.ActiveRequests)
		if connections < minConnections {
			minConnections = connections
			selected = server
		}
	}

	return selected
}

func (lb *LoadBalancer) selectWeightedRoundRobin(backend *Backend, servers []*BackendServer) *BackendServer {
	// Simplified weighted round robin
	totalWeight := 0
	for _, server := range servers {
		totalWeight += server.Weight
	}

	if totalWeight == 0 {
		return lb.selectRoundRobin(backend, servers)
	}

	index := atomic.AddUint32(&backend.currentIndex, 1)
	target := int(index) % totalWeight

	current := 0
	for _, server := range servers {
		current += server.Weight
		if current > target {
			return server
		}
	}

	return servers[0]
}

// RateLimiter implementation

func NewRateLimiter(enabled bool) *RateLimiter {
	return &RateLimiter{
		clients: make(map[string]*TokenBucket),
		enabled: enabled,
	}
}

func (rl *RateLimiter) Allow(clientID string) bool {
	if !rl.enabled {
		return true
	}

	rl.mutex.Lock()
	bucket, exists := rl.clients[clientID]
	if !exists {
		bucket = &TokenBucket{
			tokens:    100,
			capacity:  100,
			refillRate: 10,
			lastRefill: time.Now(),
		}
		rl.clients[clientID] = bucket
	}
	rl.mutex.Unlock()

	return bucket.Take()
}

func (tb *TokenBucket) Take() bool {
	tb.mutex.Lock()
	defer tb.mutex.Unlock()

	now := time.Now()
	elapsed := now.Sub(tb.lastRefill)
	
	// Refill tokens
	tokensToAdd := int(elapsed.Seconds()) * tb.refillRate
	tb.tokens = min(tb.capacity, tb.tokens+tokensToAdd)
	tb.lastRefill = now

	if tb.tokens > 0 {
		tb.tokens--
		return true
	}

	return false
}

// CircuitBreaker implementation

func NewCircuitBreaker(enabled bool) *CircuitBreaker {
	return &CircuitBreaker{
		backends: make(map[string]*CircuitState),
		enabled:  enabled,
	}
}

func (cb *CircuitBreaker) Allow(backendID string) bool {
	if !cb.enabled {
		return true
	}

	cb.mutex.Lock()
	state, exists := cb.backends[backendID]
	if !exists {
		state = &CircuitState{
			state:     Closed,
			threshold: 5,
			timeout:   30 * time.Second,
		}
		cb.backends[backendID] = state
	}
	cb.mutex.Unlock()

	return state.Allow()
}

func (cb *CircuitBreaker) RecordSuccess(backendID string) {
	if !cb.enabled {
		return
	}

	cb.mutex.RLock()
	state, exists := cb.backends[backendID]
	cb.mutex.RUnlock()

	if exists {
		state.RecordSuccess()
	}
}

func (cb *CircuitBreaker) RecordFailure(backendID string) {
	if !cb.enabled {
		return
	}

	cb.mutex.RLock()
	state, exists := cb.backends[backendID]
	cb.mutex.RUnlock()

	if exists {
		state.RecordFailure()
	}
}

func (cs *CircuitState) Allow() bool {
	cs.mutex.Lock()
	defer cs.mutex.Unlock()

	switch cs.state {
	case Closed:
		return true
	case Open:
		if time.Now().After(cs.nextAttempt) {
			cs.state = HalfOpen
			return true
		}
		return false
	case HalfOpen:
		return true
	default:
		return false
	}
}

func (cs *CircuitState) RecordSuccess() {
	cs.mutex.Lock()
	defer cs.mutex.Unlock()

	atomic.AddInt32(&cs.requests, 1)

	if cs.state == HalfOpen {
		cs.state = Closed
		atomic.StoreInt32(&cs.failures, 0)
	}
}

func (cs *CircuitState) RecordFailure() {
	cs.mutex.Lock()
	defer cs.mutex.Unlock()

	atomic.AddInt32(&cs.failures, 1)
	atomic.AddInt32(&cs.requests, 1)
	cs.lastFailure = time.Now()

	failures := atomic.LoadInt32(&cs.failures)
	if failures >= int32(cs.threshold) {
		cs.state = Open
		cs.nextAttempt = time.Now().Add(cs.timeout)
	}
}

// Cache implementation

func NewCache(enabled bool) *Cache {
	return &Cache{
		entries: make(map[string]*CacheEntry),
		enabled: enabled,
	}
}

func (c *Cache) Get(key string) *CacheEntry {
	if !c.enabled {
		return nil
	}

	c.mutex.RLock()
	defer c.mutex.RUnlock()

	entry, exists := c.entries[key]
	if !exists || time.Now().After(entry.ExpiresAt) {
		return nil
	}

	return entry
}

func (c *Cache) Set(key string, entry *CacheEntry) {
	if !c.enabled {
		return
	}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.entries[key] = entry
}

// Metrics implementation

func NewMetrics() *Metrics {
	return &Metrics{}
}

func (m *Metrics) Copy() *Metrics {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	return &Metrics{
		TotalRequests:       atomic.LoadInt64(&m.TotalRequests),
		SuccessfulRequests:  atomic.LoadInt64(&m.SuccessfulRequests),
		FailedRequests:      atomic.LoadInt64(&m.FailedRequests),
		AverageResponseTime: m.AverageResponseTime,
		ActiveConnections:   atomic.LoadInt32(&m.ActiveConnections),
		RequestsPerSecond:   m.RequestsPerSecond,
	}
}

// HealthChecker implementation

func NewHealthChecker(gateway *APIGateway, interval time.Duration) *HealthChecker {
	return &HealthChecker{
		gateway:  gateway,
		interval: interval,
		stopChan: make(chan struct{}),
	}
}

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
				hc.checkHealth()
			case <-hc.stopChan:
				return
			}
		}
	}()
}

func (hc *HealthChecker) Stop() {
	if !hc.running {
		return
	}

	hc.running = false
	close(hc.stopChan)
	hc.wg.Wait()
}

func (hc *HealthChecker) checkHealth() {
	hc.gateway.mutex.RLock()
	backends := make([]*Backend, 0, len(hc.gateway.backends))
	for _, backend := range hc.gateway.backends {
		backends = append(backends, backend)
	}
	hc.gateway.mutex.RUnlock()

	var wg sync.WaitGroup
	for _, backend := range backends {
		for _, server := range backend.Servers {
			wg.Add(1)
			go func(s *BackendServer, b *Backend) {
				defer wg.Done()
				hc.checkServerHealth(s, b.HealthCheck)
			}(server, backend)
		}
	}
	wg.Wait()
}

func (hc *HealthChecker) checkServerHealth(server *BackendServer, config HealthCheckConfig) {
	if !config.Enabled {
		return
	}

	url := server.URL.String() + config.Path
	if config.Path == "" {
		url = server.URL.String() + "/health"
	}

	client := &http.Client{
		Timeout: config.Timeout,
	}

	method := config.Method
	if method == "" {
		method = "GET"
	}

	start := time.Now()
	resp, err := client.Get(url)
	responseTime := time.Since(start)

	server.mutex.Lock()
	server.LastHealthCheck = time.Now()
	server.ResponseTime = responseTime

	if err != nil || resp.StatusCode >= 400 {
		server.Health = Unhealthy
	} else {
		server.Health = Healthy
	}
	server.mutex.Unlock()

	if resp != nil {
		resp.Body.Close()
	}
}

// Utility functions

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Middleware helpers

func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
	})
}

func CORSMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func SecurityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		next.ServeHTTP(w, r)
	})
}