package concurrentapigateway

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestNewAPIGateway(t *testing.T) {
	config := GatewayConfig{
		Port:                  8080,
		MaxConcurrentRequests: 100,
		HealthCheckInterval:   30 * time.Second,
		ReadTimeout:          10 * time.Second,
		WriteTimeout:         10 * time.Second,
	}

	gateway := NewAPIGateway(config)

	if gateway.config.Port != 8080 {
		t.Errorf("Expected port 8080, got %d", gateway.config.Port)
	}

	if gateway.config.MaxConcurrentRequests != 100 {
		t.Errorf("Expected max concurrent requests 100, got %d", gateway.config.MaxConcurrentRequests)
	}

	if len(gateway.backends) != 0 {
		t.Errorf("Expected empty backends map, got %d entries", len(gateway.backends))
	}

	if len(gateway.routes) != 0 {
		t.Errorf("Expected empty routes map, got %d entries", len(gateway.routes))
	}
}

func TestDefaultConfiguration(t *testing.T) {
	config := GatewayConfig{}
	gateway := NewAPIGateway(config)

	if gateway.config.Port != 8080 {
		t.Errorf("Expected default port 8080, got %d", gateway.config.Port)
	}

	if gateway.config.ReadTimeout != 30*time.Second {
		t.Errorf("Expected default read timeout 30s, got %v", gateway.config.ReadTimeout)
	}

	if gateway.config.MaxConcurrentRequests != 1000 {
		t.Errorf("Expected default max concurrent requests 1000, got %d", gateway.config.MaxConcurrentRequests)
	}
}

func TestAddBackend(t *testing.T) {
	gateway := NewAPIGateway(GatewayConfig{})

	serverURL, _ := url.Parse("http://localhost:9001")
	server := &BackendServer{
		ID:     "server1",
		URL:    serverURL,
		Weight: 1,
		Health: Healthy,
	}

	backend := &Backend{
		ID:      "test-backend",
		Servers: []*BackendServer{server},
		Strategy: RoundRobin,
	}

	err := gateway.AddBackend(backend)
	if err != nil {
		t.Fatalf("Failed to add backend: %v", err)
	}

	if len(gateway.backends) != 1 {
		t.Errorf("Expected 1 backend, got %d", len(gateway.backends))
	}

	// Test duplicate backend
	err = gateway.AddBackend(backend)
	if err == nil {
		t.Error("Expected error when adding duplicate backend")
	}
}

func TestAddRoute(t *testing.T) {
	gateway := NewAPIGateway(GatewayConfig{})

	// Add backend first
	serverURL, _ := url.Parse("http://localhost:9001")
	server := &BackendServer{
		ID:     "server1",
		URL:    serverURL,
		Weight: 1,
		Health: Healthy,
	}

	backend := &Backend{
		ID:      "test-backend",
		Servers: []*BackendServer{server},
		Strategy: RoundRobin,
	}

	gateway.AddBackend(backend)

	route := &Route{
		ID:      "test-route",
		Pattern: "/api/",
		Methods: []string{"GET", "POST"},
		Backend: "test-backend",
		Timeout: 30 * time.Second,
	}

	err := gateway.AddRoute(route)
	if err != nil {
		t.Fatalf("Failed to add route: %v", err)
	}

	if len(gateway.routes) != 1 {
		t.Errorf("Expected 1 route, got %d", len(gateway.routes))
	}

	// Test duplicate route
	err = gateway.AddRoute(route)
	if err == nil {
		t.Error("Expected error when adding duplicate route")
	}

	// Test route with non-existent backend
	invalidRoute := &Route{
		ID:      "invalid-route",
		Pattern: "/invalid/",
		Backend: "non-existent-backend",
	}

	err = gateway.AddRoute(invalidRoute)
	if err == nil {
		t.Error("Expected error when adding route with non-existent backend")
	}
}

func TestLoadBalancerRoundRobin(t *testing.T) {
	lb := NewLoadBalancer()

	serverURL1, _ := url.Parse("http://localhost:9001")
	serverURL2, _ := url.Parse("http://localhost:9002")
	serverURL3, _ := url.Parse("http://localhost:9003")

	servers := []*BackendServer{
		{ID: "server1", URL: serverURL1, Health: Healthy},
		{ID: "server2", URL: serverURL2, Health: Healthy},
		{ID: "server3", URL: serverURL3, Health: Healthy},
	}

	backend := &Backend{
		ID:       "test-backend",
		Servers:  servers,
		Strategy: RoundRobin,
	}

	lb.AddBackend(backend)

	// Test round robin selection
	selectedServers := make(map[string]int)
	for i := 0; i < 12; i++ {
		server, err := lb.SelectServer(backend)
		if err != nil {
			t.Fatalf("Failed to select server: %v", err)
		}
		selectedServers[server.ID]++
	}

	// Each server should be selected 4 times
	for serverID, count := range selectedServers {
		if count != 4 {
			t.Errorf("Server %s selected %d times, expected 4", serverID, count)
		}
	}
}

func TestLoadBalancerLeastConnections(t *testing.T) {
	lb := NewLoadBalancer()

	serverURL1, _ := url.Parse("http://localhost:9001")
	serverURL2, _ := url.Parse("http://localhost:9002")

	server1 := &BackendServer{ID: "server1", URL: serverURL1, Health: Healthy}
	server2 := &BackendServer{ID: "server2", URL: serverURL2, Health: Healthy}

	// Simulate different connection counts
	atomic.StoreInt32(&server1.ActiveRequests, 5)
	atomic.StoreInt32(&server2.ActiveRequests, 2)

	backend := &Backend{
		ID:       "test-backend",
		Servers:  []*BackendServer{server1, server2},
		Strategy: LeastConnections,
	}

	server, err := lb.SelectServer(backend)
	if err != nil {
		t.Fatalf("Failed to select server: %v", err)
	}

	if server.ID != "server2" {
		t.Errorf("Expected server2 (least connections), got %s", server.ID)
	}
}

func TestLoadBalancerNoHealthyServers(t *testing.T) {
	lb := NewLoadBalancer()

	serverURL, _ := url.Parse("http://localhost:9001")
	server := &BackendServer{
		ID:     "server1",
		URL:    serverURL,
		Health: Unhealthy,
	}

	backend := &Backend{
		ID:      "test-backend",
		Servers: []*BackendServer{server},
		Strategy: RoundRobin,
	}

	_, err := lb.SelectServer(backend)
	if err == nil {
		t.Error("Expected error when no healthy servers available")
	}
}

func TestRateLimiter(t *testing.T) {
	rateLimiter := NewRateLimiter(true)

	clientID := "test-client"

	// First request should be allowed
	if !rateLimiter.Allow(clientID) {
		t.Error("First request should be allowed")
	}

	// Exhaust the token bucket
	for i := 0; i < 100; i++ {
		rateLimiter.Allow(clientID)
	}

	// Next request should be denied
	if rateLimiter.Allow(clientID) {
		t.Error("Request should be denied after exhausting tokens")
	}
}

func TestRateLimiterDisabled(t *testing.T) {
	rateLimiter := NewRateLimiter(false)

	clientID := "test-client"

	// Should always allow when disabled
	for i := 0; i < 200; i++ {
		if !rateLimiter.Allow(clientID) {
			t.Error("Rate limiter should allow all requests when disabled")
		}
	}
}

func TestCircuitBreaker(t *testing.T) {
	circuitBreaker := NewCircuitBreaker(true)
	backendID := "test-backend"

	// Initially should allow requests
	if !circuitBreaker.Allow(backendID) {
		t.Error("Circuit breaker should initially allow requests")
	}

	// Record failures to trip the circuit
	for i := 0; i < 5; i++ {
		circuitBreaker.RecordFailure(backendID)
	}

	// Should now deny requests (circuit open)
	if circuitBreaker.Allow(backendID) {
		t.Error("Circuit breaker should deny requests when open")
	}

	// Record success to close circuit (in half-open state)
	circuitBreaker.RecordSuccess(backendID)

	// Should allow requests again
	if !circuitBreaker.Allow(backendID) {
		t.Error("Circuit breaker should allow requests after success")
	}
}

func TestCircuitBreakerDisabled(t *testing.T) {
	circuitBreaker := NewCircuitBreaker(false)
	backendID := "test-backend"

	// Should always allow when disabled
	for i := 0; i < 10; i++ {
		if !circuitBreaker.Allow(backendID) {
			t.Error("Circuit breaker should allow all requests when disabled")
		}
		circuitBreaker.RecordFailure(backendID)
	}
}

func TestCache(t *testing.T) {
	cache := NewCache(true)

	key := "test-key"
	entry := &CacheEntry{
		Data:      []byte("test data"),
		Headers:   make(http.Header),
		Status:    200,
		ExpiresAt: time.Now().Add(1 * time.Hour),
	}

	// Cache should be empty initially
	if cached := cache.Get(key); cached != nil {
		t.Error("Cache should be empty initially")
	}

	// Set and get
	cache.Set(key, entry)
	if cached := cache.Get(key); cached == nil {
		t.Error("Should retrieve cached entry")
	}

	// Test expiration
	expiredEntry := &CacheEntry{
		Data:      []byte("expired data"),
		Headers:   make(http.Header),
		Status:    200,
		ExpiresAt: time.Now().Add(-1 * time.Hour),
	}

	expiredKey := "expired-key"
	cache.Set(expiredKey, expiredEntry)

	if cached := cache.Get(expiredKey); cached != nil {
		t.Error("Should not retrieve expired entry")
	}
}

func TestCacheDisabled(t *testing.T) {
	cache := NewCache(false)

	key := "test-key"
	entry := &CacheEntry{
		Data:    []byte("test data"),
		Headers: make(http.Header),
		Status:  200,
	}

	cache.Set(key, entry)

	// Should not retrieve when disabled
	if cached := cache.Get(key); cached != nil {
		t.Error("Cache should not work when disabled")
	}
}

func TestTokenBucket(t *testing.T) {
	bucket := &TokenBucket{
		tokens:     5,
		capacity:   10,
		refillRate: 2,
		lastRefill: time.Now(),
	}

	// Should allow requests while tokens available
	for i := 0; i < 5; i++ {
		if !bucket.Take() {
			t.Errorf("Should allow request %d", i+1)
		}
	}

	// Should deny when no tokens
	if bucket.Take() {
		t.Error("Should deny request when no tokens")
	}

	// Wait and allow refill
	time.Sleep(1 * time.Second)

	// Should allow after refill
	if !bucket.Take() {
		t.Error("Should allow request after refill")
	}
}

func TestPatternMatching(t *testing.T) {
	gateway := NewAPIGateway(GatewayConfig{})

	tests := []struct {
		path    string
		pattern string
		match   bool
	}{
		{"/api/users", "/api/", true},
		{"/api/users/123", "/api/", true},
		{"/health", "/health", true},
		{"/health/check", "/health", true},
		{"/users", "/api/", false},
		{"/", "/", true},
		{"/anything", "/", true},
	}

	for _, test := range tests {
		result := gateway.matchPattern(test.path, test.pattern)
		if result != test.match {
			t.Errorf("Pattern matching %s vs %s: expected %v, got %v",
				test.path, test.pattern, test.match, result)
		}
	}
}

func TestMethodMatching(t *testing.T) {
	gateway := NewAPIGateway(GatewayConfig{})

	methods := []string{"GET", "POST", "PUT"}

	if !gateway.containsMethod(methods, "GET") {
		t.Error("Should contain GET method")
	}

	if !gateway.containsMethod(methods, "POST") {
		t.Error("Should contain POST method")
	}

	if gateway.containsMethod(methods, "DELETE") {
		t.Error("Should not contain DELETE method")
	}
}

func TestAuthentication(t *testing.T) {
	gateway := NewAPIGateway(GatewayConfig{AuthenticationEnabled: true})

	// Test valid token
	req := httptest.NewRequest("GET", "/api/test", nil)
	req.Header.Set("Authorization", "Bearer valid-token")

	authCtx, err := gateway.authenticate(req)
	if err != nil {
		t.Errorf("Authentication should succeed with valid token: %v", err)
	}

	if authCtx.UserID != "user123" {
		t.Errorf("Expected user ID 'user123', got '%s'", authCtx.UserID)
	}

	// Test invalid token
	req.Header.Set("Authorization", "Bearer invalid-token")
	_, err = gateway.authenticate(req)
	if err == nil {
		t.Error("Authentication should fail with invalid token")
	}

	// Test missing token
	req.Header.Del("Authorization")
	_, err = gateway.authenticate(req)
	if err == nil {
		t.Error("Authentication should fail with missing token")
	}
}

func TestClientIPExtraction(t *testing.T) {
	gateway := NewAPIGateway(GatewayConfig{})

	tests := []struct {
		headers  map[string]string
		remoteAddr string
		expected string
	}{
		{
			headers:  map[string]string{"X-Forwarded-For": "192.168.1.1,10.0.0.1"},
			expected: "192.168.1.1",
		},
		{
			headers:  map[string]string{"X-Real-IP": "203.0.113.1"},
			expected: "203.0.113.1",
		},
		{
			headers:    map[string]string{},
			remoteAddr: "198.51.100.1:12345",
			expected:   "198.51.100.1",
		},
	}

	for _, test := range tests {
		req := httptest.NewRequest("GET", "/", nil)
		for key, value := range test.headers {
			req.Header.Set(key, value)
		}
		if test.remoteAddr != "" {
			req.RemoteAddr = test.remoteAddr
		}

		ip := gateway.getClientIP(req)
		if ip != test.expected {
			t.Errorf("Expected IP %s, got %s", test.expected, ip)
		}
	}
}

func TestHealthChecker(t *testing.T) {
	// Create a test server
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	}))
	defer testServer.Close()

	gateway := NewAPIGateway(GatewayConfig{
		HealthCheckInterval: 100 * time.Millisecond,
	})

	serverURL, _ := url.Parse(testServer.URL)
	server := &BackendServer{
		ID:     "test-server",
		URL:    serverURL,
		Health: Unknown,
	}

	backend := &Backend{
		ID:      "test-backend",
		Servers: []*BackendServer{server},
		HealthCheck: HealthCheckConfig{
			Enabled: true,
			Path:    "/health",
			Timeout: 5 * time.Second,
		},
	}

	gateway.AddBackend(backend)

	// Start health checker
	gateway.healthChecker.Start()
	defer gateway.healthChecker.Stop()

	// Wait for health check
	time.Sleep(200 * time.Millisecond)

	if server.Health != Healthy {
		t.Errorf("Expected server to be healthy, got %v", server.Health)
	}
}

func TestConcurrentRequests(t *testing.T) {
	config := GatewayConfig{
		MaxConcurrentRequests: 5,
		RateLimitEnabled:      false,
	}

	gateway := NewAPIGateway(config)

	// Create test backend
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond) // Simulate processing time
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	}))
	defer testServer.Close()

	serverURL, _ := url.Parse(testServer.URL)
	server := &BackendServer{
		ID:     "test-server",
		URL:    serverURL,
		Health: Healthy,
	}

	backend := &Backend{
		ID:      "test-backend",
		Servers: []*BackendServer{server},
		Strategy: RoundRobin,
	}

	gateway.AddBackend(backend)

	route := &Route{
		ID:      "test-route",
		Pattern: "/",
		Backend: "test-backend",
	}

	gateway.AddRoute(route)

	// Test concurrent requests
	numRequests := 10
	var wg sync.WaitGroup
	results := make([]int, numRequests)

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()

			req := httptest.NewRequest("GET", "/test", nil)
			w := httptest.NewRecorder()

			gateway.handleRequest(w, req)
			results[index] = w.Code
		}(i)
	}

	wg.Wait()

	// Check results
	successful := 0
	rateLimited := 0

	for _, code := range results {
		switch code {
		case http.StatusOK:
			successful++
		case http.StatusTooManyRequests:
			rateLimited++
		}
	}

	// Should have some successful and some rate limited due to concurrency limit
	if successful == 0 {
		t.Error("Expected some successful requests")
	}

	if successful > config.MaxConcurrentRequests {
		t.Errorf("Expected at most %d concurrent successful requests, got %d",
			config.MaxConcurrentRequests, successful)
	}
}

func TestMetricsCollection(t *testing.T) {
	gateway := NewAPIGateway(GatewayConfig{})

	// Initial metrics should be zero
	metrics := gateway.GetMetrics()
	if metrics.TotalRequests != 0 {
		t.Errorf("Expected 0 total requests, got %d", metrics.TotalRequests)
	}

	// Simulate some metrics
	atomic.AddInt64(&gateway.metrics.TotalRequests, 10)
	atomic.AddInt64(&gateway.metrics.SuccessfulRequests, 8)
	atomic.AddInt64(&gateway.metrics.FailedRequests, 2)

	metrics = gateway.GetMetrics()
	if metrics.TotalRequests != 10 {
		t.Errorf("Expected 10 total requests, got %d", metrics.TotalRequests)
	}

	if metrics.SuccessfulRequests != 8 {
		t.Errorf("Expected 8 successful requests, got %d", metrics.SuccessfulRequests)
	}

	if metrics.FailedRequests != 2 {
		t.Errorf("Expected 2 failed requests, got %d", metrics.FailedRequests)
	}
}

func TestMiddleware(t *testing.T) {
	gateway := NewAPIGateway(GatewayConfig{})

	// Test middleware execution order
	executionOrder := make([]string, 0)
	var mu sync.Mutex

	middleware1 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			executionOrder = append(executionOrder, "middleware1-before")
			mu.Unlock()

			next.ServeHTTP(w, r)

			mu.Lock()
			executionOrder = append(executionOrder, "middleware1-after")
			mu.Unlock()
		})
	}

	middleware2 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			executionOrder = append(executionOrder, "middleware2-before")
			mu.Unlock()

			next.ServeHTTP(w, r)

			mu.Lock()
			executionOrder = append(executionOrder, "middleware2-after")
			mu.Unlock()
		})
	}

	gateway.AddMiddleware(middleware1)
	gateway.AddMiddleware(middleware2)

	if len(gateway.middleware) != 2 {
		t.Errorf("Expected 2 middleware, got %d", len(gateway.middleware))
	}
}

func TestRouteSelection(t *testing.T) {
	gateway := NewAPIGateway(GatewayConfig{})

	// Add routes
	routes := []*Route{
		{ID: "api-route", Pattern: "/api/", Methods: []string{"GET", "POST"}},
		{ID: "health-route", Pattern: "/health", Methods: []string{"GET"}},
		{ID: "catch-all", Pattern: "/", Methods: []string{}},
	}

	for _, route := range routes {
		gateway.routes[route.ID] = route
	}

	tests := []struct {
		path         string
		method       string
		expectedRoute string
	}{
		{"/api/users", "GET", "api-route"},
		{"/api/posts", "POST", "api-route"},
		{"/health", "GET", "health-route"},
		{"/health", "POST", "catch-all"}, // health route doesn't allow POST
		{"/other", "GET", "catch-all"},
	}

	for _, test := range tests {
		req := httptest.NewRequest(test.method, test.path, nil)
		route := gateway.findRoute(req)

		if route == nil {
			t.Errorf("No route found for %s %s", test.method, test.path)
			continue
		}

		if route.ID != test.expectedRoute {
			t.Errorf("For %s %s, expected route %s, got %s",
				test.method, test.path, test.expectedRoute, route.ID)
		}
	}
}

func TestGatewayStartStop(t *testing.T) {
	config := GatewayConfig{
		Port: 0, // Use random port for testing
	}

	gateway := NewAPIGateway(config)

	// Test start and stop
	go func() {
		err := gateway.Start()
		if err != nil && err != http.ErrServerClosed {
			t.Errorf("Unexpected server error: %v", err)
		}
	}()

	// Give server time to start
	time.Sleep(100 * time.Millisecond)

	// Stop the gateway
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := gateway.Stop(ctx)
	if err != nil {
		t.Errorf("Failed to stop gateway: %v", err)
	}

	// Test stop when not running
	err = gateway.Stop(ctx)
	if err != nil {
		t.Errorf("Stop should not error when not running: %v", err)
	}
}

func BenchmarkLoadBalancer(b *testing.B) {
	lb := NewLoadBalancer()

	servers := make([]*BackendServer, 10)
	for i := 0; i < 10; i++ {
		serverURL, _ := url.Parse(fmt.Sprintf("http://localhost:900%d", i))
		servers[i] = &BackendServer{
			ID:     fmt.Sprintf("server%d", i),
			URL:    serverURL,
			Health: Healthy,
		}
	}

	backend := &Backend{
		ID:       "bench-backend",
		Servers:  servers,
		Strategy: RoundRobin,
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := lb.SelectServer(backend)
			if err != nil {
				b.Fatalf("Load balancer error: %v", err)
			}
		}
	})
}

func BenchmarkRateLimiter(b *testing.B) {
	rateLimiter := NewRateLimiter(true)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		clientID := "bench-client"
		for pb.Next() {
			rateLimiter.Allow(clientID)
		}
	})
}

func BenchmarkCircuitBreaker(b *testing.B) {
	circuitBreaker := NewCircuitBreaker(true)
	backendID := "bench-backend"

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			circuitBreaker.Allow(backendID)
		}
	})
}

func BenchmarkCache(b *testing.B) {
	cache := NewCache(true)
	
	// Pre-populate cache
	for i := 0; i < 1000; i++ {
		key := fmt.Sprintf("key-%d", i)
		entry := &CacheEntry{
			Data:      []byte(fmt.Sprintf("data-%d", i)),
			Headers:   make(http.Header),
			Status:    200,
			ExpiresAt: time.Now().Add(1 * time.Hour),
		}
		cache.Set(key, entry)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := fmt.Sprintf("key-%d", i%1000)
			cache.Get(key)
			i++
		}
	})
}