package concurrentloadbalancer

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

func TestNewLoadBalancer(t *testing.T) {
	config := LoadBalancerConfig{
		Strategy:            RoundRobin,
		Port:                8080,
		HealthCheckInterval: 30 * time.Second,
		MaxRetries:          3,
		CircuitBreakerEnabled: true,
	}

	lb := NewLoadBalancer(config)

	if lb.strategy != RoundRobin {
		t.Errorf("Expected strategy %v, got %v", RoundRobin, lb.strategy)
	}

	if lb.config.Port != 8080 {
		t.Errorf("Expected port 8080, got %d", lb.config.Port)
	}

	if len(lb.servers) != 0 {
		t.Errorf("Expected empty server list, got %d servers", len(lb.servers))
	}

	if len(lb.serverMap) != 0 {
		t.Errorf("Expected empty server map, got %d entries", len(lb.serverMap))
	}
}

func TestDefaultConfiguration(t *testing.T) {
	config := LoadBalancerConfig{}
	lb := NewLoadBalancer(config)

	if lb.config.Port != 8080 {
		t.Errorf("Expected default port 8080, got %d", lb.config.Port)
	}

	if lb.config.HealthCheckInterval != 30*time.Second {
		t.Errorf("Expected default health check interval 30s, got %v", lb.config.HealthCheckInterval)
	}

	if lb.config.MaxRetries != 3 {
		t.Errorf("Expected default max retries 3, got %d", lb.config.MaxRetries)
	}
}

func TestAddServer(t *testing.T) {
	lb := NewLoadBalancer(LoadBalancerConfig{})

	serverURL, _ := url.Parse("http://localhost:9001")
	server := &BackendServer{
		ID:     "server1",
		URL:    serverURL,
		Weight: 1,
	}

	err := lb.AddServer(server)
	if err != nil {
		t.Fatalf("Failed to add server: %v", err)
	}

	if len(lb.servers) != 1 {
		t.Errorf("Expected 1 server, got %d", len(lb.servers))
	}

	if _, exists := lb.serverMap["server1"]; !exists {
		t.Error("Server not found in server map")
	}

	// Test adding duplicate server
	err = lb.AddServer(server)
	if err == nil {
		t.Error("Expected error when adding duplicate server")
	}
}

func TestAddServerValidation(t *testing.T) {
	lb := NewLoadBalancer(LoadBalancerConfig{})

	// Test nil server
	err := lb.AddServer(nil)
	if err == nil {
		t.Error("Expected error for nil server")
	}

	// Test server with empty ID
	server := &BackendServer{
		URL: &url.URL{},
	}
	err = lb.AddServer(server)
	if err == nil {
		t.Error("Expected error for server with empty ID")
	}

	// Test server with nil URL
	server = &BackendServer{
		ID: "test",
	}
	err = lb.AddServer(server)
	if err == nil {
		t.Error("Expected error for server with nil URL")
	}
}

func TestRemoveServer(t *testing.T) {
	lb := NewLoadBalancer(LoadBalancerConfig{})

	serverURL, _ := url.Parse("http://localhost:9001")
	server := &BackendServer{
		ID:  "server1",
		URL: serverURL,
	}

	// Add server first
	lb.AddServer(server)

	// Remove server
	err := lb.RemoveServer("server1")
	if err != nil {
		t.Fatalf("Failed to remove server: %v", err)
	}

	if len(lb.servers) != 0 {
		t.Errorf("Expected 0 servers, got %d", len(lb.servers))
	}

	if _, exists := lb.serverMap["server1"]; exists {
		t.Error("Server still found in server map")
	}

	// Test removing non-existent server
	err = lb.RemoveServer("nonexistent")
	if err == nil {
		t.Error("Expected error when removing non-existent server")
	}
}

func TestRoundRobinSelection(t *testing.T) {
	lb := NewLoadBalancer(LoadBalancerConfig{Strategy: RoundRobin})

	// Add test servers
	for i := 0; i < 3; i++ {
		serverURL, _ := url.Parse(fmt.Sprintf("http://localhost:900%d", i))
		server := &BackendServer{
			ID:           fmt.Sprintf("server%d", i),
			URL:          serverURL,
			Weight:       1,
			HealthStatus: Healthy,
		}
		lb.AddServer(server)
	}

	// Test round robin selection
	selectedServers := make(map[string]int)
	for i := 0; i < 9; i++ {
		req := httptest.NewRequest("GET", "/", nil)
		server, err := lb.selectServer(req)
		if err != nil {
			t.Fatalf("Failed to select server: %v", err)
		}
		selectedServers[server.ID]++
	}

	// Each server should be selected 3 times
	for serverID, count := range selectedServers {
		if count != 3 {
			t.Errorf("Server %s selected %d times, expected 3", serverID, count)
		}
	}
}

func TestWeightedRoundRobinSelection(t *testing.T) {
	lb := NewLoadBalancer(LoadBalancerConfig{Strategy: WeightedRoundRobin})

	// Add servers with different weights
	serverURL1, _ := url.Parse("http://localhost:9001")
	server1 := &BackendServer{
		ID:           "server1",
		URL:          serverURL1,
		Weight:       3,
		HealthStatus: Healthy,
	}
	lb.AddServer(server1)

	serverURL2, _ := url.Parse("http://localhost:9002")
	server2 := &BackendServer{
		ID:           "server2",
		URL:          serverURL2,
		Weight:       1,
		HealthStatus: Healthy,
	}
	lb.AddServer(server2)

	// Test weighted selection
	selectedServers := make(map[string]int)
	for i := 0; i < 12; i++ {
		req := httptest.NewRequest("GET", "/", nil)
		server, err := lb.selectServer(req)
		if err != nil {
			t.Fatalf("Failed to select server: %v", err)
		}
		selectedServers[server.ID]++
	}

	// server1 should be selected more often due to higher weight
	if selectedServers["server1"] <= selectedServers["server2"] {
		t.Errorf("Expected server1 to be selected more often, got server1: %d, server2: %d",
			selectedServers["server1"], selectedServers["server2"])
	}
}

func TestLeastConnectionsSelection(t *testing.T) {
	lb := NewLoadBalancer(LoadBalancerConfig{Strategy: LeastConnections})

	// Add servers
	serverURL1, _ := url.Parse("http://localhost:9001")
	server1 := &BackendServer{
		ID:           "server1",
		URL:          serverURL1,
		HealthStatus: Healthy,
	}
	lb.AddServer(server1)

	serverURL2, _ := url.Parse("http://localhost:9002")
	server2 := &BackendServer{
		ID:           "server2",
		URL:          serverURL2,
		HealthStatus: Healthy,
	}
	lb.AddServer(server2)

	// Simulate different connection counts
	atomic.StoreInt64(&server1.ActiveConnections, 5)
	atomic.StoreInt64(&server2.ActiveConnections, 2)

	req := httptest.NewRequest("GET", "/", nil)
	selected, err := lb.selectServer(req)
	if err != nil {
		t.Fatalf("Failed to select server: %v", err)
	}

	if selected.ID != "server2" {
		t.Errorf("Expected server2 (least connections), got %s", selected.ID)
	}
}

func TestIPHashSelection(t *testing.T) {
	lb := NewLoadBalancer(LoadBalancerConfig{Strategy: IPHash})

	// Add servers
	for i := 0; i < 3; i++ {
		serverURL, _ := url.Parse(fmt.Sprintf("http://localhost:900%d", i))
		server := &BackendServer{
			ID:           fmt.Sprintf("server%d", i),
			URL:          serverURL,
			HealthStatus: Healthy,
		}
		lb.AddServer(server)
	}

	// Test consistent selection for same IP
	req1 := httptest.NewRequest("GET", "/", nil)
	req1.RemoteAddr = "192.168.1.1:12345"

	server1, err := lb.selectServer(req1)
	if err != nil {
		t.Fatalf("Failed to select server: %v", err)
	}

	server2, err := lb.selectServer(req1)
	if err != nil {
		t.Fatalf("Failed to select server: %v", err)
	}

	if server1.ID != server2.ID {
		t.Errorf("Expected consistent selection for same IP, got %s and %s", server1.ID, server2.ID)
	}
}

func TestNoHealthyServers(t *testing.T) {
	lb := NewLoadBalancer(LoadBalancerConfig{})

	// Add unhealthy server
	serverURL, _ := url.Parse("http://localhost:9001")
	server := &BackendServer{
		ID:           "server1",
		URL:          serverURL,
		HealthStatus: Unhealthy,
	}
	lb.AddServer(server)

	req := httptest.NewRequest("GET", "/", nil)
	_, err := lb.selectServer(req)
	if err == nil {
		t.Error("Expected error when no healthy servers available")
	}
}

func TestHealthChecker(t *testing.T) {
	// Create a test server
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("OK"))
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer testServer.Close()

	config := LoadBalancerConfig{
		HealthCheckInterval: 100 * time.Millisecond,
		HealthCheckTimeout:  1 * time.Second,
		HealthCheckPath:     "/health",
	}
	lb := NewLoadBalancer(config)

	serverURL, _ := url.Parse(testServer.URL)
	server := &BackendServer{
		ID:           "test-server",
		URL:          serverURL,
		HealthStatus: Unknown,
	}
	lb.AddServer(server)

	// Start health checker
	lb.healthChecker.Start()
	defer lb.healthChecker.Stop()

	// Wait for health check
	time.Sleep(200 * time.Millisecond)

	if server.HealthStatus != Healthy {
		t.Errorf("Expected server to be healthy, got %v", server.HealthStatus)
	}
}

func TestCircuitBreaker(t *testing.T) {
	config := CircuitBreakerConfig{
		FailureThreshold:  3,
		RecoveryThreshold: 2,
		Timeout:           100 * time.Millisecond,
		HalfOpenMaxCalls:  5,
	}

	cb := NewCircuitBreaker(config)

	// Initially should allow requests
	if !cb.AllowRequest() {
		t.Error("Circuit breaker should initially allow requests")
	}

	// Record failures to trip the circuit
	for i := 0; i < 3; i++ {
		cb.RecordFailure()
	}

	// Should now deny requests (circuit open)
	if cb.AllowRequest() {
		t.Error("Circuit breaker should deny requests when open")
	}

	// Wait for timeout
	time.Sleep(150 * time.Millisecond)

	// Should allow limited requests in half-open state
	if !cb.AllowRequest() {
		t.Error("Circuit breaker should allow requests in half-open state")
	}

	// Record successes to close circuit
	for i := 0; i < 2; i++ {
		cb.RecordSuccess()
	}

	// Should allow requests again (circuit closed)
	if !cb.AllowRequest() {
		t.Error("Circuit breaker should allow requests when closed")
	}
}

func TestSessionManager(t *testing.T) {
	sm := NewSessionManager(true, 1*time.Hour)

	// Test session creation
	w := httptest.NewRecorder()
	sm.UpdateSession("", "server1", w)

	// Check if cookie was set
	cookies := w.Result().Cookies()
	if len(cookies) == 0 {
		t.Error("Expected session cookie to be set")
	}

	sessionID := cookies[0].Value

	// Test session retrieval
	serverID := sm.GetServerForSession(sessionID)
	if serverID != "server1" {
		t.Errorf("Expected server1, got %s", serverID)
	}

	// Test session with request
	req := httptest.NewRequest("GET", "/", nil)
	req.AddCookie(&http.Cookie{Name: "LB_SESSION", Value: sessionID})

	retrievedSessionID := sm.GetSessionID(req)
	if retrievedSessionID != sessionID {
		t.Errorf("Expected session ID %s, got %s", sessionID, retrievedSessionID)
	}
}

func TestConsistentHashRing(t *testing.T) {
	chr := NewConsistentHashRing(100)

	// Add servers
	servers := make([]*BackendServer, 3)
	for i := 0; i < 3; i++ {
		serverURL, _ := url.Parse(fmt.Sprintf("http://localhost:900%d", i))
		servers[i] = &BackendServer{
			ID:  fmt.Sprintf("server%d", i),
			URL: serverURL,
		}
		chr.AddServer(servers[i])
	}

	// Test consistent selection
	key := "test-key"
	server1 := chr.GetServer(key)
	server2 := chr.GetServer(key)

	if server1.ID != server2.ID {
		t.Errorf("Expected consistent selection, got %s and %s", server1.ID, server2.ID)
	}

	// Test distribution
	keySelections := make(map[string]int)
	for i := 0; i < 1000; i++ {
		key := fmt.Sprintf("key-%d", i)
		server := chr.GetServer(key)
		keySelections[server.ID]++
	}

	// Each server should get some keys
	for serverID, count := range keySelections {
		if count == 0 {
			t.Errorf("Server %s got no keys", serverID)
		}
	}

	// Test server removal
	chr.RemoveServer(servers[0])
	server3 := chr.GetServer("new-key")
	if server3.ID == servers[0].ID {
		t.Error("Removed server was still selected")
	}
}

func TestConcurrentRequests(t *testing.T) {
	// Create test backend servers
	server1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(10 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("Server 1"))
	}))
	defer server1.Close()

	server2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(10 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("Server 2"))
	}))
	defer server2.Close()

	// Setup load balancer
	config := LoadBalancerConfig{
		Strategy:   RoundRobin,
		MaxRetries: 1,
	}
	lb := NewLoadBalancer(config)

	serverURL1, _ := url.Parse(server1.URL)
	backendServer1 := &BackendServer{
		ID:           "backend1",
		URL:          serverURL1,
		HealthStatus: Healthy,
	}
	lb.AddServer(backendServer1)

	serverURL2, _ := url.Parse(server2.URL)
	backendServer2 := &BackendServer{
		ID:           "backend2",
		URL:          serverURL2,
		HealthStatus: Healthy,
	}
	lb.AddServer(backendServer2)

	// Simulate concurrent requests
	numRequests := 20
	var wg sync.WaitGroup
	results := make([]int, numRequests)

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()

			req := httptest.NewRequest("GET", "/test", nil)
			w := httptest.NewRecorder()

			lb.handleRequest(w, req)
			results[index] = w.Code
		}(i)
	}

	wg.Wait()

	// Check that all requests were handled
	successCount := 0
	for _, code := range results {
		if code == http.StatusOK {
			successCount++
		}
	}

	if successCount != numRequests {
		t.Errorf("Expected %d successful requests, got %d", numRequests, successCount)
	}

	// Verify metrics
	if atomic.LoadInt64(&lb.metrics.TotalRequests) != int64(numRequests) {
		t.Errorf("Expected %d total requests in metrics, got %d", 
			numRequests, atomic.LoadInt64(&lb.metrics.TotalRequests))
	}
}

func TestMetricsCollection(t *testing.T) {
	lb := NewLoadBalancer(LoadBalancerConfig{MetricsEnabled: true})

	// Start metrics collection
	lb.metrics.Start()

	// Add a server to metrics
	lb.metrics.AddServer("test-server")

	// Update server metrics
	lb.metrics.UpdateServerMetrics("test-server", 100*time.Millisecond, true)
	lb.metrics.UpdateServerMetrics("test-server", 200*time.Millisecond, false)

	// Get metrics snapshot
	snapshot := lb.metrics.GetSnapshot()

	if snapshot["total_requests"] != int64(0) {
		t.Error("Total requests should be 0 initially")
	}

	servers, ok := snapshot["servers"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected servers in metrics snapshot")
	}

	serverMetrics, exists := servers["test-server"]
	if !exists {
		t.Error("Expected test-server in metrics")
	}

	if serverMetrics == nil {
		t.Error("Expected server metrics to be non-nil")
	}
}

func TestPowerOfTwoChoices(t *testing.T) {
	lb := NewLoadBalancer(LoadBalancerConfig{Strategy: PowerOfTwoChoices})

	// Add servers with different connection counts
	servers := make([]*BackendServer, 4)
	for i := 0; i < 4; i++ {
		serverURL, _ := url.Parse(fmt.Sprintf("http://localhost:900%d", i))
		servers[i] = &BackendServer{
			ID:           fmt.Sprintf("server%d", i),
			URL:          serverURL,
			HealthStatus: Healthy,
		}
		atomic.StoreInt64(&servers[i].ActiveConnections, int64(i*2))
		lb.AddServer(servers[i])
	}

	// Test multiple selections
	selections := make(map[string]int)
	for i := 0; i < 100; i++ {
		req := httptest.NewRequest("GET", "/", nil)
		server, err := lb.selectServer(req)
		if err != nil {
			t.Fatalf("Failed to select server: %v", err)
		}
		selections[server.ID]++
	}

	// Servers with fewer connections should be selected more often
	if selections["server0"] < selections["server3"] {
		t.Error("Expected server with fewer connections to be selected more often")
	}
}

func TestRandomSelection(t *testing.T) {
	lb := NewLoadBalancer(LoadBalancerConfig{Strategy: RandomSelection})

	// Add servers
	for i := 0; i < 3; i++ {
		serverURL, _ := url.Parse(fmt.Sprintf("http://localhost:900%d", i))
		server := &BackendServer{
			ID:           fmt.Sprintf("server%d", i),
			URL:          serverURL,
			HealthStatus: Healthy,
		}
		lb.AddServer(server)
	}

	// Test random distribution
	selections := make(map[string]int)
	for i := 0; i < 300; i++ {
		req := httptest.NewRequest("GET", "/", nil)
		server, err := lb.selectServer(req)
		if err != nil {
			t.Fatalf("Failed to select server: %v", err)
		}
		selections[server.ID]++
	}

	// Each server should get some selections (allowing for randomness)
	for serverID, count := range selections {
		if count < 50 || count > 150 {
			t.Errorf("Server %s selected %d times, expected roughly 100", serverID, count)
		}
	}
}

func TestRetryMechanism(t *testing.T) {
	// Create a server that fails initially then succeeds
	requestCount := int32(0)
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&requestCount, 1)
		if count <= 2 {
			w.WriteHeader(http.StatusInternalServerError)
		} else {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("Success"))
		}
	}))
	defer testServer.Close()

	config := LoadBalancerConfig{
		Strategy:     RoundRobin,
		MaxRetries:   3,
		RetryBackoff: 10 * time.Millisecond,
	}
	lb := NewLoadBalancer(config)

	serverURL, _ := url.Parse(testServer.URL)
	server := &BackendServer{
		ID:           "test-server",
		URL:          serverURL,
		HealthStatus: Healthy,
	}
	lb.AddServer(server)

	// Another server for retry
	server2URL, _ := url.Parse("http://localhost:9999") // Non-existent
	server2 := &BackendServer{
		ID:           "test-server2",
		URL:          server2URL,
		HealthStatus: Healthy,
	}
	lb.AddServer(server2)

	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()

	lb.handleRequest(w, req)

	// Should eventually succeed after retries
	if w.Code != http.StatusOK {
		t.Errorf("Expected request to succeed after retries, got %d", w.Code)
	}
}

func BenchmarkRoundRobinSelection(b *testing.B) {
	lb := NewLoadBalancer(LoadBalancerConfig{Strategy: RoundRobin})

	// Add test servers
	for i := 0; i < 10; i++ {
		serverURL, _ := url.Parse(fmt.Sprintf("http://localhost:900%d", i))
		server := &BackendServer{
			ID:           fmt.Sprintf("server%d", i),
			URL:          serverURL,
			HealthStatus: Healthy,
		}
		lb.AddServer(server)
	}

	req := httptest.NewRequest("GET", "/", nil)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := lb.selectServer(req)
			if err != nil {
				b.Fatalf("Server selection failed: %v", err)
			}
		}
	})
}

func BenchmarkLeastConnectionsSelection(b *testing.B) {
	lb := NewLoadBalancer(LoadBalancerConfig{Strategy: LeastConnections})

	// Add test servers
	for i := 0; i < 10; i++ {
		serverURL, _ := url.Parse(fmt.Sprintf("http://localhost:900%d", i))
		server := &BackendServer{
			ID:           fmt.Sprintf("server%d", i),
			URL:          serverURL,
			HealthStatus: Healthy,
		}
		atomic.StoreInt64(&server.ActiveConnections, int64(i))
		lb.AddServer(server)
	}

	req := httptest.NewRequest("GET", "/", nil)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := lb.selectServer(req)
			if err != nil {
				b.Fatalf("Server selection failed: %v", err)
			}
		}
	})
}

func BenchmarkConsistentHashing(b *testing.B) {
	chr := NewConsistentHashRing(100)

	// Add servers
	for i := 0; i < 10; i++ {
		serverURL, _ := url.Parse(fmt.Sprintf("http://localhost:900%d", i))
		server := &BackendServer{
			ID:  fmt.Sprintf("server%d", i),
			URL: serverURL,
		}
		chr.AddServer(server)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := fmt.Sprintf("key-%d", i)
			chr.GetServer(key)
			i++
		}
	})
}

func BenchmarkCircuitBreaker(b *testing.B) {
	config := CircuitBreakerConfig{
		FailureThreshold:  5,
		RecoveryThreshold: 3,
		Timeout:           30 * time.Second,
		HalfOpenMaxCalls:  10,
	}

	cb := NewCircuitBreaker(config)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			cb.AllowRequest()
		}
	})
}