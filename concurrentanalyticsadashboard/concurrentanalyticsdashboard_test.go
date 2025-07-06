package concurrentanalyticsdashboard

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestDefaultDashboardConfig(t *testing.T) {
	config := DefaultDashboardConfig()
	
	if config.HTTPPort != 8080 {
		t.Errorf("Expected HTTP port 8080, got %d", config.HTTPPort)
	}
	
	if config.WebSocketPort != 8081 {
		t.Errorf("Expected WebSocket port 8081, got %d", config.WebSocketPort)
	}
	
	if config.MaxConnections != 1000 {
		t.Errorf("Expected max connections 1000, got %d", config.MaxConnections)
	}
	
	if config.EventBufferSize != 100000 {
		t.Errorf("Expected event buffer size 100000, got %d", config.EventBufferSize)
	}
	
	if !config.EnableAuthentication {
		t.Error("Expected authentication to be enabled by default")
	}
	
	if !config.EnableCaching {
		t.Error("Expected caching to be enabled by default")
	}
	
	if !config.EnableMetrics {
		t.Error("Expected metrics to be enabled by default")
	}
}

func TestNewAnalyticsDashboard(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0 // Disable HTTP server for testing
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create analytics dashboard: %v", err)
	}
	
	if dashboard == nil {
		t.Fatal("Dashboard should not be nil")
	}
	
	if dashboard.eventBuffer == nil {
		t.Error("Event buffer should be initialized")
	}
	
	if dashboard.metrics == nil {
		t.Error("Metrics map should be initialized")
	}
	
	if dashboard.dashboards == nil {
		t.Error("Dashboards map should be initialized")
	}
	
	if dashboard.users == nil {
		t.Error("Users map should be initialized")
	}
	
	if dashboard.cache == nil {
		t.Error("Cache should be initialized")
	}
	
	if dashboard.statistics == nil {
		t.Error("Statistics should be initialized")
	}
}

func TestInvalidConfigurations(t *testing.T) {
	testCases := []struct {
		name   string
		config DashboardConfig
	}{
		{
			name: "Zero event buffer size",
			config: DashboardConfig{
				EventBufferSize: 0,
			},
		},
		{
			name: "Negative event buffer size",
			config: DashboardConfig{
				EventBufferSize: -1,
			},
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewAnalyticsDashboard(tc.config)
			if err == nil {
				t.Error("Expected error for invalid configuration")
			}
		})
	}
}

func TestStartStopDashboard(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0 // Use available port
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	// Test start
	err = dashboard.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	
	if !dashboard.running {
		t.Error("Dashboard should be running after start")
	}
	
	// Test double start
	err = dashboard.Start()
	if err == nil {
		t.Error("Expected error on double start")
	}
	
	// Give some time for background workers to start
	time.Sleep(100 * time.Millisecond)
	
	// Test stop
	err = dashboard.Stop()
	if err != nil {
		t.Errorf("Failed to stop dashboard: %v", err)
	}
	
	if dashboard.running {
		t.Error("Dashboard should not be running after stop")
	}
	
	// Test double stop
	err = dashboard.Stop()
	if err == nil {
		t.Error("Expected error on double stop")
	}
}

func TestTrackEvent(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	// Test tracking events
	events := []*AnalyticsEvent{
		{
			Type:   PageView,
			UserID: "user1",
			Properties: map[string]interface{}{
				"page": "/home",
				"referrer": "google.com",
			},
			Value: 1.0,
		},
		{
			Type:   UserAction,
			UserID: "user1",
			Properties: map[string]interface{}{
				"action": "click",
				"element": "button",
			},
			Value: 1.0,
		},
		{
			Type:   CustomEvent,
			UserID: "user2",
			Properties: map[string]interface{}{
				"event_name": "purchase",
				"amount": 99.99,
			},
			Value: 99.99,
		},
	}
	
	for i, event := range events {
		err := dashboard.TrackEvent(event)
		if err != nil {
			t.Errorf("Failed to track event %d: %v", i, err)
		}
		
		if event.ID == "" {
			t.Errorf("Event %d should have been assigned an ID", i)
		}
		
		if event.Timestamp.IsZero() {
			t.Errorf("Event %d should have been assigned a timestamp", i)
		}
	}
	
	// Test tracking event when stopped
	dashboard.Stop()
	
	err = dashboard.TrackEvent(&AnalyticsEvent{Type: PageView})
	if err == nil {
		t.Error("Expected error when tracking event on stopped dashboard")
	}
}

func TestCreateDashboard(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	// Test creating a dashboard
	userDashboard := &Dashboard{
		Name:   "Test Dashboard",
		UserID: "user1",
		Widgets: []*Widget{
			{
				Type:  "chart",
				Title: "Page Views",
				Query: Query{
					Metric:      "page_views",
					Aggregation: Count,
					TimeWindow:  Last1Hour,
				},
				Position: Position{X: 0, Y: 0},
				Size:     Size{Width: 4, Height: 3},
			},
			{
				Type:  "metric",
				Title: "Total Users",
				Query: Query{
					Metric:      "unique_users",
					Aggregation: UniqueCount,
					TimeWindow:  Last24Hours,
				},
				Position: Position{X: 4, Y: 0},
				Size:     Size{Width: 2, Height: 2},
			},
		},
		Filters: []Filter{
			{
				Field:    "source",
				Operator: "eq",
				Value:    "web",
			},
		},
		RefreshRate: time.Minute,
		IsPublic:    false,
	}
	
	err = dashboard.CreateDashboard(userDashboard)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	if userDashboard.ID == "" {
		t.Error("Dashboard should have been assigned an ID")
	}
	
	if userDashboard.CreatedAt.IsZero() {
		t.Error("Dashboard should have creation timestamp")
	}
	
	if userDashboard.UpdatedAt.IsZero() {
		t.Error("Dashboard should have update timestamp")
	}
	
	// Test retrieving the dashboard
	retrieved, err := dashboard.GetDashboard(userDashboard.ID)
	if err != nil {
		t.Fatalf("Failed to get dashboard: %v", err)
	}
	
	if retrieved.ID != userDashboard.ID {
		t.Error("Retrieved dashboard should match created dashboard")
	}
	
	if retrieved.Name != userDashboard.Name {
		t.Error("Dashboard name should match")
	}
	
	if len(retrieved.Widgets) != len(userDashboard.Widgets) {
		t.Error("Dashboard widgets should match")
	}
}

func TestDashboardLimits(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	config.MaxDashboardsPerUser = 2
	config.MaxWidgetsPerDashboard = 1
	
	dashboardSystem, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboardSystem.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboardSystem.Stop()
	
	userID := "test_user"
	
	// Create dashboards up to limit
	for i := 0; i < config.MaxDashboardsPerUser; i++ {
		dashboard := &Dashboard{
			Name:   fmt.Sprintf("Dashboard %d", i+1),
			UserID: userID,
			Widgets: []*Widget{
				{
					Type:  "chart",
					Title: "Test Widget",
					Query: Query{Metric: "test_metric"},
				},
			},
		}
		
		err := dashboardSystem.CreateDashboard(dashboard)
		if err != nil {
			t.Errorf("Failed to create dashboard %d: %v", i+1, err)
		}
	}
	
	// Try to create one more dashboard (should fail)
	extraDashboard := &Dashboard{
		Name:   "Extra Dashboard",
		UserID: userID,
		Widgets: []*Widget{
			{Type: "chart", Title: "Test"},
		},
	}
	
	err = dashboardSystem.CreateDashboard(extraDashboard)
	if err == nil {
		t.Error("Expected error when exceeding dashboard limit")
	}
	
	// Test widget limit
	tooManyWidgets := &Dashboard{
		Name:   "Widget Test",
		UserID: "different_user",
		Widgets: []*Widget{
			{Type: "chart", Title: "Widget 1"},
			{Type: "chart", Title: "Widget 2"}, // Exceeds limit
		},
	}
	
	err = dashboardSystem.CreateDashboard(tooManyWidgets)
	if err == nil {
		t.Error("Expected error when exceeding widget limit")
	}
}

func TestUpdateDashboard(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	// Create initial dashboard
	userDashboard := &Dashboard{
		Name:   "Original Dashboard",
		UserID: "user1",
		Widgets: []*Widget{
			{Type: "chart", Title: "Original Widget"},
		},
		IsPublic: false,
	}
	
	err = dashboard.CreateDashboard(userDashboard)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	originalUpdateTime := userDashboard.UpdatedAt
	
	// Wait a bit to ensure timestamp difference
	time.Sleep(10 * time.Millisecond)
	
	// Update dashboard
	userDashboard.Name = "Updated Dashboard"
	userDashboard.Widgets = append(userDashboard.Widgets, &Widget{
		Type:  "metric",
		Title: "New Widget",
	})
	
	err = dashboard.UpdateDashboard(userDashboard)
	if err != nil {
		t.Fatalf("Failed to update dashboard: %v", err)
	}
	
	// Verify update
	updated, err := dashboard.GetDashboard(userDashboard.ID)
	if err != nil {
		t.Fatalf("Failed to get updated dashboard: %v", err)
	}
	
	if updated.Name != "Updated Dashboard" {
		t.Error("Dashboard name should be updated")
	}
	
	if len(updated.Widgets) != 2 {
		t.Error("Dashboard should have 2 widgets after update")
	}
	
	if !updated.UpdatedAt.After(originalUpdateTime) {
		t.Error("UpdatedAt should be more recent after update")
	}
	
	// Test unauthorized update
	unauthorized := &Dashboard{
		ID:     userDashboard.ID,
		UserID: "different_user",
		Name:   "Hacked Dashboard",
	}
	
	err = dashboard.UpdateDashboard(unauthorized)
	if err == nil {
		t.Error("Expected error for unauthorized update")
	}
}

func TestCreateUser(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	// Test creating users
	users := []*User{
		{
			Username: "john_doe",
			Email:    "john@example.com",
			Role:     "admin",
			Permissions: []string{"read", "write", "admin"},
		},
		{
			Username: "jane_smith",
			Email:    "jane@example.com",
			Role:     "user",
			Permissions: []string{"read"},
		},
	}
	
	for i, user := range users {
		err := dashboard.CreateUser(user)
		if err != nil {
			t.Errorf("Failed to create user %d: %v", i, err)
		}
		
		if user.ID == "" {
			t.Errorf("User %d should have been assigned an ID", i)
		}
		
		if user.LastActive.IsZero() {
			t.Errorf("User %d should have last active timestamp", i)
		}
		
		if user.Dashboards == nil {
			t.Errorf("User %d should have initialized dashboards slice", i)
		}
	}
}

func TestConcurrentEventTracking(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	config.EventBufferSize = 10000
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	numGoroutines := 10
	eventsPerGoroutine := 100
	
	var wg sync.WaitGroup
	
	// Track events concurrently
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()
			
			for j := 0; j < eventsPerGoroutine; j++ {
				event := &AnalyticsEvent{
					Type:   PageView,
					UserID: fmt.Sprintf("user_%d", goroutineID),
					Properties: map[string]interface{}{
						"page":       fmt.Sprintf("/page_%d_%d", goroutineID, j),
						"goroutine":  goroutineID,
						"iteration":  j,
					},
					Value: float64(j),
				}
				
				err := dashboard.TrackEvent(event)
				if err != nil {
					t.Errorf("Goroutine %d: Failed to track event %d: %v", goroutineID, j, err)
				}
			}
		}(i)
	}
	
	wg.Wait()
	
	// Verify statistics
	stats := dashboard.GetStatistics()
	expectedEvents := int64(numGoroutines * eventsPerGoroutine)
	
	if stats.EventsProcessed < expectedEvents {
		t.Errorf("Expected at least %d events processed, got %d", expectedEvents, stats.EventsProcessed)
	}
}

func TestQueryMetrics(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	config.EnableCaching = true
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	// Test different query types
	queries := []Query{
		{
			Metric:      "page_views",
			Aggregation: Count,
			TimeWindow:  Last1Hour,
			Filters: []Filter{
				{Field: "source", Operator: "eq", Value: "web"},
			},
		},
		{
			Metric:      "revenue",
			Aggregation: Sum,
			TimeWindow:  Last24Hours,
			GroupBy:     []string{"product_category"},
			OrderBy:     "value",
			Limit:       10,
		},
		{
			Metric:      "response_time",
			Aggregation: Percentile95,
			TimeWindow:  Last15Minutes,
		},
		{
			Metric:      "unique_users",
			Aggregation: UniqueCount,
			TimeWindow:  Last7Days,
		},
	}
	
	for i, query := range queries {
		metrics, err := dashboard.QueryMetrics(query)
		if err != nil {
			t.Errorf("Query %d failed: %v", i, err)
			continue
		}
		
		if metrics == nil {
			t.Errorf("Query %d should return non-nil metrics", i)
		}
		
		// Query again to test caching
		cachedMetrics, err := dashboard.QueryMetrics(query)
		if err != nil {
			t.Errorf("Cached query %d failed: %v", i, err)
		}
		
		if cachedMetrics == nil {
			t.Errorf("Cached query %d should return non-nil metrics", i)
		}
	}
	
	// Verify query statistics
	stats := dashboard.GetStatistics()
	if stats.QueriesExecuted == 0 {
		t.Error("Should have executed some queries")
	}
}

func TestMetricCache(t *testing.T) {
	cache := &MetricCache{
		data:    make(map[string]*CacheEntry),
		maxSize: 3,
	}
	
	// Test setting and getting values
	cache.Set("key1", "value1", time.Minute)
	cache.Set("key2", "value2", time.Minute)
	cache.Set("key3", "value3", time.Minute)
	
	if val := cache.Get("key1"); val != "value1" {
		t.Errorf("Expected 'value1', got %v", val)
	}
	
	if val := cache.Get("key2"); val != "value2" {
		t.Errorf("Expected 'value2', got %v", val)
	}
	
	if val := cache.Get("key3"); val != "value3" {
		t.Errorf("Expected 'value3', got %v", val)
	}
	
	// Test cache eviction
	cache.Set("key4", "value4", time.Minute)
	
	if len(cache.data) > cache.maxSize {
		t.Errorf("Cache size %d exceeds max size %d", len(cache.data), cache.maxSize)
	}
	
	// Test expiration
	cache.Set("expiring", "expires_soon", time.Millisecond)
	time.Sleep(2 * time.Millisecond)
	
	if val := cache.Get("expiring"); val != nil {
		t.Error("Expired entry should return nil")
	}
	
	// Test non-existent key
	if val := cache.Get("nonexistent"); val != nil {
		t.Error("Non-existent key should return nil")
	}
}

func TestHTTPEndpoints(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	// Test event creation endpoint
	t.Run("POST /api/events", func(t *testing.T) {
		event := &AnalyticsEvent{
			Type:   PageView,
			UserID: "test_user",
			Properties: map[string]interface{}{
				"page": "/test",
			},
			Value: 1.0,
		}
		
		body, _ := json.Marshal(event)
		req := httptest.NewRequest(http.MethodPost, "/api/events", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		
		dashboard.handleEvents(w, req)
		
		if w.Code != http.StatusCreated {
			t.Errorf("Expected status %d, got %d", http.StatusCreated, w.Code)
		}
		
		var response map[string]string
		json.Unmarshal(w.Body.Bytes(), &response)
		
		if response["status"] != "created" {
			t.Error("Expected status 'created' in response")
		}
		
		if response["id"] == "" {
			t.Error("Expected event ID in response")
		}
	})
	
	// Test dashboard creation endpoint
	t.Run("POST /api/dashboards", func(t *testing.T) {
		dashboardData := &Dashboard{
			Name:   "API Test Dashboard",
			UserID: "api_user",
			Widgets: []*Widget{
				{
					Type:  "chart",
					Title: "API Test Widget",
					Query: Query{
						Metric:      "api_calls",
						Aggregation: Count,
						TimeWindow:  Last1Hour,
					},
				},
			},
		}
		
		body, _ := json.Marshal(dashboardData)
		req := httptest.NewRequest(http.MethodPost, "/api/dashboards", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		
		dashboard.handleDashboards(w, req)
		
		if w.Code != http.StatusCreated {
			t.Errorf("Expected status %d, got %d", http.StatusCreated, w.Code)
		}
		
		var response Dashboard
		json.Unmarshal(w.Body.Bytes(), &response)
		
		if response.ID == "" {
			t.Error("Expected dashboard ID in response")
		}
		
		if response.Name != dashboardData.Name {
			t.Error("Dashboard name should match")
		}
	})
	
	// Test metrics query endpoint
	t.Run("POST /api/metrics", func(t *testing.T) {
		query := Query{
			Metric:      "test_metric",
			Aggregation: Count,
			TimeWindow:  Last1Hour,
		}
		
		body, _ := json.Marshal(query)
		req := httptest.NewRequest(http.MethodPost, "/api/metrics", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		
		dashboard.handleMetrics(w, req)
		
		if w.Code != http.StatusOK {
			t.Errorf("Expected status %d, got %d", http.StatusOK, w.Code)
		}
		
		var metrics []Metric
		json.Unmarshal(w.Body.Bytes(), &metrics)
		
		// Should return empty array for test
		if metrics == nil {
			t.Error("Expected metrics array in response")
		}
	})
	
	// Test statistics endpoint
	t.Run("GET /api/statistics", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/statistics", nil)
		w := httptest.NewRecorder()
		
		dashboard.handleStatistics(w, req)
		
		if w.Code != http.StatusOK {
			t.Errorf("Expected status %d, got %d", http.StatusOK, w.Code)
		}
		
		var stats DashboardStatistics
		json.Unmarshal(w.Body.Bytes(), &stats)
		
		if stats.StartTime.IsZero() {
			t.Error("Expected start time in statistics")
		}
	})
	
	// Test invalid JSON
	t.Run("Invalid JSON", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/events", strings.NewReader("invalid json"))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		
		dashboard.handleEvents(w, req)
		
		if w.Code != http.StatusBadRequest {
			t.Errorf("Expected status %d for invalid JSON, got %d", http.StatusBadRequest, w.Code)
		}
	})
	
	// Test method not allowed
	t.Run("Method Not Allowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodDelete, "/api/events", nil)
		w := httptest.NewRecorder()
		
		dashboard.handleEvents(w, req)
		
		if w.Code != http.StatusMethodNotAllowed {
			t.Errorf("Expected status %d for invalid method, got %d", http.StatusMethodNotAllowed, w.Code)
		}
	})
}

func TestWebSocketConnection(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	config.WebSocketPort = 0
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	// Test WebSocket connection creation
	conn := &WebSocketConnection{
		ID:         "test_conn",
		LastPing:   time.Now(),
		SendQueue:  make(chan []byte, 10),
		Subscriptions: make([]string, 0),
	}
	
	if conn.ID != "test_conn" {
		t.Error("Connection ID should match")
	}
	
	if conn.SendQueue == nil {
		t.Error("Send queue should be initialized")
	}
	
	if conn.Subscriptions == nil {
		t.Error("Subscriptions should be initialized")
	}
	
	// Test WebSocket message handling
	testMessage := map[string]interface{}{
		"type":      "subscribe",
		"dashboard": "test_dashboard",
	}
	
	dashboard.handleWebSocketMessage(conn, testMessage)
	
	if conn.Dashboard != "test_dashboard" {
		t.Error("Dashboard subscription should be set")
	}
	
	if len(conn.Subscriptions) != 1 || conn.Subscriptions[0] != "test_dashboard" {
		t.Error("Subscription should be added")
	}
	
	// Test unsubscribe
	unsubscribeMessage := map[string]interface{}{
		"type":      "unsubscribe",
		"dashboard": "test_dashboard",
	}
	
	dashboard.handleWebSocketMessage(conn, unsubscribeMessage)
	
	if len(conn.Subscriptions) != 0 {
		t.Error("Subscription should be removed")
	}
}

func TestBroadcastToAll(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	// Create test connections
	numConnections := 3
	connections := make([]*WebSocketConnection, numConnections)
	
	for i := 0; i < numConnections; i++ {
		conn := &WebSocketConnection{
			ID:        fmt.Sprintf("conn_%d", i),
			SendQueue: make(chan []byte, 10),
		}
		connections[i] = conn
		dashboard.connections[conn.ID] = conn
	}
	
	// Broadcast message
	message := map[string]interface{}{
		"type": "test",
		"data": "broadcast test",
	}
	
	dashboard.BroadcastToAll(message)
	
	// Verify all connections received the message
	for i, conn := range connections {
		select {
		case data := <-conn.SendQueue:
			var received map[string]interface{}
			json.Unmarshal(data, &received)
			
			if received["type"] != "test" {
				t.Errorf("Connection %d: Expected type 'test', got %v", i, received["type"])
			}
			
			if received["data"] != "broadcast test" {
				t.Errorf("Connection %d: Expected data 'broadcast test', got %v", i, received["data"])
			}
		case <-time.After(100 * time.Millisecond):
			t.Errorf("Connection %d: Did not receive broadcast message", i)
		}
	}
}

func TestGetStatistics(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		t.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	// Track some events to generate statistics
	for i := 0; i < 10; i++ {
		event := &AnalyticsEvent{
			Type:   PageView,
			UserID: fmt.Sprintf("user_%d", i),
			Value:  float64(i),
		}
		dashboard.TrackEvent(event)
	}
	
	// Create some users and dashboards
	for i := 0; i < 5; i++ {
		user := &User{
			Username: fmt.Sprintf("user_%d", i),
			Email:    fmt.Sprintf("user%d@example.com", i),
		}
		dashboard.CreateUser(user)
		
		userDashboard := &Dashboard{
			Name:   fmt.Sprintf("Dashboard %d", i),
			UserID: user.ID,
			Widgets: []*Widget{
				{Type: "chart", Title: fmt.Sprintf("Widget %d", i)},
			},
		}
		dashboard.CreateDashboard(userDashboard)
	}
	
	// Get statistics
	stats := dashboard.GetStatistics()
	
	if stats.StartTime.IsZero() {
		t.Error("Start time should be set")
	}
	
	if stats.EventsProcessed < 10 {
		t.Errorf("Expected at least 10 events processed, got %d", stats.EventsProcessed)
	}
	
	if stats.ActiveUsers < 5 {
		t.Errorf("Expected at least 5 active users, got %d", stats.ActiveUsers)
	}
	
	if stats.ActiveDashboards < 5 {
		t.Errorf("Expected at least 5 active dashboards, got %d", stats.ActiveDashboards)
	}
	
	// Test cache hit rate calculation
	dashboard.cache.hits = 80
	dashboard.cache.misses = 20
	
	stats = dashboard.GetStatistics()
	expectedHitRate := 80.0 / 100.0
	
	if stats.CacheHitRate != expectedHitRate {
		t.Errorf("Expected cache hit rate %f, got %f", expectedHitRate, stats.CacheHitRate)
	}
}

func TestEventGeneration(t *testing.T) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		t.Fatalf("Failed to create dashboard: %v", err)
	}
	
	// Test ID generation
	id1 := dashboard.generateEventID()
	id2 := dashboard.generateEventID()
	
	if id1 == "" {
		t.Error("Event ID should not be empty")
	}
	
	if id1 == id2 {
		t.Error("Event IDs should be unique")
	}
	
	if !strings.HasPrefix(id1, "event_") {
		t.Error("Event ID should have 'event_' prefix")
	}
	
	// Test dashboard ID generation
	dashboardID1 := dashboard.generateDashboardID()
	dashboardID2 := dashboard.generateDashboardID()
	
	if dashboardID1 == dashboardID2 {
		t.Error("Dashboard IDs should be unique")
	}
	
	if !strings.HasPrefix(dashboardID1, "dashboard_") {
		t.Error("Dashboard ID should have 'dashboard_' prefix")
	}
	
	// Test user ID generation
	userID := dashboard.generateUserID()
	if !strings.HasPrefix(userID, "user_") {
		t.Error("User ID should have 'user_' prefix")
	}
	
	// Test connection ID generation
	connID := dashboard.generateConnectionID()
	if !strings.HasPrefix(connID, "conn_") {
		t.Error("Connection ID should have 'conn_' prefix")
	}
	
	// Test cache key generation
	query := Query{
		Metric:      "test_metric",
		Aggregation: Count,
		TimeWindow:  Last1Hour,
	}
	
	cacheKey := dashboard.generateCacheKey(query)
	expectedKey := fmt.Sprintf("query_%s_%d_%d", query.Metric, query.Aggregation, query.TimeWindow)
	
	if cacheKey != expectedKey {
		t.Errorf("Expected cache key '%s', got '%s'", expectedKey, cacheKey)
	}
}

// Benchmark tests

func BenchmarkTrackEvent(b *testing.B) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	config.EventBufferSize = 1000000
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		b.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		b.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	event := &AnalyticsEvent{
		Type:   PageView,
		UserID: "benchmark_user",
		Properties: map[string]interface{}{
			"page": "/benchmark",
		},
		Value: 1.0,
	}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			dashboard.TrackEvent(event)
		}
	})
}

func BenchmarkQueryMetrics(b *testing.B) {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	config.EnableCaching = true
	
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		b.Fatalf("Failed to create dashboard: %v", err)
	}
	
	err = dashboard.Start()
	if err != nil {
		b.Fatalf("Failed to start dashboard: %v", err)
	}
	defer dashboard.Stop()
	
	query := Query{
		Metric:      "benchmark_metric",
		Aggregation: Count,
		TimeWindow:  Last1Hour,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dashboard.QueryMetrics(query)
	}
}

func BenchmarkCacheOperations(b *testing.B) {
	cache := &MetricCache{
		data:    make(map[string]*CacheEntry),
		maxSize: 10000,
	}
	
	// Pre-populate cache
	for i := 0; i < 1000; i++ {
		key := fmt.Sprintf("key_%d", i)
		cache.Set(key, fmt.Sprintf("value_%d", i), time.Hour)
	}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := fmt.Sprintf("key_%d", i%1000)
			cache.Get(key)
			i++
		}
	})
}

// Example functions

func ExampleNewAnalyticsDashboard() {
	// Create configuration
	config := DefaultDashboardConfig()
	config.HTTPPort = 8080
	config.MaxConnections = 1000
	config.EventBufferSize = 50000
	config.EnableCaching = true
	
	// Create dashboard
	dashboard, err := NewAnalyticsDashboard(config)
	if err != nil {
		fmt.Printf("Failed to create dashboard: %v\n", err)
		return
	}
	
	// Start the dashboard
	err = dashboard.Start()
	if err != nil {
		fmt.Printf("Failed to start dashboard: %v\n", err)
		return
	}
	defer dashboard.Stop()
	
	fmt.Printf("Analytics dashboard started on port %d\n", config.HTTPPort)
	
	// Output:
	// Analytics dashboard started on port 8080
}

func ExampleAnalyticsDashboard_TrackEvent() {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0 // Disable HTTP for example
	
	dashboard, _ := NewAnalyticsDashboard(config)
	dashboard.Start()
	defer dashboard.Stop()
	
	// Track different types of events
	events := []*AnalyticsEvent{
		{
			Type:   PageView,
			UserID: "user123",
			Properties: map[string]interface{}{
				"page":     "/home",
				"referrer": "google.com",
				"device":   "mobile",
			},
			Value: 1.0,
		},
		{
			Type:   UserAction,
			UserID: "user123",
			Properties: map[string]interface{}{
				"action": "click",
				"element": "signup_button",
			},
			Value: 1.0,
		},
		{
			Type:   CustomEvent,
			UserID: "user456",
			Properties: map[string]interface{}{
				"event_name": "purchase",
				"product_id": "prod_123",
				"amount":     99.99,
			},
			Value: 99.99,
		},
	}
	
	for _, event := range events {
		err := dashboard.TrackEvent(event)
		if err != nil {
			fmt.Printf("Failed to track event: %v\n", err)
		} else {
			fmt.Printf("Tracked event: %s (ID: %s)\n", 
				[]string{"PageView", "UserAction", "CustomEvent"}[event.Type], event.ID)
		}
	}
	
	// Output:
	// Tracked event: PageView (ID: event_1234567890_1)
	// Tracked event: UserAction (ID: event_1234567890_2)
	// Tracked event: CustomEvent (ID: event_1234567890_3)
}

func ExampleAnalyticsDashboard_CreateDashboard() {
	config := DefaultDashboardConfig()
	config.HTTPPort = 0
	
	dashboard, _ := NewAnalyticsDashboard(config)
	dashboard.Start()
	defer dashboard.Stop()
	
	// Create a user first
	user := &User{
		Username: "data_analyst",
		Email:    "analyst@company.com",
		Role:     "analyst",
		Permissions: []string{"read", "write"},
	}
	dashboard.CreateUser(user)
	
	// Create a comprehensive dashboard
	userDashboard := &Dashboard{
		Name:   "Website Analytics Dashboard",
		UserID: user.ID,
		Widgets: []*Widget{
			{
				Type:  "chart",
				Title: "Page Views Over Time",
				Query: Query{
					Metric:      "page_views",
					Aggregation: Count,
					TimeWindow:  Last24Hours,
				},
				Position: Position{X: 0, Y: 0},
				Size:     Size{Width: 6, Height: 4},
			},
			{
				Type:  "metric",
				Title: "Total Unique Users",
				Query: Query{
					Metric:      "unique_users",
					Aggregation: UniqueCount,
					TimeWindow:  Last24Hours,
				},
				Position: Position{X: 6, Y: 0},
				Size:     Size{Width: 3, Height: 2},
			},
			{
				Type:  "table",
				Title: "Top Pages",
				Query: Query{
					Metric:      "page_views",
					Aggregation: Count,
					TimeWindow:  Last24Hours,
					GroupBy:     []string{"page"},
					OrderBy:     "value",
					Limit:       10,
				},
				Position: Position{X: 0, Y: 4},
				Size:     Size{Width: 9, Height: 3},
			},
		},
		Filters: []Filter{
			{
				Field:    "source",
				Operator: "eq",
				Value:    "web",
			},
		},
		RefreshRate: 30 * time.Second,
		IsPublic:    false,
	}
	
	err := dashboard.CreateDashboard(userDashboard)
	if err != nil {
		fmt.Printf("Failed to create dashboard: %v\n", err)
		return
	}
	
	fmt.Printf("Created dashboard: %s (ID: %s)\n", userDashboard.Name, userDashboard.ID)
	fmt.Printf("Dashboard has %d widgets\n", len(userDashboard.Widgets))
	fmt.Printf("Refresh rate: %v\n", userDashboard.RefreshRate)
	
	// Output:
	// Created dashboard: Website Analytics Dashboard (ID: dashboard_1234567890)
	// Dashboard has 3 widgets
	// Refresh rate: 30s
}