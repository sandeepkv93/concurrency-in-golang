package concurrentanalyticsdashboard

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// EventType defines different types of analytics events
type EventType int

const (
	PageView EventType = iota
	UserAction
	CustomEvent
	MetricUpdate
	AlertEvent
	SystemEvent
	ErrorEvent
	PerformanceEvent
)

// AggregationType defines how metrics are aggregated
type AggregationType int

const (
	Sum AggregationType = iota
	Average
	Count
	Max
	Min
	Percentile95
	Percentile99
	UniqueCount
	Rate
	Histogram
)

// TimeWindow defines time window for analytics
type TimeWindow int

const (
	RealTime TimeWindow = iota
	Last1Minute
	Last5Minutes
	Last15Minutes
	Last1Hour
	Last24Hours
	Last7Days
	Last30Days
	Custom
)

// AnalyticsEvent represents a single analytics event
type AnalyticsEvent struct {
	ID         string                 `json:"id"`
	Type       EventType              `json:"type"`
	Timestamp  time.Time              `json:"timestamp"`
	UserID     string                 `json:"user_id,omitempty"`
	SessionID  string                 `json:"session_id,omitempty"`
	Properties map[string]interface{} `json:"properties"`
	Value      float64                `json:"value,omitempty"`
	Tags       []string               `json:"tags,omitempty"`
	Source     string                 `json:"source,omitempty"`
}

// Metric represents a calculated metric
type Metric struct {
	Name         string          `json:"name"`
	Value        float64         `json:"value"`
	Aggregation  AggregationType `json:"aggregation"`
	TimeWindow   TimeWindow      `json:"time_window"`
	Timestamp    time.Time       `json:"timestamp"`
	Dimensions   map[string]string `json:"dimensions,omitempty"`
	Count        int64           `json:"count,omitempty"`
	Tags         []string        `json:"tags,omitempty"`
}

// Dashboard represents a user dashboard configuration
type Dashboard struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	UserID      string    `json:"user_id"`
	Widgets     []*Widget `json:"widgets"`
	Filters     []Filter  `json:"filters"`
	RefreshRate time.Duration `json:"refresh_rate"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	IsPublic    bool      `json:"is_public"`
	mutex       sync.RWMutex
}

// Widget represents a dashboard widget
type Widget struct {
	ID            string          `json:"id"`
	Type          string          `json:"type"` // chart, table, metric, alert
	Title         string          `json:"title"`
	Query         Query           `json:"query"`
	Position      Position        `json:"position"`
	Size          Size            `json:"size"`
	Configuration map[string]interface{} `json:"configuration"`
	LastUpdate    time.Time       `json:"last_update"`
	Data          interface{}     `json:"data,omitempty"`
}

// Query represents a metric query
type Query struct {
	Metric      string            `json:"metric"`
	Aggregation AggregationType   `json:"aggregation"`
	TimeWindow  TimeWindow        `json:"time_window"`
	Filters     []Filter          `json:"filters"`
	GroupBy     []string          `json:"group_by"`
	OrderBy     string            `json:"order_by"`
	Limit       int               `json:"limit"`
	CustomTime  *TimeRange        `json:"custom_time,omitempty"`
}

// Filter represents a query filter
type Filter struct {
	Field    string      `json:"field"`
	Operator string      `json:"operator"` // eq, neq, gt, lt, gte, lte, in, contains
	Value    interface{} `json:"value"`
}

// Position represents widget position
type Position struct {
	X int `json:"x"`
	Y int `json:"y"`
}

// Size represents widget size
type Size struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// TimeRange represents a custom time range
type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// User represents a dashboard user
type User struct {
	ID          string    `json:"id"`
	Username    string    `json:"username"`
	Email       string    `json:"email"`
	Role        string    `json:"role"`
	Permissions []string  `json:"permissions"`
	LastActive  time.Time `json:"last_active"`
	Dashboards  []string  `json:"dashboards"`
	mutex       sync.RWMutex
}

// WebSocketConnection represents a WebSocket connection
type WebSocketConnection struct {
	ID         string
	User       *User
	Conn       *websocket.Conn
	Dashboard  string
	Subscriptions []string
	LastPing   time.Time
	SendQueue  chan []byte
	mutex      sync.RWMutex
}

// DashboardConfig contains configuration for the analytics dashboard
type DashboardConfig struct {
	HTTPPort              int
	WebSocketPort         int
	MaxConnections        int
	MaxEventsPerSecond    int
	EventBufferSize       int
	MetricRetentionPeriod time.Duration
	WebSocketPingInterval time.Duration
	WebSocketPongTimeout  time.Duration
	WorkerPoolSize        int
	EnableAuthentication  bool
	EnableRateLimiting    bool
	EnableCaching         bool
	CacheSize             int
	DatabaseURL           string
	RedisURL              string
	EnableLogging         bool
	LogLevel              string
	EnableMetrics         bool
	MaxDashboardsPerUser  int
	MaxWidgetsPerDashboard int
	EnableAlerts          bool
	AlertCheckInterval    time.Duration
}

// DefaultDashboardConfig returns default configuration
func DefaultDashboardConfig() DashboardConfig {
	return DashboardConfig{
		HTTPPort:              8080,
		WebSocketPort:         8081,
		MaxConnections:        1000,
		MaxEventsPerSecond:    10000,
		EventBufferSize:       100000,
		MetricRetentionPeriod: 30 * 24 * time.Hour, // 30 days
		WebSocketPingInterval: 30 * time.Second,
		WebSocketPongTimeout:  60 * time.Second,
		WorkerPoolSize:        runtime.NumCPU() * 2,
		EnableAuthentication:  true,
		EnableRateLimiting:    true,
		EnableCaching:         true,
		CacheSize:             10000,
		EnableLogging:         true,
		LogLevel:              "INFO",
		EnableMetrics:         true,
		MaxDashboardsPerUser:  10,
		MaxWidgetsPerDashboard: 20,
		EnableAlerts:          true,
		AlertCheckInterval:    10 * time.Second,
	}
}

// Analytics Dashboard System
type AnalyticsDashboard struct {
	config        DashboardConfig
	eventBuffer   chan *AnalyticsEvent
	metrics       map[string]*MetricStore
	dashboards    map[string]*Dashboard
	users         map[string]*User
	connections   map[string]*WebSocketConnection
	aggregators   map[string]*MetricAggregator
	alertRules    map[string]*AlertRule
	cache         *MetricCache
	statistics    *DashboardStatistics
	httpServer    *http.Server
	wsUpgrader    websocket.Upgrader
	running       bool
	ctx           context.Context
	cancel        context.CancelFunc
	workerPool    chan struct{}
	mutex         sync.RWMutex
	connMutex     sync.RWMutex
	wg            sync.WaitGroup
}

// MetricStore stores time-series metric data
type MetricStore struct {
	Name       string
	DataPoints []DataPoint
	Aggregated map[TimeWindow]map[AggregationType]float64
	mutex      sync.RWMutex
}

// DataPoint represents a single metric data point
type DataPoint struct {
	Timestamp  time.Time
	Value      float64
	Dimensions map[string]string
	Tags       []string
}

// MetricAggregator aggregates metrics in real-time
type MetricAggregator struct {
	Name        string
	Aggregation AggregationType
	TimeWindow  TimeWindow
	Values      []float64
	LastUpdate  time.Time
	mutex       sync.RWMutex
}

// AlertRule defines an alert condition
type AlertRule struct {
	ID           string
	Name         string
	Metric       string
	Condition    string // gt, lt, eq, etc.
	Threshold    float64
	TimeWindow   TimeWindow
	Enabled      bool
	LastTriggered time.Time
	Actions      []AlertAction
}

// AlertAction defines what happens when alert triggers
type AlertAction struct {
	Type   string // email, webhook, notification
	Target string
	Config map[string]interface{}
}

// MetricCache provides caching for frequently accessed metrics
type MetricCache struct {
	data   map[string]*CacheEntry
	mutex  sync.RWMutex
	maxSize int
	hits    int64
	misses  int64
}

// CacheEntry represents a cached metric
type CacheEntry struct {
	Key        string
	Value      interface{}
	Expiry     time.Time
	AccessTime time.Time
}

// DashboardStatistics tracks system performance
type DashboardStatistics struct {
	StartTime           time.Time
	EventsProcessed     int64
	MetricsCalculated   int64
	ActiveConnections   int64
	ActiveDashboards    int64
	ActiveUsers         int64
	CacheHitRate        float64
	AverageResponseTime time.Duration
	ErrorCount          int64
	AlertsTriggered     int64
	DataPointsStored    int64
	QueriesExecuted     int64
	mutex               sync.RWMutex
}

// NewAnalyticsDashboard creates a new analytics dashboard
func NewAnalyticsDashboard(config DashboardConfig) (*AnalyticsDashboard, error) {
	if config.EventBufferSize <= 0 {
		return nil, errors.New("event buffer size must be positive")
	}
	
	if config.WorkerPoolSize <= 0 {
		config.WorkerPoolSize = runtime.NumCPU()
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	dashboard := &AnalyticsDashboard{
		config:      config,
		eventBuffer: make(chan *AnalyticsEvent, config.EventBufferSize),
		metrics:     make(map[string]*MetricStore),
		dashboards:  make(map[string]*Dashboard),
		users:       make(map[string]*User),
		connections: make(map[string]*WebSocketConnection),
		aggregators: make(map[string]*MetricAggregator),
		alertRules:  make(map[string]*AlertRule),
		cache: &MetricCache{
			data:    make(map[string]*CacheEntry),
			maxSize: config.CacheSize,
		},
		statistics: &DashboardStatistics{
			StartTime: time.Now(),
		},
		workerPool: make(chan struct{}, config.WorkerPoolSize),
		running:    false,
		ctx:        ctx,
		cancel:     cancel,
		wsUpgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // In production, implement proper origin checking
			},
		},
	}
	
	// Initialize HTTP server
	dashboard.setupHTTPHandlers()
	
	return dashboard, nil
}

// Start starts the analytics dashboard
func (ad *AnalyticsDashboard) Start() error {
	ad.mutex.Lock()
	defer ad.mutex.Unlock()
	
	if ad.running {
		return errors.New("dashboard is already running")
	}
	
	ad.running = true
	
	// Start background workers
	ad.wg.Add(5)
	go ad.eventProcessor()
	go ad.metricAggregator()
	go ad.connectionManager()
	go ad.alertManager()
	go ad.cleanupManager()
	
	// Start HTTP server
	ad.httpServer = &http.Server{
		Addr: fmt.Sprintf(":%d", ad.config.HTTPPort),
	}
	
	go func() {
		if err := ad.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()
	
	if ad.config.EnableLogging {
		log.Printf("Analytics dashboard started on port %d", ad.config.HTTPPort)
	}
	
	return nil
}

// Stop stops the analytics dashboard
func (ad *AnalyticsDashboard) Stop() error {
	ad.mutex.Lock()
	defer ad.mutex.Unlock()
	
	if !ad.running {
		return errors.New("dashboard is not running")
	}
	
	ad.running = false
	ad.cancel()
	
	// Close all WebSocket connections
	ad.connMutex.Lock()
	for _, conn := range ad.connections {
		conn.Conn.Close()
	}
	ad.connMutex.Unlock()
	
	// Stop HTTP server
	if ad.httpServer != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		ad.httpServer.Shutdown(ctx)
	}
	
	// Wait for workers to finish
	ad.wg.Wait()
	
	if ad.config.EnableLogging {
		log.Printf("Analytics dashboard stopped")
	}
	
	return nil
}

// TrackEvent adds an analytics event to the processing queue
func (ad *AnalyticsDashboard) TrackEvent(event *AnalyticsEvent) error {
	if !ad.running {
		return errors.New("dashboard is not running")
	}
	
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}
	
	if event.ID == "" {
		event.ID = ad.generateEventID()
	}
	
	select {
	case ad.eventBuffer <- event:
		atomic.AddInt64(&ad.statistics.EventsProcessed, 1)
		return nil
	default:
		atomic.AddInt64(&ad.statistics.ErrorCount, 1)
		return errors.New("event buffer is full")
	}
}

// CreateDashboard creates a new dashboard
func (ad *AnalyticsDashboard) CreateDashboard(dashboard *Dashboard) error {
	if dashboard.ID == "" {
		dashboard.ID = ad.generateDashboardID()
	}
	
	if dashboard.CreatedAt.IsZero() {
		dashboard.CreatedAt = time.Now()
	}
	dashboard.UpdatedAt = time.Now()
	
	ad.mutex.Lock()
	defer ad.mutex.Unlock()
	
	// Check user dashboard limit
	userDashboards := 0
	for _, d := range ad.dashboards {
		if d.UserID == dashboard.UserID {
			userDashboards++
		}
	}
	
	if userDashboards >= ad.config.MaxDashboardsPerUser {
		return fmt.Errorf("user has reached maximum dashboard limit (%d)", ad.config.MaxDashboardsPerUser)
	}
	
	// Validate widgets
	if len(dashboard.Widgets) > ad.config.MaxWidgetsPerDashboard {
		return fmt.Errorf("dashboard has too many widgets (max: %d)", ad.config.MaxWidgetsPerDashboard)
	}
	
	ad.dashboards[dashboard.ID] = dashboard
	atomic.AddInt64(&ad.statistics.ActiveDashboards, 1)
	
	return nil
}

// GetDashboard retrieves a dashboard by ID
func (ad *AnalyticsDashboard) GetDashboard(dashboardID string) (*Dashboard, error) {
	ad.mutex.RLock()
	defer ad.mutex.RUnlock()
	
	dashboard, exists := ad.dashboards[dashboardID]
	if !exists {
		return nil, errors.New("dashboard not found")
	}
	
	return dashboard, nil
}

// UpdateDashboard updates a dashboard
func (ad *AnalyticsDashboard) UpdateDashboard(dashboard *Dashboard) error {
	ad.mutex.Lock()
	defer ad.mutex.Unlock()
	
	existing, exists := ad.dashboards[dashboard.ID]
	if !exists {
		return errors.New("dashboard not found")
	}
	
	// Check permissions
	if existing.UserID != dashboard.UserID && !existing.IsPublic {
		return errors.New("unauthorized to update dashboard")
	}
	
	dashboard.UpdatedAt = time.Now()
	ad.dashboards[dashboard.ID] = dashboard
	
	// Notify connected clients
	ad.broadcastDashboardUpdate(dashboard.ID)
	
	return nil
}

// QueryMetrics executes a metric query
func (ad *AnalyticsDashboard) QueryMetrics(query Query) ([]Metric, error) {
	atomic.AddInt64(&ad.statistics.QueriesExecuted, 1)
	
	// Check cache first
	if ad.config.EnableCaching {
		cacheKey := ad.generateCacheKey(query)
		if cached := ad.cache.Get(cacheKey); cached != nil {
			atomic.AddInt64(&ad.cache.hits, 1)
			return cached.([]Metric), nil
		}
		atomic.AddInt64(&ad.cache.misses, 1)
	}
	
	// Execute query
	results, err := ad.executeQuery(query)
	if err != nil {
		atomic.AddInt64(&ad.statistics.ErrorCount, 1)
		return nil, err
	}
	
	// Cache results
	if ad.config.EnableCaching {
		cacheKey := ad.generateCacheKey(query)
		ad.cache.Set(cacheKey, results, time.Minute*5)
	}
	
	return results, nil
}

// CreateUser creates a new user
func (ad *AnalyticsDashboard) CreateUser(user *User) error {
	if user.ID == "" {
		user.ID = ad.generateUserID()
	}
	
	user.LastActive = time.Now()
	user.Dashboards = make([]string, 0)
	
	ad.mutex.Lock()
	defer ad.mutex.Unlock()
	
	ad.users[user.ID] = user
	atomic.AddInt64(&ad.statistics.ActiveUsers, 1)
	
	return nil
}

// Background worker functions

// eventProcessor processes incoming analytics events
func (ad *AnalyticsDashboard) eventProcessor() {
	defer ad.wg.Done()
	
	for {
		select {
		case <-ad.ctx.Done():
			return
		case event := <-ad.eventBuffer:
			ad.processEvent(event)
		}
	}
}

// processEvent processes a single analytics event
func (ad *AnalyticsDashboard) processEvent(event *AnalyticsEvent) {
	// Acquire worker
	ad.workerPool <- struct{}{}
	defer func() { <-ad.workerPool }()
	
	// Extract metrics from event
	metrics := ad.extractMetricsFromEvent(event)
	
	for _, metric := range metrics {
		ad.storeMetric(metric)
		ad.updateAggregators(metric)
	}
	
	// Broadcast real-time updates
	ad.broadcastEventUpdate(event)
	
	atomic.AddInt64(&ad.statistics.MetricsCalculated, int64(len(metrics)))
}

// metricAggregator runs real-time metric aggregation
func (ad *AnalyticsDashboard) metricAggregator() {
	defer ad.wg.Done()
	
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ad.ctx.Done():
			return
		case <-ticker.C:
			ad.runAggregation()
		}
	}
}

// connectionManager manages WebSocket connections
func (ad *AnalyticsDashboard) connectionManager() {
	defer ad.wg.Done()
	
	ticker := time.NewTicker(ad.config.WebSocketPingInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ad.ctx.Done():
			return
		case <-ticker.C:
			ad.pingConnections()
			ad.cleanupStaleConnections()
		}
	}
}

// alertManager checks alert conditions
func (ad *AnalyticsDashboard) alertManager() {
	defer ad.wg.Done()
	
	ticker := time.NewTicker(ad.config.AlertCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ad.ctx.Done():
			return
		case <-ticker.C:
			if ad.config.EnableAlerts {
				ad.checkAlerts()
			}
		}
	}
}

// cleanupManager performs periodic cleanup
func (ad *AnalyticsDashboard) cleanupManager() {
	defer ad.wg.Done()
	
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()
	
	for {
		select {
		case <-ad.ctx.Done():
			return
		case <-ticker.C:
			ad.cleanupOldData()
			ad.cleanupCache()
			ad.updateStatistics()
		}
	}
}

// HTTP handlers setup
func (ad *AnalyticsDashboard) setupHTTPHandlers() {
	mux := http.NewServeMux()
	
	// API endpoints
	mux.HandleFunc("/api/events", ad.handleEvents)
	mux.HandleFunc("/api/dashboards", ad.handleDashboards)
	mux.HandleFunc("/api/metrics", ad.handleMetrics)
	mux.HandleFunc("/api/users", ad.handleUsers)
	mux.HandleFunc("/api/alerts", ad.handleAlerts)
	mux.HandleFunc("/api/statistics", ad.handleStatistics)
	
	// WebSocket endpoint
	mux.HandleFunc("/ws", ad.handleWebSocket)
	
	// Static files (dashboard UI)
	mux.Handle("/", http.FileServer(http.Dir("./static/")))
	
	http.DefaultServeMux = mux
}

// HTTP handlers

func (ad *AnalyticsDashboard) handleEvents(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		ad.handleCreateEvent(w, r)
	case http.MethodGet:
		ad.handleGetEvents(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (ad *AnalyticsDashboard) handleCreateEvent(w http.ResponseWriter, r *http.Request) {
	var event AnalyticsEvent
	if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	if err := ad.TrackEvent(&event); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"status": "created", "id": event.ID})
}

func (ad *AnalyticsDashboard) handleGetEvents(w http.ResponseWriter, r *http.Request) {
	// Implementation would query stored events
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "events endpoint"})
}

func (ad *AnalyticsDashboard) handleDashboards(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		ad.handleCreateDashboard(w, r)
	case http.MethodGet:
		ad.handleGetDashboard(w, r)
	case http.MethodPut:
		ad.handleUpdateDashboard(w, r)
	case http.MethodDelete:
		ad.handleDeleteDashboard(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (ad *AnalyticsDashboard) handleCreateDashboard(w http.ResponseWriter, r *http.Request) {
	var dashboard Dashboard
	if err := json.NewDecoder(r.Body).Decode(&dashboard); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	if err := ad.CreateDashboard(&dashboard); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(dashboard)
}

func (ad *AnalyticsDashboard) handleGetDashboard(w http.ResponseWriter, r *http.Request) {
	dashboardID := r.URL.Query().Get("id")
	if dashboardID == "" {
		// Return all dashboards for user
		userID := r.URL.Query().Get("user_id")
		dashboards := ad.getDashboardsForUser(userID)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(dashboards)
		return
	}
	
	dashboard, err := ad.GetDashboard(dashboardID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(dashboard)
}

func (ad *AnalyticsDashboard) handleUpdateDashboard(w http.ResponseWriter, r *http.Request) {
	var dashboard Dashboard
	if err := json.NewDecoder(r.Body).Decode(&dashboard); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	if err := ad.UpdateDashboard(&dashboard); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(dashboard)
}

func (ad *AnalyticsDashboard) handleDeleteDashboard(w http.ResponseWriter, r *http.Request) {
	dashboardID := r.URL.Query().Get("id")
	if dashboardID == "" {
		http.Error(w, "Dashboard ID required", http.StatusBadRequest)
		return
	}
	
	if err := ad.deleteDashboard(dashboardID); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	w.WriteHeader(http.StatusNoContent)
}

func (ad *AnalyticsDashboard) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var query Query
	if err := json.NewDecoder(r.Body).Decode(&query); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	metrics, err := ad.QueryMetrics(query)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (ad *AnalyticsDashboard) handleUsers(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		ad.handleCreateUser(w, r)
	case http.MethodGet:
		ad.handleGetUser(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (ad *AnalyticsDashboard) handleCreateUser(w http.ResponseWriter, r *http.Request) {
	var user User
	if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	if err := ad.CreateUser(&user); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(user)
}

func (ad *AnalyticsDashboard) handleGetUser(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("id")
	user, err := ad.getUser(userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(user)
}

func (ad *AnalyticsDashboard) handleAlerts(w http.ResponseWriter, r *http.Request) {
	// Implementation for alert management
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "alerts endpoint"})
}

func (ad *AnalyticsDashboard) handleStatistics(w http.ResponseWriter, r *http.Request) {
	stats := ad.GetStatistics()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// WebSocket handler
func (ad *AnalyticsDashboard) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := ad.wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	
	// Create connection
	wsConn := &WebSocketConnection{
		ID:         ad.generateConnectionID(),
		Conn:       conn,
		LastPing:   time.Now(),
		SendQueue:  make(chan []byte, 100),
		Subscriptions: make([]string, 0),
	}
	
	// Store connection
	ad.connMutex.Lock()
	ad.connections[wsConn.ID] = wsConn
	atomic.AddInt64(&ad.statistics.ActiveConnections, 1)
	ad.connMutex.Unlock()
	
	// Handle connection
	go ad.handleWebSocketConnection(wsConn)
}

func (ad *AnalyticsDashboard) handleWebSocketConnection(conn *WebSocketConnection) {
	defer func() {
		conn.Conn.Close()
		ad.connMutex.Lock()
		delete(ad.connections, conn.ID)
		atomic.AddInt64(&ad.statistics.ActiveConnections, -1)
		ad.connMutex.Unlock()
	}()
	
	// Start message sender
	go ad.websocketSender(conn)
	
	// Handle incoming messages
	for {
		var msg map[string]interface{}
		if err := conn.Conn.ReadJSON(&msg); err != nil {
			break
		}
		
		ad.handleWebSocketMessage(conn, msg)
	}
}

func (ad *AnalyticsDashboard) websocketSender(conn *WebSocketConnection) {
	ticker := time.NewTicker(ad.config.WebSocketPingInterval)
	defer ticker.Stop()
	
	for {
		select {
		case data := <-conn.SendQueue:
			if err := conn.Conn.WriteMessage(websocket.TextMessage, data); err != nil {
				return
			}
		case <-ticker.C:
			if err := conn.Conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
			conn.mutex.Lock()
			conn.LastPing = time.Now()
			conn.mutex.Unlock()
		}
	}
}

func (ad *AnalyticsDashboard) handleWebSocketMessage(conn *WebSocketConnection, msg map[string]interface{}) {
	msgType, ok := msg["type"].(string)
	if !ok {
		return
	}
	
	switch msgType {
	case "subscribe":
		if dashboard, ok := msg["dashboard"].(string); ok {
			conn.mutex.Lock()
			conn.Dashboard = dashboard
			conn.Subscriptions = append(conn.Subscriptions, dashboard)
			conn.mutex.Unlock()
		}
	case "unsubscribe":
		if dashboard, ok := msg["dashboard"].(string); ok {
			conn.mutex.Lock()
			for i, sub := range conn.Subscriptions {
				if sub == dashboard {
					conn.Subscriptions = append(conn.Subscriptions[:i], conn.Subscriptions[i+1:]...)
					break
				}
			}
			conn.mutex.Unlock()
		}
	case "pong":
		conn.mutex.Lock()
		conn.LastPing = time.Now()
		conn.mutex.Unlock()
	}
}

// Utility functions

func (ad *AnalyticsDashboard) generateEventID() string {
	return fmt.Sprintf("event_%d_%d", time.Now().UnixNano(), atomic.AddInt64(&ad.statistics.EventsProcessed, 0))
}

func (ad *AnalyticsDashboard) generateDashboardID() string {
	return fmt.Sprintf("dashboard_%d", time.Now().UnixNano())
}

func (ad *AnalyticsDashboard) generateUserID() string {
	return fmt.Sprintf("user_%d", time.Now().UnixNano())
}

func (ad *AnalyticsDashboard) generateConnectionID() string {
	return fmt.Sprintf("conn_%d", time.Now().UnixNano())
}

func (ad *AnalyticsDashboard) generateCacheKey(query Query) string {
	return fmt.Sprintf("query_%s_%d_%d", query.Metric, query.Aggregation, query.TimeWindow)
}

// Public API methods

// GetStatistics returns current dashboard statistics
func (ad *AnalyticsDashboard) GetStatistics() DashboardStatistics {
	ad.statistics.mutex.RLock()
	defer ad.statistics.mutex.RUnlock()
	
	stats := *ad.statistics
	
	// Calculate cache hit rate
	if ad.cache.hits+ad.cache.misses > 0 {
		stats.CacheHitRate = float64(ad.cache.hits) / float64(ad.cache.hits+ad.cache.misses)
	}
	
	return stats
}

// BroadcastToAll sends a message to all connected clients
func (ad *AnalyticsDashboard) BroadcastToAll(message interface{}) {
	data, err := json.Marshal(message)
	if err != nil {
		return
	}
	
	ad.connMutex.RLock()
	defer ad.connMutex.RUnlock()
	
	for _, conn := range ad.connections {
		select {
		case conn.SendQueue <- data:
		default:
			// Skip if queue is full
		}
	}
}

// Placeholder implementations for remaining functionality
func (ad *AnalyticsDashboard) extractMetricsFromEvent(event *AnalyticsEvent) []Metric {
	// Implementation would extract relevant metrics from the event
	return []Metric{
		{
			Name:        "page_views",
			Value:       1,
			Aggregation: Count,
			TimeWindow:  RealTime,
			Timestamp:   event.Timestamp,
		},
	}
}

func (ad *AnalyticsDashboard) storeMetric(metric Metric) {
	// Implementation would store metric in time-series database
}

func (ad *AnalyticsDashboard) updateAggregators(metric Metric) {
	// Implementation would update real-time aggregators
}

func (ad *AnalyticsDashboard) broadcastEventUpdate(event *AnalyticsEvent) {
	// Implementation would broadcast real-time updates
}

func (ad *AnalyticsDashboard) runAggregation() {
	// Implementation would run periodic aggregation
}

func (ad *AnalyticsDashboard) pingConnections() {
	// Implementation would ping WebSocket connections
}

func (ad *AnalyticsDashboard) cleanupStaleConnections() {
	// Implementation would cleanup stale connections
}

func (ad *AnalyticsDashboard) checkAlerts() {
	// Implementation would check alert conditions
}

func (ad *AnalyticsDashboard) cleanupOldData() {
	// Implementation would cleanup old metric data
}

func (ad *AnalyticsDashboard) cleanupCache() {
	// Implementation would cleanup expired cache entries
}

func (ad *AnalyticsDashboard) updateStatistics() {
	// Implementation would update system statistics
}

func (ad *AnalyticsDashboard) executeQuery(query Query) ([]Metric, error) {
	// Implementation would execute metric queries
	return []Metric{}, nil
}

func (ad *AnalyticsDashboard) broadcastDashboardUpdate(dashboardID string) {
	// Implementation would broadcast dashboard updates
}

func (ad *AnalyticsDashboard) getDashboardsForUser(userID string) []*Dashboard {
	// Implementation would return user dashboards
	return []*Dashboard{}
}

func (ad *AnalyticsDashboard) deleteDashboard(dashboardID string) error {
	// Implementation would delete dashboard
	return nil
}

func (ad *AnalyticsDashboard) getUser(userID string) (*User, error) {
	// Implementation would retrieve user
	return nil, errors.New("not implemented")
}

// MetricCache methods
func (mc *MetricCache) Get(key string) interface{} {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()
	
	entry, exists := mc.data[key]
	if !exists || time.Now().After(entry.Expiry) {
		return nil
	}
	
	entry.AccessTime = time.Now()
	return entry.Value
}

func (mc *MetricCache) Set(key string, value interface{}, ttl time.Duration) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	
	// Check if cache is full
	if len(mc.data) >= mc.maxSize {
		// Remove oldest entry
		mc.evictOldest()
	}
	
	mc.data[key] = &CacheEntry{
		Key:        key,
		Value:      value,
		Expiry:     time.Now().Add(ttl),
		AccessTime: time.Now(),
	}
}

func (mc *MetricCache) evictOldest() {
	var oldestKey string
	var oldestTime time.Time
	
	for key, entry := range mc.data {
		if oldestKey == "" || entry.AccessTime.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.AccessTime
		}
	}
	
	if oldestKey != "" {
		delete(mc.data, oldestKey)
	}
}