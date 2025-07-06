# Concurrent Real-time Analytics Dashboard

A high-performance, scalable real-time analytics dashboard system in Go featuring concurrent event processing, real-time data visualization, multi-user support, WebSocket-based live updates, and comprehensive dashboard management for building modern analytics applications.

## Features

### Core Analytics Engine
- **Real-time Event Processing**: High-throughput concurrent event ingestion and processing
- **Multi-dimensional Metrics**: Support for complex metric aggregations across multiple dimensions
- **Time-series Data Storage**: Efficient storage and retrieval of time-series analytics data
- **Real-time Aggregation**: Live metric calculation with configurable time windows
- **Custom Event Types**: Flexible event schema supporting various analytics use cases
- **Data Retention Management**: Automatic cleanup of old data based on retention policies

### Dashboard Management
- **Dynamic Dashboards**: Create, update, and delete dashboards with real-time configuration
- **Widget System**: Flexible widget architecture supporting charts, tables, metrics, and alerts
- **Custom Queries**: Powerful query engine with filtering, grouping, and aggregation
- **Dashboard Sharing**: Public and private dashboards with permission management
- **Template System**: Pre-built dashboard templates for common use cases
- **Export Capabilities**: Dashboard and data export in multiple formats

### Real-time Communication
- **WebSocket Integration**: Real-time updates to connected dashboard clients
- **Live Data Streaming**: Continuous data streaming to active dashboards
- **Connection Management**: Efficient WebSocket connection pooling and management
- **Subscription Model**: Selective data subscription based on dashboard requirements
- **Broadcast System**: Efficient message broadcasting to multiple clients
- **Auto-reconnection**: Robust connection handling with automatic reconnection

### User Management & Security
- **Multi-user Support**: Comprehensive user management with role-based access control
- **Authentication System**: Secure user authentication and session management
- **Permission Control**: Granular permissions for dashboard creation and access
- **User Dashboards**: Personal dashboard management for individual users
- **Audit Logging**: Comprehensive logging of user actions and system events
- **Rate Limiting**: Protection against abuse with configurable rate limits

### Performance & Scalability
- **Concurrent Processing**: High-performance concurrent event processing with worker pools
- **Caching System**: Intelligent caching for frequently accessed metrics and queries
- **Connection Pooling**: Efficient resource management for database and network connections
- **Memory Optimization**: Efficient memory usage with object pooling and garbage collection
- **Horizontal Scaling**: Design supports horizontal scaling across multiple instances
- **Load Balancing**: Built-in support for load balancing across multiple nodes

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Web UI    │  │   Mobile    │  │   API       │        │
│  │             │  │    App      │  │  Client     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                           │
                    HTTP/WebSocket
                           │
┌─────────────────────────────────────────────────────────────┐
│                 Analytics Dashboard Server                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 HTTP/WebSocket Layer                │   │
│  │                                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │ │  REST API   │ │  WebSocket  │ │ Static File │    │   │
│  │ │  Handlers   │ │   Manager   │ │   Server    │    │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Processing Layer                     │   │
│  │                                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │ │   Event     │ │   Metric    │ │ Dashboard   │    │   │
│  │ │ Processor   │ │ Aggregator  │ │  Manager    │    │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  │                                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │ │    User     │ │    Alert    │ │ Connection  │    │   │
│  │ │  Manager    │ │  Manager    │ │  Manager    │    │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Storage Layer                       │   │
│  │                                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │ │   Metric    │ │    Cache    │ │   Session   │    │   │
│  │ │   Store     │ │   System    │ │   Store     │    │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                  External Services                          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Database   │  │    Redis    │  │  External   │        │
│  │  (Time-     │  │   Cache     │  │    APIs     │        │
│  │   series)   │  │             │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Event Ingestion → Event Processing → Metric Aggregation → Real-time Updates
      │                  │                  │                    │
      ▼                  ▼                  ▼                    ▼
 ┌─────────┐      ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
 │ HTTP    │      │   Worker    │    │   Time      │    │  WebSocket  │
 │ API     │ ───► │   Pool      │ ──►│   Series    │ ──►│  Broadcast  │
 │ Events  │      │ Processing  │    │   Store     │    │   System    │
 └─────────┘      └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                    │
      ▼                  ▼                  ▼                    ▼
 ┌─────────┐      ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
 │ Event   │      │ Concurrent  │    │   Cache     │    │ Dashboard   │
 │ Buffer  │      │ Aggregation │    │   Layer     │    │   Client    │
 │ Queue   │      │   Engine    │    │             │    │  Updates    │
 └─────────┘      └─────────────┘    └─────────────┘    └─────────────┘
```

## Usage Examples

### Basic Setup

```go
package main

import (
    "fmt"
    "log"
    "time"
    
    "github.com/yourusername/concurrency-in-golang/concurrentanalyticsadashboard"
)

func main() {
    // Create dashboard configuration
    config := concurrentanalyticsadashboard.DefaultDashboardConfig()
    config.HTTPPort = 8080
    config.WebSocketPort = 8081
    config.MaxConnections = 1000
    config.EventBufferSize = 100000
    config.EnableCaching = true
    config.EnableAuthentication = true
    
    // Create analytics dashboard
    dashboard, err := concurrentanalyticsadashboard.NewAnalyticsDashboard(config)
    if err != nil {
        log.Fatalf("Failed to create dashboard: %v", err)
    }
    
    // Start the dashboard server
    err = dashboard.Start()
    if err != nil {
        log.Fatalf("Failed to start dashboard: %v", err)
    }
    defer dashboard.Stop()
    
    fmt.Printf("Analytics dashboard running on:\n")
    fmt.Printf("  HTTP API: http://localhost:%d\n", config.HTTPPort)
    fmt.Printf("  WebSocket: ws://localhost:%d/ws\n", config.HTTPPort)
    
    // Keep the server running
    select {}
}
```

### Event Tracking

```go
func trackingExample(dashboard *concurrentanalyticsadashboard.AnalyticsDashboard) {
    // Track different types of events
    events := []*concurrentanalyticsadashboard.AnalyticsEvent{
        {
            Type:      concurrentanalyticsadashboard.PageView,
            UserID:    "user123",
            SessionID: "session456",
            Properties: map[string]interface{}{
                "page":      "/home",
                "title":     "Homepage",
                "referrer":  "google.com",
                "device":    "desktop",
                "browser":   "chrome",
                "country":   "US",
            },
            Value: 1.0,
            Tags:  []string{"web", "homepage"},
            Source: "web_app",
        },
        {
            Type:      concurrentanalyticsadashboard.UserAction,
            UserID:    "user123",
            SessionID: "session456",
            Properties: map[string]interface{}{
                "action":    "click",
                "element":   "signup_button",
                "position":  "header",
                "campaign":  "summer_sale",
            },
            Value: 1.0,
            Tags:  []string{"conversion", "signup"},
        },
        {
            Type:      concurrentanalyticsadashboard.CustomEvent,
            UserID:    "user789",
            SessionID: "session012",
            Properties: map[string]interface{}{
                "event_name":   "purchase",
                "product_id":   "prod_456",
                "product_name": "Premium Subscription",
                "category":     "subscription",
                "amount":       29.99,
                "currency":     "USD",
                "payment_method": "credit_card",
            },
            Value: 29.99,
            Tags:  []string{"revenue", "subscription"},
        },
        {
            Type:      concurrentanalyticsadashboard.PerformanceEvent,
            UserID:    "user123",
            SessionID: "session456",
            Properties: map[string]interface{}{
                "metric":        "page_load_time",
                "duration_ms":   1250,
                "page":          "/dashboard",
                "connection":    "4g",
            },
            Value: 1250.0,
            Tags:  []string{"performance", "page_load"},
        },
    }
    
    // Track events concurrently
    for _, event := range events {
        err := dashboard.TrackEvent(event)
        if err != nil {
            log.Printf("Failed to track event: %v", err)
        } else {
            fmt.Printf("Tracked %s event (ID: %s)\n", 
                getEventTypeName(event.Type), event.ID)
        }
    }
}

func getEventTypeName(eventType concurrentanalyticsadashboard.EventType) string {
    names := []string{"PageView", "UserAction", "CustomEvent", "MetricUpdate", 
                     "AlertEvent", "SystemEvent", "ErrorEvent", "PerformanceEvent"}
    if int(eventType) < len(names) {
        return names[eventType]
    }
    return "Unknown"
}
```

### Dashboard Creation

```go
func createDashboardExample(dashboard *concurrentanalyticsadashboard.AnalyticsDashboard) {
    // First, create a user
    user := &concurrentanalyticsadashboard.User{
        Username: "data_analyst",
        Email:    "analyst@company.com",
        Role:     "analyst",
        Permissions: []string{"read", "write", "dashboard_create"},
    }
    
    err := dashboard.CreateUser(user)
    if err != nil {
        log.Printf("Failed to create user: %v", err)
        return
    }
    
    // Create a comprehensive analytics dashboard
    analyticsDashboard := &concurrentanalyticsadashboard.Dashboard{
        Name:   "E-commerce Analytics Dashboard",
        UserID: user.ID,
        Widgets: []*concurrentanalyticsadashboard.Widget{
            {
                Type:  "chart",
                Title: "Revenue Over Time",
                Query: concurrentanalyticsadashboard.Query{
                    Metric:      "revenue",
                    Aggregation: concurrentanalyticsadashboard.Sum,
                    TimeWindow:  concurrentanalyticsadashboard.Last24Hours,
                    Filters: []concurrentanalyticsadashboard.Filter{
                        {
                            Field:    "event_name",
                            Operator: "eq",
                            Value:    "purchase",
                        },
                    },
                    GroupBy: []string{"hour"},
                },
                Position: concurrentanalyticsadashboard.Position{X: 0, Y: 0},
                Size:     concurrentanalyticsadashboard.Size{Width: 8, Height: 4},
                Configuration: map[string]interface{}{
                    "chart_type": "line",
                    "color":      "#2563eb",
                    "show_grid":  true,
                },
            },
            {
                Type:  "metric",
                Title: "Total Revenue Today",
                Query: concurrentanalyticsadashboard.Query{
                    Metric:      "revenue",
                    Aggregation: concurrentanalyticsadashboard.Sum,
                    TimeWindow:  concurrentanalyticsadashboard.Last24Hours,
                    Filters: []concurrentanalyticsadashboard.Filter{
                        {
                            Field:    "event_name",
                            Operator: "eq",
                            Value:    "purchase",
                        },
                    },
                },
                Position: concurrentanalyticsadashboard.Position{X: 8, Y: 0},
                Size:     concurrentanalyticsadashboard.Size{Width: 4, Height: 2},
                Configuration: map[string]interface{}{
                    "format":      "currency",
                    "currency":    "USD",
                    "show_change": true,
                },
            },
            {
                Type:  "metric",
                Title: "Unique Visitors",
                Query: concurrentanalyticsadashboard.Query{
                    Metric:      "page_views",
                    Aggregation: concurrentanalyticsadashboard.UniqueCount,
                    TimeWindow:  concurrentanalyticsadashboard.Last24Hours,
                    GroupBy:     []string{"user_id"},
                },
                Position: concurrentanalyticsadashboard.Position{X: 8, Y: 2},
                Size:     concurrentanalyticsadashboard.Size{Width: 4, Height: 2},
                Configuration: map[string]interface{}{
                    "format":      "number",
                    "show_change": true,
                },
            },
            {
                Type:  "table",
                Title: "Top Products by Revenue",
                Query: concurrentanalyticsadashboard.Query{
                    Metric:      "revenue",
                    Aggregation: concurrentanalyticsadashboard.Sum,
                    TimeWindow:  concurrentanalyticsadashboard.Last7Days,
                    Filters: []concurrentanalyticsadashboard.Filter{
                        {
                            Field:    "event_name",
                            Operator: "eq",
                            Value:    "purchase",
                        },
                    },
                    GroupBy: []string{"product_name"},
                    OrderBy: "value",
                    Limit:   10,
                },
                Position: concurrentanalyticsadashboard.Position{X: 0, Y: 4},
                Size:     concurrentanalyticsadashboard.Size{Width: 6, Height: 4},
                Configuration: map[string]interface{}{
                    "show_rank": true,
                    "sortable":  true,
                },
            },
            {
                Type:  "chart",
                Title: "Conversion Funnel",
                Query: concurrentanalyticsadashboard.Query{
                    Metric:      "conversion_rate",
                    Aggregation: concurrentanalyticsadashboard.Rate,
                    TimeWindow:  concurrentanalyticsadashboard.Last7Days,
                    GroupBy:     []string{"funnel_step"},
                },
                Position: concurrentanalyticsadashboard.Position{X: 6, Y: 4},
                Size:     concurrentanalyticsadashboard.Size{Width: 6, Height: 4},
                Configuration: map[string]interface{}{
                    "chart_type": "funnel",
                    "show_percentages": true,
                },
            },
        },
        Filters: []concurrentanalyticsadashboard.Filter{
            {
                Field:    "source",
                Operator: "in",
                Value:    []string{"web", "mobile_app"},
            },
        },
        RefreshRate: 30 * time.Second,
        IsPublic:    false,
    }
    
    err = dashboard.CreateDashboard(analyticsDashboard)
    if err != nil {
        log.Printf("Failed to create dashboard: %v", err)
        return
    }
    
    fmt.Printf("Created dashboard: %s (ID: %s)\n", 
        analyticsDashboard.Name, analyticsDashboard.ID)
    fmt.Printf("Dashboard contains %d widgets\n", len(analyticsDashboard.Widgets))
    fmt.Printf("Refresh rate: %v\n", analyticsDashboard.RefreshRate)
}
```

### Real-time Metrics Query

```go
func queryMetricsExample(dashboard *concurrentanalyticsadashboard.AnalyticsDashboard) {
    // Define various metric queries
    queries := []struct {
        name  string
        query concurrentanalyticsadashboard.Query
    }{
        {
            name: "Page Views Last Hour",
            query: concurrentanalyticsadashboard.Query{
                Metric:      "page_views",
                Aggregation: concurrentanalyticsadashboard.Count,
                TimeWindow:  concurrentanalyticsadashboard.Last1Hour,
                Filters: []concurrentanalyticsadashboard.Filter{
                    {Field: "source", Operator: "eq", Value: "web"},
                },
            },
        },
        {
            name: "Average Page Load Time",
            query: concurrentanalyticsadashboard.Query{
                Metric:      "page_load_time",
                Aggregation: concurrentanalyticsadashboard.Average,
                TimeWindow:  concurrentanalyticsadashboard.Last15Minutes,
                Filters: []concurrentanalyticsadashboard.Filter{
                    {Field: "device", Operator: "neq", Value: "bot"},
                },
            },
        },
        {
            name: "95th Percentile Response Time",
            query: concurrentanalyticsadashboard.Query{
                Metric:      "response_time",
                Aggregation: concurrentanalyticsadashboard.Percentile95,
                TimeWindow:  concurrentanalyticsadashboard.Last5Minutes,
            },
        },
        {
            name: "Revenue by Product Category",
            query: concurrentanalyticsadashboard.Query{
                Metric:      "revenue",
                Aggregation: concurrentanalyticsadashboard.Sum,
                TimeWindow:  concurrentanalyticsadashboard.Last24Hours,
                Filters: []concurrentanalyticsadashboard.Filter{
                    {Field: "event_name", Operator: "eq", Value: "purchase"},
                },
                GroupBy: []string{"product_category"},
                OrderBy: "value",
                Limit:   5,
            },
        },
        {
            name: "Unique Users by Country",
            query: concurrentanalyticsadashboard.Query{
                Metric:      "unique_users",
                Aggregation: concurrentanalyticsadashboard.UniqueCount,
                TimeWindow:  concurrentanalyticsadashboard.Last7Days,
                GroupBy:     []string{"country"},
                OrderBy:     "value",
                Limit:       10,
            },
        },
    }
    
    // Execute queries and display results
    for _, q := range queries {
        fmt.Printf("\n--- %s ---\n", q.name)
        
        metrics, err := dashboard.QueryMetrics(q.query)
        if err != nil {
            log.Printf("Query failed: %v", err)
            continue
        }
        
        if len(metrics) == 0 {
            fmt.Println("No data available")
            continue
        }
        
        for _, metric := range metrics {
            fmt.Printf("Metric: %s, Value: %.2f", metric.Name, metric.Value)
            if metric.Count > 0 {
                fmt.Printf(" (Count: %d)", metric.Count)
            }
            if len(metric.Dimensions) > 0 {
                fmt.Printf(" Dimensions: %v", metric.Dimensions)
            }
            fmt.Printf(" Timestamp: %s\n", metric.Timestamp.Format("15:04:05"))
        }
    }
}
```

### Real-time Dashboard Updates

```go
func realTimeUpdatesExample(dashboard *concurrentanalyticsadashboard.AnalyticsDashboard) {
    // Simulate real-time event generation
    go func() {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()
        
        userIDs := []string{"user1", "user2", "user3", "user4", "user5"}
        pages := []string{"/home", "/products", "/about", "/contact", "/checkout"}
        sources := []string{"google", "facebook", "direct", "twitter"}
        
        for {
            select {
            case <-ticker.C:
                // Generate random events
                for i := 0; i < 5; i++ {
                    event := &concurrentanalyticsadashboard.AnalyticsEvent{
                        Type:   concurrentanalyticsadashboard.PageView,
                        UserID: userIDs[i%len(userIDs)],
                        Properties: map[string]interface{}{
                            "page":     pages[i%len(pages)],
                            "referrer": sources[i%len(sources)],
                            "duration": 1000 + (i * 500),
                        },
                        Value: 1.0,
                    }
                    
                    dashboard.TrackEvent(event)
                }
                
                // Generate some revenue events
                if time.Now().Second()%10 == 0 {
                    event := &concurrentanalyticsadashboard.AnalyticsEvent{
                        Type:   concurrentanalyticsadashboard.CustomEvent,
                        UserID: userIDs[0],
                        Properties: map[string]interface{}{
                            "event_name": "purchase",
                            "amount":     float64(10 + (time.Now().Second() % 50)),
                            "product":    "Premium Plan",
                        },
                        Value: float64(10 + (time.Now().Second() % 50)),
                    }
                    
                    dashboard.TrackEvent(event)
                }
            }
        }
    }()
    
    // Monitor statistics
    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()
        
        for {
            select {
            case <-ticker.C:
                stats := dashboard.GetStatistics()
                fmt.Printf("\n=== Dashboard Statistics ===\n")
                fmt.Printf("Events Processed: %d\n", stats.EventsProcessed)
                fmt.Printf("Active Connections: %d\n", stats.ActiveConnections)
                fmt.Printf("Active Dashboards: %d\n", stats.ActiveDashboards)
                fmt.Printf("Cache Hit Rate: %.2f%%\n", stats.CacheHitRate*100)
                fmt.Printf("Queries Executed: %d\n", stats.QueriesExecuted)
                fmt.Printf("Uptime: %v\n", time.Since(stats.StartTime))
            }
        }
    }()
    
    // Broadcast periodic updates
    go func() {
        ticker := time.NewTicker(5 * time.Second)
        defer ticker.Stop()
        
        for {
            select {
            case <-ticker.C:
                // Query current metrics
                metrics, err := dashboard.QueryMetrics(concurrentanalyticsadashboard.Query{
                    Metric:      "page_views",
                    Aggregation: concurrentanalyticsadashboard.Count,
                    TimeWindow:  concurrentanalyticsadashboard.Last5Minutes,
                })
                
                if err == nil && len(metrics) > 0 {
                    update := map[string]interface{}{
                        "type":      "metric_update",
                        "metric":    "page_views",
                        "value":     metrics[0].Value,
                        "timestamp": time.Now(),
                    }
                    
                    dashboard.BroadcastToAll(update)
                    fmt.Printf("Broadcasted update: %.0f page views\n", metrics[0].Value)
                }
            }
        }
    }()
}
```

### Advanced Analytics with Custom Aggregations

```go
func advancedAnalyticsExample(dashboard *concurrentanalyticsadashboard.AnalyticsDashboard) {
    // Complex analytics scenarios
    queries := []struct {
        name        string
        description string
        query       concurrentanalyticsadashboard.Query
    }{
        {
            name:        "Customer Lifetime Value",
            description: "Average revenue per customer over their lifetime",
            query: concurrentanalyticsadashboard.Query{
                Metric:      "customer_ltv",
                Aggregation: concurrentanalyticsadashboard.Average,
                TimeWindow:  concurrentanalyticsadashboard.Last30Days,
                Filters: []concurrentanalyticsadashboard.Filter{
                    {Field: "event_name", Operator: "eq", Value: "purchase"},
                    {Field: "user_type", Operator: "eq", Value: "returning"},
                },
                GroupBy: []string{"user_id"},
            },
        },
        {
            name:        "Conversion Rate by Traffic Source",
            description: "Conversion rates segmented by traffic source",
            query: concurrentanalyticsadashboard.Query{
                Metric:      "conversion_rate",
                Aggregation: concurrentanalyticsadashboard.Rate,
                TimeWindow:  concurrentanalyticsadashboard.Last7Days,
                GroupBy:     []string{"traffic_source"},
                OrderBy:     "value",
            },
        },
        {
            name:        "High-Value Customer Segments",
            description: "Customers with purchases above 95th percentile",
            query: concurrentanalyticsadashboard.Query{
                Metric:      "purchase_amount",
                Aggregation: concurrentanalyticsadashboard.Percentile95,
                TimeWindow:  concurrentanalyticsadashboard.Last30Days,
                Filters: []concurrentanalyticsadashboard.Filter{
                    {Field: "event_name", Operator: "eq", Value: "purchase"},
                    {Field: "amount", Operator: "gt", Value: 100},
                },
                GroupBy: []string{"customer_segment"},
            },
        },
        {
            name:        "Product Performance Histogram",
            description: "Distribution of product sales across price ranges",
            query: concurrentanalyticsadashboard.Query{
                Metric:      "product_sales",
                Aggregation: concurrentanalyticsadashboard.Histogram,
                TimeWindow:  concurrentanalyticsadashboard.Last24Hours,
                Filters: []concurrentanalyticsadashboard.Filter{
                    {Field: "product_category", Operator: "in", 
                     Value: []string{"electronics", "books", "clothing"}},
                },
                GroupBy: []string{"price_range"},
            },
        },
        {
            name:        "User Engagement Trends",
            description: "Session duration trends over time",
            query: concurrentanalyticsadashboard.Query{
                Metric:      "session_duration",
                Aggregation: concurrentanalyticsadashboard.Average,
                TimeWindow:  concurrentanalyticsadashboard.Last7Days,
                GroupBy:     []string{"date", "device_type"},
                OrderBy:     "timestamp",
            },
        },
    }
    
    fmt.Println("=== Advanced Analytics Results ===")
    
    for _, q := range queries {
        fmt.Printf("\n%s\n", q.name)
        fmt.Printf("Description: %s\n", q.description)
        fmt.Printf("Query: %+v\n", q.query)
        
        start := time.Now()
        metrics, err := dashboard.QueryMetrics(q.query)
        queryTime := time.Since(start)
        
        if err != nil {
            fmt.Printf("❌ Query failed: %v\n", err)
            continue
        }
        
        fmt.Printf("✅ Query completed in %v\n", queryTime)
        fmt.Printf("📊 Results: %d metrics returned\n", len(metrics))
        
        // Display sample results
        for i, metric := range metrics {
            if i >= 3 { // Show only first 3 results
                fmt.Printf("   ... and %d more results\n", len(metrics)-3)
                break
            }
            
            fmt.Printf("   %s: %.2f", metric.Name, metric.Value)
            if len(metric.Dimensions) > 0 {
                fmt.Printf(" (%v)", metric.Dimensions)
            }
            fmt.Println()
        }
    }
}
```

## Configuration Options

### DashboardConfig Fields

#### Network Configuration
- **HTTPPort**: Port for HTTP API server (default: 8080)
- **WebSocketPort**: Port for WebSocket connections (default: 8081)
- **MaxConnections**: Maximum concurrent WebSocket connections
- **WebSocketPingInterval**: Frequency of WebSocket ping messages
- **WebSocketPongTimeout**: Timeout for WebSocket pong responses

#### Performance Settings
- **MaxEventsPerSecond**: Rate limit for event ingestion
- **EventBufferSize**: Size of event processing buffer
- **WorkerPoolSize**: Number of worker goroutines for processing
- **CacheSize**: Maximum number of cached items
- **MetricRetentionPeriod**: How long to keep metric data

#### Feature Toggles
- **EnableAuthentication**: Enable user authentication system
- **EnableRateLimiting**: Enable request rate limiting
- **EnableCaching**: Enable metric result caching
- **EnableLogging**: Enable system logging
- **EnableMetrics**: Enable performance metrics collection
- **EnableAlerts**: Enable alert system

#### Limits and Quotas
- **MaxDashboardsPerUser**: Maximum dashboards per user account
- **MaxWidgetsPerDashboard**: Maximum widgets per dashboard
- **AlertCheckInterval**: Frequency of alert condition checking

#### External Services
- **DatabaseURL**: Connection string for time-series database
- **RedisURL**: Connection string for Redis cache
- **LogLevel**: Logging verbosity level

## API Endpoints

### Event API
- **POST /api/events**: Submit analytics events
- **GET /api/events**: Retrieve events with filtering

### Dashboard API
- **POST /api/dashboards**: Create new dashboard
- **GET /api/dashboards**: List dashboards for user
- **GET /api/dashboards?id={id}**: Get specific dashboard
- **PUT /api/dashboards**: Update dashboard
- **DELETE /api/dashboards?id={id}**: Delete dashboard

### Metrics API
- **POST /api/metrics**: Execute metric queries
- **GET /api/metrics/schema**: Get available metrics and dimensions

### User API
- **POST /api/users**: Create new user
- **GET /api/users?id={id}**: Get user information
- **PUT /api/users**: Update user profile

### System API
- **GET /api/statistics**: Get system performance statistics
- **GET /api/health**: Health check endpoint
- **GET /api/alerts**: Get alert configurations

### WebSocket API
- **ws://host:port/ws**: Real-time dashboard updates

## WebSocket Message Types

### Client to Server
```json
{
  "type": "subscribe",
  "dashboard": "dashboard_id"
}

{
  "type": "unsubscribe", 
  "dashboard": "dashboard_id"
}

{
  "type": "pong"
}
```

### Server to Client
```json
{
  "type": "metric_update",
  "metric": "page_views",
  "value": 1234,
  "timestamp": "2023-01-01T12:00:00Z"
}

{
  "type": "dashboard_update",
  "dashboard_id": "dash_123",
  "widgets": [...],
  "timestamp": "2023-01-01T12:00:00Z"
}

{
  "type": "alert",
  "alert_id": "alert_456",
  "message": "High error rate detected",
  "severity": "warning"
}

{
  "type": "ping"
}
```

## Event Schema

### Event Types
- **PageView**: Website page views and navigation
- **UserAction**: User interactions (clicks, form submissions)
- **CustomEvent**: Application-specific events
- **MetricUpdate**: Direct metric updates
- **AlertEvent**: System alerts and notifications
- **SystemEvent**: System and infrastructure events
- **ErrorEvent**: Error and exception tracking
- **PerformanceEvent**: Performance and timing metrics

### Event Properties
```json
{
  "id": "event_12345",
  "type": 0,
  "timestamp": "2023-01-01T12:00:00Z",
  "user_id": "user_123",
  "session_id": "session_456",
  "properties": {
    "custom_field_1": "value1",
    "custom_field_2": 123
  },
  "value": 1.0,
  "tags": ["web", "conversion"],
  "source": "web_app"
}
```

## Performance Characteristics

### Throughput
- **Event Ingestion**: 10,000+ events/second (single instance)
- **Concurrent Connections**: 1,000+ WebSocket connections
- **Query Performance**: Sub-100ms for cached queries
- **Real-time Updates**: <50ms latency for live dashboard updates

### Scalability
- **Horizontal Scaling**: Stateless design supports load balancing
- **Database Sharding**: Support for distributed time-series databases
- **Cache Distribution**: Redis cluster support for distributed caching
- **Worker Scaling**: Dynamic worker pool sizing based on load

### Memory Usage
- **Base Memory**: ~50MB for minimal configuration
- **Per Connection**: ~1KB per active WebSocket connection
- **Event Buffer**: Configurable based on EventBufferSize
- **Cache Memory**: Configurable based on CacheSize setting

## Best Practices

### Event Design
1. **Consistent Schema**: Use consistent property names across events
2. **Appropriate Granularity**: Balance detail with storage efficiency
3. **Batch Processing**: Group related events when possible
4. **Proper Typing**: Use appropriate data types for numeric values

### Dashboard Design
1. **Widget Optimization**: Limit widgets per dashboard for performance
2. **Query Efficiency**: Use appropriate time windows and filters
3. **Real-time Balance**: Not all metrics need real-time updates
4. **User Experience**: Consider load times and responsiveness

### Performance Optimization
1. **Cache Strategy**: Cache frequently accessed metrics
2. **Index Strategy**: Proper indexing on frequently queried dimensions
3. **Data Retention**: Implement appropriate data retention policies
4. **Connection Pooling**: Reuse database connections efficiently

### Security Considerations
1. **Input Validation**: Validate all incoming event data
2. **Authentication**: Implement proper user authentication
3. **Authorization**: Role-based access control for dashboards
4. **Rate Limiting**: Protect against abuse and DoS attacks

## Common Use Cases

### E-commerce Analytics
- **Sales Tracking**: Revenue, conversion rates, cart abandonment
- **Customer Analytics**: Customer lifetime value, segmentation
- **Product Analytics**: Product performance, inventory tracking
- **Marketing Analytics**: Campaign performance, attribution modeling

### Web Application Monitoring
- **User Behavior**: Page views, user flows, engagement metrics
- **Performance Monitoring**: Page load times, error rates
- **A/B Testing**: Experiment tracking and statistical analysis
- **Real-time Alerting**: System health and performance alerts

### Business Intelligence
- **KPI Dashboards**: Executive dashboards with key metrics
- **Operational Analytics**: Real-time operational monitoring
- **Financial Analytics**: Revenue tracking, cost analysis
- **Custom Reporting**: Flexible reporting for various stakeholders

### IoT and Sensor Data
- **Device Monitoring**: Sensor data collection and analysis
- **Predictive Maintenance**: Equipment health monitoring
- **Environmental Monitoring**: Temperature, humidity, air quality
- **Energy Management**: Usage tracking and optimization

## Integration Examples

### Frontend Integration
```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = function() {
  // Subscribe to dashboard updates
  ws.send(JSON.stringify({
    type: 'subscribe',
    dashboard: 'dashboard_123'
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'metric_update':
      updateMetricWidget(data.metric, data.value);
      break;
    case 'dashboard_update':
      refreshDashboard(data);
      break;
    case 'alert':
      showAlert(data.message, data.severity);
      break;
  }
};

// Track events via HTTP API
function trackEvent(eventData) {
  fetch('/api/events', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(eventData)
  });
}
```

### Mobile App Integration
```swift
// iOS Swift example
import Foundation

class AnalyticsSDK {
    private let baseURL = "https://analytics.example.com"
    
    func trackEvent(_ event: [String: Any]) {
        guard let url = URL(string: "\(baseURL)/api/events") else { return }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: event)
            
            URLSession.shared.dataTask(with: request) { data, response, error in
                // Handle response
            }.resume()
        } catch {
            print("Failed to serialize event: \(error)")
        }
    }
}
```

## Limitations and Considerations

### Current Limitations
1. **Single Node**: Current implementation is single-node (extensible to distributed)
2. **In-Memory Storage**: Metrics stored in memory (can be extended to persistent storage)
3. **Basic Authentication**: Simple authentication system (can be extended)
4. **Limited Aggregations**: Basic aggregation types (extensible)

### Production Considerations
- **Load Testing**: Test with expected production loads
- **Database Selection**: Choose appropriate time-series database
- **Monitoring**: Implement comprehensive system monitoring
- **Backup Strategy**: Plan for data backup and recovery
- **Security Hardening**: Implement production security measures

## Future Enhancements

Planned improvements for future versions:

- **Distributed Architecture**: Multi-node deployment with data sharding
- **Advanced Analytics**: Machine learning-powered insights and predictions
- **Enhanced Visualizations**: More chart types and interactive visualizations
- **Mobile SDKs**: Native mobile analytics SDKs for iOS and Android
- **Advanced Alerting**: Complex alert conditions and notification channels
- **Data Export**: Bulk data export and integration with external systems
- **Custom Widgets**: Plugin system for custom dashboard widgets
- **Advanced Security**: OAuth integration, advanced RBAC, audit logging