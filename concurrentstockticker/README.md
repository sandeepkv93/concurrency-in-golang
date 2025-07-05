# Concurrent Stock Ticker

A comprehensive, real-time stock price monitoring system in Go that demonstrates advanced concurrency patterns for handling streaming data, event-driven architecture, and concurrent financial calculations.

## Problem Description

Real-time financial data processing requires handling multiple concurrent challenges:

- **Streaming Data**: Continuous flow of price updates from multiple sources
- **Event-Driven Architecture**: Responding to price changes with minimal latency
- **Concurrent Access**: Multiple clients accessing stock data simultaneously
- **Data Aggregation**: Combining data from multiple sources with different reliability
- **Portfolio Management**: Real-time portfolio valuation with concurrent trading
- **Memory Efficiency**: Managing large amounts of streaming data without memory leaks

## Solution Approach

This implementation provides a robust concurrent stock ticker system using several advanced Go concurrency patterns:

1. **Publisher-Subscriber Pattern**: Event-driven updates using channels and goroutines
2. **Multi-Source Aggregation**: Concurrent data fetching from multiple sources
3. **Concurrent-Safe Data Structures**: Using sync.Map for thread-safe operations
4. **Context-Based Cancellation**: Clean shutdown and resource management
5. **Atomic Operations**: Lock-free operations for performance-critical paths
6. **Buffered Channels**: Preventing blocking on slow consumers

## Key Components

### Core Types

- **StockTicker**: Main orchestrator for stock price monitoring
- **Stock**: Represents stock data with price, volume, and metadata
- **PriceUpdate**: Represents price change events with calculations
- **MockStockSource**: Simulates real-time stock data feeds
- **PriceAggregator**: Aggregates prices from multiple sources
- **PortfolioTracker**: Manages portfolio holdings and valuations

### Interfaces

- **StockSource**: Abstraction for different data sources (APIs, feeds, etc.)

## Technical Features

### Concurrency Patterns

1. **Fan-Out Pattern**: One ticker distributes updates to multiple handlers
2. **Worker Pool**: Multiple goroutines handling different stocks
3. **Producer-Consumer**: Data sources producing updates, consumers processing them
4. **Scatter-Gather**: Concurrent fetching from multiple sources, aggregating results
5. **Observer Pattern**: Multiple handlers observing the same events

### Advanced Features

- **Real-time Updates**: Sub-second price update propagation
- **Source Failover**: Automatic fallback to alternative data sources
- **Price Caching**: TTL-based caching for aggregated prices
- **Concurrent Trading**: Thread-safe buy/sell operations
- **Event Handlers**: Pluggable handlers for custom processing
- **Graceful Shutdown**: Clean termination of all goroutines

## Usage Examples

### Basic Stock Monitoring

```go
// Create a stock data source
source := NewMockStockSource()
source.Start(100 * time.Millisecond)
defer source.Stop()

// Create ticker
ticker := NewStockTicker(source)
defer ticker.Stop()

// Register update handler
ticker.OnPriceUpdate(func(update *PriceUpdate) {
    fmt.Printf("%s: $%.2f → $%.2f (%.2f%%)\n", 
        update.Symbol, update.OldPrice, update.NewPrice, update.Percent)
})

// Track stocks
symbols := []string{"AAPL", "GOOGL", "MSFT"}
for _, symbol := range symbols {
    ticker.Track(symbol)
}

// Get current prices
for _, symbol := range symbols {
    if stock, ok := ticker.GetStock(symbol); ok {
        fmt.Printf("%s: $%.2f\n", stock.Symbol, stock.Price)
    }
}
```

### Multiple Data Sources

```go
// Create multiple sources
source1 := NewMockStockSource()
source2 := NewMockStockSource()

// Create ticker with multiple sources
ticker := NewStockTicker(source1, source2)

// The ticker will try each source until one succeeds
ticker.Track("AAPL")
```

### Price Aggregation

```go
// Create aggregator with 5-second cache TTL
aggregator := NewPriceAggregator(5 * time.Second)

// Add weighted sources
aggregator.AddSource(source1, 0.6)  // 60% weight
aggregator.AddSource(source2, 0.4)  // 40% weight

// Get weighted average price
price, err := aggregator.GetAggregatedPrice("AAPL")
if err != nil {
    log.Printf("Failed to get price: %v", err)
    return
}

fmt.Printf("Aggregated price: $%.2f\n", price)
```

### Portfolio Management

```go
// Create portfolio with $10,000 initial cash
portfolio := NewPortfolioTracker(ticker, 10000.0)

// Buy stocks
err := portfolio.Buy("AAPL", 10)
if err != nil {
    log.Printf("Failed to buy: %v", err)
    return
}

err = portfolio.Buy("MSFT", 5)
if err != nil {
    log.Printf("Failed to buy: %v", err)
    return
}

// Get portfolio value
value := portfolio.GetPortfolioValue()
fmt.Printf("Portfolio value: $%.2f\n", value)

// Sell stocks
err = portfolio.Sell("AAPL", 5)
if err != nil {
    log.Printf("Failed to sell: %v", err)
    return
}
```

### Custom Event Handlers

```go
// Price change alerter
ticker.OnPriceUpdate(func(update *PriceUpdate) {
    if math.Abs(update.Percent) > 5.0 {
        fmt.Printf("ALERT: %s moved %.2f%%\n", 
            update.Symbol, update.Percent)
    }
})

// Volume tracker
ticker.OnPriceUpdate(func(update *PriceUpdate) {
    if update.Volume > 1000000 {
        fmt.Printf("HIGH VOLUME: %s volume: %d\n", 
            update.Symbol, update.Volume)
    }
})

// Data logger
ticker.OnPriceUpdate(func(update *PriceUpdate) {
    log.Printf("LOG: %s,%f,%f,%f,%d,%s\n",
        update.Symbol, update.OldPrice, update.NewPrice, 
        update.Percent, update.Volume, update.Timestamp)
})
```

### Subscription Management

```go
// Create custom source
source := NewMockStockSource()

// Subscribe to specific stock
ch, err := source.Subscribe("AAPL")
if err != nil {
    log.Printf("Failed to subscribe: %v", err)
    return
}

// Process updates
go func() {
    for stock := range ch {
        fmt.Printf("Received update: %s $%.2f\n", 
            stock.Symbol, stock.Price)
    }
}()

// Unsubscribe when done
defer source.Unsubscribe("AAPL", ch)
```

## Implementation Details

### Thread-Safe Stock Storage

The ticker uses sync.Map for concurrent access to stock data:

```go
type StockTicker struct {
    stocks sync.Map // symbol -> *Stock
    // ... other fields
}

func (st *StockTicker) GetStock(symbol string) (*Stock, bool) {
    value, ok := st.stocks.Load(symbol)
    if !ok {
        return nil, false
    }
    return value.(*Stock), true
}
```

### Event Handler Management

Handlers are managed with read-write locks for concurrent access:

```go
func (st *StockTicker) OnPriceUpdate(handler func(*PriceUpdate)) {
    st.handlerMutex.Lock()
    defer st.handlerMutex.Unlock()
    st.updateHandlers = append(st.updateHandlers, handler)
}

func (st *StockTicker) notifyHandlers(update *PriceUpdate) {
    st.handlerMutex.RLock()
    handlers := make([]func(*PriceUpdate), len(st.updateHandlers))
    copy(handlers, st.updateHandlers)
    st.handlerMutex.RUnlock()
    
    for _, handler := range handlers {
        go handler(update)  // Concurrent execution
    }
}
```

### Concurrent Price Aggregation

Price aggregation uses goroutines to fetch from multiple sources:

```go
func (pa *PriceAggregator) GetAggregatedPrice(symbol string) (float64, error) {
    results := make(chan result, len(pa.sources))
    
    // Fetch from all sources concurrently
    for _, source := range pa.sources {
        go func(s StockSource) {
            stock, err := s.GetStock(symbol)
            if err != nil {
                results <- result{err: err}
                return
            }
            results <- result{
                price:  stock.Price,
                weight: pa.weights[s],
            }
        }(source)
    }
    
    // Collect and aggregate results
    totalWeight := 0.0
    weightedSum := 0.0
    
    for i := 0; i < len(pa.sources); i++ {
        r := <-results
        if r.err != nil {
            continue
        }
        
        totalWeight += r.weight
        weightedSum += r.price * r.weight
    }
    
    return weightedSum / totalWeight, nil
}
```

### Update Processing Pipeline

Stock updates flow through a processing pipeline:

```go
func (st *StockTicker) handleUpdates(symbol string, ch <-chan *Stock, source StockSource) {
    defer st.wg.Done()
    defer source.Unsubscribe(symbol, ch)
    
    for {
        select {
        case <-st.ctx.Done():
            return
        case newStock, ok := <-ch:
            if !ok {
                return
            }
            
            // Calculate price change
            prevValue, _ := st.stocks.Load(symbol)
            prevStock := prevValue.(*Stock)
            
            update := &PriceUpdate{
                Symbol:    symbol,
                OldPrice:  prevStock.Price,
                NewPrice:  newStock.Price,
                Change:    newStock.Price - prevStock.Price,
                Percent:   ((newStock.Price - prevStock.Price) / prevStock.Price) * 100,
                Volume:    newStock.Volume,
                Timestamp: newStock.Timestamp,
            }
            
            // Update storage
            st.stocks.Store(symbol, newStock)
            
            // Notify handlers
            st.notifyHandlers(update)
        }
    }
}
```

### Portfolio Thread Safety

Portfolio operations use mutex protection for consistent state:

```go
func (pt *PortfolioTracker) Buy(symbol string, quantity int) error {
    stock, ok := pt.ticker.GetStock(symbol)
    if !ok {
        return fmt.Errorf("stock %s not found", symbol)
    }
    
    cost := stock.Price * float64(quantity)
    
    pt.mutex.Lock()
    defer pt.mutex.Unlock()
    
    if cost > pt.cashBalance {
        return fmt.Errorf("insufficient funds")
    }
    
    pt.cashBalance -= cost
    
    // Update holdings atomically
    current, _ := pt.holdings.Load(symbol)
    currentQty := 0
    if current != nil {
        currentQty = current.(int)
    }
    
    pt.holdings.Store(symbol, currentQty+quantity)
    return nil
}
```

## Testing

The package includes comprehensive tests covering:

- **Concurrent Updates**: Multiple goroutines updating prices
- **Handler Registration**: Event handler management
- **Source Failover**: Fallback to alternative sources
- **Portfolio Operations**: Concurrent buy/sell operations
- **Price Aggregation**: Weighted average calculations
- **Memory Leaks**: Proper cleanup of goroutines and channels

Run the tests:

```bash
go test -v ./concurrentstockticker
go test -race ./concurrentstockticker  # Race condition detection
```

## Performance Considerations

1. **Channel Buffering**: Prevents blocking slow consumers
2. **Goroutine Pooling**: Limits concurrent goroutines
3. **Memory Management**: Proper cleanup of subscriptions
4. **Cache TTL**: Reduces redundant API calls
5. **Atomic Operations**: Lock-free operations where possible

### Performance Tuning

```go
// High-frequency updates
source.Start(10 * time.Millisecond)

// Larger buffers for high-volume stocks
ch := make(chan *Stock, 100)

// Shorter cache TTL for real-time data
aggregator := NewPriceAggregator(1 * time.Second)
```

## Real-World Applications

This concurrent stock ticker pattern is applicable for:

- **Trading Systems**: Real-time price monitoring and execution
- **Financial Analytics**: Market data processing and analysis
- **Risk Management**: Position monitoring and alerting
- **Market Data Distribution**: Broadcasting prices to multiple clients
- **Algorithmic Trading**: Automated trading based on price events
- **Portfolio Management**: Real-time portfolio valuation

## Advanced Features

### Custom Data Sources

```go
type APIStockSource struct {
    apiKey string
    client *http.Client
}

func (api *APIStockSource) GetStock(symbol string) (*Stock, error) {
    // Implement API call
    return &Stock{}, nil
}

func (api *APIStockSource) Subscribe(symbol string) (<-chan *Stock, error) {
    // Implement WebSocket subscription
    return nil, nil
}
```

### Event Filtering

```go
ticker.OnPriceUpdate(func(update *PriceUpdate) {
    // Only process significant changes
    if math.Abs(update.Percent) > 1.0 {
        processSignificantChange(update)
    }
})
```

### Metrics and Monitoring

```go
type TickerMetrics struct {
    UpdatesProcessed int64
    HandlersExecuted int64
    SourceFailures   int64
    AverageLatency   time.Duration
}

func (st *StockTicker) GetMetrics() TickerMetrics {
    // Return performance metrics
}
```

The implementation demonstrates sophisticated Go concurrency patterns for building high-performance, real-time financial data systems with proper error handling, resource management, and concurrent safety.