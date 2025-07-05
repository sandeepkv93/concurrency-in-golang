package concurrentstockticker

import (
	"context"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestMockStockSource(t *testing.T) {
	source := NewMockStockSource()
	
	// Test GetStock
	stock, err := source.GetStock("AAPL")
	if err != nil {
		t.Errorf("Failed to get AAPL: %v", err)
	}
	if stock.Symbol != "AAPL" {
		t.Errorf("Expected symbol AAPL, got %s", stock.Symbol)
	}
	
	// Test non-existent stock
	_, err = source.GetStock("INVALID")
	if err == nil {
		t.Error("Expected error for invalid symbol")
	}
	
	// Test subscription
	ch, err := source.Subscribe("AAPL")
	if err != nil {
		t.Errorf("Failed to subscribe: %v", err)
	}
	
	// Start updates
	source.Start(10 * time.Millisecond)
	defer source.Stop()
	
	// Wait for update
	select {
	case update := <-ch:
		if update.Symbol != "AAPL" {
			t.Errorf("Expected AAPL update, got %s", update.Symbol)
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("Timeout waiting for update")
	}
	
	// Unsubscribe
	source.Unsubscribe("AAPL", ch)
}

func TestStockTicker(t *testing.T) {
	source := NewMockStockSource()
	source.Start(10 * time.Millisecond)
	defer source.Stop()
	
	ticker := NewStockTicker(source)
	defer ticker.Stop()
	
	// Track a stock
	err := ticker.Track("AAPL")
	if err != nil {
		t.Errorf("Failed to track AAPL: %v", err)
	}
	
	// Get stock data
	stock, ok := ticker.GetStock("AAPL")
	if !ok {
		t.Error("Failed to get tracked stock")
	}
	if stock.Symbol != "AAPL" {
		t.Errorf("Expected AAPL, got %s", stock.Symbol)
	}
	
	// Register update handler
	updateReceived := make(chan *PriceUpdate, 1)
	ticker.OnPriceUpdate(func(update *PriceUpdate) {
		select {
		case updateReceived <- update:
		default:
		}
	})
	
	// Wait for price update
	select {
	case update := <-updateReceived:
		if update.Symbol != "AAPL" {
			t.Errorf("Expected AAPL update, got %s", update.Symbol)
		}
		if math.IsNaN(update.Percent) {
			t.Error("Invalid percentage change")
		}
	case <-time.After(200 * time.Millisecond):
		t.Error("Timeout waiting for price update")
	}
}

func TestMultipleSources(t *testing.T) {
	source1 := NewMockStockSource()
	source2 := NewMockStockSource()
	
	// Remove AAPL from source2 to test fallback
	source2.mutex.Lock()
	delete(source2.stocks, "AAPL")
	source2.mutex.Unlock()
	
	ticker := NewStockTicker(source2, source1) // Try source2 first, fallback to source1
	defer ticker.Stop()
	
	// Should successfully track AAPL from source1
	err := ticker.Track("AAPL")
	if err != nil {
		t.Errorf("Failed to track AAPL: %v", err)
	}
	
	stock, ok := ticker.GetStock("AAPL")
	if !ok {
		t.Error("Failed to get AAPL")
	}
	if stock == nil {
		t.Error("Stock is nil")
	}
}

func TestConcurrentUpdates(t *testing.T) {
	source := NewMockStockSource()
	source.Start(5 * time.Millisecond)
	defer source.Stop()
	
	ticker := NewStockTicker(source)
	defer ticker.Stop()
	
	// Track multiple stocks
	symbols := []string{"AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"}
	for _, symbol := range symbols {
		err := ticker.Track(symbol)
		if err != nil {
			t.Errorf("Failed to track %s: %v", symbol, err)
		}
	}
	
	// Count updates
	var updateCount int32
	ticker.OnPriceUpdate(func(update *PriceUpdate) {
		atomic.AddInt32(&updateCount, 1)
	})
	
	// Let it run
	time.Sleep(100 * time.Millisecond)
	
	count := atomic.LoadInt32(&updateCount)
	if count == 0 {
		t.Error("No updates received")
	}
	
	// Verify all stocks are tracked
	allStocks := ticker.GetAllStocks()
	if len(allStocks) != len(symbols) {
		t.Errorf("Expected %d stocks, got %d", len(symbols), len(allStocks))
	}
}

func TestPriceAggregator(t *testing.T) {
	// Create sources with different prices
	source1 := NewMockStockSource()
	source2 := NewMockStockSource()
	
	// Set different prices for AAPL
	source1.stocks["AAPL"].Price = 150.0
	source2.stocks["AAPL"].Price = 152.0
	
	aggregator := NewPriceAggregator(100 * time.Millisecond)
	aggregator.AddSource(source1, 0.6) // 60% weight
	aggregator.AddSource(source2, 0.4) // 40% weight
	
	// Get aggregated price
	price, err := aggregator.GetAggregatedPrice("AAPL")
	if err != nil {
		t.Errorf("Failed to get aggregated price: %v", err)
	}
	
	// Expected: 150*0.6 + 152*0.4 = 90 + 60.8 = 150.8
	expected := 150.8
	if math.Abs(price-expected) > 0.01 {
		t.Errorf("Expected aggregated price %.2f, got %.2f", expected, price)
	}
	
	// Test caching
	source1.stocks["AAPL"].Price = 160.0 // Change price
	
	price2, _ := aggregator.GetAggregatedPrice("AAPL")
	if price2 != price {
		t.Error("Cache should return same price")
	}
	
	// Wait for cache to expire
	time.Sleep(150 * time.Millisecond)
	
	price3, _ := aggregator.GetAggregatedPrice("AAPL")
	if price3 == price {
		t.Error("Cache should have expired")
	}
}

func TestPortfolioTracker(t *testing.T) {
	source := NewMockStockSource()
	ticker := NewStockTicker(source)
	defer ticker.Stop()
	
	// Track stocks
	ticker.Track("AAPL")
	ticker.Track("MSFT")
	
	portfolio := NewPortfolioTracker(ticker, 10000.0)
	
	// Test buying
	err := portfolio.Buy("AAPL", 10)
	if err != nil {
		t.Errorf("Failed to buy AAPL: %v", err)
	}
	
	// Test insufficient funds
	err = portfolio.Buy("AAPL", 1000) // Would cost > $10k
	if err == nil {
		t.Error("Expected insufficient funds error")
	}
	
	// Test selling
	err = portfolio.Sell("AAPL", 5)
	if err != nil {
		t.Errorf("Failed to sell AAPL: %v", err)
	}
	
	// Test selling more than owned
	err = portfolio.Sell("AAPL", 100)
	if err == nil {
		t.Error("Expected insufficient shares error")
	}
	
	// Check portfolio value
	value := portfolio.GetPortfolioValue()
	if value <= 0 {
		t.Error("Portfolio value should be positive")
	}
	
	// Verify holdings
	holding, ok := portfolio.holdings.Load("AAPL")
	if !ok || holding.(int) != 5 {
		t.Errorf("Expected 5 AAPL shares, got %v", holding)
	}
}

func TestUpdateHandlers(t *testing.T) {
	source := NewMockStockSource()
	source.Start(10 * time.Millisecond)
	defer source.Stop()
	
	ticker := NewStockTicker(source)
	defer ticker.Stop()
	
	// Register multiple handlers
	handler1Count := int32(0)
	handler2Count := int32(0)
	
	ticker.OnPriceUpdate(func(update *PriceUpdate) {
		atomic.AddInt32(&handler1Count, 1)
	})
	
	ticker.OnPriceUpdate(func(update *PriceUpdate) {
		atomic.AddInt32(&handler2Count, 1)
	})
	
	ticker.Track("AAPL")
	
	// Wait for updates
	time.Sleep(100 * time.Millisecond)
	
	// Both handlers should receive updates
	if atomic.LoadInt32(&handler1Count) == 0 {
		t.Error("Handler 1 received no updates")
	}
	
	if atomic.LoadInt32(&handler2Count) == 0 {
		t.Error("Handler 2 received no updates")
	}
}

func TestStockPriceVolatility(t *testing.T) {
	source := NewMockStockSource()
	
	// Record price changes
	priceHistory := make(map[string][]float64)
	var mutex sync.Mutex
	
	// Subscribe to all stocks
	symbols := []string{"AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"}
	for _, symbol := range symbols {
		ch, _ := source.Subscribe(symbol)
		go func(sym string, updates <-chan *Stock) {
			for stock := range updates {
				mutex.Lock()
				priceHistory[sym] = append(priceHistory[sym], stock.Price)
				mutex.Unlock()
			}
		}(symbol, ch)
	}
	
	// Generate updates
	source.Start(10 * time.Millisecond)
	time.Sleep(200 * time.Millisecond)
	source.Stop()
	
	// Verify price changes are within reasonable bounds
	mutex.Lock()
	defer mutex.Unlock()
	
	for symbol, prices := range priceHistory {
		if len(prices) < 2 {
			continue
		}
		
		initialPrice := prices[0]
		for _, price := range prices[1:] {
			percentChange := math.Abs((price-initialPrice)/initialPrice) * 100
			if percentChange > 10 {
				t.Errorf("%s: Excessive price change: %.2f%%", symbol, percentChange)
			}
		}
	}
}

func BenchmarkStockTicker(b *testing.B) {
	source := NewMockStockSource()
	source.Start(1 * time.Millisecond)
	defer source.Stop()
	
	ticker := NewStockTicker(source)
	defer ticker.Stop()
	
	// Track all stocks
	symbols := []string{"AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"}
	for _, symbol := range symbols {
		ticker.Track(symbol)
	}
	
	b.Run("GetStock", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				symbol := symbols[i%len(symbols)]
				ticker.GetStock(symbol)
				i++
			}
		})
	})
	
	b.Run("GetAllStocks", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ticker.GetAllStocks()
		}
	})
}

func BenchmarkPriceAggregator(b *testing.B) {
	// Create multiple sources
	sources := make([]*MockStockSource, 5)
	for i := range sources {
		sources[i] = NewMockStockSource()
	}
	
	aggregator := NewPriceAggregator(10 * time.Millisecond)
	for _, source := range sources {
		aggregator.AddSource(source, 1.0)
	}
	
	b.ResetTimer()
	
	b.Run("WithCache", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			aggregator.GetAggregatedPrice("AAPL")
		}
	})
	
	b.Run("WithoutCache", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			aggregator.GetAggregatedPrice("AAPL")
			time.Sleep(11 * time.Millisecond) // Force cache expiry
		}
	})
}

func TestContextCancellation(t *testing.T) {
	source := NewMockStockSource()
	source.Start(10 * time.Millisecond)
	defer source.Stop()
	
	// Create ticker with custom context
	ctx, cancel := context.WithCancel(context.Background())
	ticker := &StockTicker{
		sources:        []StockSource{source},
		updateHandlers: make([]func(*PriceUpdate), 0),
		ctx:            ctx,
		cancel:         cancel,
	}
	
	ticker.Track("AAPL")
	
	// Cancel context
	cancel()
	
	// Wait a bit
	time.Sleep(50 * time.Millisecond)
	
	// Ticker should stop receiving updates
	updateCount := int32(0)
	ticker.OnPriceUpdate(func(update *PriceUpdate) {
		atomic.AddInt32(&updateCount, 1)
	})
	
	time.Sleep(100 * time.Millisecond)
	
	if atomic.LoadInt32(&updateCount) > 0 {
		t.Error("Ticker should not receive updates after context cancellation")
	}
}