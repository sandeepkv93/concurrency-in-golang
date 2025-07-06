package concurrentstockticker

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// Stock represents a stock with its current price and metadata
type Stock struct {
	Symbol        string
	Price         float64
	PreviousClose float64
	High          float64
	Low           float64
	Volume        int64
	Timestamp     time.Time
}

// PriceUpdate represents a price change event
type PriceUpdate struct {
	Symbol    string
	OldPrice  float64
	NewPrice  float64
	Change    float64
	Percent   float64
	Volume    int64
	Timestamp time.Time
}

// StockSource represents a source of stock data
type StockSource interface {
	GetStock(symbol string) (*Stock, error)
	Subscribe(symbol string) (<-chan *Stock, error)
	Unsubscribe(symbol string, ch <-chan *Stock)
}

// MockStockSource simulates a stock data source
type MockStockSource struct {
	stocks      map[string]*Stock
	subscribers map[string][]chan *Stock
	mutex       sync.RWMutex
	running     atomic.Bool
	wg          sync.WaitGroup
}

// NewMockStockSource creates a new mock stock source
func NewMockStockSource() *MockStockSource {
	source := &MockStockSource{
		stocks:      make(map[string]*Stock),
		subscribers: make(map[string][]chan *Stock),
	}
	
	// Initialize some stocks
	symbols := []string{"AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"}
	prices := []float64{150.0, 2800.0, 300.0, 3300.0, 800.0}
	
	for i, symbol := range symbols {
		source.stocks[symbol] = &Stock{
			Symbol:        symbol,
			Price:         prices[i],
			PreviousClose: prices[i],
			High:          prices[i],
			Low:           prices[i],
			Volume:        0,
			Timestamp:     time.Now(),
		}
	}
	
	return source
}

// Start begins generating random price updates
func (m *MockStockSource) Start(updateInterval time.Duration) {
	if !m.running.CompareAndSwap(false, true) {
		return
	}
	
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		ticker := time.NewTicker(updateInterval)
		defer ticker.Stop()
		
		for m.running.Load() {
			select {
			case <-ticker.C:
				m.generatePriceUpdates()
			}
		}
	}()
}

// Stop stops generating price updates
func (m *MockStockSource) Stop() {
	m.running.Store(false)
	m.wg.Wait()
}

func (m *MockStockSource) generatePriceUpdates() {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	for symbol, stock := range m.stocks {
		// Random price change (-2% to +2%)
		change := (rand.Float64() - 0.5) * 0.04
		newPrice := stock.Price * (1 + change)
		
		// Update stock
		stock.Price = newPrice
		stock.Volume += int64(rand.Intn(10000))
		stock.Timestamp = time.Now()
		
		if newPrice > stock.High {
			stock.High = newPrice
		}
		if newPrice < stock.Low {
			stock.Low = newPrice
		}
		
		// Notify subscribers
		if subs, ok := m.subscribers[symbol]; ok {
			stockCopy := *stock
			for _, ch := range subs {
				select {
				case ch <- &stockCopy:
				default:
					// Channel full, skip
				}
			}
		}
	}
}

// GetStock returns the current stock data
func (m *MockStockSource) GetStock(symbol string) (*Stock, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	stock, ok := m.stocks[symbol]
	if !ok {
		return nil, fmt.Errorf("stock %s not found", symbol)
	}
	
	stockCopy := *stock
	return &stockCopy, nil
}

// Subscribe creates a subscription for stock updates
func (m *MockStockSource) Subscribe(symbol string) (<-chan *Stock, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	if _, ok := m.stocks[symbol]; !ok {
		return nil, fmt.Errorf("stock %s not found", symbol)
	}
	
	ch := make(chan *Stock, 10)
	m.subscribers[symbol] = append(m.subscribers[symbol], ch)
	
	return ch, nil
}

// Unsubscribe removes a subscription
func (m *MockStockSource) Unsubscribe(symbol string, ch <-chan *Stock) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	subs := m.subscribers[symbol]
	for i, sub := range subs {
		if sub == ch {
			m.subscribers[symbol] = append(subs[:i], subs[i+1:]...)
			close(sub)
			break
		}
	}
}

// StockTicker manages concurrent stock price monitoring
type StockTicker struct {
	sources        []StockSource
	stocks         sync.Map // symbol -> *Stock
	updateHandlers []func(*PriceUpdate)
	handlerMutex   sync.RWMutex
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup
}

// NewStockTicker creates a new stock ticker
func NewStockTicker(sources ...StockSource) *StockTicker {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &StockTicker{
		sources:        sources,
		updateHandlers: make([]func(*PriceUpdate), 0),
		ctx:            ctx,
		cancel:         cancel,
	}
}

// AddSource adds a new data source
func (st *StockTicker) AddSource(source StockSource) {
	st.sources = append(st.sources, source)
}

// OnPriceUpdate registers a handler for price updates
func (st *StockTicker) OnPriceUpdate(handler func(*PriceUpdate)) {
	st.handlerMutex.Lock()
	defer st.handlerMutex.Unlock()
	st.updateHandlers = append(st.updateHandlers, handler)
}

// Track starts tracking a stock symbol
func (st *StockTicker) Track(symbol string) error {
	// Try each source
	for _, source := range st.sources {
		// Get initial data
		stock, err := source.GetStock(symbol)
		if err != nil {
			continue
		}
		
		st.stocks.Store(symbol, stock)
		
		// Subscribe to updates
		ch, err := source.Subscribe(symbol)
		if err != nil {
			continue
		}
		
		// Start update handler
		st.wg.Add(1)
		go st.handleUpdates(symbol, ch, source)
		
		return nil
	}
	
	return fmt.Errorf("no source available for symbol %s", symbol)
}

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
			
			// Get previous stock data
			prevValue, _ := st.stocks.Load(symbol)
			prevStock := prevValue.(*Stock)
			
			// Calculate changes
			update := &PriceUpdate{
				Symbol:    symbol,
				OldPrice:  prevStock.Price,
				NewPrice:  newStock.Price,
				Change:    newStock.Price - prevStock.Price,
				Percent:   ((newStock.Price - prevStock.Price) / prevStock.Price) * 100,
				Volume:    newStock.Volume,
				Timestamp: newStock.Timestamp,
			}
			
			// Update stored data
			st.stocks.Store(symbol, newStock)
			
			// Notify handlers
			st.notifyHandlers(update)
		}
	}
}

func (st *StockTicker) notifyHandlers(update *PriceUpdate) {
	st.handlerMutex.RLock()
	handlers := make([]func(*PriceUpdate), len(st.updateHandlers))
	copy(handlers, st.updateHandlers)
	st.handlerMutex.RUnlock()
	
	for _, handler := range handlers {
		go handler(update)
	}
}

// GetStock returns current stock data
func (st *StockTicker) GetStock(symbol string) (*Stock, bool) {
	value, ok := st.stocks.Load(symbol)
	if !ok {
		return nil, false
	}
	return value.(*Stock), true
}

// GetAllStocks returns all tracked stocks
func (st *StockTicker) GetAllStocks() map[string]*Stock {
	result := make(map[string]*Stock)
	
	st.stocks.Range(func(key, value interface{}) bool {
		result[key.(string)] = value.(*Stock)
		return true
	})
	
	return result
}

// Stop stops the ticker
func (st *StockTicker) Stop() {
	st.cancel()
	st.wg.Wait()
}

// PriceAggregator aggregates prices from multiple sources
type PriceAggregator struct {
	sources   []StockSource
	weights   map[StockSource]float64
	cache     sync.Map // symbol -> aggregatedPrice
	cacheTTL  time.Duration
	lastFetch sync.Map // symbol -> time.Time
}

// NewPriceAggregator creates a new price aggregator
func NewPriceAggregator(cacheTTL time.Duration) *PriceAggregator {
	return &PriceAggregator{
		sources:  make([]StockSource, 0),
		weights:  make(map[StockSource]float64),
		cacheTTL: cacheTTL,
	}
}

// AddSource adds a source with weight
func (pa *PriceAggregator) AddSource(source StockSource, weight float64) {
	pa.sources = append(pa.sources, source)
	pa.weights[source] = weight
}

// GetAggregatedPrice returns weighted average price
func (pa *PriceAggregator) GetAggregatedPrice(symbol string) (float64, error) {
	// Check cache
	if cached, ok := pa.cache.Load(symbol); ok {
		if lastFetch, ok := pa.lastFetch.Load(symbol); ok {
			if time.Since(lastFetch.(time.Time)) < pa.cacheTTL {
				return cached.(float64), nil
			}
		}
	}
	
	// Fetch from sources concurrently
	type result struct {
		price  float64
		weight float64
		err    error
	}
	
	results := make(chan result, len(pa.sources))
	
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
	
	// Collect results
	totalWeight := 0.0
	weightedSum := 0.0
	errors := 0
	
	for i := 0; i < len(pa.sources); i++ {
		r := <-results
		if r.err != nil {
			errors++
			continue
		}
		
		totalWeight += r.weight
		weightedSum += r.price * r.weight
	}
	
	if totalWeight == 0 {
		return 0, fmt.Errorf("no valid prices available")
	}
	
	aggregatedPrice := weightedSum / totalWeight
	
	// Update cache
	pa.cache.Store(symbol, aggregatedPrice)
	pa.lastFetch.Store(symbol, time.Now())
	
	return aggregatedPrice, nil
}

// PortfolioTracker tracks a portfolio of stocks
type PortfolioTracker struct {
	ticker     *StockTicker
	holdings   sync.Map // symbol -> quantity
	cashBalance float64
	mutex      sync.RWMutex
}

// NewPortfolioTracker creates a new portfolio tracker
func NewPortfolioTracker(ticker *StockTicker, initialCash float64) *PortfolioTracker {
	return &PortfolioTracker{
		ticker:      ticker,
		cashBalance: initialCash,
	}
}

// Buy buys shares of a stock
func (pt *PortfolioTracker) Buy(symbol string, quantity int) error {
	stock, ok := pt.ticker.GetStock(symbol)
	if !ok {
		return fmt.Errorf("stock %s not found", symbol)
	}
	
	cost := stock.Price * float64(quantity)
	
	pt.mutex.Lock()
	defer pt.mutex.Unlock()
	
	if cost > pt.cashBalance {
		return fmt.Errorf("insufficient funds: need %.2f, have %.2f", cost, pt.cashBalance)
	}
	
	pt.cashBalance -= cost
	
	current, _ := pt.holdings.Load(symbol)
	currentQty := 0
	if current != nil {
		currentQty = current.(int)
	}
	
	pt.holdings.Store(symbol, currentQty+quantity)
	
	return nil
}

// Sell sells shares of a stock
func (pt *PortfolioTracker) Sell(symbol string, quantity int) error {
	current, ok := pt.holdings.Load(symbol)
	if !ok || current.(int) < quantity {
		return fmt.Errorf("insufficient shares")
	}
	
	stock, ok := pt.ticker.GetStock(symbol)
	if !ok {
		return fmt.Errorf("stock %s not found", symbol)
	}
	
	proceeds := stock.Price * float64(quantity)
	
	pt.mutex.Lock()
	defer pt.mutex.Unlock()
	
	pt.cashBalance += proceeds
	
	newQty := current.(int) - quantity
	if newQty == 0 {
		pt.holdings.Delete(symbol)
	} else {
		pt.holdings.Store(symbol, newQty)
	}
	
	return nil
}

// GetPortfolioValue returns total portfolio value
func (pt *PortfolioTracker) GetPortfolioValue() float64 {
	pt.mutex.RLock()
	totalValue := pt.cashBalance
	pt.mutex.RUnlock()
	
	pt.holdings.Range(func(key, value interface{}) bool {
		symbol := key.(string)
		quantity := value.(int)
		
		if stock, ok := pt.ticker.GetStock(symbol); ok {
			totalValue += stock.Price * float64(quantity)
		}
		
		return true
	})
	
	return totalValue
}

// Example demonstrates concurrent stock ticker
func Example() {
	fmt.Println("=== Concurrent Stock Ticker Example ===")
	
	// Create mock source
	source := NewMockStockSource()
	source.Start(100 * time.Millisecond)
	defer source.Stop()
	
	// Create ticker
	ticker := NewStockTicker(source)
	defer ticker.Stop()
	
	// Register update handler
	updateCount := int32(0)
	ticker.OnPriceUpdate(func(update *PriceUpdate) {
		count := atomic.AddInt32(&updateCount, 1)
		if count <= 5 { // Print first 5 updates
			fmt.Printf("%s: $%.2f → $%.2f (%.2f%%)\n", 
				update.Symbol, update.OldPrice, update.NewPrice, update.Percent)
		}
	})
	
	// Track some stocks
	symbols := []string{"AAPL", "GOOGL", "MSFT"}
	for _, symbol := range symbols {
		err := ticker.Track(symbol)
		if err != nil {
			fmt.Printf("Error tracking %s: %v\n", symbol, err)
		}
	}
	
	// Let it run for a bit
	time.Sleep(500 * time.Millisecond)
	
	// Show current prices
	fmt.Println("\nCurrent Prices:")
	for _, symbol := range symbols {
		if stock, ok := ticker.GetStock(symbol); ok {
			fmt.Printf("%s: $%.2f (High: $%.2f, Low: $%.2f)\n", 
				stock.Symbol, stock.Price, stock.High, stock.Low)
		}
	}
	
	// Portfolio example
	fmt.Println("\n=== Portfolio Tracker Example ===")
	
	portfolio := NewPortfolioTracker(ticker, 10000.0)
	
	// Buy some stocks
	portfolio.Buy("AAPL", 10)
	portfolio.Buy("MSFT", 5)
	
	fmt.Printf("Initial portfolio value: $%.2f\n", portfolio.GetPortfolioValue())
	
	// Wait for price changes
	time.Sleep(300 * time.Millisecond)
	
	fmt.Printf("Final portfolio value: $%.2f\n", portfolio.GetPortfolioValue())
}