package concurrentauctionhouse

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

type BidType int

const (
	RegularBid BidType = iota
	AutoBid
	ReserveBid
)

type AuctionStatus int

const (
	Pending AuctionStatus = iota
	Active
	Ended
	Cancelled
)

type Bid struct {
	ID        string
	BidderID  string
	Amount    float64
	Timestamp time.Time
	Type      BidType
	MaxAmount float64
}

type Item struct {
	ID          string
	Title       string
	Description string
	StartPrice  float64
	ReservePrice float64
	Category    string
	Condition   string
	ImageURLs   []string
	SellerID    string
	CreatedAt   time.Time
}

type Auction struct {
	ID              string
	Item            Item
	StartTime       time.Time
	EndTime         time.Time
	Status          AuctionStatus
	CurrentBid      *Bid
	BidHistory      []Bid
	Watchers        map[string]bool
	AutoBidders     map[string]*AutoBidder
	ReserveReached  bool
	ExtensionTime   time.Duration
	BidIncrement    float64
	mutex           sync.RWMutex
	eventChan       chan AuctionEvent
	observers       []AuctionObserver
}

type AutoBidder struct {
	UserID     string
	MaxAmount  float64
	Increment  float64
	LastBid    float64
	IsActive   bool
	CreatedAt  time.Time
}

type User struct {
	ID       string
	Username string
	Email    string
	Balance  float64
	Rating   float64
	JoinedAt time.Time
	IsActive bool
	mutex    sync.RWMutex
}

type AuctionHouse struct {
	auctions    map[string]*Auction
	users       map[string]*User
	categories  map[string][]string
	mutex       sync.RWMutex
	eventBus    *EventBus
	scheduler   *AuctionScheduler
	bidValidator *BidValidator
	paymentProcessor *PaymentProcessor
	notificationService *NotificationService
}

type AuctionEvent struct {
	Type      string
	AuctionID string
	UserID    string
	Data      interface{}
	Timestamp time.Time
}

type AuctionObserver interface {
	OnBidPlaced(auction *Auction, bid *Bid)
	OnAuctionEnded(auction *Auction)
	OnReserveReached(auction *Auction)
	OnAutoBidTriggered(auction *Auction, bidder *AutoBidder, bid *Bid)
}

type EventBus struct {
	subscribers map[string][]chan AuctionEvent
	mutex       sync.RWMutex
}

type AuctionScheduler struct {
	auctionHouse *AuctionHouse
	timers       map[string]*time.Timer
	mutex        sync.RWMutex
}

type BidValidator struct {
	minBidIncrement float64
	maxBidAmount    float64
}

type PaymentProcessor struct {
	transactions map[string]*Transaction
	mutex        sync.RWMutex
}

type Transaction struct {
	ID        string
	UserID    string
	AuctionID string
	Amount    float64
	Status    string
	CreatedAt time.Time
}

type NotificationService struct {
	subscribers map[string][]chan Notification
	mutex       sync.RWMutex
}

type Notification struct {
	UserID    string
	Type      string
	Message   string
	Data      interface{}
	Timestamp time.Time
}

func NewAuctionHouse() *AuctionHouse {
	ah := &AuctionHouse{
		auctions:   make(map[string]*Auction),
		users:      make(map[string]*User),
		categories: make(map[string][]string),
		eventBus:   NewEventBus(),
		bidValidator: &BidValidator{
			minBidIncrement: 1.0,
			maxBidAmount:    1000000.0,
		},
		paymentProcessor:    NewPaymentProcessor(),
		notificationService: NewNotificationService(),
	}
	
	ah.scheduler = NewAuctionScheduler(ah)
	return ah
}

func (ah *AuctionHouse) RegisterUser(user *User) error {
	ah.mutex.Lock()
	defer ah.mutex.Unlock()
	
	if _, exists := ah.users[user.ID]; exists {
		return errors.New("user already exists")
	}
	
	ah.users[user.ID] = user
	return nil
}

func (ah *AuctionHouse) GetUser(userID string) (*User, bool) {
	ah.mutex.RLock()
	defer ah.mutex.RUnlock()
	
	user, exists := ah.users[userID]
	return user, exists
}

func (ah *AuctionHouse) CreateAuction(auction *Auction) error {
	ah.mutex.Lock()
	defer ah.mutex.Unlock()
	
	if _, exists := ah.auctions[auction.ID]; exists {
		return errors.New("auction already exists")
	}
	
	auction.Status = Pending
	auction.Watchers = make(map[string]bool)
	auction.AutoBidders = make(map[string]*AutoBidder)
	auction.BidHistory = make([]Bid, 0)
	auction.eventChan = make(chan AuctionEvent, 100)
	auction.observers = make([]AuctionObserver, 0)
	
	ah.auctions[auction.ID] = auction
	ah.scheduler.ScheduleAuction(auction)
	
	return nil
}

func (ah *AuctionHouse) GetAuction(auctionID string) (*Auction, bool) {
	ah.mutex.RLock()
	defer ah.mutex.RUnlock()
	
	auction, exists := ah.auctions[auctionID]
	return auction, exists
}

func (ah *AuctionHouse) PlaceBid(ctx context.Context, auctionID, userID string, amount float64) (*Bid, error) {
	auction, exists := ah.GetAuction(auctionID)
	if !exists {
		return nil, errors.New("auction not found")
	}
	
	user, exists := ah.GetUser(userID)
	if !exists {
		return nil, errors.New("user not found")
	}
	
	if !ah.bidValidator.ValidateBid(auction, user, amount) {
		return nil, errors.New("invalid bid")
	}
	
	bid := &Bid{
		ID:        fmt.Sprintf("bid_%d", time.Now().UnixNano()),
		BidderID:  userID,
		Amount:    amount,
		Timestamp: time.Now(),
		Type:      RegularBid,
	}
	
	if err := auction.PlaceBid(bid); err != nil {
		return nil, err
	}
	
	ah.processAutoBids(auction, bid)
	ah.eventBus.Publish("bid_placed", AuctionEvent{
		Type:      "bid_placed",
		AuctionID: auctionID,
		UserID:    userID,
		Data:      bid,
		Timestamp: time.Now(),
	})
	
	return bid, nil
}

func (ah *AuctionHouse) SetupAutoBid(auctionID, userID string, maxAmount, increment float64) error {
	auction, exists := ah.GetAuction(auctionID)
	if !exists {
		return errors.New("auction not found")
	}
	
	user, exists := ah.GetUser(userID)
	if !exists {
		return errors.New("user not found")
	}
	
	if user.Balance < maxAmount {
		return errors.New("insufficient balance")
	}
	
	autoBidder := &AutoBidder{
		UserID:    userID,
		MaxAmount: maxAmount,
		Increment: increment,
		LastBid:   0,
		IsActive:  true,
		CreatedAt: time.Now(),
	}
	
	auction.mutex.Lock()
	auction.AutoBidders[userID] = autoBidder
	auction.mutex.Unlock()
	
	return nil
}

func (ah *AuctionHouse) processAutoBids(auction *Auction, triggeredBid *Bid) {
	auction.mutex.RLock()
	autoBidders := make(map[string]*AutoBidder)
	for k, v := range auction.AutoBidders {
		if v.IsActive && k != triggeredBid.BidderID {
			autoBidders[k] = v
		}
	}
	auction.mutex.RUnlock()
	
	var highestAutoBidder *AutoBidder
	highestAmount := triggeredBid.Amount
	
	for _, autoBidder := range autoBidders {
		if autoBidder.MaxAmount > highestAmount {
			nextBidAmount := highestAmount + autoBidder.Increment
			if nextBidAmount <= autoBidder.MaxAmount {
				if highestAutoBidder == nil || autoBidder.MaxAmount > highestAutoBidder.MaxAmount {
					highestAutoBidder = autoBidder
					highestAmount = nextBidAmount
				}
			}
		}
	}
	
	if highestAutoBidder != nil {
		autoBid := &Bid{
			ID:        fmt.Sprintf("autobid_%d", time.Now().UnixNano()),
			BidderID:  highestAutoBidder.UserID,
			Amount:    highestAmount,
			Timestamp: time.Now(),
			Type:      AutoBid,
			MaxAmount: highestAutoBidder.MaxAmount,
		}
		
		auction.PlaceBid(autoBid)
		highestAutoBidder.LastBid = highestAmount
		
		for _, observer := range auction.observers {
			observer.OnAutoBidTriggered(auction, highestAutoBidder, autoBid)
		}
	}
}

func (ah *AuctionHouse) WatchAuction(auctionID, userID string) error {
	auction, exists := ah.GetAuction(auctionID)
	if !exists {
		return errors.New("auction not found")
	}
	
	auction.mutex.Lock()
	auction.Watchers[userID] = true
	auction.mutex.Unlock()
	
	return nil
}

func (ah *AuctionHouse) UnwatchAuction(auctionID, userID string) error {
	auction, exists := ah.GetAuction(auctionID)
	if !exists {
		return errors.New("auction not found")
	}
	
	auction.mutex.Lock()
	delete(auction.Watchers, userID)
	auction.mutex.Unlock()
	
	return nil
}

func (ah *AuctionHouse) GetActiveAuctions() []*Auction {
	ah.mutex.RLock()
	defer ah.mutex.RUnlock()
	
	var active []*Auction
	for _, auction := range ah.auctions {
		if auction.Status == Active {
			active = append(active, auction)
		}
	}
	
	return active
}

func (ah *AuctionHouse) GetAuctionsByCategory(category string) []*Auction {
	ah.mutex.RLock()
	defer ah.mutex.RUnlock()
	
	var auctions []*Auction
	for _, auction := range ah.auctions {
		if auction.Item.Category == category {
			auctions = append(auctions, auction)
		}
	}
	
	return auctions
}

func (ah *AuctionHouse) EndAuction(auctionID string) error {
	auction, exists := ah.GetAuction(auctionID)
	if !exists {
		return errors.New("auction not found")
	}
	
	auction.mutex.Lock()
	defer auction.mutex.Unlock()
	
	if auction.Status != Active {
		return errors.New("auction is not active")
	}
	
	auction.Status = Ended
	auction.EndTime = time.Now()
	
	if auction.CurrentBid != nil {
		ah.processWinningBid(auction)
	}
	
	for _, observer := range auction.observers {
		observer.OnAuctionEnded(auction)
	}
	
	ah.eventBus.Publish("auction_ended", AuctionEvent{
		Type:      "auction_ended",
		AuctionID: auctionID,
		Data:      auction,
		Timestamp: time.Now(),
	})
	
	return nil
}

func (ah *AuctionHouse) processWinningBid(auction *Auction) {
	if auction.CurrentBid == nil {
		return
	}
	
	winnerID := auction.CurrentBid.BidderID
	amount := auction.CurrentBid.Amount
	
	transaction := &Transaction{
		ID:        fmt.Sprintf("txn_%d", time.Now().UnixNano()),
		UserID:    winnerID,
		AuctionID: auction.ID,
		Amount:    amount,
		Status:    "pending",
		CreatedAt: time.Now(),
	}
	
	ah.paymentProcessor.ProcessTransaction(transaction)
	
	ah.notificationService.SendNotification(&Notification{
		UserID:    winnerID,
		Type:      "auction_won",
		Message:   fmt.Sprintf("You won auction %s for $%.2f", auction.ID, amount),
		Timestamp: time.Now(),
	})
	
	ah.notificationService.SendNotification(&Notification{
		UserID:    auction.Item.SellerID,
		Type:      "auction_sold",
		Message:   fmt.Sprintf("Your item %s sold for $%.2f", auction.Item.Title, amount),
		Timestamp: time.Now(),
	})
}

func (a *Auction) PlaceBid(bid *Bid) error {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	
	if a.Status != Active {
		return errors.New("auction is not active")
	}
	
	if time.Now().After(a.EndTime) {
		return errors.New("auction has ended")
	}
	
	if a.CurrentBid != nil && bid.Amount <= a.CurrentBid.Amount {
		return errors.New("bid amount must be higher than current bid")
	}
	
	if bid.Amount < a.Item.StartPrice {
		return errors.New("bid amount below starting price")
	}
	
	a.CurrentBid = bid
	a.BidHistory = append(a.BidHistory, *bid)
	
	if !a.ReserveReached && bid.Amount >= a.Item.ReservePrice {
		a.ReserveReached = true
		for _, observer := range a.observers {
			observer.OnReserveReached(a)
		}
	}
	
	if time.Until(a.EndTime) < a.ExtensionTime {
		a.EndTime = time.Now().Add(a.ExtensionTime)
	}
	
	for _, observer := range a.observers {
		observer.OnBidPlaced(a, bid)
	}
	
	return nil
}

func (a *Auction) AddObserver(observer AuctionObserver) {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	a.observers = append(a.observers, observer)
}

func (a *Auction) RemoveObserver(observer AuctionObserver) {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	
	for i, obs := range a.observers {
		if obs == observer {
			a.observers = append(a.observers[:i], a.observers[i+1:]...)
			break
		}
	}
}

func (a *Auction) GetBidHistory() []Bid {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	
	history := make([]Bid, len(a.BidHistory))
	copy(history, a.BidHistory)
	return history
}

func (a *Auction) GetWatchers() []string {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	
	var watchers []string
	for userID := range a.Watchers {
		watchers = append(watchers, userID)
	}
	return watchers
}

func (u *User) DeductBalance(amount float64) error {
	u.mutex.Lock()
	defer u.mutex.Unlock()
	
	if u.Balance < amount {
		return errors.New("insufficient balance")
	}
	
	u.Balance -= amount
	return nil
}

func (u *User) AddBalance(amount float64) {
	u.mutex.Lock()
	defer u.mutex.Unlock()
	u.Balance += amount
}

func (u *User) GetBalance() float64 {
	u.mutex.RLock()
	defer u.mutex.RUnlock()
	return u.Balance
}

func NewEventBus() *EventBus {
	return &EventBus{
		subscribers: make(map[string][]chan AuctionEvent),
	}
}

func (eb *EventBus) Subscribe(eventType string) <-chan AuctionEvent {
	eb.mutex.Lock()
	defer eb.mutex.Unlock()
	
	ch := make(chan AuctionEvent, 10)
	eb.subscribers[eventType] = append(eb.subscribers[eventType], ch)
	return ch
}

func (eb *EventBus) Publish(eventType string, event AuctionEvent) {
	eb.mutex.RLock()
	defer eb.mutex.RUnlock()
	
	for _, ch := range eb.subscribers[eventType] {
		select {
		case ch <- event:
		default:
		}
	}
}

func NewAuctionScheduler(ah *AuctionHouse) *AuctionScheduler {
	return &AuctionScheduler{
		auctionHouse: ah,
		timers:       make(map[string]*time.Timer),
	}
}

func (as *AuctionScheduler) ScheduleAuction(auction *Auction) {
	as.mutex.Lock()
	defer as.mutex.Unlock()
	
	startTimer := time.AfterFunc(time.Until(auction.StartTime), func() {
		auction.mutex.Lock()
		auction.Status = Active
		auction.mutex.Unlock()
	})
	
	endTimer := time.AfterFunc(time.Until(auction.EndTime), func() {
		as.auctionHouse.EndAuction(auction.ID)
	})
	
	as.timers[auction.ID+"_start"] = startTimer
	as.timers[auction.ID+"_end"] = endTimer
}

func (as *AuctionScheduler) CancelAuction(auctionID string) {
	as.mutex.Lock()
	defer as.mutex.Unlock()
	
	if timer, exists := as.timers[auctionID+"_start"]; exists {
		timer.Stop()
		delete(as.timers, auctionID+"_start")
	}
	
	if timer, exists := as.timers[auctionID+"_end"]; exists {
		timer.Stop()
		delete(as.timers, auctionID+"_end")
	}
}

func (bv *BidValidator) ValidateBid(auction *Auction, user *User, amount float64) bool {
	if amount <= 0 {
		return false
	}
	
	if amount > bv.maxBidAmount {
		return false
	}
	
	if user.Balance < amount {
		return false
	}
	
	if auction.CurrentBid != nil {
		minNextBid := auction.CurrentBid.Amount + bv.minBidIncrement
		if amount < minNextBid {
			return false
		}
	}
	
	return true
}

func NewPaymentProcessor() *PaymentProcessor {
	return &PaymentProcessor{
		transactions: make(map[string]*Transaction),
	}
}

func (pp *PaymentProcessor) ProcessTransaction(transaction *Transaction) error {
	pp.mutex.Lock()
	defer pp.mutex.Unlock()
	
	transaction.Status = "processing"
	pp.transactions[transaction.ID] = transaction
	
	go func() {
		time.Sleep(100 * time.Millisecond)
		
		pp.mutex.Lock()
		transaction.Status = "completed"
		pp.mutex.Unlock()
	}()
	
	return nil
}

func (pp *PaymentProcessor) GetTransaction(transactionID string) (*Transaction, bool) {
	pp.mutex.RLock()
	defer pp.mutex.RUnlock()
	
	transaction, exists := pp.transactions[transactionID]
	return transaction, exists
}

func NewNotificationService() *NotificationService {
	return &NotificationService{
		subscribers: make(map[string][]chan Notification),
	}
}

func (ns *NotificationService) Subscribe(userID string) <-chan Notification {
	ns.mutex.Lock()
	defer ns.mutex.Unlock()
	
	ch := make(chan Notification, 10)
	ns.subscribers[userID] = append(ns.subscribers[userID], ch)
	return ch
}

func (ns *NotificationService) SendNotification(notification *Notification) {
	ns.mutex.RLock()
	defer ns.mutex.RUnlock()
	
	for _, ch := range ns.subscribers[notification.UserID] {
		select {
		case ch <- *notification:
		default:
		}
	}
}

type ConsoleObserver struct {
	Name string
}

func NewConsoleObserver(name string) *ConsoleObserver {
	return &ConsoleObserver{Name: name}
}

func (co *ConsoleObserver) OnBidPlaced(auction *Auction, bid *Bid) {
	fmt.Printf("[%s] New bid on auction %s: $%.2f by %s\n", 
		co.Name, auction.ID, bid.Amount, bid.BidderID)
}

func (co *ConsoleObserver) OnAuctionEnded(auction *Auction) {
	winner := "No winner"
	amount := 0.0
	
	if auction.CurrentBid != nil {
		winner = auction.CurrentBid.BidderID
		amount = auction.CurrentBid.Amount
	}
	
	fmt.Printf("[%s] Auction %s ended. Winner: %s, Amount: $%.2f\n", 
		co.Name, auction.ID, winner, amount)
}

func (co *ConsoleObserver) OnReserveReached(auction *Auction) {
	fmt.Printf("[%s] Reserve price reached for auction %s\n", 
		co.Name, auction.ID)
}

func (co *ConsoleObserver) OnAutoBidTriggered(auction *Auction, bidder *AutoBidder, bid *Bid) {
	fmt.Printf("[%s] Auto-bid triggered for auction %s: $%.2f by %s\n", 
		co.Name, auction.ID, bid.Amount, bidder.UserID)
}

func (ah *AuctionHouse) GetAuctionStatistics() map[string]interface{} {
	ah.mutex.RLock()
	defer ah.mutex.RUnlock()
	
	stats := make(map[string]interface{})
	
	totalAuctions := len(ah.auctions)
	activeAuctions := 0
	endedAuctions := 0
	totalBids := 0
	totalValue := 0.0
	
	for _, auction := range ah.auctions {
		switch auction.Status {
		case Active:
			activeAuctions++
		case Ended:
			endedAuctions++
		}
		
		totalBids += len(auction.BidHistory)
		if auction.CurrentBid != nil {
			totalValue += auction.CurrentBid.Amount
		}
	}
	
	stats["total_auctions"] = totalAuctions
	stats["active_auctions"] = activeAuctions
	stats["ended_auctions"] = endedAuctions
	stats["total_bids"] = totalBids
	stats["total_value"] = totalValue
	stats["average_value"] = 0.0
	
	if endedAuctions > 0 {
		stats["average_value"] = totalValue / float64(endedAuctions)
	}
	
	return stats
}

func (ah *AuctionHouse) SearchAuctions(query string, category string, minPrice, maxPrice float64) []*Auction {
	ah.mutex.RLock()
	defer ah.mutex.RUnlock()
	
	var results []*Auction
	
	for _, auction := range ah.auctions {
		if category != "" && auction.Item.Category != category {
			continue
		}
		
		if auction.CurrentBid != nil {
			if auction.CurrentBid.Amount < minPrice || auction.CurrentBid.Amount > maxPrice {
				continue
			}
		} else {
			if auction.Item.StartPrice < minPrice || auction.Item.StartPrice > maxPrice {
				continue
			}
		}
		
		if query != "" {
			if !contains(auction.Item.Title, query) && !contains(auction.Item.Description, query) {
				continue
			}
		}
		
		results = append(results, auction)
	}
	
	sort.Slice(results, func(i, j int) bool {
		return results[i].EndTime.Before(results[j].EndTime)
	})
	
	return results
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		(s == substr || len(substr) == 0 || 
		(len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || 
		func() bool {
			for i := 1; i <= len(s)-len(substr); i++ {
				if s[i:i+len(substr)] == substr {
					return true
				}
			}
			return false
		}())))
}

type AuctionManager struct {
	auctionHouse *AuctionHouse
	running      bool
	stopChan     chan struct{}
	wg           sync.WaitGroup
}

func NewAuctionManager(ah *AuctionHouse) *AuctionManager {
	return &AuctionManager{
		auctionHouse: ah,
		stopChan:     make(chan struct{}),
	}
}

func (am *AuctionManager) Start(ctx context.Context) {
	am.running = true
	
	am.wg.Add(1)
	go am.monitorAuctions(ctx)
}

func (am *AuctionManager) Stop() {
	am.running = false
	close(am.stopChan)
	am.wg.Wait()
}

func (am *AuctionManager) monitorAuctions(ctx context.Context) {
	defer am.wg.Done()
	
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-am.stopChan:
			return
		case <-ticker.C:
			am.checkExpiredAuctions()
		}
	}
}

func (am *AuctionManager) checkExpiredAuctions() {
	auctions := am.auctionHouse.GetActiveAuctions()
	now := time.Now()
	
	for _, auction := range auctions {
		if now.After(auction.EndTime) {
			am.auctionHouse.EndAuction(auction.ID)
		}
	}
}