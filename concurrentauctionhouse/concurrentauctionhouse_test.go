package concurrentauctionhouse

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestNewAuctionHouse(t *testing.T) {
	ah := NewAuctionHouse()
	
	if ah.auctions == nil {
		t.Error("Expected auctions map to be initialized")
	}
	
	if ah.users == nil {
		t.Error("Expected users map to be initialized")
	}
	
	if ah.eventBus == nil {
		t.Error("Expected event bus to be initialized")
	}
	
	if ah.scheduler == nil {
		t.Error("Expected scheduler to be initialized")
	}
}

func TestRegisterUser(t *testing.T) {
	ah := NewAuctionHouse()
	
	user := &User{
		ID:       "user1",
		Username: "testuser",
		Email:    "test@example.com",
		Balance:  1000.0,
		Rating:   5.0,
		JoinedAt: time.Now(),
		IsActive: true,
	}
	
	err := ah.RegisterUser(user)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	
	retrievedUser, exists := ah.GetUser("user1")
	if !exists {
		t.Error("Expected user to exist after registration")
	}
	
	if retrievedUser.Username != "testuser" {
		t.Errorf("Expected username 'testuser', got %s", retrievedUser.Username)
	}
	
	err = ah.RegisterUser(user)
	if err == nil {
		t.Error("Expected error when registering existing user")
	}
}

func TestCreateAuction(t *testing.T) {
	ah := NewAuctionHouse()
	
	item := Item{
		ID:           "item1",
		Title:        "Test Item",
		Description:  "A test item for auction",
		StartPrice:   10.0,
		ReservePrice: 50.0,
		Category:     "Electronics",
		SellerID:     "seller1",
		CreatedAt:    time.Now(),
	}
	
	auction := &Auction{
		ID:            "auction1",
		Item:          item,
		StartTime:     time.Now().Add(1 * time.Minute),
		EndTime:       time.Now().Add(1 * time.Hour),
		BidIncrement:  1.0,
		ExtensionTime: 5 * time.Minute,
	}
	
	err := ah.CreateAuction(auction)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	
	retrievedAuction, exists := ah.GetAuction("auction1")
	if !exists {
		t.Error("Expected auction to exist after creation")
	}
	
	if retrievedAuction.Item.Title != "Test Item" {
		t.Errorf("Expected item title 'Test Item', got %s", retrievedAuction.Item.Title)
	}
	
	if retrievedAuction.Status != Pending {
		t.Errorf("Expected auction status to be Pending, got %d", retrievedAuction.Status)
	}
	
	err = ah.CreateAuction(auction)
	if err == nil {
		t.Error("Expected error when creating duplicate auction")
	}
}

func TestPlaceBid(t *testing.T) {
	ah := NewAuctionHouse()
	
	user := &User{
		ID:       "user1",
		Username: "bidder",
		Balance:  1000.0,
		IsActive: true,
	}
	ah.RegisterUser(user)
	
	item := Item{
		ID:           "item1",
		Title:        "Test Item",
		StartPrice:   10.0,
		ReservePrice: 50.0,
		Category:     "Electronics",
		SellerID:     "seller1",
	}
	
	auction := &Auction{
		ID:            "auction1",
		Item:          item,
		StartTime:     time.Now().Add(-1 * time.Minute),
		EndTime:       time.Now().Add(1 * time.Hour),
		Status:        Active,
		BidIncrement:  1.0,
		ExtensionTime: 5 * time.Minute,
	}
	
	ah.CreateAuction(auction)
	auction.Status = Active
	
	ctx := context.Background()
	bid, err := ah.PlaceBid(ctx, "auction1", "user1", 15.0)
	
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	
	if bid.Amount != 15.0 {
		t.Errorf("Expected bid amount 15.0, got %f", bid.Amount)
	}
	
	if bid.BidderID != "user1" {
		t.Errorf("Expected bidder ID 'user1', got %s", bid.BidderID)
	}
	
	retrievedAuction, _ := ah.GetAuction("auction1")
	if retrievedAuction.CurrentBid.Amount != 15.0 {
		t.Errorf("Expected current bid amount 15.0, got %f", retrievedAuction.CurrentBid.Amount)
	}
	
	_, err = ah.PlaceBid(ctx, "auction1", "user1", 12.0)
	if err == nil {
		t.Error("Expected error when placing lower bid")
	}
}

func TestBidValidation(t *testing.T) {
	validator := &BidValidator{
		minBidIncrement: 1.0,
		maxBidAmount:    1000.0,
	}
	
	user := &User{
		ID:      "user1",
		Balance: 100.0,
	}
	
	auction := &Auction{
		CurrentBid: &Bid{Amount: 50.0},
	}
	
	if validator.ValidateBid(auction, user, 0) {
		t.Error("Expected validation to fail for zero bid")
	}
	
	if validator.ValidateBid(auction, user, 2000.0) {
		t.Error("Expected validation to fail for bid exceeding max amount")
	}
	
	if validator.ValidateBid(auction, user, 150.0) {
		t.Error("Expected validation to fail for bid exceeding user balance")
	}
	
	if validator.ValidateBid(auction, user, 50.5) {
		t.Error("Expected validation to fail for insufficient increment")
	}
	
	if !validator.ValidateBid(auction, user, 52.0) {
		t.Error("Expected validation to pass for valid bid")
	}
}

func TestAutoBid(t *testing.T) {
	ah := NewAuctionHouse()
	
	user1 := &User{ID: "user1", Balance: 1000.0}
	user2 := &User{ID: "user2", Balance: 1000.0}
	ah.RegisterUser(user1)
	ah.RegisterUser(user2)
	
	item := Item{
		ID:         "item1",
		StartPrice: 10.0,
		SellerID:   "seller1",
	}
	
	auction := &Auction{
		ID:            "auction1",
		Item:          item,
		StartTime:     time.Now().Add(-1 * time.Minute),
		EndTime:       time.Now().Add(1 * time.Hour),
		Status:        Active,
		BidIncrement:  1.0,
		ExtensionTime: 5 * time.Minute,
	}
	
	ah.CreateAuction(auction)
	auction.Status = Active
	
	err := ah.SetupAutoBid("auction1", "user2", 100.0, 5.0)
	if err != nil {
		t.Errorf("Expected no error setting up auto bid, got %v", err)
	}
	
	ctx := context.Background()
	_, err = ah.PlaceBid(ctx, "auction1", "user1", 15.0)
	if err != nil {
		t.Errorf("Expected no error placing bid, got %v", err)
	}
	
	time.Sleep(100 * time.Millisecond)
	
	retrievedAuction, _ := ah.GetAuction("auction1")
	if retrievedAuction.CurrentBid.Amount <= 15.0 {
		t.Error("Expected auto bid to be triggered and increase current bid")
	}
	
	if retrievedAuction.CurrentBid.BidderID != "user2" {
		t.Error("Expected auto bidder to become current high bidder")
	}
}

func TestWatchAuction(t *testing.T) {
	ah := NewAuctionHouse()
	
	item := Item{ID: "item1", StartPrice: 10.0, SellerID: "seller1"}
	auction := &Auction{
		ID:   "auction1",
		Item: item,
	}
	
	ah.CreateAuction(auction)
	
	err := ah.WatchAuction("auction1", "user1")
	if err != nil {
		t.Errorf("Expected no error watching auction, got %v", err)
	}
	
	retrievedAuction, _ := ah.GetAuction("auction1")
	watchers := retrievedAuction.GetWatchers()
	
	found := false
	for _, watcher := range watchers {
		if watcher == "user1" {
			found = true
			break
		}
	}
	
	if !found {
		t.Error("Expected user1 to be in watchers list")
	}
	
	err = ah.UnwatchAuction("auction1", "user1")
	if err != nil {
		t.Errorf("Expected no error unwatching auction, got %v", err)
	}
	
	watchers = retrievedAuction.GetWatchers()
	if len(watchers) != 0 {
		t.Error("Expected watchers list to be empty after unwatching")
	}
}

func TestAuctionObserver(t *testing.T) {
	observer := NewConsoleObserver("test")
	
	item := Item{
		ID:         "item1",
		StartPrice: 10.0,
		SellerID:   "seller1",
	}
	
	auction := &Auction{
		ID:     "auction1",
		Item:   item,
		Status: Active,
	}
	
	auction.AddObserver(observer)
	
	bid := &Bid{
		ID:       "bid1",
		BidderID: "user1",
		Amount:   15.0,
		Type:     RegularBid,
	}
	
	err := auction.PlaceBid(bid)
	if err != nil {
		t.Errorf("Expected no error placing bid, got %v", err)
	}
	
	auction.RemoveObserver(observer)
	
	bid2 := &Bid{
		ID:       "bid2",
		BidderID: "user2",
		Amount:   20.0,
		Type:     RegularBid,
	}
	
	err = auction.PlaceBid(bid2)
	if err != nil {
		t.Errorf("Expected no error placing bid after removing observer, got %v", err)
	}
}

func TestEventBus(t *testing.T) {
	eb := NewEventBus()
	
	ch := eb.Subscribe("test_event")
	
	event := AuctionEvent{
		Type:      "test_event",
		AuctionID: "auction1",
		UserID:    "user1",
		Timestamp: time.Now(),
	}
	
	eb.Publish("test_event", event)
	
	select {
	case receivedEvent := <-ch:
		if receivedEvent.Type != "test_event" {
			t.Errorf("Expected event type 'test_event', got %s", receivedEvent.Type)
		}
		if receivedEvent.AuctionID != "auction1" {
			t.Errorf("Expected auction ID 'auction1', got %s", receivedEvent.AuctionID)
		}
	case <-time.After(1 * time.Second):
		t.Error("Expected to receive event within 1 second")
	}
}

func TestUserBalance(t *testing.T) {
	user := &User{
		ID:      "user1",
		Balance: 100.0,
	}
	
	if user.GetBalance() != 100.0 {
		t.Errorf("Expected balance 100.0, got %f", user.GetBalance())
	}
	
	user.AddBalance(50.0)
	if user.GetBalance() != 150.0 {
		t.Errorf("Expected balance 150.0 after adding 50.0, got %f", user.GetBalance())
	}
	
	err := user.DeductBalance(75.0)
	if err != nil {
		t.Errorf("Expected no error deducting 75.0, got %v", err)
	}
	
	if user.GetBalance() != 75.0 {
		t.Errorf("Expected balance 75.0 after deducting 75.0, got %f", user.GetBalance())
	}
	
	err = user.DeductBalance(100.0)
	if err == nil {
		t.Error("Expected error when deducting more than balance")
	}
}

func TestPaymentProcessor(t *testing.T) {
	pp := NewPaymentProcessor()
	
	transaction := &Transaction{
		ID:        "txn1",
		UserID:    "user1",
		AuctionID: "auction1",
		Amount:    50.0,
		CreatedAt: time.Now(),
	}
	
	err := pp.ProcessTransaction(transaction)
	if err != nil {
		t.Errorf("Expected no error processing transaction, got %v", err)
	}
	
	retrievedTxn, exists := pp.GetTransaction("txn1")
	if !exists {
		t.Error("Expected transaction to exist after processing")
	}
	
	if retrievedTxn.Amount != 50.0 {
		t.Errorf("Expected transaction amount 50.0, got %f", retrievedTxn.Amount)
	}
	
	time.Sleep(200 * time.Millisecond)
	
	retrievedTxn, _ = pp.GetTransaction("txn1")
	if retrievedTxn.Status != "completed" {
		t.Errorf("Expected transaction status 'completed', got %s", retrievedTxn.Status)
	}
}

func TestNotificationService(t *testing.T) {
	ns := NewNotificationService()
	
	ch := ns.Subscribe("user1")
	
	notification := &Notification{
		UserID:    "user1",
		Type:      "auction_won",
		Message:   "You won the auction!",
		Timestamp: time.Now(),
	}
	
	ns.SendNotification(notification)
	
	select {
	case receivedNotification := <-ch:
		if receivedNotification.Type != "auction_won" {
			t.Errorf("Expected notification type 'auction_won', got %s", receivedNotification.Type)
		}
		if receivedNotification.Message != "You won the auction!" {
			t.Errorf("Expected message 'You won the auction!', got %s", receivedNotification.Message)
		}
	case <-time.After(1 * time.Second):
		t.Error("Expected to receive notification within 1 second")
	}
}

func TestAuctionStatistics(t *testing.T) {
	ah := NewAuctionHouse()
	
	item1 := Item{ID: "item1", StartPrice: 10.0, SellerID: "seller1"}
	item2 := Item{ID: "item2", StartPrice: 20.0, SellerID: "seller1"}
	
	auction1 := &Auction{
		ID:     "auction1",
		Item:   item1,
		Status: Active,
	}
	
	auction2 := &Auction{
		ID:     "auction2",
		Item:   item2,
		Status: Ended,
		CurrentBid: &Bid{Amount: 50.0},
	}
	
	ah.CreateAuction(auction1)
	ah.CreateAuction(auction2)
	
	stats := ah.GetAuctionStatistics()
	
	if stats["total_auctions"].(int) != 2 {
		t.Errorf("Expected 2 total auctions, got %d", stats["total_auctions"].(int))
	}
	
	if stats["active_auctions"].(int) != 1 {
		t.Errorf("Expected 1 active auction, got %d", stats["active_auctions"].(int))
	}
	
	if stats["ended_auctions"].(int) != 1 {
		t.Errorf("Expected 1 ended auction, got %d", stats["ended_auctions"].(int))
	}
}

func TestSearchAuctions(t *testing.T) {
	ah := NewAuctionHouse()
	
	item1 := Item{
		ID:          "item1",
		Title:       "iPhone 12",
		Category:    "Electronics",
		StartPrice:  100.0,
		SellerID:    "seller1",
	}
	
	item2 := Item{
		ID:          "item2",
		Title:       "Samsung Galaxy",
		Category:    "Electronics",
		StartPrice:  150.0,
		SellerID:    "seller1",
	}
	
	item3 := Item{
		ID:          "item3",
		Title:       "Book",
		Category:    "Books",
		StartPrice:  10.0,
		SellerID:    "seller1",
	}
	
	auction1 := &Auction{ID: "auction1", Item: item1}
	auction2 := &Auction{ID: "auction2", Item: item2}
	auction3 := &Auction{ID: "auction3", Item: item3}
	
	ah.CreateAuction(auction1)
	ah.CreateAuction(auction2)
	ah.CreateAuction(auction3)
	
	results := ah.SearchAuctions("iPhone", "", 0, 1000)
	if len(results) != 1 {
		t.Errorf("Expected 1 result for iPhone search, got %d", len(results))
	}
	
	results = ah.SearchAuctions("", "Electronics", 0, 1000)
	if len(results) != 2 {
		t.Errorf("Expected 2 results for Electronics category, got %d", len(results))
	}
	
	results = ah.SearchAuctions("", "", 50, 200)
	if len(results) != 2 {
		t.Errorf("Expected 2 results for price range 50-200, got %d", len(results))
	}
}

func TestConcurrentBidding(t *testing.T) {
	ah := NewAuctionHouse()
	
	for i := 0; i < 10; i++ {
		user := &User{
			ID:      fmt.Sprintf("user%d", i),
			Balance: 1000.0,
		}
		ah.RegisterUser(user)
	}
	
	item := Item{
		ID:         "item1",
		StartPrice: 10.0,
		SellerID:   "seller1",
	}
	
	auction := &Auction{
		ID:            "auction1",
		Item:          item,
		StartTime:     time.Now().Add(-1 * time.Minute),
		EndTime:       time.Now().Add(1 * time.Hour),
		Status:        Active,
		BidIncrement:  1.0,
		ExtensionTime: 5 * time.Minute,
	}
	
	ah.CreateAuction(auction)
	auction.Status = Active
	
	var wg sync.WaitGroup
	ctx := context.Background()
	
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(userID int) {
			defer wg.Done()
			
			for j := 0; j < 5; j++ {
				bidAmount := float64(20 + userID*10 + j)
				ah.PlaceBid(ctx, "auction1", fmt.Sprintf("user%d", userID), bidAmount)
				time.Sleep(10 * time.Millisecond)
			}
		}(i)
	}
	
	wg.Wait()
	
	retrievedAuction, _ := ah.GetAuction("auction1")
	if retrievedAuction.CurrentBid == nil {
		t.Error("Expected auction to have a current bid after concurrent bidding")
	}
	
	if len(retrievedAuction.BidHistory) == 0 {
		t.Error("Expected auction to have bid history after concurrent bidding")
	}
}

func TestAuctionScheduler(t *testing.T) {
	ah := NewAuctionHouse()
	
	item := Item{
		ID:         "item1",
		StartPrice: 10.0,
		SellerID:   "seller1",
	}
	
	auction := &Auction{
		ID:        "auction1",
		Item:      item,
		StartTime: time.Now().Add(100 * time.Millisecond),
		EndTime:   time.Now().Add(300 * time.Millisecond),
	}
	
	ah.CreateAuction(auction)
	
	if auction.Status != Pending {
		t.Error("Expected auction to be pending initially")
	}
	
	time.Sleep(150 * time.Millisecond)
	
	if auction.Status != Active {
		t.Error("Expected auction to be active after start time")
	}
	
	time.Sleep(200 * time.Millisecond)
	
	if auction.Status != Ended {
		t.Error("Expected auction to be ended after end time")
	}
}

func TestAuctionManager(t *testing.T) {
	ah := NewAuctionHouse()
	manager := NewAuctionManager(ah)
	
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	manager.Start(ctx)
	defer manager.Stop()
	
	item := Item{
		ID:         "item1",
		StartPrice: 10.0,
		SellerID:   "seller1",
	}
	
	auction := &Auction{
		ID:        "auction1",
		Item:      item,
		StartTime: time.Now().Add(-1 * time.Minute),
		EndTime:   time.Now().Add(-1 * time.Second),
		Status:    Active,
	}
	
	ah.CreateAuction(auction)
	auction.Status = Active
	
	time.Sleep(2 * time.Second)
	
	if auction.Status != Ended {
		t.Error("Expected auction manager to end expired auction")
	}
}

func BenchmarkPlaceBid(b *testing.B) {
	ah := NewAuctionHouse()
	
	user := &User{
		ID:      "user1",
		Balance: 1000000.0,
	}
	ah.RegisterUser(user)
	
	item := Item{
		ID:         "item1",
		StartPrice: 10.0,
		SellerID:   "seller1",
	}
	
	auction := &Auction{
		ID:            "auction1",
		Item:          item,
		Status:        Active,
		BidIncrement:  1.0,
		ExtensionTime: 5 * time.Minute,
		EndTime:       time.Now().Add(1 * time.Hour),
	}
	
	ah.CreateAuction(auction)
	auction.Status = Active
	
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ah.PlaceBid(ctx, "auction1", "user1", float64(20+i))
	}
}

func BenchmarkConcurrentBidding(b *testing.B) {
	ah := NewAuctionHouse()
	
	for i := 0; i < 100; i++ {
		user := &User{
			ID:      fmt.Sprintf("user%d", i),
			Balance: 1000000.0,
		}
		ah.RegisterUser(user)
	}
	
	item := Item{
		ID:         "item1",
		StartPrice: 10.0,
		SellerID:   "seller1",
	}
	
	auction := &Auction{
		ID:            "auction1",
		Item:          item,
		Status:        Active,
		BidIncrement:  1.0,
		ExtensionTime: 5 * time.Minute,
		EndTime:       time.Now().Add(1 * time.Hour),
	}
	
	ah.CreateAuction(auction)
	auction.Status = Active
	
	ctx := context.Background()
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			userID := fmt.Sprintf("user%d", i%100)
			ah.PlaceBid(ctx, "auction1", userID, float64(20+i))
			i++
		}
	})
}