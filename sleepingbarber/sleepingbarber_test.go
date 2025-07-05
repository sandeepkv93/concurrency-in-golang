package sleepingbarber

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBasicBarberShop(t *testing.T) {
	shop := NewBarberShop("Test Shop", 3)
	shop.Open()
	
	// Add a few customers
	customer1 := &Customer{
		id:          1,
		name:        "Customer 1",
		arrivalTime: time.Now(),
		serviceTime: 50 * time.Millisecond,
	}
	
	customer2 := &Customer{
		id:          2,
		name:        "Customer 2",
		arrivalTime: time.Now(),
		serviceTime: 50 * time.Millisecond,
	}
	
	// Add customers
	if !shop.AddCustomer(customer1) {
		t.Error("Failed to add customer 1")
	}
	
	if !shop.AddCustomer(customer2) {
		t.Error("Failed to add customer 2")
	}
	
	// Wait for service
	time.Sleep(200 * time.Millisecond)
	
	// Close shop
	shop.Close()
	
	// Check statistics
	stats := shop.GetStatistics()
	if stats.ServedCustomers != 2 {
		t.Errorf("Expected 2 served customers, got %d", stats.ServedCustomers)
	}
	
	if stats.TurnedAway != 0 {
		t.Errorf("Expected 0 turned away, got %d", stats.TurnedAway)
	}
}

func TestWaitingRoomCapacity(t *testing.T) {
	shop := NewBarberShop("Small Shop", 2) // Small waiting room
	shop.Open()
	
	// Create customers with long service time
	customers := make([]*Customer, 5)
	for i := 0; i < 5; i++ {
		customers[i] = &Customer{
			id:          i,
			name:        fmt.Sprintf("Customer %d", i),
			arrivalTime: time.Now(),
			serviceTime: 100 * time.Millisecond,
		}
	}
	
	// Add all customers quickly
	accepted := 0
	for _, customer := range customers {
		if shop.AddCustomer(customer) {
			accepted++
		}
	}
	
	// Should accept at most 3 (1 being served + 2 waiting)
	if accepted > 3 {
		t.Errorf("Accepted %d customers, but waiting room size is 2", accepted)
	}
	
	// Wait and close
	time.Sleep(500 * time.Millisecond)
	shop.Close()
	
	stats := shop.GetStatistics()
	if stats.TurnedAway == 0 {
		t.Error("Expected some customers to be turned away")
	}
}

func TestBarberSleeping(t *testing.T) {
	shop := NewBarberShop("Quiet Shop", 5)
	shop.Open()
	
	// Let barber sleep for a while
	time.Sleep(200 * time.Millisecond)
	
	// Add a customer
	customer := &Customer{
		id:          1,
		name:        "Wake-up Customer",
		arrivalTime: time.Now(),
		serviceTime: 50 * time.Millisecond,
	}
	
	shop.AddCustomer(customer)
	
	// Wait for service
	time.Sleep(100 * time.Millisecond)
	
	shop.Close()
	
	stats := shop.GetStatistics()
	if stats.BarberSleepTime == 0 {
		t.Error("Barber should have some sleep time")
	}
	
	if stats.ServedCustomers != 1 {
		t.Errorf("Expected 1 served customer, got %d", stats.ServedCustomers)
	}
}

func TestCustomerGenerator(t *testing.T) {
	shop := NewBarberShop("Generated Shop", 3)
	shop.Open()
	
	generator := NewCustomerGenerator(shop, 50*time.Millisecond, 30*time.Millisecond)
	generator.Start()
	
	// Run for a while
	time.Sleep(500 * time.Millisecond)
	
	generator.Stop()
	shop.Close()
	
	stats := shop.GetStatistics()
	if stats.TotalCustomers == 0 {
		t.Error("No customers were generated")
	}
	
	if stats.ServedCustomers == 0 {
		t.Error("No customers were served")
	}
	
	t.Logf("Generated %d customers, served %d, turned away %d",
		stats.TotalCustomers, stats.ServedCustomers, stats.TurnedAway)
}

func TestMultiBarberShop(t *testing.T) {
	shop := NewMultiBarberShop("Multi Shop", 3, 5)
	shop.Open()
	
	// Add many customers quickly
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			customer := &Customer{
				id:          id,
				name:        fmt.Sprintf("Customer %d", id),
				arrivalTime: time.Now(),
				serviceTime: 100 * time.Millisecond,
			}
			shop.AddCustomer(customer)
		}(i)
	}
	
	wg.Wait()
	
	// Wait for service
	time.Sleep(500 * time.Millisecond)
	
	shop.Close()
	
	// Check that multiple barbers worked
	activeBarbers := 0
	for _, barber := range shop.barbers {
		if atomic.LoadInt32(&barber.totalServed) > 0 {
			activeBarbers++
		}
	}
	
	if activeBarbers < 2 {
		t.Errorf("Expected at least 2 active barbers, got %d", activeBarbers)
	}
	
	totalServed := atomic.LoadInt32(&shop.servedCustomers)
	if totalServed == 0 {
		t.Error("No customers were served")
	}
}

func TestConcurrentCustomers(t *testing.T) {
	shop := NewBarberShop("Concurrent Shop", 10)
	shop.Open()
	
	// Many customers arriving simultaneously
	numCustomers := 20
	var wg sync.WaitGroup
	accepted := int32(0)
	
	for i := 0; i < numCustomers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			customer := &Customer{
				id:          id,
				name:        fmt.Sprintf("Customer %d", id),
				arrivalTime: time.Now(),
				serviceTime: 50 * time.Millisecond,
			}
			if shop.AddCustomer(customer) {
				atomic.AddInt32(&accepted, 1)
			}
		}(i)
	}
	
	wg.Wait()
	
	// Should accept at most 11 (1 being served + 10 waiting)
	if atomic.LoadInt32(&accepted) > 11 {
		t.Errorf("Accepted %d customers, but max should be 11", accepted)
	}
	
	// Wait for all to be served
	time.Sleep(1 * time.Second)
	
	shop.Close()
	
	stats := shop.GetStatistics()
	if stats.ServedCustomers != atomic.LoadInt32(&accepted) {
		t.Errorf("Served %d customers, but accepted %d", 
			stats.ServedCustomers, accepted)
	}
}

func TestShopClosing(t *testing.T) {
	shop := NewBarberShop("Closing Shop", 5)
	shop.Open()
	
	// Add customers continuously
	stopAdding := make(chan bool)
	go func() {
		i := 0
		for {
			select {
			case <-stopAdding:
				return
			default:
				customer := &Customer{
					id:          i,
					name:        fmt.Sprintf("Customer %d", i),
					arrivalTime: time.Now(),
					serviceTime: 50 * time.Millisecond,
				}
				shop.AddCustomer(customer)
				i++
				time.Sleep(30 * time.Millisecond)
			}
		}
	}()
	
	// Run for a bit
	time.Sleep(200 * time.Millisecond)
	
	// Stop adding customers and close shop
	close(stopAdding)
	shop.Close()
	
	// Verify shop is closed
	customer := &Customer{
		id:          999,
		name:        "Late Customer",
		arrivalTime: time.Now(),
		serviceTime: 50 * time.Millisecond,
	}
	
	if shop.AddCustomer(customer) {
		t.Error("Shop accepted customer after closing")
	}
}

func TestStatistics(t *testing.T) {
	shop := NewBarberShop("Stats Shop", 3)
	shop.Open()
	
	// Add customers with known service times
	serviceTime := 50 * time.Millisecond
	numCustomers := 5
	
	for i := 0; i < numCustomers; i++ {
		customer := &Customer{
			id:          i,
			name:        fmt.Sprintf("Customer %d", i),
			arrivalTime: time.Now(),
			serviceTime: serviceTime,
		}
		shop.AddCustomer(customer)
		time.Sleep(10 * time.Millisecond) // Small delay between arrivals
	}
	
	// Wait for all to be served
	time.Sleep(time.Duration(numCustomers) * serviceTime * 2)
	
	shop.Close()
	
	stats := shop.GetStatistics()
	
	// Verify statistics
	if stats.ServedCustomers != int32(numCustomers) {
		t.Errorf("Expected %d served customers, got %d", numCustomers, stats.ServedCustomers)
	}
	
	// Average service time should be close to serviceTime
	tolerance := 10 * time.Millisecond
	if stats.AvgServiceTime < serviceTime-tolerance || 
	   stats.AvgServiceTime > serviceTime+tolerance {
		t.Errorf("Average service time %v is not close to expected %v", 
			stats.AvgServiceTime, serviceTime)
	}
}

func BenchmarkBarberShop(b *testing.B) {
	configurations := []struct {
		name            string
		waitingRoomSize int
		arrivalRate     time.Duration
		serviceTime     time.Duration
	}{
		{"Small_FastService", 3, 50 * time.Millisecond, 30 * time.Millisecond},
		{"Large_SlowService", 10, 30 * time.Millisecond, 60 * time.Millisecond},
		{"Balanced", 5, 40 * time.Millisecond, 40 * time.Millisecond},
	}
	
	for _, config := range configurations {
		b.Run(config.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				shop := NewBarberShop("Bench Shop", config.waitingRoomSize)
				shop.Open()
				
				generator := NewCustomerGenerator(shop, config.arrivalRate, config.serviceTime)
				generator.Start()
				
				time.Sleep(500 * time.Millisecond)
				
				generator.Stop()
				shop.Close()
			}
		})
	}
}

func BenchmarkMultiBarberShop(b *testing.B) {
	numBarbers := []int{1, 3, 5}
	
	for _, n := range numBarbers {
		b.Run(fmt.Sprintf("Barbers_%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				shop := NewMultiBarberShop("Bench Multi Shop", n, 10)
				shop.Open()
				
				// Simulate customer arrivals
				stop := make(chan bool)
				go func() {
					id := 0
					for {
						select {
						case <-stop:
							return
						default:
							customer := &Customer{
								id:          id,
								name:        fmt.Sprintf("Customer %d", id),
								arrivalTime: time.Now(),
								serviceTime: 50 * time.Millisecond,
							}
							shop.AddCustomer(customer)
							id++
							time.Sleep(20 * time.Millisecond)
						}
					}
				}()
				
				time.Sleep(500 * time.Millisecond)
				close(stop)
				shop.Close()
			}
		})
	}
}