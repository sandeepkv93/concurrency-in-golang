package sleepingbarber

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// Shop interface for different types of shops
type Shop interface {
	AddCustomer(customer *Customer) bool
}

// BarberShop represents the barber shop
type BarberShop struct {
	name              string
	waitingRoom       chan *Customer
	barberReady       chan bool
	customerReady     chan bool
	customerDone      chan bool
	barberDone        chan bool
	waitingRoomSize   int
	totalCustomers    int32
	servedCustomers   int32
	turnedAway        int32
	totalWaitTime     int64
	totalServiceTime  int64
	barberSleepTime   int64
	barberWorkTime    int64
	shopOpen          atomic.Bool
	wg                sync.WaitGroup
}

// Customer represents a customer
type Customer struct {
	id          int
	name        string
	arrivalTime time.Time
	serviceTime time.Duration
}

// NewBarberShop creates a new barber shop
func NewBarberShop(name string, waitingRoomSize int) *BarberShop {
	return &BarberShop{
		name:            name,
		waitingRoom:     make(chan *Customer, waitingRoomSize),
		barberReady:     make(chan bool, 1),
		customerReady:   make(chan bool, 1),
		customerDone:    make(chan bool, 1),
		barberDone:      make(chan bool, 1),
		waitingRoomSize: waitingRoomSize,
	}
}

// Open opens the barber shop
func (bs *BarberShop) Open() {
	bs.shopOpen.Store(true)
	bs.wg.Add(1)
	go bs.barber()
}

// Close closes the barber shop
func (bs *BarberShop) Close() {
	bs.shopOpen.Store(false)
	
	// Wake up barber if sleeping
	select {
	case bs.waitingRoom <- nil:
	default:
	}
	
	bs.wg.Wait()
}

// barber represents the barber's routine
func (bs *BarberShop) barber() {
	defer bs.wg.Done()
	
	fmt.Printf("Barber at %s is ready for work\n", bs.name)
	
	for bs.shopOpen.Load() || len(bs.waitingRoom) > 0 {
		// Check if there are customers waiting
		select {
		case customer := <-bs.waitingRoom:
			if customer == nil {
				// Shop closing signal
				continue
			}
			
			// Customer available, start cutting
			bs.cutHair(customer)
			
		case <-time.After(100 * time.Millisecond):
			// No customers, barber sleeps
			sleepStart := time.Now()
			
			// Wait for customer or shop closing
			customer := <-bs.waitingRoom
			
			atomic.AddInt64(&bs.barberSleepTime, int64(time.Since(sleepStart)))
			
			if customer != nil && bs.shopOpen.Load() {
				// Woken up by customer
				fmt.Printf("Barber woken up by %s\n", customer.name)
				bs.cutHair(customer)
			}
		}
	}
	
	fmt.Printf("Barber at %s is going home\n", bs.name)
}

// cutHair simulates cutting a customer's hair
func (bs *BarberShop) cutHair(customer *Customer) {
	workStart := time.Now()
	
	// Signal customer that barber is ready
	select {
	case bs.barberReady <- true:
	default:
	}
	
	// Wait for customer to sit down
	<-bs.customerReady
	
	// Cut hair
	fmt.Printf("Barber is cutting %s's hair\n", customer.name)
	time.Sleep(customer.serviceTime)
	
	// Signal that haircut is done
	bs.barberDone <- true
	
	// Wait for customer to leave chair
	<-bs.customerDone
	
	atomic.AddInt64(&bs.barberWorkTime, int64(time.Since(workStart)))
	atomic.AddInt32(&bs.servedCustomers, 1)
	
	// Update service time
	atomic.AddInt64(&bs.totalServiceTime, int64(customer.serviceTime))
	
	// Update wait time
	waitTime := time.Since(customer.arrivalTime) - customer.serviceTime
	atomic.AddInt64(&bs.totalWaitTime, int64(waitTime))
	
	fmt.Printf("Barber finished cutting %s's hair\n", customer.name)
}

// AddCustomer adds a customer to the shop
func (bs *BarberShop) AddCustomer(customer *Customer) bool {
	if !bs.shopOpen.Load() {
		return false
	}
	
	atomic.AddInt32(&bs.totalCustomers, 1)
	
	// Try to enter waiting room
	select {
	case bs.waitingRoom <- customer:
		fmt.Printf("%s entered the waiting room\n", customer.name)
		
		// Customer waits for service
		bs.wg.Add(1)
		go bs.customerRoutine(customer)
		return true
		
	default:
		// Waiting room is full
		fmt.Printf("%s left because waiting room is full\n", customer.name)
		atomic.AddInt32(&bs.turnedAway, 1)
		return false
	}
}

// customerRoutine represents a customer getting a haircut
func (bs *BarberShop) customerRoutine(customer *Customer) {
	defer bs.wg.Done()
	
	// Wait for barber to be ready
	<-bs.barberReady
	
	// Sit in barber chair
	fmt.Printf("%s is sitting in the barber chair\n", customer.name)
	bs.customerReady <- true
	
	// Wait for haircut to finish
	<-bs.barberDone
	
	// Leave the chair
	fmt.Printf("%s is happy with the haircut and leaving\n", customer.name)
	bs.customerDone <- true
}

// GetStatistics returns shop statistics
func (bs *BarberShop) GetStatistics() Statistics {
	totalCustomers := atomic.LoadInt32(&bs.totalCustomers)
	servedCustomers := atomic.LoadInt32(&bs.servedCustomers)
	turnedAway := atomic.LoadInt32(&bs.turnedAway)
	
	stats := Statistics{
		TotalCustomers:  totalCustomers,
		ServedCustomers: servedCustomers,
		TurnedAway:      turnedAway,
		BarberSleepTime: time.Duration(atomic.LoadInt64(&bs.barberSleepTime)),
		BarberWorkTime:  time.Duration(atomic.LoadInt64(&bs.barberWorkTime)),
	}
	
	if servedCustomers > 0 {
		stats.AvgWaitTime = time.Duration(atomic.LoadInt64(&bs.totalWaitTime) / int64(servedCustomers))
		stats.AvgServiceTime = time.Duration(atomic.LoadInt64(&bs.totalServiceTime) / int64(servedCustomers))
	}
	
	return stats
}

// Statistics holds barber shop statistics
type Statistics struct {
	TotalCustomers  int32
	ServedCustomers int32
	TurnedAway      int32
	AvgWaitTime     time.Duration
	AvgServiceTime  time.Duration
	BarberSleepTime time.Duration
	BarberWorkTime  time.Duration
}

// CustomerGenerator generates customers at random intervals
type CustomerGenerator struct {
	shop            Shop
	avgArrivalTime  time.Duration
	avgServiceTime  time.Duration
	stopChan        chan bool
	wg              sync.WaitGroup
}

// NewCustomerGenerator creates a new customer generator
func NewCustomerGenerator(shop Shop, avgArrivalTime, avgServiceTime time.Duration) *CustomerGenerator {
	return &CustomerGenerator{
		shop:           shop,
		avgArrivalTime: avgArrivalTime,
		avgServiceTime: avgServiceTime,
		stopChan:       make(chan bool),
	}
}

// Start begins generating customers
func (cg *CustomerGenerator) Start() {
	cg.wg.Add(1)
	go cg.generateCustomers()
}

// Stop stops generating customers
func (cg *CustomerGenerator) Stop() {
	close(cg.stopChan)
	cg.wg.Wait()
}

func (cg *CustomerGenerator) generateCustomers() {
	defer cg.wg.Done()
	
	customerID := 0
	
	for {
		select {
		case <-cg.stopChan:
			return
		default:
			// Random arrival time
			arrivalTime := time.Duration(float64(cg.avgArrivalTime) * (0.5 + rand.Float64()))
			time.Sleep(arrivalTime)
			
			// Create customer
			customerID++
			customer := &Customer{
				id:          customerID,
				name:        fmt.Sprintf("Customer %d", customerID),
				arrivalTime: time.Now(),
				serviceTime: time.Duration(float64(cg.avgServiceTime) * (0.5 + rand.Float64())),
			}
			
			// Try to add customer
			cg.shop.AddCustomer(customer)
		}
	}
}

// MultiBarberShop represents a shop with multiple barbers
type MultiBarberShop struct {
	name            string
	numBarbers      int
	waitingRoom     chan *Customer
	barberAvailable chan int
	waitingRoomSize int
	barbers         []*Barber
	shopOpen        atomic.Bool
	wg              sync.WaitGroup
	totalCustomers  int32
	servedCustomers int32
	turnedAway      int32
}

// Barber represents a barber in multi-barber shop
type Barber struct {
	id            int
	shop          *MultiBarberShop
	totalServed   int32
	totalWorkTime int64
}

// NewMultiBarberShop creates a shop with multiple barbers
func NewMultiBarberShop(name string, numBarbers, waitingRoomSize int) *MultiBarberShop {
	shop := &MultiBarberShop{
		name:            name,
		numBarbers:      numBarbers,
		waitingRoom:     make(chan *Customer, waitingRoomSize),
		barberAvailable: make(chan int, numBarbers),
		waitingRoomSize: waitingRoomSize,
		barbers:         make([]*Barber, numBarbers),
	}
	
	// Create barbers
	for i := 0; i < numBarbers; i++ {
		shop.barbers[i] = &Barber{
			id:   i,
			shop: shop,
		}
	}
	
	return shop
}

// Open opens the multi-barber shop
func (mbs *MultiBarberShop) Open() {
	mbs.shopOpen.Store(true)
	
	// Start all barbers
	for _, barber := range mbs.barbers {
		mbs.wg.Add(1)
		go mbs.barberRoutine(barber)
	}
}

// Close closes the multi-barber shop
func (mbs *MultiBarberShop) Close() {
	mbs.shopOpen.Store(false)
	
	// Wake up all barbers
	for i := 0; i < mbs.numBarbers; i++ {
		select {
		case mbs.waitingRoom <- nil:
		default:
		}
	}
	
	mbs.wg.Wait()
}

func (mbs *MultiBarberShop) barberRoutine(barber *Barber) {
	defer mbs.wg.Done()
	
	fmt.Printf("Barber %d is ready for work\n", barber.id)
	
	for mbs.shopOpen.Load() || len(mbs.waitingRoom) > 0 {
		// Signal availability
		select {
		case mbs.barberAvailable <- barber.id:
		default:
		}
		
		// Wait for customer
		customer := <-mbs.waitingRoom
		if customer == nil {
			continue
		}
		
		// Serve customer
		startTime := time.Now()
		fmt.Printf("Barber %d is cutting %s's hair\n", barber.id, customer.name)
		time.Sleep(customer.serviceTime)
		fmt.Printf("Barber %d finished cutting %s's hair\n", barber.id, customer.name)
		
		atomic.AddInt32(&barber.totalServed, 1)
		atomic.AddInt64(&barber.totalWorkTime, int64(time.Since(startTime)))
		atomic.AddInt32(&mbs.servedCustomers, 1)
	}
	
	fmt.Printf("Barber %d is going home\n", barber.id)
}

// AddCustomer adds a customer to the multi-barber shop
func (mbs *MultiBarberShop) AddCustomer(customer *Customer) bool {
	if !mbs.shopOpen.Load() {
		return false
	}
	
	atomic.AddInt32(&mbs.totalCustomers, 1)
	
	// Try to enter waiting room
	select {
	case mbs.waitingRoom <- customer:
		fmt.Printf("%s entered the waiting room (multi-barber shop)\n", customer.name)
		return true
	default:
		fmt.Printf("%s left because waiting room is full (multi-barber shop)\n", customer.name)
		atomic.AddInt32(&mbs.turnedAway, 1)
		return false
	}
}

// Example demonstrates the sleeping barber problem
func Example() {
	fmt.Println("=== Sleeping Barber Problem ===")
	
	// Single barber shop
	fmt.Println("\n--- Single Barber Shop ---")
	shop := NewBarberShop("Classic Cuts", 3)
	shop.Open()
	
	// Generate customers
	generator := NewCustomerGenerator(shop, 200*time.Millisecond, 150*time.Millisecond)
	generator.Start()
	
	// Run for 2 seconds
	time.Sleep(2 * time.Second)
	
	// Stop generating customers
	generator.Stop()
	
	// Close shop
	shop.Close()
	
	// Print statistics
	stats := shop.GetStatistics()
	fmt.Printf("\nSingle Barber Shop Statistics:\n")
	fmt.Printf("Total customers: %d\n", stats.TotalCustomers)
	fmt.Printf("Served customers: %d\n", stats.ServedCustomers)
	fmt.Printf("Turned away: %d\n", stats.TurnedAway)
	fmt.Printf("Average wait time: %v\n", stats.AvgWaitTime)
	fmt.Printf("Average service time: %v\n", stats.AvgServiceTime)
	fmt.Printf("Barber sleep time: %v\n", stats.BarberSleepTime)
	fmt.Printf("Barber work time: %v\n", stats.BarberWorkTime)
	
	// Multi-barber shop
	fmt.Println("\n--- Multi-Barber Shop (3 barbers) ---")
	multiShop := NewMultiBarberShop("Modern Cuts", 3, 5)
	multiShop.Open()
	
	// Generate customers
	multiGenerator := NewCustomerGenerator(multiShop, 100*time.Millisecond, 200*time.Millisecond)
	multiGenerator.Start()
	
	// Run for 2 seconds
	time.Sleep(2 * time.Second)
	
	// Stop
	multiGenerator.Stop()
	multiShop.Close()
	
	// Print statistics
	fmt.Printf("\nMulti-Barber Shop Statistics:\n")
	fmt.Printf("Total customers: %d\n", atomic.LoadInt32(&multiShop.totalCustomers))
	fmt.Printf("Served customers: %d\n", atomic.LoadInt32(&multiShop.servedCustomers))
	fmt.Printf("Turned away: %d\n", atomic.LoadInt32(&multiShop.turnedAway))
	
	for _, barber := range multiShop.barbers {
		fmt.Printf("Barber %d served: %d customers\n", 
			barber.id, atomic.LoadInt32(&barber.totalServed))
	}
}