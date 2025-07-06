package readerswriters

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Database represents a shared resource with read/write access
type Database struct {
	data            map[string]interface{}
	strategy        RWStrategy
	readCount       int32
	writeCount      int32
	totalReads      int64
	totalWrites     int64
	readWaitTime    int64
	writeWaitTime   int64
}

// RWStrategy defines the strategy for handling readers and writers
type RWStrategy interface {
	StartRead()
	EndRead()
	StartWrite()
	EndWrite()
	Name() string
}

// NewDatabase creates a new database with the given strategy
func NewDatabase(strategy RWStrategy) *Database {
	return &Database{
		data:     make(map[string]interface{}),
		strategy: strategy,
	}
}

// Read performs a read operation
func (db *Database) Read(key string) (interface{}, bool) {
	start := time.Now()
	db.strategy.StartRead()
	atomic.AddInt64(&db.readWaitTime, int64(time.Since(start)))
	
	atomic.AddInt32(&db.readCount, 1)
	defer func() {
		atomic.AddInt32(&db.readCount, -1)
		db.strategy.EndRead()
		atomic.AddInt64(&db.totalReads, 1)
	}()
	
	// Simulate read operation
	time.Sleep(time.Duration(10+time.Now().UnixNano()%10) * time.Millisecond)
	
	value, exists := db.data[key]
	return value, exists
}

// Write performs a write operation
func (db *Database) Write(key string, value interface{}) {
	start := time.Now()
	db.strategy.StartWrite()
	atomic.AddInt64(&db.writeWaitTime, int64(time.Since(start)))
	
	atomic.AddInt32(&db.writeCount, 1)
	defer func() {
		atomic.AddInt32(&db.writeCount, -1)
		db.strategy.EndWrite()
		atomic.AddInt64(&db.totalWrites, 1)
	}()
	
	// Simulate write operation
	time.Sleep(time.Duration(20+time.Now().UnixNano()%10) * time.Millisecond)
	
	db.data[key] = value
}

// GetStatistics returns database access statistics
func (db *Database) GetStatistics() Statistics {
	totalReads := atomic.LoadInt64(&db.totalReads)
	totalWrites := atomic.LoadInt64(&db.totalWrites)
	
	stats := Statistics{
		TotalReads:  totalReads,
		TotalWrites: totalWrites,
	}
	
	if totalReads > 0 {
		stats.AvgReadWaitTime = time.Duration(atomic.LoadInt64(&db.readWaitTime) / totalReads)
	}
	
	if totalWrites > 0 {
		stats.AvgWriteWaitTime = time.Duration(atomic.LoadInt64(&db.writeWaitTime) / totalWrites)
	}
	
	return stats
}

// Statistics holds access statistics
type Statistics struct {
	TotalReads       int64
	TotalWrites      int64
	AvgReadWaitTime  time.Duration
	AvgWriteWaitTime time.Duration
}

// Strategies

// ReadersPreferenceStrategy gives preference to readers
type ReadersPreferenceStrategy struct {
	mutex      sync.Mutex
	readCount  int
	writeMutex sync.Mutex
}

func (rp *ReadersPreferenceStrategy) StartRead() {
	rp.mutex.Lock()
	rp.readCount++
	if rp.readCount == 1 {
		rp.writeMutex.Lock()
	}
	rp.mutex.Unlock()
}

func (rp *ReadersPreferenceStrategy) EndRead() {
	rp.mutex.Lock()
	rp.readCount--
	if rp.readCount == 0 {
		rp.writeMutex.Unlock()
	}
	rp.mutex.Unlock()
}

func (rp *ReadersPreferenceStrategy) StartWrite() {
	rp.writeMutex.Lock()
}

func (rp *ReadersPreferenceStrategy) EndWrite() {
	rp.writeMutex.Unlock()
}

func (rp *ReadersPreferenceStrategy) Name() string {
	return "Readers Preference"
}

// WritersPreferenceStrategy gives preference to writers
type WritersPreferenceStrategy struct {
	readMutex    sync.Mutex
	writeMutex   sync.Mutex
	readTry      sync.Mutex
	resource     sync.Mutex
	readCount    int
	writeCount   int
}

func (wp *WritersPreferenceStrategy) StartRead() {
	wp.readTry.Lock()
	wp.readMutex.Lock()
	wp.readCount++
	if wp.readCount == 1 {
		wp.resource.Lock()
	}
	wp.readMutex.Unlock()
	wp.readTry.Unlock()
}

func (wp *WritersPreferenceStrategy) EndRead() {
	wp.readMutex.Lock()
	wp.readCount--
	if wp.readCount == 0 {
		wp.resource.Unlock()
	}
	wp.readMutex.Unlock()
}

func (wp *WritersPreferenceStrategy) StartWrite() {
	wp.writeMutex.Lock()
	wp.writeCount++
	if wp.writeCount == 1 {
		wp.readTry.Lock()
	}
	wp.writeMutex.Unlock()
	
	wp.resource.Lock()
}

func (wp *WritersPreferenceStrategy) EndWrite() {
	wp.resource.Unlock()
	
	wp.writeMutex.Lock()
	wp.writeCount--
	if wp.writeCount == 0 {
		wp.readTry.Unlock()
	}
	wp.writeMutex.Unlock()
}

func (wp *WritersPreferenceStrategy) Name() string {
	return "Writers Preference"
}

// FairStrategy provides fair access using a queue
type FairStrategy struct {
	queue      chan request
	done       chan bool
	resource   sync.RWMutex
	activeReads int32
	currentReq *request
}

type request struct {
	isWrite bool
	ready   chan bool
	done    chan bool
}

func NewFairStrategy() *FairStrategy {
	fs := &FairStrategy{
		queue: make(chan request, 1000),
		done:  make(chan bool),
	}
	go fs.scheduler()
	return fs
}

func (fs *FairStrategy) scheduler() {
	for {
		select {
		case req := <-fs.queue:
			if req.isWrite {
				// Wait for all readers to finish
				for atomic.LoadInt32(&fs.activeReads) > 0 {
					time.Sleep(time.Millisecond)
				}
				fs.resource.Lock()
				req.ready <- true
				<-req.done
				fs.resource.Unlock()
			} else {
				// Multiple readers can proceed
				atomic.AddInt32(&fs.activeReads, 1)
				fs.resource.RLock()
				req.ready <- true
				<-req.done
				fs.resource.RUnlock()
				atomic.AddInt32(&fs.activeReads, -1)
			}
		case <-fs.done:
			return
		}
	}
}

func (fs *FairStrategy) StartRead() {
	req := request{
		isWrite: false,
		ready:   make(chan bool),
		done:    make(chan bool),
	}
	fs.queue <- req
	<-req.ready
	fs.currentReq = &req
}

func (fs *FairStrategy) EndRead() {
	if fs.currentReq.done != nil {
		fs.currentReq.done <- true
	}
}

func (fs *FairStrategy) StartWrite() {
	req := request{
		isWrite: true,
		ready:   make(chan bool),
		done:    make(chan bool),
	}
	fs.queue <- req
	<-req.ready
	fs.currentReq = &req
}

func (fs *FairStrategy) EndWrite() {
	if fs.currentReq.done != nil {
		fs.currentReq.done <- true
	}
}

func (fs *FairStrategy) Name() string {
	return "Fair (Queue-based)"
}

func (fs *FairStrategy) Stop() {
	close(fs.done)
}

var currentReq request

// RWMutexStrategy uses Go's built-in RWMutex
type RWMutexStrategy struct {
	rwMutex sync.RWMutex
}

func (rw *RWMutexStrategy) StartRead() {
	rw.rwMutex.RLock()
}

func (rw *RWMutexStrategy) EndRead() {
	rw.rwMutex.RUnlock()
}

func (rw *RWMutexStrategy) StartWrite() {
	rw.rwMutex.Lock()
}

func (rw *RWMutexStrategy) EndWrite() {
	rw.rwMutex.Unlock()
}

func (rw *RWMutexStrategy) Name() string {
	return "Go RWMutex"
}

// Simulation runs a readers-writers simulation
type Simulation struct {
	db          *Database
	numReaders  int
	numWriters  int
	duration    time.Duration
	stopChan    chan bool
	wg          sync.WaitGroup
}

// NewSimulation creates a new simulation
func NewSimulation(db *Database, numReaders, numWriters int, duration time.Duration) *Simulation {
	return &Simulation{
		db:         db,
		numReaders: numReaders,
		numWriters: numWriters,
		duration:   duration,
		stopChan:   make(chan bool),
	}
}

// Run starts the simulation
func (s *Simulation) Run() Statistics {
	fmt.Printf("Running %s strategy with %d readers and %d writers for %v\n",
		s.db.strategy.Name(), s.numReaders, s.numWriters, s.duration)
	
	// Start readers
	for i := 0; i < s.numReaders; i++ {
		s.wg.Add(1)
		go s.reader(i)
	}
	
	// Start writers
	for i := 0; i < s.numWriters; i++ {
		s.wg.Add(1)
		go s.writer(i)
	}
	
	// Run for duration
	time.Sleep(s.duration)
	close(s.stopChan)
	
	// Wait for all goroutines
	s.wg.Wait()
	
	return s.db.GetStatistics()
}

func (s *Simulation) reader(id int) {
	defer s.wg.Done()
	
	keys := []string{"key1", "key2", "key3", "key4", "key5"}
	
	for {
		select {
		case <-s.stopChan:
			return
		default:
			key := keys[id%len(keys)]
			s.db.Read(key)
			time.Sleep(time.Duration(5+id%5) * time.Millisecond)
		}
	}
}

func (s *Simulation) writer(id int) {
	defer s.wg.Done()
	
	keys := []string{"key1", "key2", "key3", "key4", "key5"}
	
	for {
		select {
		case <-s.stopChan:
			return
		default:
			key := keys[id%len(keys)]
			value := fmt.Sprintf("value_%d_%d", id, time.Now().UnixNano())
			s.db.Write(key, value)
			time.Sleep(time.Duration(50+id%10) * time.Millisecond)
		}
	}
}

// Example demonstrates different readers-writers solutions
func Example() {
	fmt.Println("=== Readers-Writers Problem ===\n")
	
	strategies := []RWStrategy{
		&ReadersPreferenceStrategy{},
		&WritersPreferenceStrategy{},
		NewFairStrategy(),
		&RWMutexStrategy{},
	}
	
	for _, strategy := range strategies {
		db := NewDatabase(strategy)
		sim := NewSimulation(db, 5, 2, 2*time.Second)
		stats := sim.Run()
		
		fmt.Printf("\nResults for %s:\n", strategy.Name())
		fmt.Printf("Total reads: %d\n", stats.TotalReads)
		fmt.Printf("Total writes: %d\n", stats.TotalWrites)
		fmt.Printf("Read/Write ratio: %.2f\n", float64(stats.TotalReads)/float64(stats.TotalWrites))
		fmt.Printf("Avg read wait time: %v\n", stats.AvgReadWaitTime)
		fmt.Printf("Avg write wait time: %v\n", stats.AvgWriteWaitTime)
		
		// Clean up fair strategy
		if fs, ok := strategy.(*FairStrategy); ok {
			fs.Stop()
		}
		
		time.Sleep(500 * time.Millisecond)
	}
}

// Monitor provides real-time monitoring of database access
type Monitor struct {
	db       *Database
	interval time.Duration
	stopChan chan bool
}

// NewMonitor creates a new monitor
func NewMonitor(db *Database, interval time.Duration) *Monitor {
	return &Monitor{
		db:       db,
		interval: interval,
		stopChan: make(chan bool),
	}
}

// Start begins monitoring
func (m *Monitor) Start() {
	go func() {
		ticker := time.NewTicker(m.interval)
		defer ticker.Stop()
		
		for {
			select {
			case <-ticker.C:
				readCount := atomic.LoadInt32(&m.db.readCount)
				writeCount := atomic.LoadInt32(&m.db.writeCount)
				fmt.Printf("Active: %d readers, %d writers\n", readCount, writeCount)
			case <-m.stopChan:
				return
			}
		}
	}()
}

// Stop stops monitoring
func (m *Monitor) Stop() {
	close(m.stopChan)
}