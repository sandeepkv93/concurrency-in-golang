package parallelbloomfilter

import (
	"context"
	"crypto/md5"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"hash/crc64"
	"hash/fnv"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// HashFunction defines different hash functions available
type HashFunction int

const (
	FNV1 HashFunction = iota
	FNV1a
	CRC32
	CRC64
	MD5Hash
	SHA1Hash
	SHA256Hash
	MurmurHash3
	XXHash
	CityHash
)

// BloomFilterType defines different Bloom filter implementations
type BloomFilterType int

const (
	StandardBloom BloomFilterType = iota
	CountingBloom
	ScalableBloom
	PartitionedBloom
	TimingBloom
	DistributedBloom
)

// BloomFilterConfig contains configuration for the Bloom filter
type BloomFilterConfig struct {
	ExpectedElements uint64
	FalsePositiveRate float64
	MaxElements      uint64
	HashFunctions    []HashFunction
	FilterType       BloomFilterType
	NumPartitions    int
	EnableCounting   bool
	EnableScaling    bool
	EnableTiming     bool
	TimeWindow       time.Duration
	MaxFilters       int
	NumWorkers       int
	EnableMetrics    bool
	EnableOptimizations bool
	SeedValue        uint64
}

// DefaultBloomFilterConfig returns default configuration
func DefaultBloomFilterConfig() BloomFilterConfig {
	return BloomFilterConfig{
		ExpectedElements:   1000000,
		FalsePositiveRate:  0.01,
		MaxElements:        10000000,
		HashFunctions:      []HashFunction{FNV1a, CRC32, MurmurHash3},
		FilterType:         StandardBloom,
		NumPartitions:      runtime.NumCPU(),
		EnableCounting:     false,
		EnableScaling:      false,
		EnableTiming:       false,
		TimeWindow:         time.Hour,
		MaxFilters:         10,
		NumWorkers:         runtime.NumCPU(),
		EnableMetrics:      true,
		EnableOptimizations: true,
		SeedValue:          0,
	}
}

// BitArray represents a thread-safe bit array
type BitArray struct {
	bits     []uint64
	size     uint64
	mutex    sync.RWMutex
	segments []BitSegment
}

// BitSegment represents a segment of the bit array for partitioned access
type BitSegment struct {
	bits     []uint64
	size     uint64
	mutex    sync.RWMutex
	offset   uint64
}

// CountingArray represents a thread-safe counting array for counting Bloom filters
type CountingArray struct {
	counts   []uint32
	size     uint64
	mutex    sync.RWMutex
	segments []CountingSegment
}

// CountingSegment represents a segment of the counting array
type CountingSegment struct {
	counts []uint32
	size   uint64
	mutex  sync.RWMutex
	offset uint64
}

// BloomFilter represents the main Bloom filter interface
type BloomFilter interface {
	Add(data []byte) error
	AddString(s string) error
	Contains(data []byte) (bool, error)
	ContainsString(s string) (bool, error)
	Remove(data []byte) error
	Clear()
	Size() uint64
	ElementCount() uint64
	FalsePositiveRate() float64
	GetStatistics() *BloomFilterStatistics
	Merge(other BloomFilter) error
	Export() ([]byte, error)
	Import(data []byte) error
}

// ParallelBloomFilter implements a thread-safe Bloom filter
type ParallelBloomFilter struct {
	config       BloomFilterConfig
	bitArray     *BitArray
	countArray   *CountingArray
	size         uint64
	hashCount    uint64
	elementCount uint64
	hashers      []Hasher
	workers      []*Worker
	taskQueue    chan Task
	resultQueue  chan Result
	statistics   *BloomFilterStatistics
	filters      []*ParallelBloomFilter // For scalable filters
	timeFilters  map[time.Time]*ParallelBloomFilter // For timing filters
	partitions   []*Partition
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
	mutex        sync.RWMutex
	running      bool
}

// Hasher wraps hash functions with additional functionality
type Hasher struct {
	hashFunc HashFunction
	hasher   hash.Hash
	seed     uint64
	mutex    sync.Mutex
}

// Worker represents a worker for parallel operations
type Worker struct {
	id        int
	filter    *ParallelBloomFilter
	taskQueue chan Task
	resultQueue chan Result
	ctx       context.Context
}

// Task represents a task for worker processing
type Task struct {
	Type     string
	Data     []byte
	Hash     uint64
	Position uint64
	Value    interface{}
}

// Result represents the result of a task
type Result struct {
	Success bool
	Value   interface{}
	Error   error
}

// Partition represents a partition of the Bloom filter
type Partition struct {
	id       int
	filter   *ParallelBloomFilter
	bitArray *BitArray
	mutex    sync.RWMutex
}

// BloomFilterStatistics tracks Bloom filter metrics
type BloomFilterStatistics struct {
	ElementsAdded     uint64    `json:"elements_added"`
	ElementsRemoved   uint64    `json:"elements_removed"`
	QueriesPerformed  uint64    `json:"queries_performed"`
	TruePositives     uint64    `json:"true_positives"`
	FalsePositives    uint64    `json:"false_positives"`
	TrueNegatives     uint64    `json:"true_negatives"`
	BitsSet           uint64    `json:"bits_set"`
	LoadFactor        float64   `json:"load_factor"`
	ActualFPRate      float64   `json:"actual_fp_rate"`
	HashCollisions    uint64    `json:"hash_collisions"`
	MemoryUsage       uint64    `json:"memory_usage"`
	OperationTimes    []time.Duration `json:"-"`
	AverageOpTime     time.Duration   `json:"average_op_time"`
	ThroughputOps     float64   `json:"throughput_ops"`
	ConcurrencyLevel  int       `json:"concurrency_level"`
	CreatedAt         time.Time `json:"created_at"`
	LastUpdated       time.Time `json:"last_updated"`
	mutex             sync.RWMutex `json:"-"`
}

// ScalableBloomFilter implements scalable Bloom filter
type ScalableBloomFilter struct {
	*ParallelBloomFilter
	filters     []*ParallelBloomFilter
	growth      float64
	currentSize uint64
	maxSize     uint64
}

// CountingBloomFilter implements counting Bloom filter
type CountingBloomFilter struct {
	*ParallelBloomFilter
	countArray *CountingArray
	maxCount   uint32
}

// TimingBloomFilter implements time-based Bloom filter
type TimingBloomFilter struct {
	*ParallelBloomFilter
	timeWindows map[int64]*ParallelBloomFilter
	windowSize  time.Duration
	maxWindows  int
}

// NewParallelBloomFilter creates a new parallel Bloom filter
func NewParallelBloomFilter(config BloomFilterConfig) (*ParallelBloomFilter, error) {
	// Calculate optimal parameters
	size, hashCount := calculateOptimalParams(config.ExpectedElements, config.FalsePositiveRate)
	
	if size == 0 || hashCount == 0 {
		return nil, errors.New("invalid configuration parameters")
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	bf := &ParallelBloomFilter{
		config:      config,
		size:        size,
		hashCount:   hashCount,
		statistics:  NewBloomFilterStatistics(),
		hashers:     make([]Hasher, len(config.HashFunctions)),
		taskQueue:   make(chan Task, config.NumWorkers*10),
		resultQueue: make(chan Result, config.NumWorkers*10),
		ctx:         ctx,
		cancel:      cancel,
		running:     true,
	}
	
	// Initialize bit array
	if config.NumPartitions > 1 {
		bf.bitArray = NewPartitionedBitArray(size, config.NumPartitions)
	} else {
		bf.bitArray = NewBitArray(size)
	}
	
	// Initialize counting array if needed
	if config.EnableCounting || config.FilterType == CountingBloom {
		bf.countArray = NewCountingArray(size)
	}
	
	// Initialize hashers
	for i, hashFunc := range config.HashFunctions {
		bf.hashers[i] = NewHasher(hashFunc, config.SeedValue+uint64(i))
	}
	
	// Initialize workers
	if config.NumWorkers > 0 {
		bf.initializeWorkers()
	}
	
	// Initialize partitions for partitioned filter
	if config.FilterType == PartitionedBloom {
		bf.initializePartitions()
	}
	
	return bf, nil
}

// NewBitArray creates a new bit array
func NewBitArray(size uint64) *BitArray {
	wordCount := (size + 63) / 64
	return &BitArray{
		bits: make([]uint64, wordCount),
		size: size,
	}
}

// NewPartitionedBitArray creates a partitioned bit array
func NewPartitionedBitArray(size uint64, numPartitions int) *BitArray {
	wordCount := (size + 63) / 64
	segmentSize := wordCount / uint64(numPartitions)
	if wordCount%uint64(numPartitions) != 0 {
		segmentSize++
	}
	
	ba := &BitArray{
		bits:     make([]uint64, wordCount),
		size:     size,
		segments: make([]BitSegment, numPartitions),
	}
	
	for i := 0; i < numPartitions; i++ {
		start := uint64(i) * segmentSize
		end := start + segmentSize
		if end > wordCount {
			end = wordCount
		}
		
		ba.segments[i] = BitSegment{
			bits:   ba.bits[start:end],
			size:   (end - start) * 64,
			offset: start * 64,
		}
	}
	
	return ba
}

// NewCountingArray creates a new counting array
func NewCountingArray(size uint64) *CountingArray {
	return &CountingArray{
		counts: make([]uint32, size),
		size:   size,
	}
}

// NewHasher creates a new hasher
func NewHasher(hashFunc HashFunction, seed uint64) Hasher {
	hasher := Hasher{
		hashFunc: hashFunc,
		seed:     seed,
	}
	
	switch hashFunc {
	case FNV1:
		hasher.hasher = fnv.New64()
	case FNV1a:
		hasher.hasher = fnv.New64a()
	case CRC32:
		hasher.hasher = crc32.NewIEEE()
	case CRC64:
		hasher.hasher = crc64.New(crc64.MakeTable(crc64.ECMA))
	case MD5Hash:
		hasher.hasher = md5.New()
	case SHA1Hash:
		hasher.hasher = sha1.New()
	case SHA256Hash:
		hasher.hasher = sha256.New()
	default:
		hasher.hasher = fnv.New64a()
	}
	
	return hasher
}

// NewBloomFilterStatistics creates new statistics instance
func NewBloomFilterStatistics() *BloomFilterStatistics {
	return &BloomFilterStatistics{
		OperationTimes:   make([]time.Duration, 0),
		ConcurrencyLevel: runtime.NumCPU(),
		CreatedAt:        time.Now(),
		LastUpdated:      time.Now(),
	}
}

// Hash computes hash value for data
func (h *Hasher) Hash(data []byte) uint64 {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	
	h.hasher.Reset()
	
	// Add seed
	if h.seed != 0 {
		seedBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(seedBytes, h.seed)
		h.hasher.Write(seedBytes)
	}
	
	h.hasher.Write(data)
	
	switch h.hashFunc {
	case CRC32:
		return uint64(binary.LittleEndian.Uint32(h.hasher.Sum(nil)))
	case CRC64, FNV1, FNV1a:
		return binary.LittleEndian.Uint64(h.hasher.Sum(nil))
	default:
		sum := h.hasher.Sum(nil)
		if len(sum) >= 8 {
			return binary.LittleEndian.Uint64(sum[:8])
		}
		// Extend shorter hashes
		result := uint64(0)
		for i, b := range sum {
			result |= uint64(b) << (i * 8)
		}
		return result
	}
}

// Set sets a bit in the bit array
func (ba *BitArray) Set(index uint64) {
	if index >= ba.size {
		return
	}
	
	wordIndex := index / 64
	bitIndex := index % 64
	
	if len(ba.segments) > 0 {
		// Partitioned access
		segmentIndex := int(wordIndex * uint64(len(ba.segments)) / uint64(len(ba.bits)))
		if segmentIndex >= len(ba.segments) {
			segmentIndex = len(ba.segments) - 1
		}
		
		segment := &ba.segments[segmentIndex]
		localIndex := wordIndex - segment.offset/64
		
		if localIndex < uint64(len(segment.bits)) {
			segment.mutex.Lock()
			segment.bits[localIndex] |= 1 << bitIndex
			segment.mutex.Unlock()
		}
	} else {
		// Standard access
		ba.mutex.Lock()
		ba.bits[wordIndex] |= 1 << bitIndex
		ba.mutex.Unlock()
	}
}

// Get gets a bit from the bit array
func (ba *BitArray) Get(index uint64) bool {
	if index >= ba.size {
		return false
	}
	
	wordIndex := index / 64
	bitIndex := index % 64
	
	if len(ba.segments) > 0 {
		// Partitioned access
		segmentIndex := int(wordIndex * uint64(len(ba.segments)) / uint64(len(ba.bits)))
		if segmentIndex >= len(ba.segments) {
			segmentIndex = len(ba.segments) - 1
		}
		
		segment := &ba.segments[segmentIndex]
		localIndex := wordIndex - segment.offset/64
		
		if localIndex < uint64(len(segment.bits)) {
			segment.mutex.RLock()
			result := (segment.bits[localIndex] & (1 << bitIndex)) != 0
			segment.mutex.RUnlock()
			return result
		}
	} else {
		// Standard access
		ba.mutex.RLock()
		result := (ba.bits[wordIndex] & (1 << bitIndex)) != 0
		ba.mutex.RUnlock()
		return result
	}
	
	return false
}

// Clear clears all bits in the bit array
func (ba *BitArray) Clear() {
	if len(ba.segments) > 0 {
		for i := range ba.segments {
			ba.segments[i].mutex.Lock()
			for j := range ba.segments[i].bits {
				ba.segments[i].bits[j] = 0
			}
			ba.segments[i].mutex.Unlock()
		}
	} else {
		ba.mutex.Lock()
		for i := range ba.bits {
			ba.bits[i] = 0
		}
		ba.mutex.Unlock()
	}
}

// Count returns the number of set bits
func (ba *BitArray) Count() uint64 {
	count := uint64(0)
	
	if len(ba.segments) > 0 {
		for i := range ba.segments {
			ba.segments[i].mutex.RLock()
			for _, word := range ba.segments[i].bits {
				count += uint64(popcount(word))
			}
			ba.segments[i].mutex.RUnlock()
		}
	} else {
		ba.mutex.RLock()
		for _, word := range ba.bits {
			count += uint64(popcount(word))
		}
		ba.mutex.RUnlock()
	}
	
	return count
}

// Increment increments a counter in the counting array
func (ca *CountingArray) Increment(index uint64) {
	if index >= ca.size {
		return
	}
	
	ca.mutex.Lock()
	if ca.counts[index] < ^uint32(0) { // Prevent overflow
		ca.counts[index]++
	}
	ca.mutex.Unlock()
}

// Decrement decrements a counter in the counting array
func (ca *CountingArray) Decrement(index uint64) {
	if index >= ca.size {
		return
	}
	
	ca.mutex.Lock()
	if ca.counts[index] > 0 {
		ca.counts[index]--
	}
	ca.mutex.Unlock()
}

// Get gets a counter value
func (ca *CountingArray) Get(index uint64) uint32 {
	if index >= ca.size {
		return 0
	}
	
	ca.mutex.RLock()
	value := ca.counts[index]
	ca.mutex.RUnlock()
	
	return value
}

// Clear clears all counters
func (ca *CountingArray) Clear() {
	ca.mutex.Lock()
	for i := range ca.counts {
		ca.counts[i] = 0
	}
	ca.mutex.Unlock()
}

// initializeWorkers initializes worker goroutines
func (bf *ParallelBloomFilter) initializeWorkers() {
	bf.workers = make([]*Worker, bf.config.NumWorkers)
	
	for i := 0; i < bf.config.NumWorkers; i++ {
		worker := &Worker{
			id:          i,
			filter:      bf,
			taskQueue:   bf.taskQueue,
			resultQueue: bf.resultQueue,
			ctx:         bf.ctx,
		}
		bf.workers[i] = worker
		
		bf.wg.Add(1)
		go worker.start(&bf.wg)
	}
}

// initializePartitions initializes filter partitions
func (bf *ParallelBloomFilter) initializePartitions() {
	bf.partitions = make([]*Partition, bf.config.NumPartitions)
	
	partitionSize := bf.size / uint64(bf.config.NumPartitions)
	
	for i := 0; i < bf.config.NumPartitions; i++ {
		bf.partitions[i] = &Partition{
			id:       i,
			filter:   bf,
			bitArray: NewBitArray(partitionSize),
		}
	}
}

// start starts a worker
func (w *Worker) start(wg *sync.WaitGroup) {
	defer wg.Done()
	
	for {
		select {
		case task := <-w.taskQueue:
			result := w.processTask(task)
			select {
			case w.resultQueue <- result:
			case <-w.ctx.Done():
				return
			}
		case <-w.ctx.Done():
			return
		}
	}
}

// processTask processes a worker task
func (w *Worker) processTask(task Task) Result {
	switch task.Type {
	case "add":
		return w.processAdd(task)
	case "contains":
		return w.processContains(task)
	case "remove":
		return w.processRemove(task)
	default:
		return Result{
			Success: false,
			Error:   fmt.Errorf("unknown task type: %s", task.Type),
		}
	}
}

// processAdd processes add operation
func (w *Worker) processAdd(task Task) Result {
	hashes := w.computeHashes(task.Data)
	
	for _, hash := range hashes {
		index := hash % w.filter.size
		w.filter.bitArray.Set(index)
		
		if w.filter.countArray != nil {
			w.filter.countArray.Increment(index)
		}
	}
	
	atomic.AddUint64(&w.filter.elementCount, 1)
	
	return Result{Success: true}
}

// processContains processes contains operation
func (w *Worker) processContains(task Task) Result {
	hashes := w.computeHashes(task.Data)
	
	for _, hash := range hashes {
		index := hash % w.filter.size
		if !w.filter.bitArray.Get(index) {
			return Result{Success: true, Value: false}
		}
	}
	
	return Result{Success: true, Value: true}
}

// processRemove processes remove operation (for counting filters)
func (w *Worker) processRemove(task Task) Result {
	if w.filter.countArray == nil {
		return Result{
			Success: false,
			Error:   errors.New("remove operation not supported in standard Bloom filter"),
		}
	}
	
	hashes := w.computeHashes(task.Data)
	
	// Check if element exists first
	for _, hash := range hashes {
		index := hash % w.filter.size
		if w.filter.countArray.Get(index) == 0 {
			return Result{Success: true, Value: false} // Element not present
		}
	}
	
	// Decrement all counters
	for _, hash := range hashes {
		index := hash % w.filter.size
		w.filter.countArray.Decrement(index)
		
		// Clear bit if counter reaches zero
		if w.filter.countArray.Get(index) == 0 {
			// Note: This is a simplified approach
			// In practice, you'd need to check if other elements use this bit
		}
	}
	
	atomic.AddUint64(&w.filter.statistics.ElementsRemoved, 1)
	
	return Result{Success: true, Value: true}
}

// computeHashes computes all hash values for data
func (w *Worker) computeHashes(data []byte) []uint64 {
	hashes := make([]uint64, len(w.filter.hashers))
	
	for i, hasher := range w.filter.hashers {
		hashes[i] = hasher.Hash(data)
	}
	
	return hashes
}

// Add adds an element to the Bloom filter
func (bf *ParallelBloomFilter) Add(data []byte) error {
	if !bf.running {
		return errors.New("Bloom filter is not running")
	}
	
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		bf.updateStatistics(duration, "add")
	}()
	
	if bf.config.NumWorkers > 0 {
		// Use worker pool
		task := Task{
			Type: "add",
			Data: data,
		}
		
		select {
		case bf.taskQueue <- task:
		case <-bf.ctx.Done():
			return errors.New("operation cancelled")
		}
		
		select {
		case result := <-bf.resultQueue:
			if !result.Success {
				return result.Error
			}
		case <-bf.ctx.Done():
			return errors.New("operation cancelled")
		}
	} else {
		// Direct operation
		bf.addDirect(data)
	}
	
	atomic.AddUint64(&bf.statistics.ElementsAdded, 1)
	return nil
}

// AddString adds a string to the Bloom filter
func (bf *ParallelBloomFilter) AddString(s string) error {
	return bf.Add([]byte(s))
}

// Contains checks if an element might be in the Bloom filter
func (bf *ParallelBloomFilter) Contains(data []byte) (bool, error) {
	if !bf.running {
		return false, errors.New("Bloom filter is not running")
	}
	
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		bf.updateStatistics(duration, "contains")
	}()
	
	atomic.AddUint64(&bf.statistics.QueriesPerformed, 1)
	
	if bf.config.NumWorkers > 0 {
		// Use worker pool
		task := Task{
			Type: "contains",
			Data: data,
		}
		
		select {
		case bf.taskQueue <- task:
		case <-bf.ctx.Done():
			return false, errors.New("operation cancelled")
		}
		
		select {
		case result := <-bf.resultQueue:
			if !result.Success {
				return false, result.Error
			}
			return result.Value.(bool), nil
		case <-bf.ctx.Done():
			return false, errors.New("operation cancelled")
		}
	} else {
		// Direct operation
		return bf.containsDirect(data), nil
	}
}

// ContainsString checks if a string might be in the Bloom filter
func (bf *ParallelBloomFilter) ContainsString(s string) (bool, error) {
	return bf.Contains([]byte(s))
}

// Remove removes an element from counting Bloom filter
func (bf *ParallelBloomFilter) Remove(data []byte) error {
	if bf.countArray == nil {
		return errors.New("remove operation not supported in standard Bloom filter")
	}
	
	if !bf.running {
		return errors.New("Bloom filter is not running")
	}
	
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		bf.updateStatistics(duration, "remove")
	}()
	
	if bf.config.NumWorkers > 0 {
		// Use worker pool
		task := Task{
			Type: "remove",
			Data: data,
		}
		
		select {
		case bf.taskQueue <- task:
		case <-bf.ctx.Done():
			return errors.New("operation cancelled")
		}
		
		select {
		case result := <-bf.resultQueue:
			if !result.Success {
				return result.Error
			}
		case <-bf.ctx.Done():
			return errors.New("operation cancelled")
		}
	} else {
		// Direct operation
		return bf.removeDirect(data)
	}
	
	return nil
}

// addDirect performs direct add operation
func (bf *ParallelBloomFilter) addDirect(data []byte) {
	for i, hasher := range bf.hashers {
		hash := hasher.Hash(data)
		// Use double hashing for additional hash functions
		for j := uint64(0); j < bf.hashCount/uint64(len(bf.hashers)); j++ {
			combinedHash := hash + uint64(i)*j
			index := combinedHash % bf.size
			bf.bitArray.Set(index)
			
			if bf.countArray != nil {
				bf.countArray.Increment(index)
			}
		}
	}
	
	atomic.AddUint64(&bf.elementCount, 1)
}

// containsDirect performs direct contains operation
func (bf *ParallelBloomFilter) containsDirect(data []byte) bool {
	for i, hasher := range bf.hashers {
		hash := hasher.Hash(data)
		// Use double hashing for additional hash functions
		for j := uint64(0); j < bf.hashCount/uint64(len(bf.hashers)); j++ {
			combinedHash := hash + uint64(i)*j
			index := combinedHash % bf.size
			if !bf.bitArray.Get(index) {
				return false
			}
		}
	}
	
	return true
}

// removeDirect performs direct remove operation
func (bf *ParallelBloomFilter) removeDirect(data []byte) error {
	if bf.countArray == nil {
		return errors.New("remove operation not supported")
	}
	
	// First check if element exists
	if !bf.containsDirect(data) {
		return nil // Element not present
	}
	
	for i, hasher := range bf.hashers {
		hash := hasher.Hash(data)
		for j := uint64(0); j < bf.hashCount/uint64(len(bf.hashers)); j++ {
			combinedHash := hash + uint64(i)*j
			index := combinedHash % bf.size
			bf.countArray.Decrement(index)
		}
	}
	
	return nil
}

// Clear clears the Bloom filter
func (bf *ParallelBloomFilter) Clear() {
	bf.bitArray.Clear()
	if bf.countArray != nil {
		bf.countArray.Clear()
	}
	atomic.StoreUint64(&bf.elementCount, 0)
	
	// Reset statistics
	bf.statistics.mutex.Lock()
	bf.statistics.ElementsAdded = 0
	bf.statistics.ElementsRemoved = 0
	bf.statistics.QueriesPerformed = 0
	bf.statistics.BitsSet = 0
	bf.statistics.mutex.Unlock()
}

// Size returns the size of the bit array
func (bf *ParallelBloomFilter) Size() uint64 {
	return bf.size
}

// ElementCount returns the number of elements added
func (bf *ParallelBloomFilter) ElementCount() uint64 {
	return atomic.LoadUint64(&bf.elementCount)
}

// FalsePositiveRate returns the current false positive rate
func (bf *ParallelBloomFilter) FalsePositiveRate() float64 {
	bitsSet := bf.bitArray.Count()
	if bitsSet == 0 {
		return 0.0
	}
	
	// Calculate actual false positive rate
	loadFactor := float64(bitsSet) / float64(bf.size)
	return math.Pow(loadFactor, float64(bf.hashCount))
}

// GetStatistics returns current statistics
func (bf *ParallelBloomFilter) GetStatistics() *BloomFilterStatistics {
	bf.statistics.mutex.RLock()
	defer bf.statistics.mutex.RUnlock()
	
	stats := *bf.statistics
	stats.BitsSet = bf.bitArray.Count()
	stats.LoadFactor = float64(stats.BitsSet) / float64(bf.size)
	stats.ActualFPRate = bf.FalsePositiveRate()
	stats.LastUpdated = time.Now()
	
	return &stats
}

// Merge merges another Bloom filter into this one
func (bf *ParallelBloomFilter) Merge(other BloomFilter) error {
	otherBF, ok := other.(*ParallelBloomFilter)
	if !ok {
		return errors.New("can only merge with another ParallelBloomFilter")
	}
	
	if bf.size != otherBF.size {
		return errors.New("Bloom filters must have the same size")
	}
	
	// Merge bit arrays
	bf.bitArray.mutex.Lock()
	otherBF.bitArray.mutex.RLock()
	
	for i := range bf.bitArray.bits {
		bf.bitArray.bits[i] |= otherBF.bitArray.bits[i]
	}
	
	otherBF.bitArray.mutex.RUnlock()
	bf.bitArray.mutex.Unlock()
	
	// Update element count (approximation)
	atomic.AddUint64(&bf.elementCount, otherBF.ElementCount())
	
	return nil
}

// Export exports the Bloom filter data
func (bf *ParallelBloomFilter) Export() ([]byte, error) {
	// Simplified export - in practice would use proper serialization
	bf.bitArray.mutex.RLock()
	defer bf.bitArray.mutex.RUnlock()
	
	data := make([]byte, len(bf.bitArray.bits)*8)
	for i, word := range bf.bitArray.bits {
		binary.LittleEndian.PutUint64(data[i*8:], word)
	}
	
	return data, nil
}

// Import imports Bloom filter data
func (bf *ParallelBloomFilter) Import(data []byte) error {
	if len(data)%8 != 0 {
		return errors.New("invalid data length")
	}
	
	wordCount := len(data) / 8
	if wordCount != len(bf.bitArray.bits) {
		return errors.New("data size mismatch")
	}
	
	bf.bitArray.mutex.Lock()
	defer bf.bitArray.mutex.Unlock()
	
	for i := 0; i < wordCount; i++ {
		bf.bitArray.bits[i] = binary.LittleEndian.Uint64(data[i*8:])
	}
	
	return nil
}

// updateStatistics updates filter statistics
func (bf *ParallelBloomFilter) updateStatistics(duration time.Duration, operation string) {
	if !bf.config.EnableMetrics {
		return
	}
	
	bf.statistics.mutex.Lock()
	defer bf.statistics.mutex.Unlock()
	
	bf.statistics.OperationTimes = append(bf.statistics.OperationTimes, duration)
	
	// Keep only recent operation times
	if len(bf.statistics.OperationTimes) > 1000 {
		bf.statistics.OperationTimes = bf.statistics.OperationTimes[len(bf.statistics.OperationTimes)-1000:]
	}
	
	// Calculate average operation time
	total := time.Duration(0)
	for _, t := range bf.statistics.OperationTimes {
		total += t
	}
	bf.statistics.AverageOpTime = total / time.Duration(len(bf.statistics.OperationTimes))
	
	// Calculate throughput
	if len(bf.statistics.OperationTimes) > 1 {
		timeWindow := bf.statistics.OperationTimes[len(bf.statistics.OperationTimes)-1] - 
			bf.statistics.OperationTimes[0]
		if timeWindow > 0 {
			bf.statistics.ThroughputOps = float64(len(bf.statistics.OperationTimes)) / 
				timeWindow.Seconds()
		}
	}
	
	bf.statistics.LastUpdated = time.Now()
}

// Shutdown gracefully shuts down the Bloom filter
func (bf *ParallelBloomFilter) Shutdown() error {
	if !bf.running {
		return errors.New("Bloom filter is not running")
	}
	
	bf.running = false
	bf.cancel()
	
	// Wait for workers to finish
	bf.wg.Wait()
	
	return nil
}

// Utility functions

// calculateOptimalParams calculates optimal Bloom filter parameters
func calculateOptimalParams(expectedElements uint64, falsePositiveRate float64) (uint64, uint64) {
	if expectedElements == 0 || falsePositiveRate <= 0 || falsePositiveRate >= 1 {
		return 0, 0
	}
	
	// Calculate optimal bit array size
	size := uint64(-1.0 * float64(expectedElements) * math.Log(falsePositiveRate) / (math.Log(2) * math.Log(2)))
	
	// Calculate optimal number of hash functions
	hashCount := uint64(float64(size) / float64(expectedElements) * math.Log(2))
	
	if hashCount == 0 {
		hashCount = 1
	}
	
	return size, hashCount
}

// popcount counts the number of set bits in a word
func popcount(x uint64) int {
	count := 0
	for x != 0 {
		count++
		x &= x - 1
	}
	return count
}

// NewScalableBloomFilter creates a scalable Bloom filter
func NewScalableBloomFilter(config BloomFilterConfig) (*ScalableBloomFilter, error) {
	baseBF, err := NewParallelBloomFilter(config)
	if err != nil {
		return nil, err
	}
	
	sbf := &ScalableBloomFilter{
		ParallelBloomFilter: baseBF,
		filters:            []*ParallelBloomFilter{baseBF},
		growth:             2.0,
		currentSize:        config.ExpectedElements,
		maxSize:           config.MaxElements,
	}
	
	return sbf, nil
}

// Add adds element to scalable Bloom filter
func (sbf *ScalableBloomFilter) Add(data []byte) error {
	// Check if we need to scale up
	if sbf.ElementCount() >= sbf.currentSize {
		if sbf.currentSize*uint64(sbf.growth) <= sbf.maxSize {
			if err := sbf.scaleUp(); err != nil {
				return err
			}
		}
	}
	
	// Add to the latest filter
	return sbf.filters[len(sbf.filters)-1].Add(data)
}

// Contains checks if element might be in scalable Bloom filter
func (sbf *ScalableBloomFilter) Contains(data []byte) (bool, error) {
	// Check all filters
	for _, filter := range sbf.filters {
		contains, err := filter.Contains(data)
		if err != nil {
			return false, err
		}
		if contains {
			return true, nil
		}
	}
	return false, nil
}

// scaleUp adds a new filter to the scalable Bloom filter
func (sbf *ScalableBloomFilter) scaleUp() error {
	newConfig := sbf.config
	newConfig.ExpectedElements = uint64(float64(sbf.currentSize) * sbf.growth)
	
	newFilter, err := NewParallelBloomFilter(newConfig)
	if err != nil {
		return err
	}
	
	sbf.filters = append(sbf.filters, newFilter)
	sbf.currentSize = newConfig.ExpectedElements
	
	return nil
}

// NewCountingBloomFilter creates a counting Bloom filter
func NewCountingBloomFilter(config BloomFilterConfig) (*CountingBloomFilter, error) {
	config.EnableCounting = true
	baseBF, err := NewParallelBloomFilter(config)
	if err != nil {
		return nil, err
	}
	
	cbf := &CountingBloomFilter{
		ParallelBloomFilter: baseBF,
		countArray:         baseBF.countArray,
		maxCount:           ^uint32(0),
	}
	
	return cbf, nil
}

// NewTimingBloomFilter creates a time-based Bloom filter
func NewTimingBloomFilter(config BloomFilterConfig) (*TimingBloomFilter, error) {
	baseBF, err := NewParallelBloomFilter(config)
	if err != nil {
		return nil, err
	}
	
	tbf := &TimingBloomFilter{
		ParallelBloomFilter: baseBF,
		timeWindows:        make(map[int64]*ParallelBloomFilter),
		windowSize:         config.TimeWindow,
		maxWindows:         config.MaxFilters,
	}
	
	return tbf, nil
}

// Add adds element to timing Bloom filter
func (tbf *TimingBloomFilter) Add(data []byte) error {
	now := time.Now()
	windowKey := now.Unix() / int64(tbf.windowSize.Seconds())
	
	// Get or create window filter
	windowFilter, exists := tbf.timeWindows[windowKey]
	if !exists {
		windowFilter, _ = NewParallelBloomFilter(tbf.config)
		tbf.timeWindows[windowKey] = windowFilter
		
		// Clean old windows
		tbf.cleanOldWindows(windowKey)
	}
	
	return windowFilter.Add(data)
}

// Contains checks if element might be in timing Bloom filter
func (tbf *TimingBloomFilter) Contains(data []byte) (bool, error) {
	// Check current and recent windows
	now := time.Now()
	currentWindow := now.Unix() / int64(tbf.windowSize.Seconds())
	
	for windowKey := currentWindow - int64(tbf.maxWindows); windowKey <= currentWindow; windowKey++ {
		if filter, exists := tbf.timeWindows[windowKey]; exists {
			contains, err := filter.Contains(data)
			if err != nil {
				return false, err
			}
			if contains {
				return true, nil
			}
		}
	}
	
	return false, nil
}

// cleanOldWindows removes old time windows
func (tbf *TimingBloomFilter) cleanOldWindows(currentWindow int64) {
	cutoff := currentWindow - int64(tbf.maxWindows)
	
	for windowKey := range tbf.timeWindows {
		if windowKey < cutoff {
			tbf.timeWindows[windowKey].Shutdown()
			delete(tbf.timeWindows, windowKey)
		}
	}
}