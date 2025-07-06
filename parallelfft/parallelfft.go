package parallelfft

import (
	"context"
	"errors"
	"fmt"
	"math"
		"runtime"
	"sync"
	"time"
)

// Complex represents a complex number with real and imaginary parts
type Complex struct {
	Real, Imag float64
}

// Add adds two complex numbers
func (c Complex) Add(other Complex) Complex {
	return Complex{c.Real + other.Real, c.Imag + other.Imag}
}

// Sub subtracts two complex numbers
func (c Complex) Sub(other Complex) Complex {
	return Complex{c.Real - other.Real, c.Imag - other.Imag}
}

// Mul multiplies two complex numbers
func (c Complex) Mul(other Complex) Complex {
	return Complex{
		c.Real*other.Real - c.Imag*other.Imag,
		c.Real*other.Imag + c.Imag*other.Real,
	}
}

// MulScalar multiplies complex number by scalar
func (c Complex) MulScalar(scalar float64) Complex {
	return Complex{c.Real * scalar, c.Imag * scalar}
}

// Abs returns the magnitude of complex number
func (c Complex) Abs() float64 {
	return math.Sqrt(c.Real*c.Real + c.Imag*c.Imag)
}

// Phase returns the phase of complex number
func (c Complex) Phase() float64 {
	return math.Atan2(c.Imag, c.Real)
}

// NewComplex creates a complex number from real and imaginary parts
func NewComplex(real, imag float64) Complex {
	return Complex{Real: real, Imag: imag}
}

// NewComplexPolar creates complex number from magnitude and phase
func NewComplexPolar(magnitude, phase float64) Complex {
	return Complex{
		Real: magnitude * math.Cos(phase),
		Imag: magnitude * math.Sin(phase),
	}
}

// FFTConfig contains configuration for FFT operations
type FFTConfig struct {
	Algorithm    FFTAlgorithm
	NumWorkers   int
	ChunkSize    int
	UseInPlace   bool
	WindowFunc   WindowFunction
	Precision    Precision
	EnableCache  bool
	MaxCacheSize int
}

// FFTAlgorithm represents different FFT algorithm types
type FFTAlgorithm int

const (
	CooleyTukey FFTAlgorithm = iota
	Radix4
	MixedRadix
	Bluestein
	RealFFT
)

// WindowFunction represents windowing functions
type WindowFunction int

const (
	NoWindow WindowFunction = iota
	Hamming
	Hanning
	Blackman
	Kaiser
	Gaussian
)

// Precision represents computation precision
type Precision int

const (
	Float32 Precision = iota
	Float64
)

// FFTProcessor handles FFT computations
type FFTProcessor struct {
	config       FFTConfig
	workerPool   *WorkerPool
	twiddleCache map[int][]Complex
	planCache    map[PlanKey]*FFTPlan
	mutex        sync.RWMutex
	stats        *FFTStats
}

// WorkerPool manages parallel workers
type WorkerPool struct {
	workers   []*Worker
	taskQueue chan Task
	wg        sync.WaitGroup
	ctx       context.Context
	cancel    context.CancelFunc
}

// Worker represents a computational worker
type Worker struct {
	id         int
	processor  *FFTProcessor
	taskQueue  chan Task
	resultChan chan Result
}

// Task represents an FFT computation task
type Task struct {
	Type     TaskType
	Data     []Complex
	StartIdx int
	EndIdx   int
	Level    int
	Stride   int
	TaskID   string
}

// TaskType represents different types of FFT tasks
type TaskType int

const (
	ButterflyTask TaskType = iota
	TwiddleTask
	PermutationTask
	WindowingTask
)

// Result represents computation result
type Result struct {
	TaskID string
	Data   []Complex
	Error  error
	Stats  TaskStats
}

// TaskStats contains task execution statistics
type TaskStats struct {
	StartTime    time.Time
	EndTime      time.Time
	Duration     time.Duration
	WorkerID     int
	Operations   int64
	MemoryUsed   int64
}

// FFTPlan represents a cached computation plan
type FFTPlan struct {
	Size        int
	Algorithm   FFTAlgorithm
	TwiddleFactors []Complex
	BitReversed []int
	Stages      []FFTStage
	CreatedAt   time.Time
}

// FFTStage represents a stage in the FFT computation
type FFTStage struct {
	Level    int
	Stride   int
	NumTasks int
	Tasks    []Task
}

// PlanKey is used for caching FFT plans
type PlanKey struct {
	Size      int
	Algorithm FFTAlgorithm
	Direction int // 1 for forward, -1 for inverse
}

// FFTStats contains performance statistics
type FFTStats struct {
	TotalTransforms   int64
	TotalSamples      int64
	TotalTime         time.Duration
	ParallelEfficiency float64
	CacheHitRate      float64
	AverageLatency    time.Duration
	WorkerUtilization []float64
	mutex             sync.RWMutex
}

// NewFFTProcessor creates a new FFT processor
func NewFFTProcessor(config FFTConfig) *FFTProcessor {
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	if config.ChunkSize <= 0 {
		config.ChunkSize = 1024
	}
	if config.MaxCacheSize <= 0 {
		config.MaxCacheSize = 100
	}

	processor := &FFTProcessor{
		config:       config,
		twiddleCache: make(map[int][]Complex),
		planCache:    make(map[PlanKey]*FFTPlan),
		stats:        &FFTStats{WorkerUtilization: make([]float64, config.NumWorkers)},
	}

	processor.workerPool = NewWorkerPool(processor, config.NumWorkers)
	return processor
}

// FFT performs Fast Fourier Transform
func (fft *FFTProcessor) FFT(input []Complex) ([]Complex, error) {
	return fft.fftWithDirection(input, 1)
}

// IFFT performs Inverse Fast Fourier Transform
func (fft *FFTProcessor) IFFT(input []Complex) ([]Complex, error) {
	result, err := fft.fftWithDirection(input, -1)
	if err != nil {
		return nil, err
	}

	// Scale by 1/N for inverse transform
	n := float64(len(input))
	for i := range result {
		result[i] = result[i].MulScalar(1.0 / n)
	}

	return result, nil
}

// fftWithDirection performs FFT with specified direction
func (fft *FFTProcessor) fftWithDirection(input []Complex, direction int) ([]Complex, error) {
	if len(input) == 0 {
		return nil, errors.New("input cannot be empty")
	}

	start := time.Now()
	defer func() {
		fft.updateStats(len(input), time.Since(start))
	}()

	n := len(input)
	
	// Get or create plan
	planKey := PlanKey{Size: n, Algorithm: fft.config.Algorithm, Direction: direction}
	plan := fft.getOrCreatePlan(planKey)

	// Apply windowing if specified
	data := make([]Complex, n)
	copy(data, input)
	
	if fft.config.WindowFunc != NoWindow {
		fft.applyWindow(data, fft.config.WindowFunc)
	}

	// Choose algorithm based on configuration and input size
	switch fft.config.Algorithm {
	case CooleyTukey:
		return fft.cooleyTukeyFFT(data, direction, plan)
	case Radix4:
		return fft.radix4FFT(data, direction, plan)
	case MixedRadix:
		return fft.mixedRadixFFT(data, direction)
	case Bluestein:
		return fft.bluesteinFFT(data, direction)
	case RealFFT:
		return fft.realFFT(data, direction)
	default:
		return fft.cooleyTukeyFFT(data, direction, plan)
	}
}

// cooleyTukeyFFT implements the Cooley-Tukey FFT algorithm
func (fft *FFTProcessor) cooleyTukeyFFT(data []Complex, direction int, plan *FFTPlan) ([]Complex, error) {
	n := len(data)
	
	if n == 1 {
		return data, nil
	}

	if !isPowerOfTwo(n) {
		return nil, fmt.Errorf("Cooley-Tukey FFT requires power-of-2 size, got %d", n)
	}

	// Bit-reverse permutation
	result := make([]Complex, n)
	for i := 0; i < n; i++ {
		result[i] = data[plan.BitReversed[i]]
	}

	// Iterative FFT with parallel butterfly operations
	for stage, stageInfo := range plan.Stages {
		if fft.shouldParallelize(stageInfo.NumTasks) {
			fft.parallelButterflyStage(result, stage, stageInfo, direction, plan.TwiddleFactors)
		} else {
			fft.serialButterflyStage(result, stage, stageInfo, direction, plan.TwiddleFactors)
		}
	}

	return result, nil
}

// parallelButterflyStage performs butterfly operations in parallel
func (fft *FFTProcessor) parallelButterflyStage(data []Complex, stage int, stageInfo FFTStage, direction int, twiddles []Complex) {
	numTasks := stageInfo.NumTasks
	chunkSize := (numTasks + fft.config.NumWorkers - 1) / fft.config.NumWorkers

	var wg sync.WaitGroup
	for worker := 0; worker < fft.config.NumWorkers; worker++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			start := workerID * chunkSize
			end := min(start+chunkSize, numTasks)

			for taskIdx := start; taskIdx < end; taskIdx++ {
				fft.performButterflyOperation(data, stage, taskIdx, stageInfo, direction, twiddles)
			}
		}(worker)
	}
	wg.Wait()
}

// serialButterflyStage performs butterfly operations serially
func (fft *FFTProcessor) serialButterflyStage(data []Complex, stage int, stageInfo FFTStage, direction int, twiddles []Complex) {
	for taskIdx := 0; taskIdx < stageInfo.NumTasks; taskIdx++ {
		fft.performButterflyOperation(data, stage, taskIdx, stageInfo, direction, twiddles)
	}
}

// performButterflyOperation performs a single butterfly operation
func (fft *FFTProcessor) performButterflyOperation(data []Complex, stage, taskIdx int, stageInfo FFTStage, direction int, twiddles []Complex) {
	stride := stageInfo.Stride
	m := 1 << (stage + 1) // 2^(stage+1)
	
	for i := taskIdx; i < len(data); i += m * fft.config.NumWorkers {
		for j := 0; j < stride; j++ {
			pos := i + j
			if pos+stride >= len(data) {
				break
			}

			// Calculate twiddle factor
			twiddleIdx := (j * len(data) / m) % len(twiddles)
			twiddle := twiddles[twiddleIdx]
			
			if direction == -1 {
				twiddle = Complex{twiddle.Real, -twiddle.Imag} // Complex conjugate for IFFT
			}

			// Butterfly operation
			even := data[pos]
			odd := data[pos+stride].Mul(twiddle)

			data[pos] = even.Add(odd)
			data[pos+stride] = even.Sub(odd)
		}
	}
}

// radix4FFT implements radix-4 FFT for better cache performance
func (fft *FFTProcessor) radix4FFT(data []Complex, direction int, plan *FFTPlan) ([]Complex, error) {
	n := len(data)
	if n%4 != 0 {
		return fft.cooleyTukeyFFT(data, direction, plan) // Fallback to radix-2
	}

	result := make([]Complex, n)
	copy(result, data)

	// Bit-reverse for radix-4
	for i := 0; i < n; i++ {
		j := bitReverseRadix4(i, n)
		if i < j {
			result[i], result[j] = result[j], result[i]
		}
	}

	// Radix-4 stages
	stride := 4
	for stride <= n {
		fft.radix4Stage(result, stride, direction)
		stride *= 4
	}

	return result, nil
}

// radix4Stage performs one stage of radix-4 FFT
func (fft *FFTProcessor) radix4Stage(data []Complex, stride, direction int) {
	n := len(data)
	numGroups := n / stride

	var wg sync.WaitGroup
	chunkSize := (numGroups + fft.config.NumWorkers - 1) / fft.config.NumWorkers

	for worker := 0; worker < fft.config.NumWorkers; worker++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			start := workerID * chunkSize
			end := min(start+chunkSize, numGroups)

			for group := start; group < end; group++ {
				fft.radix4Butterfly(data, group, stride, direction)
			}
		}(worker)
	}
	wg.Wait()
}

// radix4Butterfly performs radix-4 butterfly operation
func (fft *FFTProcessor) radix4Butterfly(data []Complex, group, stride, direction int) {
	base := group * stride
	quarter := stride / 4

	for i := 0; i < quarter; i++ {
		// Calculate indices
		i0 := base + i
		i1 := i0 + quarter
		i2 := i1 + quarter
		i3 := i2 + quarter

		// Get data points
		x0 := data[i0]
		x1 := data[i1]
		x2 := data[i2]
		x3 := data[i3]

		// Calculate twiddle factors
		angle := 2 * math.Pi * float64(i) / float64(stride)
		w1 := NewComplexPolar(1, float64(direction)*angle)
		w2 := NewComplexPolar(1, float64(direction)*2*angle)
		w3 := NewComplexPolar(1, float64(direction)*3*angle)

		// Apply twiddle factors
		x1 = x1.Mul(w1)
		x2 = x2.Mul(w2)
		x3 = x3.Mul(w3)

		// Radix-4 butterfly computation
		a0 := x0.Add(x2)
		a1 := x0.Sub(x2)
		a2 := x1.Add(x3)
		a3 := x1.Sub(x3)

		// Multiply a3 by -j for forward transform or +j for inverse
		if direction == 1 {
			a3 = Complex{a3.Imag, -a3.Real} // Multiply by -j
		} else {
			a3 = Complex{-a3.Imag, a3.Real} // Multiply by +j
		}

		// Final butterfly outputs
		data[i0] = a0.Add(a2)
		data[i1] = a1.Add(a3)
		data[i2] = a0.Sub(a2)
		data[i3] = a1.Sub(a3)
	}
}

// mixedRadixFFT handles non-power-of-2 sizes using mixed radix
func (fft *FFTProcessor) mixedRadixFFT(data []Complex, direction int) ([]Complex, error) {
	n := len(data)
	factors := factorize(n)

	if len(factors) == 1 && factors[0] == n {
		// Prime size, use Bluestein's algorithm
		return fft.bluesteinFFT(data, direction)
	}

	result := make([]Complex, n)
	copy(result, data)

	// Apply mixed-radix decomposition
	stride := 1
	for _, factor := range factors {
		for start := 0; start < stride; start++ {
			fft.mixedRadixStage(result, start, stride, factor, direction)
		}
		stride *= factor
	}

	return result, nil
}

// mixedRadixStage performs one stage of mixed-radix FFT
func (fft *FFTProcessor) mixedRadixStage(data []Complex, start, stride, radix, direction int) {
	n := len(data)
	groupSize := stride * radix
	numGroups := n / groupSize

	for group := 0; group < numGroups; group++ {
		base := group*groupSize + start

		// Extract radix points
		points := make([]Complex, radix)
		for i := 0; i < radix; i++ {
			points[i] = data[base+i*stride]
		}

		// Perform DFT of size 'radix'
		dftResult := fft.dft(points, direction)

		// Apply twiddle factors and store back
		for i := 0; i < radix; i++ {
			angle := 2 * math.Pi * float64(i*start) / float64(groupSize)
			twiddle := NewComplexPolar(1, float64(direction)*angle)
			data[base+i*stride] = dftResult[i].Mul(twiddle)
		}
	}
}

// dft performs direct DFT for small sizes
func (fft *FFTProcessor) dft(data []Complex, direction int) []Complex {
	n := len(data)
	result := make([]Complex, n)

	for k := 0; k < n; k++ {
		sum := Complex{0, 0}
		for j := 0; j < n; j++ {
			angle := 2 * math.Pi * float64(k*j) / float64(n)
			twiddle := NewComplexPolar(1, float64(direction)*angle)
			sum = sum.Add(data[j].Mul(twiddle))
		}
		result[k] = sum
	}

	return result
}

// bluesteinFFT implements Bluestein's algorithm for arbitrary sizes
func (fft *FFTProcessor) bluesteinFFT(data []Complex, direction int) ([]Complex, error) {
	n := len(data)
	
	// Find next power of 2 greater than 2*n-1
	m := nextPowerOfTwo(2*n - 1)

	// Prepare sequences
	a := make([]Complex, m)
	b := make([]Complex, m)

	// Fill sequence a
	for i := 0; i < n; i++ {
		angle := math.Pi * float64(i*i) / float64(n)
		chirp := NewComplexPolar(1, float64(direction)*angle)
		a[i] = data[i].Mul(chirp)
	}

	// Fill sequence b
	for i := 0; i < n; i++ {
		angle := -math.Pi * float64(i*i) / float64(n)
		b[i] = NewComplexPolar(1, float64(direction)*angle)
	}
	for i := 1; i < n; i++ {
		angle := -math.Pi * float64(i*i) / float64(n)
		b[m-i] = NewComplexPolar(1, float64(direction)*angle)
	}

	// Convolve using FFT
	aFFT, err := fft.cooleyTukeyFFT(a, 1, fft.getOrCreatePlan(PlanKey{Size: m, Algorithm: CooleyTukey, Direction: 1}))
	if err != nil {
		return nil, err
	}

	bFFT, err := fft.cooleyTukeyFFT(b, 1, fft.getOrCreatePlan(PlanKey{Size: m, Algorithm: CooleyTukey, Direction: 1}))
	if err != nil {
		return nil, err
	}

	// Pointwise multiplication
	conv := make([]Complex, m)
	for i := 0; i < m; i++ {
		conv[i] = aFFT[i].Mul(bFFT[i])
	}

	// Inverse FFT
	convIFFT, err := fft.cooleyTukeyFFT(conv, -1, fft.getOrCreatePlan(PlanKey{Size: m, Algorithm: CooleyTukey, Direction: -1}))
	if err != nil {
		return nil, err
	}

	// Scale by 1/m
	for i := range convIFFT {
		convIFFT[i] = convIFFT[i].MulScalar(1.0 / float64(m))
	}

	// Extract result and apply final chirp
	result := make([]Complex, n)
	for i := 0; i < n; i++ {
		angle := math.Pi * float64(i*i) / float64(n)
		chirp := NewComplexPolar(1, float64(direction)*angle)
		result[i] = convIFFT[i].Mul(chirp)
	}

	return result, nil
}

// realFFT performs FFT optimized for real-valued input
func (fft *FFTProcessor) realFFT(data []Complex, direction int) ([]Complex, error) {
	n := len(data)
	
	// Check if input is real
	isReal := true
	for _, c := range data {
		if math.Abs(c.Imag) > 1e-10 {
			isReal = false
			break
		}
	}

	if !isReal {
		// Fall back to complex FFT
		return fft.cooleyTukeyFFT(data, direction, fft.getOrCreatePlan(PlanKey{Size: n, Algorithm: CooleyTukey, Direction: direction}))
	}

	// For real input, we can use the fact that FFT of real signal has Hermitian symmetry
	if n%2 != 0 {
		return fft.cooleyTukeyFFT(data, direction, fft.getOrCreatePlan(PlanKey{Size: n, Algorithm: CooleyTukey, Direction: direction}))
	}

	// Pack real data into complex array of half size
	halfN := n / 2
	packed := make([]Complex, halfN)
	for i := 0; i < halfN; i++ {
		packed[i] = Complex{data[2*i].Real, data[2*i+1].Real}
	}

	// Perform FFT on packed data
	packedFFT, err := fft.cooleyTukeyFFT(packed, direction, fft.getOrCreatePlan(PlanKey{Size: halfN, Algorithm: CooleyTukey, Direction: direction}))
	if err != nil {
		return nil, err
	}

	// Unpack the result using Hermitian symmetry
	result := make([]Complex, n)
	result[0] = Complex{packedFFT[0].Real + packedFFT[0].Imag, 0}
	result[halfN] = Complex{packedFFT[0].Real - packedFFT[0].Imag, 0}

	for k := 1; k < halfN; k++ {
		angle := math.Pi * float64(k) / float64(halfN)
		cosAngle := math.Cos(angle)
		sinAngle := math.Sin(angle) * float64(direction)

		fk := packedFFT[k]
		fmk := Complex{packedFFT[halfN-k].Real, -packedFFT[halfN-k].Imag}

		h1k := Complex{
			0.5 * (fk.Real + fmk.Real),
			0.5 * (fk.Imag + fmk.Imag),
		}

		h2k := Complex{
			0.5 * (fk.Imag - fmk.Imag),
			0.5 * (-fk.Real + fmk.Real),
		}

		w := Complex{cosAngle, sinAngle}
		h2kw := h2k.Mul(w)

		result[k] = h1k.Add(h2kw)
		result[n-k] = Complex{h1k.Real - h2kw.Real, -h1k.Imag + h2kw.Imag}
	}

	return result, nil
}

// Helper functions

func (fft *FFTProcessor) getOrCreatePlan(key PlanKey) *FFTPlan {
	fft.mutex.RLock()
	plan, exists := fft.planCache[key]
	fft.mutex.RUnlock()

	if exists {
		return plan
	}

	fft.mutex.Lock()
	defer fft.mutex.Unlock()

	// Double-check after acquiring write lock
	if plan, exists := fft.planCache[key]; exists {
		return plan
	}

	// Create new plan
	plan = fft.createPlan(key)
	
	// Cache management
	if len(fft.planCache) >= fft.config.MaxCacheSize {
		fft.evictOldestPlan()
	}
	
	fft.planCache[key] = plan
	return plan
}

func (fft *FFTProcessor) createPlan(key PlanKey) *FFTPlan {
	n := key.Size
	plan := &FFTPlan{
		Size:      n,
		Algorithm: key.Algorithm,
		CreatedAt: time.Now(),
	}

	// Generate twiddle factors
	plan.TwiddleFactors = fft.generateTwiddleFactors(n, key.Direction)

	// Generate bit-reversed indices
	plan.BitReversed = generateBitReversedIndices(n)

	// Generate stages for iterative FFT
	plan.Stages = fft.generateStages(n)

	return plan
}

func (fft *FFTProcessor) generateTwiddleFactors(n, direction int) []Complex {
	factors := make([]Complex, n)
	for i := 0; i < n; i++ {
		angle := 2 * math.Pi * float64(i) / float64(n)
		factors[i] = NewComplexPolar(1, float64(direction)*angle)
	}
	return factors
}

func generateBitReversedIndices(n int) []int {
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = bitReverse(i, n)
	}
	return indices
}

func (fft *FFTProcessor) generateStages(n int) []FFTStage {
	stages := make([]FFTStage, 0)
	logN := int(math.Log2(float64(n)))

	for stage := 0; stage < logN; stage++ {
		stride := 1 << stage
		numTasks := n / (2 * stride)
		
		stages = append(stages, FFTStage{
			Level:    stage,
			Stride:   stride,
			NumTasks: numTasks,
		})
	}

	return stages
}

func (fft *FFTProcessor) shouldParallelize(numTasks int) bool {
	return numTasks >= fft.config.NumWorkers && numTasks >= fft.config.ChunkSize
}

func (fft *FFTProcessor) applyWindow(data []Complex, windowFunc WindowFunction) {
	n := len(data)
	
	switch windowFunc {
	case Hamming:
		for i := 0; i < n; i++ {
			w := 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(n-1))
			data[i] = data[i].MulScalar(w)
		}
	case Hanning:
		for i := 0; i < n; i++ {
			w := 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n-1))
			data[i] = data[i].MulScalar(w)
		}
	case Blackman:
		for i := 0; i < n; i++ {
			w := 0.42 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n-1)) + 0.08*math.Cos(4*math.Pi*float64(i)/float64(n-1))
			data[i] = data[i].MulScalar(w)
		}
	}
}

func (fft *FFTProcessor) updateStats(numSamples int, duration time.Duration) {
	fft.stats.mutex.Lock()
	defer fft.stats.mutex.Unlock()

	fft.stats.TotalTransforms++
	fft.stats.TotalSamples += int64(numSamples)
	fft.stats.TotalTime += duration

	if fft.stats.TotalTransforms > 0 {
		fft.stats.AverageLatency = fft.stats.TotalTime / time.Duration(fft.stats.TotalTransforms)
	}
}

func (fft *FFTProcessor) evictOldestPlan() {
	var oldestKey PlanKey
	var oldestTime time.Time
	first := true

	for key, plan := range fft.planCache {
		if first || plan.CreatedAt.Before(oldestTime) {
			oldestKey = key
			oldestTime = plan.CreatedAt
			first = false
		}
	}

	delete(fft.planCache, oldestKey)
}

// Worker pool implementation

func NewWorkerPool(processor *FFTProcessor, numWorkers int) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())
	
	pool := &WorkerPool{
		workers:   make([]*Worker, numWorkers),
		taskQueue: make(chan Task, numWorkers*2),
		ctx:       ctx,
		cancel:    cancel,
	}

	for i := 0; i < numWorkers; i++ {
		pool.workers[i] = &Worker{
			id:         i,
			processor:  processor,
			taskQueue:  pool.taskQueue,
			resultChan: make(chan Result, 1),
		}
	}

	return pool
}

// Utility functions

func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

func nextPowerOfTwo(n int) int {
	if isPowerOfTwo(n) {
		return n
	}
	
	power := 1
	for power < n {
		power <<= 1
	}
	return power
}

func bitReverse(x, n int) int {
	result := 0
	logN := int(math.Log2(float64(n)))
	
	for i := 0; i < logN; i++ {
		if x&(1<<i) != 0 {
			result |= 1 << (logN - 1 - i)
		}
	}
	
	return result
}

func bitReverseRadix4(x, n int) int {
	result := 0
	logN := int(math.Log2(float64(n))) / 2
	
	for i := 0; i < logN; i++ {
		result = (result << 2) | (x & 3)
		x >>= 2
	}
	
	return result
}

func factorize(n int) []int {
	factors := make([]int, 0)
	
	// Try small primes first
	primes := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
	
	for _, p := range primes {
		for n%p == 0 {
			factors = append(factors, p)
			n /= p
		}
	}
	
	// Handle remaining factor
	if n > 1 {
		factors = append(factors, n)
	}
	
	return factors
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Public utility functions

// FFT2D performs 2D FFT for image processing
func (fft *FFTProcessor) FFT2D(input [][]Complex) ([][]Complex, error) {
	rows := len(input)
	if rows == 0 {
		return nil, errors.New("input cannot be empty")
	}
	cols := len(input[0])

	// Allocate result
	result := make([][]Complex, rows)
	for i := range result {
		result[i] = make([]Complex, cols)
		copy(result[i], input[i])
	}

	// FFT along rows
	var wg sync.WaitGroup
	for i := 0; i < rows; i++ {
		wg.Add(1)
		go func(row int) {
			defer wg.Done()
			transformed, err := fft.FFT(result[row])
			if err == nil {
				copy(result[row], transformed)
			}
		}(i)
	}
	wg.Wait()

	// FFT along columns
	for j := 0; j < cols; j++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			column := make([]Complex, rows)
			for i := 0; i < rows; i++ {
				column[i] = result[i][col]
			}
			
			transformed, err := fft.FFT(column)
			if err == nil {
				for i := 0; i < rows; i++ {
					result[i][col] = transformed[i]
				}
			}
		}(j)
	}
	wg.Wait()

	return result, nil
}

// Convolution performs circular convolution using FFT
func (fft *FFTProcessor) Convolution(a, b []Complex) ([]Complex, error) {
	n := len(a)
	if len(b) != n {
		return nil, errors.New("sequences must have same length")
	}

	// Pad to next power of 2 for efficiency
	size := nextPowerOfTwo(2*n - 1)
	paddedA := make([]Complex, size)
	paddedB := make([]Complex, size)
	
	copy(paddedA, a)
	copy(paddedB, b)

	// FFT of both sequences
	fftA, err := fft.FFT(paddedA)
	if err != nil {
		return nil, err
	}

	fftB, err := fft.FFT(paddedB)
	if err != nil {
		return nil, err
	}

	// Pointwise multiplication
	product := make([]Complex, size)
	for i := 0; i < size; i++ {
		product[i] = fftA[i].Mul(fftB[i])
	}

	// Inverse FFT
	result, err := fft.IFFT(product)
	if err != nil {
		return nil, err
	}

	// Return first n elements (trim padding)
	return result[:n], nil
}

// GetStats returns current FFT processor statistics
func (fft *FFTProcessor) GetStats() *FFTStats {
	fft.stats.mutex.RLock()
	defer fft.stats.mutex.RUnlock()

	// Return a copy of stats
	statsCopy := &FFTStats{
		TotalTransforms:   fft.stats.TotalTransforms,
		TotalSamples:      fft.stats.TotalSamples,
		TotalTime:         fft.stats.TotalTime,
		ParallelEfficiency: fft.stats.ParallelEfficiency,
		CacheHitRate:      fft.stats.CacheHitRate,
		AverageLatency:    fft.stats.AverageLatency,
		WorkerUtilization: make([]float64, len(fft.stats.WorkerUtilization)),
	}
	copy(statsCopy.WorkerUtilization, fft.stats.WorkerUtilization)

	return statsCopy
}

// Cleanup releases resources used by the FFT processor
func (fft *FFTProcessor) Cleanup() {
	if fft.workerPool != nil {
		fft.workerPool.cancel()
	}
	
	fft.mutex.Lock()
	defer fft.mutex.Unlock()
	
	fft.twiddleCache = nil
	fft.planCache = nil
}