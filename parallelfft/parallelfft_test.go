package parallelfft

import (
	"context"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"
)

const epsilon = 1e-10

func TestNewComplex(t *testing.T) {
	c := NewComplex(3, 4)
	if c.Real != 3 || c.Imag != 4 {
		t.Errorf("Expected (3, 4), got (%f, %f)", c.Real, c.Imag)
	}
}

func TestNewComplexPolar(t *testing.T) {
	magnitude := 5.0
	phase := math.Pi / 4
	c := NewComplexPolar(magnitude, phase)
	
	expectedReal := magnitude * math.Cos(phase)
	expectedImag := magnitude * math.Sin(phase)
	
	if math.Abs(c.Real-expectedReal) > epsilon || math.Abs(c.Imag-expectedImag) > epsilon {
		t.Errorf("Expected (%f, %f), got (%f, %f)", expectedReal, expectedImag, c.Real, c.Imag)
	}
}

func TestComplexOperations(t *testing.T) {
	c1 := NewComplex(3, 4)
	c2 := NewComplex(1, 2)
	
	// Test addition
	sum := c1.Add(c2)
	if sum.Real != 4 || sum.Imag != 6 {
		t.Errorf("Addition failed: expected (4, 6), got (%f, %f)", sum.Real, sum.Imag)
	}
	
	// Test subtraction
	diff := c1.Sub(c2)
	if diff.Real != 2 || diff.Imag != 2 {
		t.Errorf("Subtraction failed: expected (2, 2), got (%f, %f)", diff.Real, diff.Imag)
	}
	
	// Test multiplication
	prod := c1.Mul(c2)
	expected := NewComplex(-5, 10) // (3+4i)(1+2i) = 3+6i+4i+8i² = 3+10i-8 = -5+10i
	if math.Abs(prod.Real-expected.Real) > epsilon || math.Abs(prod.Imag-expected.Imag) > epsilon {
		t.Errorf("Multiplication failed: expected (%f, %f), got (%f, %f)", 
			expected.Real, expected.Imag, prod.Real, prod.Imag)
	}
	
	// Test scalar multiplication
	scaled := c1.MulScalar(2)
	if scaled.Real != 6 || scaled.Imag != 8 {
		t.Errorf("Scalar multiplication failed: expected (6, 8), got (%f, %f)", scaled.Real, scaled.Imag)
	}
	
	// Test magnitude
	magnitude := c1.Abs()
	expectedMag := math.Sqrt(3*3 + 4*4)
	if math.Abs(magnitude-expectedMag) > epsilon {
		t.Errorf("Magnitude failed: expected %f, got %f", expectedMag, magnitude)
	}
	
	// Test phase
	phase := c1.Phase()
	expectedPhase := math.Atan2(4, 3)
	if math.Abs(phase-expectedPhase) > epsilon {
		t.Errorf("Phase failed: expected %f, got %f", expectedPhase, phase)
	}
}

func TestNewFFTProcessor(t *testing.T) {
	config := FFTConfig{
		Algorithm:    CooleyTukey,
		NumWorkers:   4,
		ChunkSize:    512,
		EnableCache:  true,
		MaxCacheSize: 50,
	}
	
	processor := NewFFTProcessor(config)
	if processor == nil {
		t.Fatal("Failed to create FFT processor")
	}
	
	if processor.config.NumWorkers != 4 {
		t.Errorf("Expected 4 workers, got %d", processor.config.NumWorkers)
	}
	
	if processor.config.ChunkSize != 512 {
		t.Errorf("Expected chunk size 512, got %d", processor.config.ChunkSize)
	}
}

func TestDefaultFFTConfig(t *testing.T) {
	config := FFTConfig{}
	processor := NewFFTProcessor(config)
	
	if processor.config.NumWorkers <= 0 {
		t.Error("Default number of workers should be positive")
	}
	
	if processor.config.ChunkSize != 1024 {
		t.Errorf("Expected default chunk size 1024, got %d", processor.config.ChunkSize)
	}
	
	if processor.config.MaxCacheSize != 100 {
		t.Errorf("Expected default max cache size 100, got %d", processor.config.MaxCacheSize)
	}
}

func TestFFTBasicProperties(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: CooleyTukey})
	defer processor.Cleanup()
	
	// Test empty input
	_, err := processor.FFT([]Complex{})
	if err == nil {
		t.Error("Expected error for empty input")
	}
	
	// Test single element
	input := []Complex{NewComplex(1, 0)}
	result, err := processor.FFT(input)
	if err != nil {
		t.Fatalf("FFT failed: %v", err)
	}
	
	if len(result) != 1 {
		t.Errorf("Expected length 1, got %d", len(result))
	}
	
	if math.Abs(result[0].Real-1) > epsilon || math.Abs(result[0].Imag) > epsilon {
		t.Errorf("Expected (1, 0), got (%f, %f)", result[0].Real, result[0].Imag)
	}
}

func TestFFTAndIFFT(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: CooleyTukey})
	defer processor.Cleanup()
	
	// Test with power-of-2 size
	input := []Complex{
		NewComplex(1, 0),
		NewComplex(2, 0),
		NewComplex(3, 0),
		NewComplex(4, 0),
	}
	
	// Forward FFT
	fftResult, err := processor.FFT(input)
	if err != nil {
		t.Fatalf("FFT failed: %v", err)
	}
	
	// Inverse FFT
	ifftResult, err := processor.IFFT(fftResult)
	if err != nil {
		t.Fatalf("IFFT failed: %v", err)
	}
	
	// Check if we get back the original
	if len(ifftResult) != len(input) {
		t.Errorf("Expected length %d, got %d", len(input), len(ifftResult))
	}
	
	for i := range input {
		if math.Abs(ifftResult[i].Real-input[i].Real) > epsilon || 
		   math.Abs(ifftResult[i].Imag-input[i].Imag) > epsilon {
			t.Errorf("IFFT[%d]: expected (%f, %f), got (%f, %f)", 
				i, input[i].Real, input[i].Imag, ifftResult[i].Real, ifftResult[i].Imag)
		}
	}
}

func TestFFTLinearity(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: CooleyTukey})
	defer processor.Cleanup()
	
	input1 := []Complex{
		NewComplex(1, 0),
		NewComplex(0, 1),
		NewComplex(-1, 0),
		NewComplex(0, -1),
	}
	
	input2 := []Complex{
		NewComplex(2, 1),
		NewComplex(1, -2),
		NewComplex(-2, 1),
		NewComplex(1, 2),
	}
	
	// FFT of individual inputs
	fft1, _ := processor.FFT(input1)
	fft2, _ := processor.FFT(input2)
	
	// Sum inputs and compute FFT
	sumInput := make([]Complex, len(input1))
	for i := range input1 {
		sumInput[i] = input1[i].Add(input2[i])
	}
	fftSum, _ := processor.FFT(sumInput)
	
	// FFT(a + b) should equal FFT(a) + FFT(b)
	for i := range fft1 {
		expected := fft1[i].Add(fft2[i])
		if math.Abs(fftSum[i].Real-expected.Real) > epsilon || 
		   math.Abs(fftSum[i].Imag-expected.Imag) > epsilon {
			t.Errorf("Linearity test failed at index %d", i)
		}
	}
}

func TestRadix4FFT(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: Radix4})
	defer processor.Cleanup()
	
	// Test with size that's a power of 4
	input := make([]Complex, 16)
	for i := range input {
		input[i] = NewComplex(float64(i), 0)
	}
	
	result, err := processor.FFT(input)
	if err != nil {
		t.Fatalf("Radix-4 FFT failed: %v", err)
	}
	
	if len(result) != len(input) {
		t.Errorf("Expected length %d, got %d", len(input), len(result))
	}
	
	// Test round-trip
	recovered, err := processor.IFFT(result)
	if err != nil {
		t.Fatalf("Radix-4 IFFT failed: %v", err)
	}
	
	for i := range input {
		if math.Abs(recovered[i].Real-input[i].Real) > epsilon {
			t.Errorf("Round-trip failed at index %d: expected %f, got %f", 
				i, input[i].Real, recovered[i].Real)
		}
	}
}

func TestMixedRadixFFT(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: MixedRadix})
	defer processor.Cleanup()
	
	// Test with non-power-of-2 size
	input := make([]Complex, 12) // 12 = 3 * 4
	for i := range input {
		input[i] = NewComplex(float64(i+1), 0)
	}
	
	result, err := processor.FFT(input)
	if err != nil {
		t.Fatalf("Mixed-radix FFT failed: %v", err)
	}
	
	if len(result) != len(input) {
		t.Errorf("Expected length %d, got %d", len(input), len(result))
	}
	
	// Test round-trip
	recovered, err := processor.IFFT(result)
	if err != nil {
		t.Fatalf("Mixed-radix IFFT failed: %v", err)
	}
	
	for i := range input {
		if math.Abs(recovered[i].Real-input[i].Real) > epsilon {
			t.Errorf("Round-trip failed at index %d: expected %f, got %f", 
				i, input[i].Real, recovered[i].Real)
		}
	}
}

func TestBluesteinFFT(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: Bluestein})
	defer processor.Cleanup()
	
	// Test with prime size
	input := make([]Complex, 7)
	for i := range input {
		input[i] = NewComplex(float64(i), float64(i))
	}
	
	result, err := processor.FFT(input)
	if err != nil {
		t.Fatalf("Bluestein FFT failed: %v", err)
	}
	
	if len(result) != len(input) {
		t.Errorf("Expected length %d, got %d", len(input), len(result))
	}
	
	// Test round-trip
	recovered, err := processor.IFFT(result)
	if err != nil {
		t.Fatalf("Bluestein IFFT failed: %v", err)
	}
	
	for i := range input {
		if math.Abs(recovered[i].Real-input[i].Real) > epsilon ||
		   math.Abs(recovered[i].Imag-input[i].Imag) > epsilon {
			t.Errorf("Round-trip failed at index %d: expected (%f, %f), got (%f, %f)", 
				i, input[i].Real, input[i].Imag, recovered[i].Real, recovered[i].Imag)
		}
	}
}

func TestRealFFT(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: RealFFT})
	defer processor.Cleanup()
	
	// Test with real-valued input
	input := make([]Complex, 8)
	for i := range input {
		input[i] = NewComplex(float64(i+1), 0) // Only real part
	}
	
	result, err := processor.FFT(input)
	if err != nil {
		t.Fatalf("Real FFT failed: %v", err)
	}
	
	if len(result) != len(input) {
		t.Errorf("Expected length %d, got %d", len(input), len(result))
	}
	
	// For real input, FFT should have Hermitian symmetry
	n := len(result)
	for i := 1; i < n/2; i++ {
		conjugate := Complex{result[n-i].Real, -result[n-i].Imag}
		if math.Abs(result[i].Real-conjugate.Real) > epsilon ||
		   math.Abs(result[i].Imag-conjugate.Imag) > epsilon {
			t.Errorf("Hermitian symmetry violated at index %d", i)
		}
	}
}

func TestFFT2D(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: CooleyTukey})
	defer processor.Cleanup()
	
	// Create 2D input
	rows, cols := 4, 4
	input := make([][]Complex, rows)
	for i := range input {
		input[i] = make([]Complex, cols)
		for j := range input[i] {
			input[i][j] = NewComplex(float64(i*cols+j), 0)
		}
	}
	
	result, err := processor.FFT2D(input)
	if err != nil {
		t.Fatalf("2D FFT failed: %v", err)
	}
	
	if len(result) != rows || len(result[0]) != cols {
		t.Errorf("Expected %dx%d result, got %dx%d", rows, cols, len(result), len(result[0]))
	}
}

func TestConvolution(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: CooleyTukey})
	defer processor.Cleanup()
	
	// Simple convolution test
	a := []Complex{
		NewComplex(1, 0),
		NewComplex(2, 0),
		NewComplex(3, 0),
		NewComplex(0, 0),
	}
	
	b := []Complex{
		NewComplex(1, 0),
		NewComplex(1, 0),
		NewComplex(0, 0),
		NewComplex(0, 0),
	}
	
	result, err := processor.Convolution(a, b)
	if err != nil {
		t.Fatalf("Convolution failed: %v", err)
	}
	
	// Expected result: [1, 3, 5, 3]
	expected := []float64{1, 3, 5, 3}
	
	for i, exp := range expected {
		if math.Abs(result[i].Real-exp) > epsilon {
			t.Errorf("Convolution result[%d]: expected %f, got %f", i, exp, result[i].Real)
		}
	}
}

func TestWindowFunctions(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{
		Algorithm:  CooleyTukey,
		WindowFunc: Hamming,
	})
	defer processor.Cleanup()
	
	input := make([]Complex, 8)
	for i := range input {
		input[i] = NewComplex(1, 0) // Constant signal
	}
	
	result, err := processor.FFT(input)
	if err != nil {
		t.Fatalf("Windowed FFT failed: %v", err)
	}
	
	if len(result) != len(input) {
		t.Errorf("Expected length %d, got %d", len(input), len(result))
	}
}

func TestPlanCaching(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{
		Algorithm:    CooleyTukey,
		EnableCache:  true,
		MaxCacheSize: 5,
	})
	defer processor.Cleanup()
	
	input := make([]Complex, 8)
	for i := range input {
		input[i] = NewComplex(float64(i), 0)
	}
	
	// First FFT should create and cache plan
	_, err := processor.FFT(input)
	if err != nil {
		t.Fatalf("First FFT failed: %v", err)
	}
	
	planKey := PlanKey{Size: 8, Algorithm: CooleyTukey, Direction: 1}
	processor.mutex.RLock()
	_, exists := processor.planCache[planKey]
	processor.mutex.RUnlock()
	
	if !exists {
		t.Error("Plan should be cached")
	}
	
	// Second FFT should use cached plan
	_, err = processor.FFT(input)
	if err != nil {
		t.Fatalf("Second FFT failed: %v", err)
	}
}

func TestConcurrentFFT(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{
		Algorithm:  CooleyTukey,
		NumWorkers: 4,
	})
	defer processor.Cleanup()
	
	const numGoroutines = 10
	const inputSize = 16
	
	var wg sync.WaitGroup
	results := make([][]Complex, numGoroutines)
	errors := make([]error, numGoroutines)
	
	// Create input
	input := make([]Complex, inputSize)
	for i := range input {
		input[i] = NewComplex(float64(i), 0)
	}
	
	// Run concurrent FFTs
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx], errors[idx] = processor.FFT(input)
		}(i)
	}
	
	wg.Wait()
	
	// Check results
	for i := 0; i < numGoroutines; i++ {
		if errors[i] != nil {
			t.Errorf("Goroutine %d failed: %v", i, errors[i])
		}
		
		if len(results[i]) != inputSize {
			t.Errorf("Goroutine %d: expected length %d, got %d", i, inputSize, len(results[i]))
		}
		
		// All results should be identical
		if i > 0 {
			for j := range results[i] {
				if math.Abs(results[i][j].Real-results[0][j].Real) > epsilon ||
				   math.Abs(results[i][j].Imag-results[0][j].Imag) > epsilon {
					t.Errorf("Results differ between goroutines at index %d", j)
				}
			}
		}
	}
}

func TestStatistics(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: CooleyTukey})
	defer processor.Cleanup()
	
	input := make([]Complex, 8)
	for i := range input {
		input[i] = NewComplex(float64(i), 0)
	}
	
	// Perform some FFTs
	for i := 0; i < 5; i++ {
		_, err := processor.FFT(input)
		if err != nil {
			t.Fatalf("FFT %d failed: %v", i, err)
		}
	}
	
	stats := processor.GetStats()
	if stats.TotalTransforms != 5 {
		t.Errorf("Expected 5 transforms, got %d", stats.TotalTransforms)
	}
	
	if stats.TotalSamples != 5*8 {
		t.Errorf("Expected %d samples, got %d", 5*8, stats.TotalSamples)
	}
	
	if stats.TotalTime == 0 {
		t.Error("Total time should be greater than 0")
	}
}

func TestUtilityFunctions(t *testing.T) {
	// Test isPowerOfTwo
	testCases := []struct {
		input    int
		expected bool
	}{
		{1, true},
		{2, true},
		{4, true},
		{8, true},
		{16, true},
		{3, false},
		{5, false},
		{6, false},
		{7, false},
		{9, false},
	}
	
	for _, tc := range testCases {
		if isPowerOfTwo(tc.input) != tc.expected {
			t.Errorf("isPowerOfTwo(%d): expected %v, got %v", tc.input, tc.expected, !tc.expected)
		}
	}
	
	// Test nextPowerOfTwo
	testCases2 := []struct {
		input    int
		expected int
	}{
		{1, 1},
		{2, 2},
		{3, 4},
		{5, 8},
		{9, 16},
		{15, 16},
		{17, 32},
	}
	
	for _, tc := range testCases2 {
		if nextPowerOfTwo(tc.input) != tc.expected {
			t.Errorf("nextPowerOfTwo(%d): expected %d, got %d", tc.input, tc.expected, nextPowerOfTwo(tc.input))
		}
	}
	
	// Test bitReverse
	if bitReverse(0, 8) != 0 {
		t.Error("bitReverse(0, 8) should be 0")
	}
	if bitReverse(1, 8) != 4 {
		t.Error("bitReverse(1, 8) should be 4")
	}
	if bitReverse(2, 8) != 2 {
		t.Error("bitReverse(2, 8) should be 2")
	}
	if bitReverse(3, 8) != 6 {
		t.Error("bitReverse(3, 8) should be 6")
	}
}

func TestFactorize(t *testing.T) {
	testCases := []struct {
		input    int
		expected []int
	}{
		{1, []int{}},
		{2, []int{2}},
		{4, []int{2, 2}},
		{6, []int{2, 3}},
		{12, []int{2, 2, 3}},
		{15, []int{3, 5}},
		{30, []int{2, 3, 5}},
	}
	
	for _, tc := range testCases {
		result := factorize(tc.input)
		if len(result) != len(tc.expected) {
			t.Errorf("factorize(%d): expected %v, got %v", tc.input, tc.expected, result)
			continue
		}
		
		for i, factor := range result {
			if factor != tc.expected[i] {
				t.Errorf("factorize(%d): expected %v, got %v", tc.input, tc.expected, result)
				break
			}
		}
	}
}

func BenchmarkFFTCooleyTukey(b *testing.B) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: CooleyTukey})
	defer processor.Cleanup()
	
	sizes := []int{64, 256, 1024, 4096}
	
	for _, size := range sizes {
		input := make([]Complex, size)
		for i := range input {
			input[i] = NewComplex(rand.Float64(), rand.Float64())
		}
		
		b.Run(fmt.Sprintf("Size-%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := processor.FFT(input)
				if err != nil {
					b.Fatalf("FFT failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkFFTRadix4(b *testing.B) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: Radix4})
	defer processor.Cleanup()
	
	sizes := []int{64, 256, 1024, 4096}
	
	for _, size := range sizes {
		input := make([]Complex, size)
		for i := range input {
			input[i] = NewComplex(rand.Float64(), rand.Float64())
		}
		
		b.Run(fmt.Sprintf("Size-%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := processor.FFT(input)
				if err != nil {
					b.Fatalf("FFT failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkFFTParallelWorkers(b *testing.B) {
	workers := []int{1, 2, 4, 8}
	size := 1024
	
	for _, numWorkers := range workers {
		processor := NewFFTProcessor(FFTConfig{
			Algorithm:  CooleyTukey,
			NumWorkers: numWorkers,
		})
		
		input := make([]Complex, size)
		for i := range input {
			input[i] = NewComplex(rand.Float64(), rand.Float64())
		}
		
		b.Run(fmt.Sprintf("Workers-%d", numWorkers), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := processor.FFT(input)
				if err != nil {
					b.Fatalf("FFT failed: %v", err)
				}
			}
		})
		
		processor.Cleanup()
	}
}

func BenchmarkConvolution(b *testing.B) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: CooleyTukey})
	defer processor.Cleanup()
	
	sizes := []int{64, 256, 1024}
	
	for _, size := range sizes {
		a := make([]Complex, size)
		b := make([]Complex, size)
		for i := range a {
			a[i] = NewComplex(rand.Float64(), rand.Float64())
			b[i] = NewComplex(rand.Float64(), rand.Float64())
		}
		
		b.Run(fmt.Sprintf("Size-%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := processor.Convolution(a, b)
				if err != nil {
					b.Fatalf("Convolution failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkFFT2D(b *testing.B) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: CooleyTukey})
	defer processor.Cleanup()
	
	sizes := []int{32, 64, 128}
	
	for _, size := range sizes {
		input := make([][]Complex, size)
		for i := range input {
			input[i] = make([]Complex, size)
			for j := range input[i] {
				input[i][j] = NewComplex(rand.Float64(), rand.Float64())
			}
		}
		
		b.Run(fmt.Sprintf("Size-%dx%d", size, size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := processor.FFT2D(input)
				if err != nil {
					b.Fatalf("2D FFT failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkPlanCaching(b *testing.B) {
	processor := NewFFTProcessor(FFTConfig{
		Algorithm:    CooleyTukey,
		EnableCache:  true,
		MaxCacheSize: 100,
	})
	defer processor.Cleanup()
	
	input := make([]Complex, 256)
	for i := range input {
		input[i] = NewComplex(rand.Float64(), rand.Float64())
	}
	
	// Warm up cache
	processor.FFT(input)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := processor.FFT(input)
		if err != nil {
			b.Fatalf("FFT failed: %v", err)
		}
	}
}

func TestErrorHandling(t *testing.T) {
	processor := NewFFTProcessor(FFTConfig{Algorithm: CooleyTukey})
	defer processor.Cleanup()
	
	// Test with non-power-of-2 size for Cooley-Tukey
	input := make([]Complex, 7) // Prime number
	_, err := processor.FFT(input)
	if err == nil {
		t.Error("Expected error for non-power-of-2 input with Cooley-Tukey algorithm")
	}
	
	// Test convolution with different sized inputs
	a := make([]Complex, 4)
	b := make([]Complex, 6)
	_, err = processor.Convolution(a, b)
	if err == nil {
		t.Error("Expected error for convolution with different sized inputs")
	}
	
	// Test 2D FFT with empty input
	emptyInput := [][]Complex{}
	_, err = processor.FFT2D(emptyInput)
	if err == nil {
		t.Error("Expected error for empty 2D input")
	}
}

func ExampleFFTProcessor_FFT() {
	// Create FFT processor
	processor := NewFFTProcessor(FFTConfig{
		Algorithm:  CooleyTukey,
		NumWorkers: 4,
	})
	defer processor.Cleanup()
	
	// Create input signal
	input := make([]Complex, 8)
	for i := range input {
		input[i] = NewComplex(float64(i), 0)
	}
	
	// Compute FFT
	result, err := processor.FFT(input)
	if err != nil {
		panic(err)
	}
	
	fmt.Printf("FFT computed for %d samples\n", len(result))
	// Output: FFT computed for 8 samples
}

func ExampleFFTProcessor_Convolution() {
	processor := NewFFTProcessor(FFTConfig{Algorithm: CooleyTukey})
	defer processor.Cleanup()
	
	// Create two signals
	signal := []Complex{
		NewComplex(1, 0),
		NewComplex(2, 0),
		NewComplex(3, 0),
		NewComplex(0, 0),
	}
	
	kernel := []Complex{
		NewComplex(1, 0),
		NewComplex(1, 0),
		NewComplex(0, 0),
		NewComplex(0, 0),
	}
	
	// Compute convolution
	result, err := processor.Convolution(signal, kernel)
	if err != nil {
		panic(err)
	}
	
	fmt.Printf("Convolution result has %d samples\n", len(result))
	// Output: Convolution result has 4 samples
}