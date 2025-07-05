package parallelmatrixmultiplication

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func TestMatrixMultiplicationBasic(t *testing.T) {
	// Test case 1: Simple 2x3 × 3x2
	a := NewMatrixFromSlice([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	
	b := NewMatrixFromSlice([][]float64{
		{7, 8},
		{9, 10},
		{11, 12},
	})
	
	expected := NewMatrixFromSlice([][]float64{
		{58, 64},
		{139, 154},
	})
	
	// Test sequential
	result, err := MultiplySequential(a, b)
	if err != nil {
		t.Fatalf("Sequential multiplication failed: %v", err)
	}
	
	if !matricesEqual(result, expected) {
		t.Error("Sequential multiplication gave incorrect result")
	}
	
	// Test parallel
	result, err = MultiplyParallel(a, b)
	if err != nil {
		t.Fatalf("Parallel multiplication failed: %v", err)
	}
	
	if !matricesEqual(result, expected) {
		t.Error("Parallel multiplication gave incorrect result")
	}
	
	// Test blocked parallel
	result, err = MultiplyParallelBlocked(a, b)
	if err != nil {
		t.Fatalf("Blocked parallel multiplication failed: %v", err)
	}
	
	if !matricesEqual(result, expected) {
		t.Error("Blocked parallel multiplication gave incorrect result")
	}
}

func TestMatrixMultiplicationIdentity(t *testing.T) {
	// Multiply by identity matrix
	size := 10
	a := randomMatrix(size, size)
	identity := identityMatrix(size)
	
	result, _ := MultiplyParallel(a, identity)
	
	if !matricesEqual(a, result) {
		t.Error("Multiplication by identity matrix should return original matrix")
	}
}

func TestMatrixMultiplicationDimensionError(t *testing.T) {
	a := NewMatrix(2, 3)
	b := NewMatrix(4, 2) // Incompatible dimensions
	
	_, err := MultiplySequential(a, b)
	if err == nil {
		t.Error("Expected dimension error for sequential multiplication")
	}
	
	_, err = MultiplyParallel(a, b)
	if err == nil {
		t.Error("Expected dimension error for parallel multiplication")
	}
	
	_, err = MultiplyParallelBlocked(a, b)
	if err == nil {
		t.Error("Expected dimension error for blocked multiplication")
	}
}

func TestMatrixMultiplicationLarge(t *testing.T) {
	sizes := []int{50, 100, 200}
	
	for _, size := range sizes {
		t.Run(fmt.Sprintf("Size%d", size), func(t *testing.T) {
			a := randomMatrix(size, size)
			b := randomMatrix(size, size)
			
			// Compare sequential and parallel results
			seqResult, _ := MultiplySequential(a, b)
			parResult, _ := MultiplyParallel(a, b)
			blockedResult, _ := MultiplyParallelBlocked(a, b)
			
			if !matricesAlmostEqual(seqResult, parResult, 1e-10) {
				t.Error("Parallel result differs from sequential")
			}
			
			if !matricesAlmostEqual(seqResult, blockedResult, 1e-10) {
				t.Error("Blocked result differs from sequential")
			}
		})
	}
}

func TestStrassenMultiplication(t *testing.T) {
	// Test with power-of-2 size
	size := 128
	a := randomMatrix(size, size)
	b := randomMatrix(size, size)
	
	seqResult, _ := MultiplySequential(a, b)
	strassenResult, err := StrassenMultiply(a, b)
	
	if err != nil {
		t.Fatalf("Strassen multiplication failed: %v", err)
	}
	
	if !matricesAlmostEqual(seqResult, strassenResult, 1e-10) {
		t.Error("Strassen result differs from sequential")
	}
	
	// Test with non-power-of-2 size
	a = randomMatrix(100, 100)
	b = randomMatrix(100, 100)
	
	seqResult, _ = MultiplySequential(a, b)
	strassenResult, err = StrassenMultiply(a, b)
	
	if err != nil {
		t.Fatalf("Strassen multiplication failed for non-power-of-2: %v", err)
	}
	
	if !matricesAlmostEqual(seqResult, strassenResult, 1e-10) {
		t.Error("Strassen result differs from sequential for non-power-of-2")
	}
}

func TestConcurrentMultiplications(t *testing.T) {
	// Run multiple matrix multiplications concurrently
	numGoroutines := 10
	size := 50
	done := make(chan bool, numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		go func() {
			a := randomMatrix(size, size)
			b := randomMatrix(size, size)
			
			seqResult, _ := MultiplySequential(a, b)
			parResult, _ := MultiplyParallel(a, b)
			
			if !matricesAlmostEqual(seqResult, parResult, 1e-10) {
				t.Error("Concurrent multiplication gave incorrect result")
			}
			
			done <- true
		}()
	}
	
	// Wait for all goroutines
	for i := 0; i < numGoroutines; i++ {
		<-done
	}
}

// Helper functions

func randomMatrix(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.data[i][j] = rand.Float64() * 10
		}
	}
	return m
}

func identityMatrix(size int) *Matrix {
	m := NewMatrix(size, size)
	for i := 0; i < size; i++ {
		m.data[i][i] = 1
	}
	return m
}

func matricesEqual(a, b *Matrix) bool {
	if a.rows != b.rows || a.cols != b.cols {
		return false
	}
	
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			if a.data[i][j] != b.data[i][j] {
				return false
			}
		}
	}
	return true
}

func matricesAlmostEqual(a, b *Matrix, tolerance float64) bool {
	if a.rows != b.rows || a.cols != b.cols {
		return false
	}
	
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			if math.Abs(a.data[i][j]-b.data[i][j]) > tolerance {
				return false
			}
		}
	}
	return true
}

// Benchmarks

func BenchmarkMatrixMultiplication(b *testing.B) {
	sizes := []int{64, 128, 256, 512}
	
	for _, size := range sizes {
		a := randomMatrix(size, size)
		bMatrix := randomMatrix(size, size)
		
		b.Run(fmt.Sprintf("Sequential_%dx%d", size, size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MultiplySequential(a, bMatrix)
			}
		})
		
		b.Run(fmt.Sprintf("Parallel_%dx%d", size, size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MultiplyParallel(a, bMatrix)
			}
		})
		
		b.Run(fmt.Sprintf("Blocked_%dx%d", size, size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MultiplyParallelBlocked(a, bMatrix)
			}
		})
		
		if size >= 128 {
			b.Run(fmt.Sprintf("Strassen_%dx%d", size, size), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					StrassenMultiply(a, bMatrix)
				}
			})
		}
	}
}

func BenchmarkLargeMatrixMultiplication(b *testing.B) {
	size := 1024
	a := randomMatrix(size, size)
	bMatrix := randomMatrix(size, size)
	
	b.Run("Parallel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			MultiplyParallel(a, bMatrix)
		}
	})
	
	b.Run("Blocked", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			MultiplyParallelBlocked(a, bMatrix)
		}
	})
}