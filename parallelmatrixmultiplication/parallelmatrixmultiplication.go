package parallelmatrixmultiplication

import (
	"fmt"
	"runtime"
	"sync"
)

// Matrix represents a 2D matrix
type Matrix struct {
	rows, cols int
	data       [][]float64
}

// NewMatrix creates a new matrix with given dimensions
func NewMatrix(rows, cols int) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{rows: rows, cols: cols, data: data}
}

// NewMatrixFromSlice creates a matrix from a 2D slice
func NewMatrixFromSlice(data [][]float64) *Matrix {
	if len(data) == 0 {
		return &Matrix{rows: 0, cols: 0, data: nil}
	}
	rows := len(data)
	cols := len(data[0])
	
	// Verify all rows have same number of columns
	for _, row := range data {
		if len(row) != cols {
			panic("all rows must have the same number of columns")
		}
	}
	
	// Deep copy the data
	matrixData := make([][]float64, rows)
	for i := range data {
		matrixData[i] = make([]float64, cols)
		copy(matrixData[i], data[i])
	}
	
	return &Matrix{rows: rows, cols: cols, data: matrixData}
}

// Get returns the value at position (i, j)
func (m *Matrix) Get(i, j int) float64 {
	return m.data[i][j]
}

// Set sets the value at position (i, j)
func (m *Matrix) Set(i, j int, val float64) {
	m.data[i][j] = val
}

// Dimensions returns the dimensions of the matrix
func (m *Matrix) Dimensions() (rows, cols int) {
	return m.rows, m.cols
}

// MultiplySequential performs sequential matrix multiplication
func MultiplySequential(a, b *Matrix) (*Matrix, error) {
	if a.cols != b.rows {
		return nil, fmt.Errorf("incompatible dimensions: A is %dx%d, B is %dx%d", 
			a.rows, a.cols, b.rows, b.cols)
	}
	
	result := NewMatrix(a.rows, b.cols)
	
	for i := 0; i < a.rows; i++ {
		for j := 0; j < b.cols; j++ {
			sum := 0.0
			for k := 0; k < a.cols; k++ {
				sum += a.data[i][k] * b.data[k][j]
			}
			result.data[i][j] = sum
		}
	}
	
	return result, nil
}

// MultiplyParallel performs parallel matrix multiplication
func MultiplyParallel(a, b *Matrix) (*Matrix, error) {
	if a.cols != b.rows {
		return nil, fmt.Errorf("incompatible dimensions: A is %dx%d, B is %dx%d", 
			a.rows, a.cols, b.rows, b.cols)
	}
	
	result := NewMatrix(a.rows, b.cols)
	
	// Use goroutines to compute rows in parallel
	numWorkers := runtime.NumCPU()
	rowsPerWorker := (a.rows + numWorkers - 1) / numWorkers
	
	var wg sync.WaitGroup
	
	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > a.rows {
			endRow = a.rows
		}
		
		if startRow >= endRow {
			continue
		}
		
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < b.cols; j++ {
					sum := 0.0
					for k := 0; k < a.cols; k++ {
						sum += a.data[i][k] * b.data[k][j]
					}
					result.data[i][j] = sum
				}
			}
		}(startRow, endRow)
	}
	
	wg.Wait()
	return result, nil
}

// MultiplyParallelBlocked performs blocked parallel matrix multiplication for better cache performance
func MultiplyParallelBlocked(a, b *Matrix) (*Matrix, error) {
	if a.cols != b.rows {
		return nil, fmt.Errorf("incompatible dimensions: A is %dx%d, B is %dx%d", 
			a.rows, a.cols, b.rows, b.cols)
	}
	
	result := NewMatrix(a.rows, b.cols)
	blockSize := 64 // Tune based on cache size
	
	// Transpose B for better cache locality
	bT := transposeMatrix(b)
	
	numWorkers := runtime.NumCPU()
	blocks := make(chan blockRange, 100)
	
	var wg sync.WaitGroup
	
	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for block := range blocks {
				multiplyBlock(a, bT, result, block, blockSize)
			}
		}()
	}
	
	// Generate blocks
	for i := 0; i < a.rows; i += blockSize {
		for j := 0; j < b.cols; j += blockSize {
			blocks <- blockRange{
				rowStart: i,
				rowEnd:   min(i+blockSize, a.rows),
				colStart: j,
				colEnd:   min(j+blockSize, b.cols),
			}
		}
	}
	close(blocks)
	
	wg.Wait()
	return result, nil
}

type blockRange struct {
	rowStart, rowEnd int
	colStart, colEnd int
}

func multiplyBlock(a, bT, result *Matrix, block blockRange, blockSize int) {
	for i := block.rowStart; i < block.rowEnd; i++ {
		for j := block.colStart; j < block.colEnd; j++ {
			sum := 0.0
			// Since B is transposed, we access bT.data[j] instead of column j
			for k := 0; k < a.cols; k++ {
				sum += a.data[i][k] * bT.data[j][k]
			}
			result.data[i][j] = sum
		}
	}
}

func transposeMatrix(m *Matrix) *Matrix {
	result := NewMatrix(m.cols, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.data[j][i] = m.data[i][j]
		}
	}
	return result
}

// StrassenMultiply performs matrix multiplication using Strassen's algorithm
func StrassenMultiply(a, b *Matrix) (*Matrix, error) {
	if a.cols != b.rows {
		return nil, fmt.Errorf("incompatible dimensions: A is %dx%d, B is %dx%d", 
			a.rows, a.cols, b.rows, b.cols)
	}
	
	// For small matrices, use regular multiplication
	if a.rows <= 64 || a.cols <= 64 || b.cols <= 64 {
		return MultiplyParallel(a, b)
	}
	
	// Pad matrices to next power of 2 if necessary
	size := nextPowerOf2(max(a.rows, a.cols, b.rows, b.cols))
	aPadded := padMatrix(a, size, size)
	bPadded := padMatrix(b, size, size)
	
	result := strassenRecursive(aPadded, bPadded)
	
	// Remove padding
	return extractSubmatrix(result, 0, 0, a.rows, b.cols), nil
}

func strassenRecursive(a, b *Matrix) *Matrix {
	n := a.rows
	
	// Base case
	if n <= 64 {
		result, _ := MultiplySequential(a, b)
		return result
	}
	
	// Divide matrices into quadrants
	half := n / 2
	
	a11 := extractSubmatrix(a, 0, 0, half, half)
	a12 := extractSubmatrix(a, 0, half, half, half)
	a21 := extractSubmatrix(a, half, 0, half, half)
	a22 := extractSubmatrix(a, half, half, half, half)
	
	b11 := extractSubmatrix(b, 0, 0, half, half)
	b12 := extractSubmatrix(b, 0, half, half, half)
	b21 := extractSubmatrix(b, half, 0, half, half)
	b22 := extractSubmatrix(b, half, half, half, half)
	
	// Compute Strassen's 7 products in parallel
	var m1, m2, m3, m4, m5, m6, m7 *Matrix
	var wg sync.WaitGroup
	
	wg.Add(7)
	
	go func() {
		defer wg.Done()
		m1 = strassenRecursive(addMatrices(a11, a22), addMatrices(b11, b22))
	}()
	
	go func() {
		defer wg.Done()
		m2 = strassenRecursive(addMatrices(a21, a22), b11)
	}()
	
	go func() {
		defer wg.Done()
		m3 = strassenRecursive(a11, subtractMatrices(b12, b22))
	}()
	
	go func() {
		defer wg.Done()
		m4 = strassenRecursive(a22, subtractMatrices(b21, b11))
	}()
	
	go func() {
		defer wg.Done()
		m5 = strassenRecursive(addMatrices(a11, a12), b22)
	}()
	
	go func() {
		defer wg.Done()
		m6 = strassenRecursive(subtractMatrices(a21, a11), addMatrices(b11, b12))
	}()
	
	go func() {
		defer wg.Done()
		m7 = strassenRecursive(subtractMatrices(a12, a22), addMatrices(b21, b22))
	}()
	
	wg.Wait()
	
	// Compute result quadrants
	c11 := addMatrices(subtractMatrices(addMatrices(m1, m4), m5), m7)
	c12 := addMatrices(m3, m5)
	c21 := addMatrices(m2, m4)
	c22 := addMatrices(subtractMatrices(addMatrices(m1, m3), m2), m6)
	
	// Combine quadrants
	result := NewMatrix(n, n)
	copySubmatrix(c11, result, 0, 0)
	copySubmatrix(c12, result, 0, half)
	copySubmatrix(c21, result, half, 0)
	copySubmatrix(c22, result, half, half)
	
	return result
}

func nextPowerOf2(n int) int {
	power := 1
	for power < n {
		power *= 2
	}
	return power
}

func padMatrix(m *Matrix, newRows, newCols int) *Matrix {
	result := NewMatrix(newRows, newCols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.data[i][j] = m.data[i][j]
		}
	}
	return result
}

func extractSubmatrix(m *Matrix, rowStart, colStart, rows, cols int) *Matrix {
	result := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.data[i][j] = m.data[rowStart+i][colStart+j]
		}
	}
	return result
}

func copySubmatrix(src, dst *Matrix, rowOffset, colOffset int) {
	for i := 0; i < src.rows; i++ {
		for j := 0; j < src.cols; j++ {
			dst.data[rowOffset+i][colOffset+j] = src.data[i][j]
		}
	}
}

func addMatrices(a, b *Matrix) *Matrix {
	result := NewMatrix(a.rows, a.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			result.data[i][j] = a.data[i][j] + b.data[i][j]
		}
	}
	return result
}

func subtractMatrices(a, b *Matrix) *Matrix {
	result := NewMatrix(a.rows, a.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			result.data[i][j] = a.data[i][j] - b.data[i][j]
		}
	}
	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(nums ...int) int {
	m := nums[0]
	for _, n := range nums[1:] {
		if n > m {
			m = n
		}
	}
	return m
}

// Example demonstrates matrix multiplication
func Example() {
	fmt.Println("=== Parallel Matrix Multiplication Example ===")
	
	// Create sample matrices
	a := NewMatrixFromSlice([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	
	b := NewMatrixFromSlice([][]float64{
		{7, 8},
		{9, 10},
		{11, 12},
	})
	
	fmt.Println("Matrix A:")
	printMatrix(a)
	
	fmt.Println("\nMatrix B:")
	printMatrix(b)
	
	// Sequential multiplication
	resultSeq, _ := MultiplySequential(a, b)
	fmt.Println("\nSequential Result (A × B):")
	printMatrix(resultSeq)
	
	// Parallel multiplication
	resultPar, _ := MultiplyParallel(a, b)
	fmt.Println("\nParallel Result (A × B):")
	printMatrix(resultPar)
	
	// Blocked parallel multiplication
	resultBlocked, _ := MultiplyParallelBlocked(a, b)
	fmt.Println("\nBlocked Parallel Result (A × B):")
	printMatrix(resultBlocked)
}

func printMatrix(m *Matrix) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			fmt.Printf("%8.2f ", m.data[i][j])
		}
		fmt.Println()
	}
}