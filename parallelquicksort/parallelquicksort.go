package parallelquicksort

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

// Threshold below which we use sequential sort
const sequentialThreshold = 1000

// ParallelQuickSort sorts a slice of integers using parallel quicksort
func ParallelQuickSort(arr []int) {
	if len(arr) <= 1 {
		return
	}
	
	// Use all available CPU cores
	maxGoroutines := runtime.NumCPU()
	sem := make(chan struct{}, maxGoroutines)
	
	parallelQuickSort(arr, sem)
}

func parallelQuickSort(arr []int, sem chan struct{}) {
	if len(arr) <= 1 {
		return
	}
	
	// Use sequential sort for small arrays
	if len(arr) < sequentialThreshold {
		sequentialQuickSort(arr)
		return
	}
	
	// Partition the array
	pivotIndex := partition(arr)
	
	// Sort partitions in parallel
	var wg sync.WaitGroup
	
	// Left partition
	if pivotIndex > 0 {
		select {
		case sem <- struct{}{}:
			// Got a token, can run in parallel
			wg.Add(1)
			go func() {
				defer func() {
					<-sem
					wg.Done()
				}()
				parallelQuickSort(arr[:pivotIndex], sem)
			}()
		default:
			// No token available, run sequentially
			parallelQuickSort(arr[:pivotIndex], sem)
		}
	}
	
	// Right partition
	if pivotIndex < len(arr)-1 {
		select {
		case sem <- struct{}{}:
			// Got a token, can run in parallel
			wg.Add(1)
			go func() {
				defer func() {
					<-sem
					wg.Done()
				}()
				parallelQuickSort(arr[pivotIndex+1:], sem)
			}()
		default:
			// No token available, run sequentially
			parallelQuickSort(arr[pivotIndex+1:], sem)
		}
	}
	
	wg.Wait()
}

func sequentialQuickSort(arr []int) {
	if len(arr) <= 1 {
		return
	}
	
	pivotIndex := partition(arr)
	sequentialQuickSort(arr[:pivotIndex])
	sequentialQuickSort(arr[pivotIndex+1:])
}

func partition(arr []int) int {
	// Choose random pivot for better average case performance
	randomIndex := rand.Intn(len(arr))
	arr[randomIndex], arr[len(arr)-1] = arr[len(arr)-1], arr[randomIndex]
	
	pivot := arr[len(arr)-1]
	i := -1
	
	for j := 0; j < len(arr)-1; j++ {
		if arr[j] <= pivot {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	
	arr[i+1], arr[len(arr)-1] = arr[len(arr)-1], arr[i+1]
	return i + 1
}

// ParallelQuickSortGeneric sorts a slice of any ordered type
func ParallelQuickSortGeneric[T any](arr []T, less func(a, b T) bool) {
	if len(arr) <= 1 {
		return
	}
	
	maxGoroutines := runtime.NumCPU()
	sem := make(chan struct{}, maxGoroutines)
	
	parallelQuickSortGeneric(arr, less, sem)
}

func parallelQuickSortGeneric[T any](arr []T, less func(a, b T) bool, sem chan struct{}) {
	if len(arr) <= 1 {
		return
	}
	
	if len(arr) < sequentialThreshold {
		sequentialQuickSortGeneric(arr, less)
		return
	}
	
	pivotIndex := partitionGeneric(arr, less)
	
	var wg sync.WaitGroup
	
	// Left partition
	if pivotIndex > 0 {
		select {
		case sem <- struct{}{}:
			wg.Add(1)
			go func() {
				defer func() {
					<-sem
					wg.Done()
				}()
				parallelQuickSortGeneric(arr[:pivotIndex], less, sem)
			}()
		default:
			parallelQuickSortGeneric(arr[:pivotIndex], less, sem)
		}
	}
	
	// Right partition
	if pivotIndex < len(arr)-1 {
		select {
		case sem <- struct{}{}:
			wg.Add(1)
			go func() {
				defer func() {
					<-sem
					wg.Done()
				}()
				parallelQuickSortGeneric(arr[pivotIndex+1:], less, sem)
			}()
		default:
			parallelQuickSortGeneric(arr[pivotIndex+1:], less, sem)
		}
	}
	
	wg.Wait()
}

func sequentialQuickSortGeneric[T any](arr []T, less func(a, b T) bool) {
	if len(arr) <= 1 {
		return
	}
	
	pivotIndex := partitionGeneric(arr, less)
	sequentialQuickSortGeneric(arr[:pivotIndex], less)
	sequentialQuickSortGeneric(arr[pivotIndex+1:], less)
}

func partitionGeneric[T any](arr []T, less func(a, b T) bool) int {
	randomIndex := rand.Intn(len(arr))
	arr[randomIndex], arr[len(arr)-1] = arr[len(arr)-1], arr[randomIndex]
	
	pivot := arr[len(arr)-1]
	i := -1
	
	for j := 0; j < len(arr)-1; j++ {
		if !less(pivot, arr[j]) {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	
	arr[i+1], arr[len(arr)-1] = arr[len(arr)-1], arr[i+1]
	return i + 1
}

// ThreeWayParallelQuickSort implements three-way partitioning for better handling of duplicates
func ThreeWayParallelQuickSort(arr []int) {
	if len(arr) <= 1 {
		return
	}
	
	maxGoroutines := runtime.NumCPU()
	sem := make(chan struct{}, maxGoroutines)
	
	threeWayParallelQuickSort(arr, sem)
}

func threeWayParallelQuickSort(arr []int, sem chan struct{}) {
	if len(arr) <= 1 {
		return
	}
	
	if len(arr) < sequentialThreshold {
		threeWaySequentialQuickSort(arr)
		return
	}
	
	lt, gt := threeWayPartition(arr)
	
	var wg sync.WaitGroup
	
	// Sort elements less than pivot
	if lt > 0 {
		select {
		case sem <- struct{}{}:
			wg.Add(1)
			go func() {
				defer func() {
					<-sem
					wg.Done()
				}()
				threeWayParallelQuickSort(arr[:lt], sem)
			}()
		default:
			threeWayParallelQuickSort(arr[:lt], sem)
		}
	}
	
	// Sort elements greater than pivot
	if gt < len(arr) {
		select {
		case sem <- struct{}{}:
			wg.Add(1)
			go func() {
				defer func() {
					<-sem
					wg.Done()
				}()
				threeWayParallelQuickSort(arr[gt:], sem)
			}()
		default:
			threeWayParallelQuickSort(arr[gt:], sem)
		}
	}
	
	wg.Wait()
}

func threeWaySequentialQuickSort(arr []int) {
	if len(arr) <= 1 {
		return
	}
	
	lt, gt := threeWayPartition(arr)
	threeWaySequentialQuickSort(arr[:lt])
	threeWaySequentialQuickSort(arr[gt:])
}

func threeWayPartition(arr []int) (int, int) {
	if len(arr) == 0 {
		return 0, 0
	}
	
	// Choose random pivot
	randomIndex := rand.Intn(len(arr))
	arr[0], arr[randomIndex] = arr[randomIndex], arr[0]
	
	pivot := arr[0]
	lt, i, gt := 0, 1, len(arr)
	
	for i < gt {
		if arr[i] < pivot {
			arr[lt], arr[i] = arr[i], arr[lt]
			lt++
			i++
		} else if arr[i] > pivot {
			gt--
			arr[i], arr[gt] = arr[gt], arr[i]
		} else {
			i++
		}
	}
	
	return lt, gt
}

// Example demonstrates parallel quicksort
func Example() {
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())
	
	fmt.Println("=== Parallel QuickSort Example ===")
	
	// Create random array
	size := 20
	arr := make([]int, size)
	for i := range arr {
		arr[i] = rand.Intn(100)
	}
	
	fmt.Println("Original array:", arr)
	
	// Sort using parallel quicksort
	start := time.Now()
	ParallelQuickSort(arr)
	elapsed := time.Since(start)
	
	fmt.Println("Sorted array:", arr)
	fmt.Printf("Time taken: %v\n", elapsed)
	
	// Example with duplicates using three-way partitioning
	fmt.Println("\n=== Three-Way Parallel QuickSort (with duplicates) ===")
	
	arrWithDups := make([]int, size)
	for i := range arrWithDups {
		arrWithDups[i] = rand.Intn(10) // More duplicates
	}
	
	fmt.Println("Original array:", arrWithDups)
	
	start = time.Now()
	ThreeWayParallelQuickSort(arrWithDups)
	elapsed = time.Since(start)
	
	fmt.Println("Sorted array:", arrWithDups)
	fmt.Printf("Time taken: %v\n", elapsed)
	
	// Example with generic sort
	fmt.Println("\n=== Generic Parallel QuickSort (strings) ===")
	
	strings := []string{"apple", "zebra", "banana", "cherry", "date", "elderberry"}
	fmt.Println("Original:", strings)
	
	ParallelQuickSortGeneric(strings, func(a, b string) bool { return a < b })
	fmt.Println("Sorted:", strings)
}