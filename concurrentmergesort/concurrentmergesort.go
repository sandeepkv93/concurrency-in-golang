package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// insertionSort sorts small slices using the insertion sort algorithm.
func insertionSort(slice []int) {
	for i := 1; i < len(slice); i++ {
		j := i
		// Keep swapping adjacent pairs if they're in the wrong order.
		for j > 0 && slice[j-1] > slice[j] {
			slice[j-1], slice[j] = slice[j], slice[j-1]
			j--
		}
	}
}

// mergeSort sorts a slice recursively using the merge sort algorithm.
func mergeSort(slice []int, wg *sync.WaitGroup) {
	defer wg.Done() // Decrement the counter when the goroutine completes.

	if len(slice) < 2 {
		return // If slice has 1 or 0 elements, it's already sorted.
	}

	if len(slice) < 16 {
		insertionSort(slice) // Use insertion sort for small slices.
		return
	}

	mid := len(slice) / 2 // Determine the midpoint of the slice.

	var wgInner sync.WaitGroup
	wgInner.Add(2) // Expect two additional goroutines for sorting halves.

	// Sort the two halves concurrently.
	go mergeSort(slice[:mid], &wgInner)
	go mergeSort(slice[mid:], &wgInner)

	wgInner.Wait() // Wait for both halves to be sorted.

	merge(slice, mid) // Merge the sorted halves.
}

// merge merges two sorted halves of a slice.
func merge(slice []int, mid int) {
	temp := make([]int, len(slice))
	i, j, k := 0, mid, 0

	// Compare elements and merge.
	for i < mid && j < len(slice) {
		if slice[i] < slice[j] {
			temp[k] = slice[i]
			i++
		} else {
			temp[k] = slice[j]
			j++
		}
		k++
	}

	// Copy any remaining elements from the left half.
	for i < mid {
		temp[k] = slice[i]
		i++
		k++
	}

	// Copy any remaining elements from the right half.
	for j < len(slice) {
		temp[k] = slice[j]
		j++
		k++
	}

	copy(slice, temp) // Copy the merged data back into the original slice.
}

// concurrentSort sorts a slice using concurrent merge sort.
func concurrentSort(slice []int) {
	var wg sync.WaitGroup
	wg.Add(1) // Expect one additional goroutine.
	go mergeSort(slice, &wg)
	wg.Wait() // Wait for sorting to be complete.
}

// checkIfSorted checks if a slice is sorted in ascending order.
func checkIfSorted(slice []int) bool {
	for i := 1; i < len(slice); i++ {
		if slice[i-1] > slice[i] {
			return false
		}
	}
	return true
}

func main() {
	// Generate a random slice of integers.
	sliceSize := 1000000
	slice := make([]int, sliceSize)
	for i := 0; i < sliceSize; i++ {
		slice[i] = rand.Intn(1000000)
	}

	// Start the timer to measure performance.
	start := time.Now()

	// Concurrently sort the slice.
	concurrentSort(slice)

	// Print the time taken.
	fmt.Printf("Sorting took %v\n", time.Since(start))

	// Check if the slice is sorted.
	if !checkIfSorted(slice) {
		fmt.Println("The slice is not sorted")
		return
	}
}
