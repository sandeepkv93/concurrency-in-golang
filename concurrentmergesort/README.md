# Concurrent Merge Sort in Go

This package provides an implementation of the merge sort algorithm that can concurrently sort large slices of integers. It uses goroutines to parallelize the sorting process, making it faster on systems with multiple cores.

## Functions

### `insertionSort(slice []int)`

This function sorts small slices (less than 16 elements) using the insertion sort algorithm. It is used within the merge sort implementation to handle base cases efficiently.

### `mergeSort(slice []int, wg *sync.WaitGroup)`

This function sorts a slice recursively using the merge sort algorithm. It divides the slice into two halves and sorts them concurrently using goroutines. For small slices, it falls back to insertion sort.

### `merge(slice []int, mid int)`

This function merges two sorted halves of a slice into a single sorted slice. It's used as part of the merge sort process.

### `concurrentSort(slice []int)`

This function sorts a slice using concurrent merge sort. It's a wrapper around the `mergeSort` function that handles concurrency using a wait group.

### `checkIfSorted(slice []int) bool`

This function checks if a slice is sorted in ascending order. It returns true if the slice is sorted, and false otherwise.

## Usage

The code is self-contained and can be run directly. It includes a `main` function that demonstrates how to use the concurrent sorting functions.

1. **Generating a Slice** : The main function begins by generating a random slice of integers of size 1,000,000.
2. **Starting the Timer** : Before sorting, the code starts a timer to measure performance.
3. **Sorting** : The slice is then sorted concurrently using the `concurrentSort` function.
4. **Time Taken** : After sorting, the code prints the time taken to sort the slice.
5. **Checking if Sorted** : Finally, the code checks if the slice is sorted using the `checkIfSorted` function, printing a message if the slice is not sorted.

## Conclusion

This package provides a fast and concurrent implementation of merge sort that leverages Go's native concurrency features. It demonstrates how to use goroutines and wait groups to parallelize a classic algorithm, achieving better performance on multicore systems.
