package main

import (
	"math/rand"
	"sort"
	"testing"
)

// TestConcurrentSort_EmptySlice tests sorting an empty slice.
func TestConcurrentSort_EmptySlice(t *testing.T) {
	slice := []int{}
	concurrentSort(slice)
	if !sort.IntsAreSorted(slice) {
		t.Errorf("Failed to sort an empty slice")
	}
}

// TestConcurrentSort_SingleElement tests sorting a single-element slice.
func TestConcurrentSort_SingleElement(t *testing.T) {
	slice := []int{5}
	concurrentSort(slice)
	if !sort.IntsAreSorted(slice) {
		t.Errorf("Failed to sort a single-element slice")
	}
}

// TestConcurrentSort_SmallSortedSlice tests sorting a small sorted slice.
func TestConcurrentSort_SmallSortedSlice(t *testing.T) {
	slice := []int{1, 2, 3, 4, 5}
	concurrentSort(slice)
	if !sort.IntsAreSorted(slice) {
		t.Errorf("Failed to sort a small sorted slice")
	}
}

// TestConcurrentSort_SmallReverseSortedSlice tests sorting a small reverse-sorted slice.
func TestConcurrentSort_SmallReverseSortedSlice(t *testing.T) {
	slice := []int{5, 4, 3, 2, 1}
	concurrentSort(slice)
	if !sort.IntsAreSorted(slice) {
		t.Errorf("Failed to sort a small reverse-sorted slice")
	}
}

// TestConcurrentSort_LargeRandomSlice tests sorting a large random slice.
func TestConcurrentSort_LargeRandomSlice(t *testing.T) {
	slice := make([]int, 1000)
	for i := 0; i < 1000; i++ {
		slice[i] = rand.Intn(10000)
	}
	concurrentSort(slice)
	if !sort.IntsAreSorted(slice) {
		t.Errorf("Failed to sort a large random slice")
	}
}

// TestConcurrentSort_NegativeValues tests sorting a slice containing negative values.
func TestConcurrentSort_NegativeValues(t *testing.T) {
	slice := []int{-5, -2, -9, 1, 3}
	concurrentSort(slice)
	if !sort.IntsAreSorted(slice) {
		t.Errorf("Failed to sort a slice containing negative values")
	}
}

// TestConcurrentSort_DuplicateValues tests sorting a slice containing duplicate values.
func TestConcurrentSort_DuplicateValues(t *testing.T) {
	slice := []int{5, 3, 5, 1, 3, 5}
	concurrentSort(slice)
	if !sort.IntsAreSorted(slice) {
		t.Errorf("Failed to sort a slice containing duplicate values")
	}
}

// TestConcurrentSort_AllSameValues tests sorting a slice where all values are the same.
func TestConcurrentSort_AllSameValues(t *testing.T) {
	slice := []int{2, 2, 2, 2, 2}
	concurrentSort(slice)
	if !sort.IntsAreSorted(slice) {
		t.Errorf("Failed to sort a slice where all values are the same")
	}
}

// TestConcurrentSort_ThresholdBoundary tests sorting a slice that is exactly at the threshold between using merge sort and insertion sort.
func TestConcurrentSort_ThresholdBoundary(t *testing.T) {
	slice := make([]int, 16)
	for i := 0; i < 16; i++ {
		slice[i] = rand.Intn(100)
	}
	concurrentSort(slice)
	if !sort.IntsAreSorted(slice) {
		t.Errorf("Failed to sort a slice that is exactly at the threshold between using merge sort and insertion sort")
	}
}

// TestConcurrentSort_UniformDistribution tests sorting a slice with uniformly distributed values.
func TestConcurrentSort_UniformDistribution(t *testing.T) {
	slice := make([]int, 1000)
	for i := 0; i < 1000; i++ {
		slice[i] = i
	}
	rand.Shuffle(len(slice), func(i, j int) { slice[i], slice[j] = slice[j], slice[i] })
	concurrentSort(slice)
	if !sort.IntsAreSorted(slice) {
		t.Errorf("Failed to sort a slice with uniformly distributed values")
	}
}
