package parallelquicksort

import (
	"math/rand"
	"sort"
	"testing"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func TestParallelQuickSortBasic(t *testing.T) {
	tests := []struct {
		name string
		arr  []int
		want []int
	}{
		{"Empty", []int{}, []int{}},
		{"Single", []int{42}, []int{42}},
		{"Sorted", []int{1, 2, 3, 4, 5}, []int{1, 2, 3, 4, 5}},
		{"Reverse", []int{5, 4, 3, 2, 1}, []int{1, 2, 3, 4, 5}},
		{"Random", []int{3, 1, 4, 1, 5, 9, 2, 6}, []int{1, 1, 2, 3, 4, 5, 6, 9}},
		{"Duplicates", []int{5, 2, 5, 2, 5, 2}, []int{2, 2, 2, 5, 5, 5}},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			arr := make([]int, len(tt.arr))
			copy(arr, tt.arr)
			
			ParallelQuickSort(arr)
			
			for i := range arr {
				if arr[i] != tt.want[i] {
					t.Errorf("ParallelQuickSort() = %v, want %v", arr, tt.want)
					break
				}
			}
		})
	}
}

func TestParallelQuickSortLarge(t *testing.T) {
	sizes := []int{100, 1000, 10000, 100000}
	
	for _, size := range sizes {
		t.Run(fmt.Sprintf("Size%d", size), func(t *testing.T) {
			// Create random array
			arr := make([]int, size)
			for i := range arr {
				arr[i] = rand.Intn(size)
			}
			
			// Create copy for verification
			expected := make([]int, size)
			copy(expected, arr)
			sort.Ints(expected)
			
			// Sort using parallel quicksort
			ParallelQuickSort(arr)
			
			// Verify sorted
			for i := range arr {
				if arr[i] != expected[i] {
					t.Errorf("Array not properly sorted at index %d", i)
					break
				}
			}
		})
	}
}

func TestThreeWayParallelQuickSort(t *testing.T) {
	// Test with many duplicates
	size := 10000
	arr := make([]int, size)
	for i := range arr {
		arr[i] = rand.Intn(10) // Only 10 different values
	}
	
	expected := make([]int, size)
	copy(expected, arr)
	sort.Ints(expected)
	
	ThreeWayParallelQuickSort(arr)
	
	for i := range arr {
		if arr[i] != expected[i] {
			t.Errorf("Array not properly sorted at index %d", i)
			break
		}
	}
}

func TestParallelQuickSortGeneric(t *testing.T) {
	// Test with strings
	strings := []string{"zebra", "apple", "mango", "banana", "cherry", "apple"}
	expected := []string{"apple", "apple", "banana", "cherry", "mango", "zebra"}
	
	ParallelQuickSortGeneric(strings, func(a, b string) bool { return a < b })
	
	for i := range strings {
		if strings[i] != expected[i] {
			t.Errorf("Strings not properly sorted: got %v, want %v", strings, expected)
			break
		}
	}
	
	// Test with custom struct
	type Person struct {
		Name string
		Age  int
	}
	
	people := []Person{
		{"Alice", 30},
		{"Bob", 25},
		{"Charlie", 35},
		{"David", 25},
	}
	
	// Sort by age, then by name
	ParallelQuickSortGeneric(people, func(a, b Person) bool {
		if a.Age != b.Age {
			return a.Age < b.Age
		}
		return a.Name < b.Name
	})
	
	// Verify sort order
	if people[0].Name != "Bob" || people[1].Name != "David" ||
		people[2].Name != "Alice" || people[3].Name != "Charlie" {
		t.Errorf("People not properly sorted: %v", people)
	}
}

func TestConcurrentSorts(t *testing.T) {
	// Run multiple sorts concurrently
	numGoroutines := 10
	arraySize := 10000
	done := make(chan bool, numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		go func() {
			arr := make([]int, arraySize)
			for j := range arr {
				arr[j] = rand.Intn(arraySize)
			}
			
			ParallelQuickSort(arr)
			
			// Verify sorted
			for j := 1; j < len(arr); j++ {
				if arr[j] < arr[j-1] {
					t.Errorf("Array not sorted correctly")
					break
				}
			}
			
			done <- true
		}()
	}
	
	// Wait for all goroutines to complete
	for i := 0; i < numGoroutines; i++ {
		<-done
	}
}

func BenchmarkParallelQuickSort(b *testing.B) {
	sizes := []int{100, 1000, 10000, 100000}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			original := make([]int, size)
			for i := range original {
				original[i] = rand.Intn(size)
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				arr := make([]int, size)
				copy(arr, original)
				b.StartTimer()
				
				ParallelQuickSort(arr)
			}
		})
	}
}

func BenchmarkSequentialVsParallel(b *testing.B) {
	size := 100000
	original := make([]int, size)
	for i := range original {
		original[i] = rand.Intn(size)
	}
	
	b.Run("Sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			arr := make([]int, size)
			copy(arr, original)
			b.StartTimer()
			
			sequentialQuickSort(arr)
		}
	})
	
	b.Run("Parallel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			arr := make([]int, size)
			copy(arr, original)
			b.StartTimer()
			
			ParallelQuickSort(arr)
		}
	})
	
	b.Run("StandardSort", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			arr := make([]int, size)
			copy(arr, original)
			b.StartTimer()
			
			sort.Ints(arr)
		}
	})
}