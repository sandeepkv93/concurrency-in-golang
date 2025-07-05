package paralleldijkstra

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func TestBasicShortestPath(t *testing.T) {
	// Create a simple graph
	g := NewGraph(5)
	g.AddEdge(0, 1, 10)
	g.AddEdge(0, 2, 5)
	g.AddEdge(1, 2, 2)
	g.AddEdge(1, 3, 1)
	g.AddEdge(2, 1, 3)
	g.AddEdge(2, 3, 9)
	g.AddEdge(2, 4, 2)
	g.AddEdge(3, 4, 4)
	g.AddEdge(4, 3, 6)
	
	// Expected distances from vertex 0
	expected := []float64{0, 8, 5, 9, 7}
	
	// Test sequential
	sp1 := SequentialDijkstra(g, 0)
	for i, exp := range expected {
		if math.Abs(sp1.Distance[i]-exp) > 0.001 {
			t.Errorf("Sequential: Vertex %d - expected distance %.1f, got %.1f", 
				i, exp, sp1.Distance[i])
		}
	}
	
	// Test parallel
	sp2 := ParallelDijkstra(g, 0)
	for i, exp := range expected {
		if math.Abs(sp2.Distance[i]-exp) > 0.001 {
			t.Errorf("Parallel: Vertex %d - expected distance %.1f, got %.1f", 
				i, exp, sp2.Distance[i])
		}
	}
	
	// Test delta-stepping
	sp3 := DeltaSteppingDijkstra(g, 0, 3)
	for i, exp := range expected {
		if math.Abs(sp3.Distance[i]-exp) > 0.001 {
			t.Errorf("Delta-stepping: Vertex %d - expected distance %.1f, got %.1f", 
				i, exp, sp3.Distance[i])
		}
	}
}

func TestPath(t *testing.T) {
	g := NewGraph(4)
	g.AddEdge(0, 1, 1)
	g.AddEdge(1, 2, 2)
	g.AddEdge(2, 3, 3)
	g.AddEdge(0, 3, 10) // Direct path is more expensive
	
	sp := SequentialDijkstra(g, 0)
	
	// Path from 0 to 3 should be 0->1->2->3
	path := sp.GetPath(3)
	expected := []int{0, 1, 2, 3}
	
	if len(path) != len(expected) {
		t.Errorf("Expected path length %d, got %d", len(expected), len(path))
	}
	
	for i := range path {
		if path[i] != expected[i] {
			t.Errorf("Path differs at position %d: expected %d, got %d", 
				i, expected[i], path[i])
		}
	}
}

func TestDisconnectedGraph(t *testing.T) {
	g := NewGraph(5)
	g.AddEdge(0, 1, 1)
	g.AddEdge(1, 2, 1)
	g.AddEdge(3, 4, 1)
	// No connection between {0,1,2} and {3,4}
	
	sp := SequentialDijkstra(g, 0)
	
	// Vertices 0,1,2 should be reachable
	if math.IsInf(sp.Distance[0], 1) || math.IsInf(sp.Distance[1], 1) || math.IsInf(sp.Distance[2], 1) {
		t.Error("Connected vertices should be reachable")
	}
	
	// Vertices 3,4 should be unreachable
	if !math.IsInf(sp.Distance[3], 1) || !math.IsInf(sp.Distance[4], 1) {
		t.Error("Disconnected vertices should be unreachable")
	}
}

func TestSingleVertex(t *testing.T) {
	g := NewGraph(1)
	
	sp := SequentialDijkstra(g, 0)
	
	if sp.Distance[0] != 0 {
		t.Errorf("Distance to self should be 0, got %.1f", sp.Distance[0])
	}
}

func TestNegativeWeights(t *testing.T) {
	// Note: Dijkstra's algorithm doesn't work correctly with negative weights
	// This test just ensures it doesn't crash
	g := NewGraph(3)
	g.AddEdge(0, 1, -1)
	g.AddEdge(1, 2, 2)
	
	// Should not panic
	_ = SequentialDijkstra(g, 0)
	_ = ParallelDijkstra(g, 0)
	_ = DeltaSteppingDijkstra(g, 0, 1)
}

func TestBidirectional(t *testing.T) {
	g := NewGraph(5)
	g.AddEdge(0, 1, 10)
	g.AddEdge(0, 2, 5)
	g.AddEdge(1, 2, 2)
	g.AddEdge(1, 3, 1)
	g.AddEdge(2, 1, 3)
	g.AddEdge(2, 3, 9)
	g.AddEdge(2, 4, 2)
	g.AddEdge(3, 4, 4)
	g.AddEdge(4, 3, 6)
	
	// Bidirectional search from 0 to 4
	sp := BidirectionalDijkstra(g, 0, 4)
	
	expectedDist := 7.0
	if math.Abs(sp.Distance[4]-expectedDist) > 0.001 {
		t.Errorf("Expected distance %.1f, got %.1f", expectedDist, sp.Distance[4])
	}
	
	// Test same source and target
	sp2 := BidirectionalDijkstra(g, 2, 2)
	if sp2.Distance[2] != 0 {
		t.Errorf("Distance to self should be 0, got %.1f", sp2.Distance[2])
	}
}

func TestLargeGraph(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large graph test in short mode")
	}
	
	// Generate random graph
	vertices := 100
	g := NewGraph(vertices)
	
	// Create a connected graph
	for i := 1; i < vertices; i++ {
		// Connect to previous vertex
		weight := rand.Float64() * 10
		g.AddEdge(i-1, i, weight)
		g.AddEdge(i, i-1, weight)
		
		// Add some random edges
		for j := 0; j < 3; j++ {
			target := rand.Intn(vertices)
			if target != i {
				weight := rand.Float64() * 10
				g.AddEdge(i, target, weight)
			}
		}
	}
	
	// Compare algorithms
	source := 0
	
	start := time.Now()
	sp1 := SequentialDijkstra(g, source)
	seqTime := time.Since(start)
	
	start = time.Now()
	sp2 := ParallelDijkstra(g, source)
	parTime := time.Since(start)
	
	start = time.Now()
	sp3 := DeltaSteppingDijkstra(g, source, 2.5)
	deltaTime := time.Since(start)
	
	t.Logf("Sequential time: %v", seqTime)
	t.Logf("Parallel time: %v", parTime)
	t.Logf("Delta-stepping time: %v", deltaTime)
	
	// Verify results match
	for i := 0; i < vertices; i++ {
		if math.Abs(sp1.Distance[i]-sp2.Distance[i]) > 0.001 {
			t.Errorf("Sequential and Parallel distances differ for vertex %d: %.3f vs %.3f",
				i, sp1.Distance[i], sp2.Distance[i])
		}
		
		if math.Abs(sp1.Distance[i]-sp3.Distance[i]) > 0.001 {
			t.Errorf("Sequential and Delta distances differ for vertex %d: %.3f vs %.3f",
				i, sp1.Distance[i], sp3.Distance[i])
		}
	}
}

func TestDifferentDeltaValues(t *testing.T) {
	g := NewGraph(10)
	
	// Add edges with various weights
	for i := 0; i < 9; i++ {
		g.AddEdge(i, i+1, float64(i+1))
	}
	for i := 0; i < 8; i++ {
		g.AddEdge(i, i+2, float64(2*(i+1)))
	}
	
	source := 0
	sp1 := SequentialDijkstra(g, source)
	
	// Test different delta values
	deltas := []float64{0.5, 1.0, 2.0, 5.0, 10.0}
	
	for _, delta := range deltas {
		sp := DeltaSteppingDijkstra(g, source, delta)
		
		for i := 0; i < 10; i++ {
			if math.Abs(sp.Distance[i]-sp1.Distance[i]) > 0.001 {
				t.Errorf("Delta %.1f: Distance to vertex %d differs: %.3f vs %.3f",
					delta, i, sp.Distance[i], sp1.Distance[i])
			}
		}
	}
}

func TestConcurrentAccess(t *testing.T) {
	g := NewGraph(50)
	
	// Create a dense graph
	for i := 0; i < 50; i++ {
		for j := 0; j < 50; j++ {
			if i != j && rand.Float64() < 0.3 {
				g.AddEdge(i, j, rand.Float64()*10)
			}
		}
	}
	
	// Run multiple concurrent shortest path computations
	results := make(chan []float64, 10)
	
	for i := 0; i < 10; i++ {
		go func(source int) {
			sp := ParallelDijkstra(g, source)
			results <- sp.Distance
		}(i)
	}
	
	// Collect and verify results
	for i := 0; i < 10; i++ {
		dist := <-results
		if len(dist) != 50 {
			t.Errorf("Expected 50 distances, got %d", len(dist))
		}
	}
}

func BenchmarkDijkstraAlgorithms(b *testing.B) {
	sizes := []int{10, 50, 100, 500}
	
	for _, size := range sizes {
		// Create graph
		g := NewGraph(size)
		for i := 0; i < size; i++ {
			// Connect to next vertices
			for j := 1; j <= 5 && i+j < size; j++ {
				weight := rand.Float64() * 10
				g.AddEdge(i, i+j, weight)
				g.AddEdge(i+j, i, weight)
			}
		}
		
		b.Run(fmt.Sprintf("Sequential_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SequentialDijkstra(g, 0)
			}
		})
		
		b.Run(fmt.Sprintf("Parallel_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ParallelDijkstra(g, 0)
			}
		})
		
		b.Run(fmt.Sprintf("DeltaStepping_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				DeltaSteppingDijkstra(g, 0, 2.5)
			}
		})
		
		if size <= 100 {
			b.Run(fmt.Sprintf("Bidirectional_%d", size), func(b *testing.B) {
				target := size - 1
				for i := 0; i < b.N; i++ {
					BidirectionalDijkstra(g, 0, target)
				}
			})
		}
	}
}

func BenchmarkDenseVsSparse(b *testing.B) {
	size := 100
	
	// Sparse graph
	sparse := NewGraph(size)
	for i := 0; i < size-1; i++ {
		sparse.AddEdge(i, i+1, rand.Float64()*10)
		sparse.AddEdge(i+1, i, rand.Float64()*10)
	}
	
	// Dense graph
	dense := NewGraph(size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if i != j {
				dense.AddEdge(i, j, rand.Float64()*10)
			}
		}
	}
	
	b.Run("Sparse_Sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			SequentialDijkstra(sparse, 0)
		}
	})
	
	b.Run("Sparse_Parallel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ParallelDijkstra(sparse, 0)
		}
	})
	
	b.Run("Dense_Sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			SequentialDijkstra(dense, 0)
		}
	})
	
	b.Run("Dense_Parallel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ParallelDijkstra(dense, 0)
		}
	})
}

func TestUndirectedGraph(t *testing.T) {
	g := NewGraph(4)
	g.AddUndirectedEdge(0, 1, 1)
	g.AddUndirectedEdge(1, 2, 2)
	g.AddUndirectedEdge(2, 3, 3)
	
	sp := SequentialDijkstra(g, 3)
	
	// Should be able to reach all vertices from vertex 3
	expected := []float64{6, 5, 3, 0}
	
	for i, exp := range expected {
		if math.Abs(sp.Distance[i]-exp) > 0.001 {
			t.Errorf("Vertex %d: expected distance %.1f, got %.1f", 
				i, exp, sp.Distance[i])
		}
	}
}