package parallelprim

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

func TestBasicMST(t *testing.T) {
	// Create a simple graph
	g := NewGraph(4)
	g.AddEdge(0, 1, 10)
	g.AddEdge(0, 2, 6)
	g.AddEdge(0, 3, 5)
	g.AddEdge(1, 3, 15)
	g.AddEdge(2, 3, 4)
	
	// Expected MST has edges: 0-3(5), 0-2(6), 2-3(4) with total cost 12
	// But since 2-3(4) is cheaper, actual MST is: 0-3(5), 2-3(4), 0-2(6)
	// Wait, that would create a cycle. Correct MST: 0-3(5), 2-3(4), 0-1(10) or similar
	// Actually, the MST should be: 2-3(4), 0-3(5), 0-2(6) with total 15
	// Let me recalculate: We need 3 edges for 4 vertices
	// Start from 0: take 0-3(5), then 2-3(4), then 0-2(6) = 15
	// Or: 2-3(4), 0-3(5), 0-1(10) = 19
	// Or: 0-2(6), 2-3(4), 0-3(5) would create a cycle
	// Correct MST: 2-3(4), 0-3(5), 0-2(6) = 15
	
	expectedCost := 15.0
	
	// Test sequential
	mst1 := SequentialPrim(g, 0)
	if math.Abs(mst1.TotalCost-expectedCost) > 0.001 {
		t.Errorf("Sequential: Expected cost %.1f, got %.1f", expectedCost, mst1.TotalCost)
	}
	if len(mst1.Edges) != 3 {
		t.Errorf("Sequential: Expected 3 edges, got %d", len(mst1.Edges))
	}
	
	// Test parallel
	mst2 := ParallelPrim(g, 0)
	if math.Abs(mst2.TotalCost-expectedCost) > 0.001 {
		t.Errorf("Parallel: Expected cost %.1f, got %.1f", expectedCost, mst2.TotalCost)
	}
	if len(mst2.Edges) != 3 {
		t.Errorf("Parallel: Expected 3 edges, got %d", len(mst2.Edges))
	}
	
	// Test distributed
	mst3 := DistributedPrim(g, 0, 2)
	if math.Abs(mst3.TotalCost-expectedCost) > 0.001 {
		t.Errorf("Distributed: Expected cost %.1f, got %.1f", expectedCost, mst3.TotalCost)
	}
	if len(mst3.Edges) != 3 {
		t.Errorf("Distributed: Expected 3 edges, got %d", len(mst3.Edges))
	}
}

func TestEmptyGraph(t *testing.T) {
	g := NewGraph(0)
	
	mst := SequentialPrim(g, 0)
	if len(mst.Edges) != 0 {
		t.Error("Empty graph should have no MST edges")
	}
	if mst.TotalCost != 0 {
		t.Error("Empty graph should have zero cost")
	}
}

func TestSingleVertex(t *testing.T) {
	g := NewGraph(1)
	
	mst := SequentialPrim(g, 0)
	if len(mst.Edges) != 0 {
		t.Error("Single vertex graph should have no MST edges")
	}
	if mst.TotalCost != 0 {
		t.Error("Single vertex graph should have zero cost")
	}
}

func TestDisconnectedGraph(t *testing.T) {
	g := NewGraph(4)
	g.AddEdge(0, 1, 1)
	g.AddEdge(2, 3, 1)
	// No edge between {0,1} and {2,3}
	
	mst := SequentialPrim(g, 0)
	// Should only include vertices reachable from start vertex
	if len(mst.Edges) != 1 {
		t.Errorf("Expected 1 edge in MST, got %d", len(mst.Edges))
	}
}

func TestCompleteGraph(t *testing.T) {
	// Complete graph K5
	g := NewGraph(5)
	weights := [][]float64{
		{0, 2, 3, 1, 4},
		{2, 0, 5, 2, 3},
		{3, 5, 0, 3, 6},
		{1, 2, 3, 0, 1},
		{4, 3, 6, 1, 0},
	}
	
	for i := 0; i < 5; i++ {
		for j := i + 1; j < 5; j++ {
			g.AddEdge(i, j, weights[i][j])
		}
	}
	
	// All algorithms should produce MST with same cost
	mst1 := SequentialPrim(g, 0)
	mst2 := ParallelPrim(g, 0)
	mst3 := DistributedPrim(g, 0, 3)
	
	if len(mst1.Edges) != 4 {
		t.Errorf("Expected 4 edges in MST, got %d", len(mst1.Edges))
	}
	
	// All should have same cost
	if math.Abs(mst1.TotalCost-mst2.TotalCost) > 0.001 {
		t.Errorf("Sequential and Parallel costs differ: %.1f vs %.1f", 
			mst1.TotalCost, mst2.TotalCost)
	}
	
	if math.Abs(mst1.TotalCost-mst3.TotalCost) > 0.001 {
		t.Errorf("Sequential and Distributed costs differ: %.1f vs %.1f", 
			mst1.TotalCost, mst3.TotalCost)
	}
}

func TestLargeRandomGraph(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large graph test in short mode")
	}
	
	// Generate random graph
	vertices := 100
	g := GenerateRandomGraph(vertices, 0.1)
	
	// Compare all algorithms
	start := time.Now()
	mst1 := SequentialPrim(g, 0)
	seqTime := time.Since(start)
	
	start = time.Now()
	mst2 := ParallelPrim(g, 0)
	parTime := time.Since(start)
	
	start = time.Now()
	mst3 := DistributedPrim(g, 0, 4)
	distTime := time.Since(start)
	
	t.Logf("Sequential time: %v", seqTime)
	t.Logf("Parallel time: %v", parTime)
	t.Logf("Distributed time: %v", distTime)
	
	// All should produce MST with same number of edges
	expectedEdges := vertices - 1
	if len(mst1.Edges) != expectedEdges {
		t.Errorf("Sequential: Expected %d edges, got %d", expectedEdges, len(mst1.Edges))
	}
	if len(mst2.Edges) != expectedEdges {
		t.Errorf("Parallel: Expected %d edges, got %d", expectedEdges, len(mst2.Edges))
	}
	if len(mst3.Edges) != expectedEdges {
		t.Errorf("Distributed: Expected %d edges, got %d", expectedEdges, len(mst3.Edges))
	}
	
	// Costs should be very close (might have slight differences due to tie-breaking)
	if math.Abs(mst1.TotalCost-mst2.TotalCost) > 0.001 {
		t.Errorf("Sequential and Parallel costs differ significantly: %.3f vs %.3f", 
			mst1.TotalCost, mst2.TotalCost)
	}
}

func TestConcurrency(t *testing.T) {
	// Test that parallel algorithms handle concurrent access correctly
	g := GenerateRandomGraph(50, 0.2)
	
	// Run multiple parallel MST computations concurrently
	results := make(chan *MST, 10)
	
	for i := 0; i < 10; i++ {
		go func() {
			mst := ParallelPrim(g, 0)
			results <- mst
		}()
	}
	
	// Collect results
	var firstCost float64
	for i := 0; i < 10; i++ {
		mst := <-results
		if i == 0 {
			firstCost = mst.TotalCost
		} else if math.Abs(mst.TotalCost-firstCost) > 0.001 {
			t.Errorf("Concurrent execution produced different costs: %.3f vs %.3f", 
				firstCost, mst.TotalCost)
		}
	}
}

func TestDifferentStartVertices(t *testing.T) {
	g := GenerateRandomGraph(20, 0.3)
	
	// MST should have same total cost regardless of start vertex
	costs := make([]float64, 5)
	for i := 0; i < 5; i++ {
		mst := SequentialPrim(g, i)
		costs[i] = mst.TotalCost
	}
	
	// All costs should be the same
	for i := 1; i < 5; i++ {
		if math.Abs(costs[0]-costs[i]) > 0.001 {
			t.Errorf("Different start vertices produced different costs: %.3f vs %.3f", 
				costs[0], costs[i])
		}
	}
}

func BenchmarkPrimAlgorithms(b *testing.B) {
	sizes := []int{10, 50, 100, 500}
	
	for _, size := range sizes {
		g := GenerateRandomGraph(size, 0.1)
		
		b.Run(fmt.Sprintf("Sequential_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SequentialPrim(g, 0)
			}
		})
		
		b.Run(fmt.Sprintf("Parallel_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ParallelPrim(g, 0)
			}
		})
		
		b.Run(fmt.Sprintf("Distributed_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				DistributedPrim(g, 0, 4)
			}
		})
	}
}

func BenchmarkLargeGraphs(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping large graph benchmark in short mode")
	}
	
	g := GenerateRandomGraph(1000, 0.05)
	
	b.Run("Sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			SequentialPrim(g, 0)
		}
	})
	
	b.Run("Parallel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ParallelPrim(g, 0)
		}
	})
	
	b.Run("Distributed_4", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			DistributedPrim(g, 0, 4)
		}
	})
	
	b.Run("Distributed_8", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			DistributedPrim(g, 0, 8)
		}
	})
}

func TestParentMap(t *testing.T) {
	g := NewGraph(5)
	g.AddEdge(0, 1, 1)
	g.AddEdge(0, 2, 2)
	g.AddEdge(1, 3, 3)
	g.AddEdge(2, 4, 4)
	
	mst := SequentialPrim(g, 0)
	
	// Check parent map
	if len(mst.ParentMap) != 4 {
		t.Errorf("Expected 4 entries in parent map, got %d", len(mst.ParentMap))
	}
	
	// Vertex 0 should not have a parent
	if _, exists := mst.ParentMap[0]; exists {
		t.Error("Root vertex should not have a parent")
	}
	
	// All other vertices should have parents
	for i := 1; i < 5; i++ {
		if _, exists := mst.ParentMap[i]; !exists {
			t.Errorf("Vertex %d should have a parent", i)
		}
	}
}