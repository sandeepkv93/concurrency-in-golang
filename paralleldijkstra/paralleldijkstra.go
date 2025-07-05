package paralleldijkstra

import (
	"container/heap"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// Edge represents a directed weighted edge
type Edge struct {
	To     int
	Weight float64
}

// Graph represents a directed weighted graph
type Graph struct {
	vertices  int
	adjacency [][]Edge
}

// NewGraph creates a new graph with n vertices
func NewGraph(n int) *Graph {
	return &Graph{
		vertices:  n,
		adjacency: make([][]Edge, n),
	}
}

// AddEdge adds a directed edge to the graph
func (g *Graph) AddEdge(from, to int, weight float64) {
	g.adjacency[from] = append(g.adjacency[from], Edge{To: to, Weight: weight})
}

// AddUndirectedEdge adds an undirected edge to the graph
func (g *Graph) AddUndirectedEdge(u, v int, weight float64) {
	g.AddEdge(u, v, weight)
	g.AddEdge(v, u, weight)
}

// ShortestPaths represents the result of shortest path computation
type ShortestPaths struct {
	Source   int
	Distance []float64
	Previous []int
	Paths    map[int][]int
}

// GetPath returns the shortest path from source to target
func (sp *ShortestPaths) GetPath(target int) []int {
	if sp.Previous[target] == -1 && target != sp.Source {
		return nil // No path exists
	}
	
	path := []int{}
	current := target
	
	for current != -1 {
		path = append([]int{current}, path...)
		current = sp.Previous[current]
	}
	
	return path
}

// Node represents a node in the priority queue
type Node struct {
	vertex   int
	distance float64
	index    int
}

// PriorityQueue implements heap.Interface
type PriorityQueue []*Node

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].distance < pq[j].distance
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	node := x.(*Node)
	node.index = n
	*pq = append(*pq, node)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	node := old[n-1]
	node.index = -1
	*pq = old[0 : n-1]
	return node
}

// SequentialDijkstra implements the classic Dijkstra's algorithm
func SequentialDijkstra(g *Graph, source int) *ShortestPaths {
	dist := make([]float64, g.vertices)
	prev := make([]int, g.vertices)
	visited := make([]bool, g.vertices)
	
	// Initialize distances
	for i := 0; i < g.vertices; i++ {
		dist[i] = math.Inf(1)
		prev[i] = -1
	}
	dist[source] = 0
	
	// Priority queue
	pq := &PriorityQueue{}
	heap.Init(pq)
	heap.Push(pq, &Node{vertex: source, distance: 0})
	
	for pq.Len() > 0 {
		node := heap.Pop(pq).(*Node)
		u := node.vertex
		
		if visited[u] {
			continue
		}
		visited[u] = true
		
		// Relax edges
		for _, edge := range g.adjacency[u] {
			v := edge.To
			alt := dist[u] + edge.Weight
			
			if alt < dist[v] {
				dist[v] = alt
				prev[v] = u
				heap.Push(pq, &Node{vertex: v, distance: alt})
			}
		}
	}
	
	return &ShortestPaths{
		Source:   source,
		Distance: dist,
		Previous: prev,
	}
}

// ParallelDijkstra implements parallel Dijkstra using concurrent edge relaxation
func ParallelDijkstra(g *Graph, source int) *ShortestPaths {
	dist := make([]atomic.Value, g.vertices)
	prev := make([]atomic.Int32, g.vertices)
	inQueue := make([]atomic.Bool, g.vertices)
	
	// Initialize
	for i := 0; i < g.vertices; i++ {
		dist[i].Store(math.Inf(1))
		prev[i].Store(int32(-1))
	}
	dist[source].Store(0.0)
	
	// Work queue for vertices to process
	workQueue := make(chan int, g.vertices)
	var wg sync.WaitGroup
	
	// Number of workers
	numWorkers := 4
	
	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			for u := range workQueue {
				currentDist := dist[u].Load().(float64)
				
				// Process all edges from u
				for _, edge := range g.adjacency[u] {
					v := edge.To
					newDist := currentDist + edge.Weight
					
					// Try to update distance
					for {
						oldDist := dist[v].Load().(float64)
						if newDist >= oldDist {
							break
						}
						
						if dist[v].CompareAndSwap(oldDist, newDist) {
							prev[v].Store(int32(u))
							
							// Add to queue if not already there
							if !inQueue[v].Swap(true) {
								workQueue <- v
							}
							break
						}
					}
				}
				
				// Mark as not in queue
				inQueue[u].Store(false)
			}
		}()
	}
	
	// Start with source
	workQueue <- source
	inQueue[source].Store(true)
	
	// Wait for completion (when queue is empty and all workers are done)
	done := make(chan bool)
	go func() {
		for {
			time.Sleep(10 * time.Millisecond)
			
			// Check if any vertex is still in queue
			anyInQueue := false
			for i := 0; i < g.vertices; i++ {
				if inQueue[i].Load() {
					anyInQueue = true
					break
				}
			}
			
			if !anyInQueue && len(workQueue) == 0 {
				close(workQueue)
				done <- true
				return
			}
		}
	}()
	
	<-done
	wg.Wait()
	
	// Convert results
	finalDist := make([]float64, g.vertices)
	finalPrev := make([]int, g.vertices)
	
	for i := 0; i < g.vertices; i++ {
		finalDist[i] = dist[i].Load().(float64)
		finalPrev[i] = int(prev[i].Load())
	}
	
	return &ShortestPaths{
		Source:   source,
		Distance: finalDist,
		Previous: finalPrev,
	}
}

// DeltaSteppingDijkstra implements the delta-stepping parallel algorithm
func DeltaSteppingDijkstra(g *Graph, source int, delta float64) *ShortestPaths {
	if delta <= 0 {
		delta = 1.0
	}
	
	dist := make([]atomic.Value, g.vertices)
	prev := make([]atomic.Int32, g.vertices)
	
	// Initialize
	for i := 0; i < g.vertices; i++ {
		dist[i].Store(math.Inf(1))
		prev[i].Store(int32(-1))
	}
	dist[source].Store(0.0)
	
	// Buckets for vertices
	bucketMutex := sync.Mutex{}
	buckets := make(map[int]map[int]bool)
	
	// Add vertex to bucket
	addToBucket := func(v int, d float64) {
		bucket := int(d / delta)
		bucketMutex.Lock()
		if buckets[bucket] == nil {
			buckets[bucket] = make(map[int]bool)
		}
		buckets[bucket][v] = true
		bucketMutex.Unlock()
	}
	
	// Remove vertex from bucket
	removeFromBucket := func(v int, bucket int) {
		bucketMutex.Lock()
		delete(buckets[bucket], v)
		if len(buckets[bucket]) == 0 {
			delete(buckets, bucket)
		}
		bucketMutex.Unlock()
	}
	
	// Get minimum bucket
	getMinBucket := func() (int, bool) {
		bucketMutex.Lock()
		defer bucketMutex.Unlock()
		
		minBucket := -1
		for b := range buckets {
			if minBucket == -1 || b < minBucket {
				minBucket = b
			}
		}
		
		return minBucket, minBucket != -1
	}
	
	// Process edges
	relaxEdge := func(u int, edge Edge) {
		v := edge.To
		currentDistU := dist[u].Load().(float64)
		newDist := currentDistU + edge.Weight
		
		for {
			oldDist := dist[v].Load().(float64)
			if newDist >= oldDist {
				break
			}
			
			if dist[v].CompareAndSwap(oldDist, newDist) {
				prev[v].Store(int32(u))
				addToBucket(v, newDist)
				break
			}
		}
	}
	
	// Add source to bucket
	addToBucket(source, 0)
	
	// Process buckets
	for {
		bucket, exists := getMinBucket()
		if !exists {
			break
		}
		
		// Get all vertices in current bucket
		bucketMutex.Lock()
		vertices := make([]int, 0, len(buckets[bucket]))
		for v := range buckets[bucket] {
			vertices = append(vertices, v)
		}
		bucketMutex.Unlock()
		
		// Process vertices in parallel
		var wg sync.WaitGroup
		for _, u := range vertices {
			removeFromBucket(u, bucket)
			
			wg.Add(1)
			go func(vertex int) {
				defer wg.Done()
				
				// Relax light edges (weight <= delta)
				for _, edge := range g.adjacency[vertex] {
					if edge.Weight <= delta {
						relaxEdge(vertex, edge)
					}
				}
			}(u)
		}
		wg.Wait()
		
		// Relax heavy edges (weight > delta)
		for _, u := range vertices {
			wg.Add(1)
			go func(vertex int) {
				defer wg.Done()
				
				for _, edge := range g.adjacency[vertex] {
					if edge.Weight > delta {
						relaxEdge(vertex, edge)
					}
				}
			}(u)
		}
		wg.Wait()
	}
	
	// Convert results
	finalDist := make([]float64, g.vertices)
	finalPrev := make([]int, g.vertices)
	
	for i := 0; i < g.vertices; i++ {
		finalDist[i] = dist[i].Load().(float64)
		finalPrev[i] = int(prev[i].Load())
	}
	
	return &ShortestPaths{
		Source:   source,
		Distance: finalDist,
		Previous: finalPrev,
	}
}

// BidirectionalDijkstra implements bidirectional search
func BidirectionalDijkstra(g *Graph, source, target int) *ShortestPaths {
	if source == target {
		result := &ShortestPaths{
			Source:   source,
			Distance: make([]float64, g.vertices),
			Previous: make([]int, g.vertices),
		}
		for i := 0; i < g.vertices; i++ {
			result.Distance[i] = math.Inf(1)
			result.Previous[i] = -1
		}
		result.Distance[source] = 0
		return result
	}
	
	// Forward and backward distances
	distF := make([]float64, g.vertices)
	distB := make([]float64, g.vertices)
	prevF := make([]int, g.vertices)
	prevB := make([]int, g.vertices)
	visitedF := make([]bool, g.vertices)
	visitedB := make([]bool, g.vertices)
	
	// Initialize
	for i := 0; i < g.vertices; i++ {
		distF[i] = math.Inf(1)
		distB[i] = math.Inf(1)
		prevF[i] = -1
		prevB[i] = -1
	}
	distF[source] = 0
	distB[target] = 0
	
	// Priority queues
	pqF := &PriorityQueue{}
	pqB := &PriorityQueue{}
	heap.Init(pqF)
	heap.Init(pqB)
	
	heap.Push(pqF, &Node{vertex: source, distance: 0})
	heap.Push(pqB, &Node{vertex: target, distance: 0})
	
	bestPath := math.Inf(1)
	meetingPoint := -1
	
	// Alternate between forward and backward search
	for pqF.Len() > 0 || pqB.Len() > 0 {
		// Forward step
		if pqF.Len() > 0 {
			node := heap.Pop(pqF).(*Node)
			u := node.vertex
			
			if !visitedF[u] {
				visitedF[u] = true
				
				// Check if we can improve the path
				if visitedB[u] && distF[u]+distB[u] < bestPath {
					bestPath = distF[u] + distB[u]
					meetingPoint = u
				}
				
				// Relax edges
				for _, edge := range g.adjacency[u] {
					v := edge.To
					alt := distF[u] + edge.Weight
					
					if alt < distF[v] {
						distF[v] = alt
						prevF[v] = u
						heap.Push(pqF, &Node{vertex: v, distance: alt})
					}
				}
			}
		}
		
		// Backward step
		if pqB.Len() > 0 {
			node := heap.Pop(pqB).(*Node)
			u := node.vertex
			
			if !visitedB[u] {
				visitedB[u] = true
				
				// Check if we can improve the path
				if visitedF[u] && distF[u]+distB[u] < bestPath {
					bestPath = distF[u] + distB[u]
					meetingPoint = u
				}
				
				// Relax edges (in reverse direction)
				for _, edge := range g.adjacency[u] {
					v := edge.To
					alt := distB[u] + edge.Weight
					
					if alt < distB[v] {
						distB[v] = alt
						prevB[v] = u
					}
				}
			}
		}
		
		// Early termination
		minF := math.Inf(1)
		if pqF.Len() > 0 {
			minF = (*pqF)[0].distance
		}
		minB := math.Inf(1)
		if pqB.Len() > 0 {
			minB = (*pqB)[0].distance
		}
		
		if minF+minB >= bestPath {
			break
		}
	}
	
	// Reconstruct path
	result := &ShortestPaths{
		Source:   source,
		Distance: distF,
		Previous: prevF,
	}
	
	// Update distances for target
	if meetingPoint != -1 {
		result.Distance[target] = bestPath
	}
	
	return result
}

// Example demonstrates Dijkstra's algorithm implementations
func Example() {
	fmt.Println("=== Parallel Dijkstra's Algorithm Example ===")
	
	// Create a sample graph
	g := NewGraph(6)
	g.AddEdge(0, 1, 4)
	g.AddEdge(0, 2, 2)
	g.AddEdge(1, 2, 1)
	g.AddEdge(1, 3, 5)
	g.AddEdge(2, 3, 8)
	g.AddEdge(2, 4, 10)
	g.AddEdge(3, 4, 2)
	g.AddEdge(3, 5, 6)
	g.AddEdge(4, 5, 3)
	
	fmt.Println("Graph edges:")
	for u := 0; u < 6; u++ {
		for _, edge := range g.adjacency[u] {
			fmt.Printf("  %d -> %d : %.0f\n", u, edge.To, edge.Weight)
		}
	}
	
	// Sequential Dijkstra
	fmt.Println("\nSequential Dijkstra from vertex 0:")
	sp1 := SequentialDijkstra(g, 0)
	printShortestPaths(sp1)
	
	// Parallel Dijkstra
	fmt.Println("\nParallel Dijkstra from vertex 0:")
	sp2 := ParallelDijkstra(g, 0)
	printShortestPaths(sp2)
	
	// Delta-stepping
	fmt.Println("\nDelta-stepping Dijkstra from vertex 0 (delta=3):")
	sp3 := DeltaSteppingDijkstra(g, 0, 3)
	printShortestPaths(sp3)
	
	// Bidirectional search
	fmt.Println("\nBidirectional Dijkstra from 0 to 5:")
	sp4 := BidirectionalDijkstra(g, 0, 5)
	fmt.Printf("Distance: %.0f\n", sp4.Distance[5])
	fmt.Printf("Path: %v\n", sp4.GetPath(5))
}

func printShortestPaths(sp *ShortestPaths) {
	for v := 0; v < len(sp.Distance); v++ {
		if math.IsInf(sp.Distance[v], 1) {
			fmt.Printf("  Vertex %d: unreachable\n", v)
		} else {
			fmt.Printf("  Vertex %d: distance=%.0f, path=%v\n", 
				v, sp.Distance[v], sp.GetPath(v))
		}
	}
}