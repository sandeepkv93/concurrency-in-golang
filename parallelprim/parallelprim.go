package parallelprim

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
)

// Edge represents a weighted edge in the graph
type Edge struct {
	From   int
	To     int
	Weight float64
}

// Graph represents an undirected weighted graph
type Graph struct {
	vertices  int
	adjacency map[int][]Edge
}

// NewGraph creates a new graph with n vertices
func NewGraph(n int) *Graph {
	g := &Graph{
		vertices:  n,
		adjacency: make(map[int][]Edge),
	}
	for i := 0; i < n; i++ {
		g.adjacency[i] = make([]Edge, 0)
	}
	return g
}

// AddEdge adds an undirected edge to the graph
func (g *Graph) AddEdge(from, to int, weight float64) {
	g.adjacency[from] = append(g.adjacency[from], Edge{From: from, To: to, Weight: weight})
	g.adjacency[to] = append(g.adjacency[to], Edge{From: to, To: from, Weight: weight})
}

// MST represents a minimum spanning tree
type MST struct {
	Edges      []Edge
	TotalCost  float64
	ParentMap  map[int]int
}

// PriorityQueue implements heap.Interface for edges
type PriorityQueue []Edge

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].Weight < pq[j].Weight
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(Edge))
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// SequentialPrim implements sequential Prim's algorithm
func SequentialPrim(g *Graph, start int) *MST {
	if g.vertices == 0 {
		return &MST{Edges: []Edge{}, ParentMap: make(map[int]int)}
	}

	inMST := make([]bool, g.vertices)
	key := make([]float64, g.vertices)
	parent := make([]int, g.vertices)
	
	for i := 0; i < g.vertices; i++ {
		key[i] = math.Inf(1)
		parent[i] = -1
	}
	
	pq := &PriorityQueue{}
	heap.Init(pq)
	
	key[start] = 0
	heap.Push(pq, Edge{From: -1, To: start, Weight: 0})
	
	mstEdges := make([]Edge, 0)
	totalCost := 0.0
	
	for pq.Len() > 0 {
		edge := heap.Pop(pq).(Edge)
		u := edge.To
		
		if inMST[u] {
			continue
		}
		
		inMST[u] = true
		
		if edge.From != -1 {
			mstEdges = append(mstEdges, edge)
			totalCost += edge.Weight
		}
		
		// Update adjacent vertices
		for _, e := range g.adjacency[u] {
			v := e.To
			if !inMST[v] && e.Weight < key[v] {
				key[v] = e.Weight
				parent[v] = u
				heap.Push(pq, Edge{From: u, To: v, Weight: e.Weight})
			}
		}
	}
	
	parentMap := make(map[int]int)
	for i, p := range parent {
		if p != -1 {
			parentMap[i] = p
		}
	}
	
	return &MST{
		Edges:     mstEdges,
		TotalCost: totalCost,
		ParentMap: parentMap,
	}
}

// ParallelPrim implements parallel Prim's algorithm using concurrent edge processing
func ParallelPrim(g *Graph, start int) *MST {
	if g.vertices == 0 {
		return &MST{Edges: []Edge{}, ParentMap: make(map[int]int)}
	}

	inMST := make([]atomic.Bool, g.vertices)
	key := make([]atomic.Value, g.vertices)
	parent := make([]atomic.Int32, g.vertices)
	
	// Initialize
	for i := 0; i < g.vertices; i++ {
		key[i].Store(math.Inf(1))
		parent[i].Store(int32(-1))
	}
	
	key[start].Store(0.0)
	
	mstEdges := make([]Edge, 0)
	totalCost := 0.0
	edgeMutex := sync.Mutex{}
	
	// Process vertices in parallel
	numWorkers := 4
	vertexChan := make(chan int, g.vertices)
	var wg sync.WaitGroup
	
	// Worker function
	worker := func() {
		defer wg.Done()
		localPQ := &PriorityQueue{}
		heap.Init(localPQ)
		
		for u := range vertexChan {
			if inMST[u].Load() {
				continue
			}
			
			// Find minimum edge to this vertex
			minWeight := key[u].Load().(float64)
			parentVertex := int(parent[u].Load())
			
			if !math.IsInf(minWeight, 1) && !inMST[u].Swap(true) {
				// Add to MST
				if parentVertex != -1 {
					edge := Edge{From: parentVertex, To: u, Weight: minWeight}
					edgeMutex.Lock()
					mstEdges = append(mstEdges, edge)
					totalCost += minWeight
					edgeMutex.Unlock()
				}
				
				// Update adjacent vertices
				for _, e := range g.adjacency[u] {
					v := e.To
					if !inMST[v].Load() {
						currentKey := key[v].Load().(float64)
						if e.Weight < currentKey {
							// Try to update key value
							for {
								old := key[v].Load().(float64)
								if e.Weight >= old || key[v].CompareAndSwap(old, e.Weight) {
									if e.Weight < old {
										parent[v].Store(int32(u))
									}
									break
								}
							}
						}
					}
				}
			}
		}
	}
	
	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker()
	}
	
	// Feed vertices to workers
	processedCount := 0
	for processedCount < g.vertices {
		// Find next vertex to process
		minKey := math.Inf(1)
		nextVertex := -1
		
		for i := 0; i < g.vertices; i++ {
			if !inMST[i].Load() {
				currentKey := key[i].Load().(float64)
				if currentKey < minKey {
					minKey = currentKey
					nextVertex = i
				}
			}
		}
		
		if nextVertex == -1 {
			break
		}
		
		vertexChan <- nextVertex
		processedCount++
	}
	
	close(vertexChan)
	wg.Wait()
	
	// Build parent map
	parentMap := make(map[int]int)
	for i := 0; i < g.vertices; i++ {
		p := int(parent[i].Load())
		if p != -1 {
			parentMap[i] = p
		}
	}
	
	return &MST{
		Edges:     mstEdges,
		TotalCost: totalCost,
		ParentMap: parentMap,
	}
}

// DistributedPrim implements a distributed version using graph partitioning
func DistributedPrim(g *Graph, start int, numPartitions int) *MST {
	if g.vertices == 0 || numPartitions <= 0 {
		return &MST{Edges: []Edge{}, ParentMap: make(map[int]int)}
	}
	
	if numPartitions == 1 {
		return SequentialPrim(g, start)
	}
	
	// Partition vertices
	partitions := make([][]int, numPartitions)
	for i := 0; i < g.vertices; i++ {
		pid := i % numPartitions
		partitions[pid] = append(partitions[pid], i)
	}
	
	// Shared data structures
	globalInMST := make([]atomic.Bool, g.vertices)
	globalKey := make([]atomic.Value, g.vertices)
	globalParent := make([]atomic.Int32, g.vertices)
	
	// Initialize
	for i := 0; i < g.vertices; i++ {
		globalKey[i].Store(math.Inf(1))
		globalParent[i].Store(int32(-1))
	}
	globalKey[start].Store(0.0)
	
	// Results collection
	mstEdges := make([]Edge, 0)
	totalCost := 0.0
	resultMutex := sync.Mutex{}
	
	// Process partitions in parallel
	var wg sync.WaitGroup
	for pid := 0; pid < numPartitions; pid++ {
		wg.Add(1)
		go func(partitionID int, vertices []int) {
			defer wg.Done()
			
			// Local priority queue for this partition
			localPQ := &PriorityQueue{}
			heap.Init(localPQ)
			
			// Process vertices in this partition
			for {
				// Find minimum key vertex in partition that's not in MST
				minKey := math.Inf(1)
				minVertex := -1
				
				for _, v := range vertices {
					if !globalInMST[v].Load() {
						key := globalKey[v].Load().(float64)
						if key < minKey {
							minKey = key
							minVertex = v
						}
					}
				}
				
				if minVertex == -1 || math.IsInf(minKey, 1) {
					break
				}
				
				// Try to claim this vertex
				if !globalInMST[minVertex].Swap(true) {
					parentVertex := int(globalParent[minVertex].Load())
					
					// Add edge to MST
					if parentVertex != -1 {
						edge := Edge{From: parentVertex, To: minVertex, Weight: minKey}
						resultMutex.Lock()
						mstEdges = append(mstEdges, edge)
						totalCost += minKey
						resultMutex.Unlock()
					}
					
					// Update neighbors
					for _, e := range g.adjacency[minVertex] {
						neighbor := e.To
						if !globalInMST[neighbor].Load() {
							for {
								oldKey := globalKey[neighbor].Load().(float64)
								if e.Weight >= oldKey {
									break
								}
								if globalKey[neighbor].CompareAndSwap(oldKey, e.Weight) {
									globalParent[neighbor].Store(int32(minVertex))
									break
								}
							}
						}
					}
				}
			}
		}(pid, partitions[pid])
	}
	
	wg.Wait()
	
	// Build parent map
	parentMap := make(map[int]int)
	for i := 0; i < g.vertices; i++ {
		p := int(globalParent[i].Load())
		if p != -1 {
			parentMap[i] = p
		}
	}
	
	return &MST{
		Edges:     mstEdges,
		TotalCost: totalCost,
		ParentMap: parentMap,
	}
}

// GenerateRandomGraph generates a random connected graph
func GenerateRandomGraph(vertices int, edgeProbability float64) *Graph {
	g := NewGraph(vertices)
	
	// Ensure connectivity with a spanning tree
	for i := 1; i < vertices; i++ {
		parent := i / 2
		weight := 1.0 + 9.0*rand.Float64()
		g.AddEdge(parent, i, weight)
	}
	
	// Add random edges
	for i := 0; i < vertices; i++ {
		for j := i + 1; j < vertices; j++ {
			if rand.Float64() < edgeProbability {
				weight := 1.0 + 9.0*rand.Float64()
				g.AddEdge(i, j, weight)
			}
		}
	}
	
	return g
}

// Example demonstrates Prim's algorithm implementations
func Example() {
	fmt.Println("=== Parallel Prim's Algorithm Example ===")
	
	// Create a sample graph
	g := NewGraph(6)
	g.AddEdge(0, 1, 4)
	g.AddEdge(0, 2, 3)
	g.AddEdge(1, 2, 1)
	g.AddEdge(1, 3, 2)
	g.AddEdge(2, 3, 4)
	g.AddEdge(3, 4, 2)
	g.AddEdge(4, 5, 6)
	
	fmt.Println("Graph edges:")
	for _, edges := range g.adjacency {
		for _, e := range edges {
			if e.From < e.To {
				fmt.Printf("  %d -- %d : %.1f\n", e.From, e.To, e.Weight)
			}
		}
	}
	
	// Sequential Prim's
	fmt.Println("\nSequential Prim's MST:")
	mst1 := SequentialPrim(g, 0)
	printMST(mst1)
	
	// Parallel Prim's
	fmt.Println("\nParallel Prim's MST:")
	mst2 := ParallelPrim(g, 0)
	printMST(mst2)
	
	// Distributed Prim's
	fmt.Println("\nDistributed Prim's MST (2 partitions):")
	mst3 := DistributedPrim(g, 0, 2)
	printMST(mst3)
}

func printMST(mst *MST) {
	fmt.Println("MST edges:")
	for _, edge := range mst.Edges {
		fmt.Printf("  %d -- %d : %.1f\n", edge.From, edge.To, edge.Weight)
	}
	fmt.Printf("Total cost: %.1f\n", mst.TotalCost)
}