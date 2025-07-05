# Parallel Dijkstra's Algorithm

A comprehensive implementation of Dijkstra's shortest path algorithm with multiple parallel variants in Go, demonstrating advanced concurrency patterns for graph algorithms including delta-stepping, bidirectional search, and concurrent edge relaxation.

## Problem Description

Finding shortest paths in large graphs presents several computational challenges:

- **Sequential Bottleneck**: Traditional Dijkstra's algorithm is inherently sequential
- **Large Graph Processing**: Million-node graphs require efficient parallel processing
- **Memory Contention**: Multiple threads updating shared data structures
- **Load Balancing**: Uneven work distribution across parallel workers
- **Race Conditions**: Concurrent updates to distance and predecessor arrays
- **Synchronization Overhead**: Balancing parallelism with coordination costs

## Solution Approach

This implementation provides multiple parallel variants of Dijkstra's algorithm:

1. **Sequential Dijkstra**: Classic algorithm using priority queue
2. **Parallel Dijkstra**: Concurrent edge relaxation with atomic operations
3. **Delta-Stepping**: Bucket-based parallel algorithm for better load balancing
4. **Bidirectional Search**: Simultaneous forward and backward exploration

## Key Components

### Core Data Structures

- **Graph**: Adjacency list representation with weighted edges
- **Edge**: Represents a directed weighted edge between vertices
- **ShortestPaths**: Results containing distances and path reconstruction
- **PriorityQueue**: Heap-based priority queue for vertex selection
- **Node**: Priority queue element with vertex and distance

### Algorithm Variants

- **SequentialDijkstra**: Single-threaded reference implementation
- **ParallelDijkstra**: Multi-threaded with atomic operations
- **DeltaSteppingDijkstra**: Bucket-based parallel algorithm
- **BidirectionalDijkstra**: Bidirectional search optimization

## Technical Features

### Concurrency Patterns

1. **Work Stealing**: Dynamic work distribution using channels
2. **Atomic Operations**: Lock-free distance updates with compare-and-swap
3. **Bucket Synchronization**: Coordinated parallel processing of vertex buckets
4. **Producer-Consumer**: Worker threads consuming vertices from queues
5. **Scatter-Gather**: Parallel edge relaxation with result aggregation

### Advanced Features

- **Lock-free Distance Updates**: Using atomic.Value for concurrent access
- **Bucket-based Parallelism**: Delta-stepping for improved load balancing
- **Bidirectional Optimization**: Reducing search space by half
- **Path Reconstruction**: Efficient shortest path recovery
- **Graph Flexibility**: Support for directed and undirected graphs

## Usage Examples

### Basic Sequential Dijkstra

```go
// Create a graph with 6 vertices
g := NewGraph(6)

// Add edges (from, to, weight)
g.AddEdge(0, 1, 4)
g.AddEdge(0, 2, 2)
g.AddEdge(1, 2, 1)
g.AddEdge(1, 3, 5)
g.AddEdge(2, 3, 8)
g.AddEdge(2, 4, 10)
g.AddEdge(3, 4, 2)
g.AddEdge(3, 5, 6)
g.AddEdge(4, 5, 3)

// Find shortest paths from vertex 0
sp := SequentialDijkstra(g, 0)

// Print distances
for v := 0; v < g.vertices; v++ {
    if math.IsInf(sp.Distance[v], 1) {
        fmt.Printf("Vertex %d: unreachable\n", v)
    } else {
        fmt.Printf("Vertex %d: distance=%.0f\n", v, sp.Distance[v])
    }
}

// Get specific path
path := sp.GetPath(5)
fmt.Printf("Path from 0 to 5: %v\n", path)
```

### Parallel Dijkstra

```go
// Use parallel implementation for large graphs
sp := ParallelDijkstra(g, 0)

// Results are identical to sequential version
fmt.Printf("Distance to vertex 5: %.0f\n", sp.Distance[5])
fmt.Printf("Path: %v\n", sp.GetPath(5))
```

### Delta-Stepping Algorithm

```go
// Delta-stepping with bucket size 3
// Optimal delta depends on graph structure
sp := DeltaSteppingDijkstra(g, 0, 3.0)

// Better parallelization for graphs with varied edge weights
fmt.Printf("Distance to vertex 5: %.0f\n", sp.Distance[5])
fmt.Printf("Path: %v\n", sp.GetPath(5))
```

### Bidirectional Search

```go
// Efficient for single source-target queries
sp := BidirectionalDijkstra(g, 0, 5)

// Only computes paths relevant to target
fmt.Printf("Distance from 0 to 5: %.0f\n", sp.Distance[5])
fmt.Printf("Path: %v\n", sp.GetPath(5))
```

### Undirected Graphs

```go
// Create undirected graph
g := NewGraph(4)
g.AddUndirectedEdge(0, 1, 1)
g.AddUndirectedEdge(1, 2, 2)
g.AddUndirectedEdge(2, 3, 3)
g.AddUndirectedEdge(0, 3, 10)

// Find shortest paths
sp := ParallelDijkstra(g, 0)
fmt.Printf("Shortest path 0->3: %v (distance: %.0f)\n", 
    sp.GetPath(3), sp.Distance[3])
```

### Performance Comparison

```go
// Large graph for performance testing
g := NewGraph(10000)
for i := 0; i < 10000; i++ {
    for j := 0; j < 5; j++ {
        target := rand.Intn(10000)
        weight := rand.Float64() * 100
        g.AddEdge(i, target, weight)
    }
}

// Sequential timing
start := time.Now()
sp1 := SequentialDijkstra(g, 0)
seqTime := time.Since(start)

// Parallel timing
start = time.Now()
sp2 := ParallelDijkstra(g, 0)
parTime := time.Since(start)

fmt.Printf("Sequential: %v\n", seqTime)
fmt.Printf("Parallel: %v\n", parTime)
fmt.Printf("Speedup: %.2fx\n", float64(seqTime)/float64(parTime))
```

## Implementation Details

### Sequential Dijkstra

The classic algorithm using a priority queue:

```go
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
    
    // Priority queue for vertices
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
        
        // Relax all edges from u
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
```

### Parallel Dijkstra with Atomic Operations

Concurrent edge relaxation using atomic operations:

```go
func ParallelDijkstra(g *Graph, source int) *ShortestPaths {
    dist := make([]atomic.Value, g.vertices)
    prev := make([]atomic.Int32, g.vertices)
    inQueue := make([]atomic.Bool, g.vertices)
    
    // Initialize atomic values
    for i := 0; i < g.vertices; i++ {
        dist[i].Store(math.Inf(1))
        prev[i].Store(int32(-1))
    }
    dist[source].Store(0.0)
    
    // Work queue for vertices to process
    workQueue := make(chan int, g.vertices)
    var wg sync.WaitGroup
    
    // Start worker goroutines
    numWorkers := 4
    for w := 0; w < numWorkers; w++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            
            for u := range workQueue {
                currentDist := dist[u].Load().(float64)
                
                // Relax all edges from u
                for _, edge := range g.adjacency[u] {
                    v := edge.To
                    newDist := currentDist + edge.Weight
                    
                    // Atomic distance update
                    for {
                        oldDist := dist[v].Load().(float64)
                        if newDist >= oldDist {
                            break
                        }
                        
                        if dist[v].CompareAndSwap(oldDist, newDist) {
                            prev[v].Store(int32(u))
                            
                            // Add to work queue if not already there
                            if !inQueue[v].Swap(true) {
                                workQueue <- v
                            }
                            break
                        }
                    }
                }
                
                inQueue[u].Store(false)
            }
        }()
    }
    
    // Start with source vertex
    workQueue <- source
    inQueue[source].Store(true)
    
    // Wait for completion
    // ... (completion detection logic)
    
    return result
}
```

### Delta-Stepping Algorithm

Bucket-based parallel algorithm:

```go
func DeltaSteppingDijkstra(g *Graph, source int, delta float64) *ShortestPaths {
    dist := make([]atomic.Value, g.vertices)
    prev := make([]atomic.Int32, g.vertices)
    
    // Initialize
    for i := 0; i < g.vertices; i++ {
        dist[i].Store(math.Inf(1))
        prev[i].Store(int32(-1))
    }
    dist[source].Store(0.0)
    
    // Bucket management
    buckets := make(map[int]map[int]bool)
    
    // Add vertex to appropriate bucket
    addToBucket := func(v int, d float64) {
        bucket := int(d / delta)
        if buckets[bucket] == nil {
            buckets[bucket] = make(map[int]bool)
        }
        buckets[bucket][v] = true
    }
    
    // Process buckets in order
    for {
        bucket := getMinBucket(buckets)
        if bucket == -1 {
            break
        }
        
        // Process all vertices in current bucket
        vertices := getVerticesInBucket(buckets, bucket)
        
        // Parallel processing of light edges
        var wg sync.WaitGroup
        for _, u := range vertices {
            wg.Add(1)
            go func(vertex int) {
                defer wg.Done()
                
                for _, edge := range g.adjacency[vertex] {
                    if edge.Weight <= delta {
                        relaxEdge(vertex, edge)
                    }
                }
            }(u)
        }
        wg.Wait()
        
        // Process heavy edges
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
    
    return result
}
```

### Bidirectional Search

Simultaneous forward and backward exploration:

```go
func BidirectionalDijkstra(g *Graph, source, target int) *ShortestPaths {
    // Forward and backward distances
    distF := make([]float64, g.vertices)
    distB := make([]float64, g.vertices)
    
    // Initialize
    for i := 0; i < g.vertices; i++ {
        distF[i] = math.Inf(1)
        distB[i] = math.Inf(1)
    }
    distF[source] = 0
    distB[target] = 0
    
    // Priority queues for both directions
    pqF := &PriorityQueue{}
    pqB := &PriorityQueue{}
    
    bestPath := math.Inf(1)
    meetingPoint := -1
    
    // Alternate between forward and backward search
    for pqF.Len() > 0 || pqB.Len() > 0 {
        // Forward step
        if pqF.Len() > 0 {
            // ... process forward direction
        }
        
        // Backward step
        if pqB.Len() > 0 {
            // ... process backward direction
        }
        
        // Check for meeting point
        if meetingPoint != -1 {
            bestPath = distF[meetingPoint] + distB[meetingPoint]
        }
        
        // Early termination condition
        if minForward + minBackward >= bestPath {
            break
        }
    }
    
    return result
}
```

## Testing

The package includes comprehensive tests covering:

- **Correctness**: Comparing parallel results with sequential algorithm
- **Performance**: Measuring speedup on various graph sizes
- **Edge Cases**: Disconnected graphs, single vertices, negative weights
- **Race Conditions**: Concurrent stress testing
- **Memory Safety**: Proper cleanup of goroutines and channels

Run the tests:

```bash
go test -v ./paralleldijkstra
go test -race ./paralleldijkstra  # Race condition detection
go test -bench=. ./paralleldijkstra  # Performance benchmarks
```

## Performance Considerations

### Algorithm Selection

- **Sequential**: Best for small graphs (< 1000 vertices)
- **Parallel**: Good for medium graphs with uniform edge weights
- **Delta-Stepping**: Optimal for graphs with varied edge weights
- **Bidirectional**: Ideal for single source-target queries

### Parameter Tuning

```go
// Delta-stepping parameter selection
func optimalDelta(g *Graph) float64 {
    // Rule of thumb: delta = average edge weight
    totalWeight := 0.0
    edgeCount := 0
    
    for u := 0; u < g.vertices; u++ {
        for _, edge := range g.adjacency[u] {
            totalWeight += edge.Weight
            edgeCount++
        }
    }
    
    return totalWeight / float64(edgeCount)
}
```

### Memory Optimization

```go
// Sparse graph representation
type SparseGraph struct {
    vertices int
    edges    []Edge
    offsets  []int
}

// Compact edge representation
type CompactEdge struct {
    to     uint32
    weight float32
}
```

## Real-World Applications

This parallel Dijkstra implementation is suitable for:

- **Route Planning**: GPS navigation systems
- **Network Routing**: Internet packet routing protocols
- **Social Networks**: Finding shortest paths between users
- **Game Development**: Pathfinding for NPCs and AI
- **Supply Chain**: Optimizing logistics and delivery routes
- **VLSI Design**: Wire routing in chip design

## Advanced Features

### Graph Preprocessing

```go
// Contraction hierarchies for faster queries
func PreprocessGraph(g *Graph) *ContractionHierarchy {
    // Implementation of node contraction
    return &ContractionHierarchy{}
}

// A* search with heuristic
func AStarDijkstra(g *Graph, source, target int, heuristic func(int, int) float64) *ShortestPaths {
    // Implementation of A* algorithm
    return &ShortestPaths{}
}
```

### Dynamic Graphs

```go
// Incremental shortest paths for dynamic graphs
type DynamicDijkstra struct {
    g     *Graph
    cache map[int]*ShortestPaths
}

func (dd *DynamicDijkstra) AddEdge(from, to int, weight float64) {
    // Update shortest paths incrementally
}
```

### Distributed Implementation

```go
// Distributed Dijkstra for massive graphs
func DistributedDijkstra(g *DistributedGraph, source int, workers []WorkerNode) *ShortestPaths {
    // Implementation for distributed computing
    return &ShortestPaths{}
}
```

The implementation showcases advanced Go concurrency patterns for parallel graph algorithms, providing a solid foundation for high-performance shortest path computations in large-scale applications.