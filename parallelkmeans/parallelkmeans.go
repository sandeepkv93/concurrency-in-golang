package parallelkmeans

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// Point represents a data point in n-dimensional space
type Point []float64

// Cluster represents a cluster with its centroid and assigned points
type Cluster struct {
	ID       int
	Centroid Point
	Points   []Point
	mutex    sync.RWMutex
}

// ClusteringConfig contains configuration for the K-means algorithm
type ClusteringConfig struct {
	K                int           // Number of clusters
	MaxIterations    int           // Maximum number of iterations
	Tolerance        float64       // Convergence tolerance
	InitMethod       InitMethod    // Initialization method
	DistanceMetric   DistanceFunc  // Distance function
	NumWorkers       int           // Number of parallel workers
	BatchSize        int           // Batch size for parallel processing
	RandomSeed       int64         // Random seed for reproducibility
	ConvergenceCheck bool          // Enable convergence checking
	Verbose          bool          // Enable verbose logging
}

// InitMethod defines cluster initialization strategies
type InitMethod int

const (
	RandomInit InitMethod = iota
	KMeansPlusPlusInit
	ForgyInit
	RandomPartitionInit
)

// DistanceFunc represents a distance function between two points
type DistanceFunc func(Point, Point) float64

// KMeansClusterer implements parallel K-means clustering
type KMeansClusterer struct {
	config        ClusteringConfig
	clusters      []*Cluster
	points        []Point
	dimensions    int
	workers       []*Worker
	workQueue     chan WorkItem
	resultQueue   chan WorkResult
	stats         *ClusteringStats
	converged     bool
	iteration     int
	mutex         sync.RWMutex
	wg            sync.WaitGroup
	ctx           context.Context
	cancel        context.CancelFunc
}

// Worker represents a parallel processing worker
type Worker struct {
	ID            int
	clusterer     *KMeansClusterer
	localClusters []*LocalCluster
	assignedWork  int64
	processedWork int64
}

// LocalCluster represents a worker's local view of a cluster
type LocalCluster struct {
	ID     int
	Sum    Point
	Count  int
	Points []Point
}

// WorkItem represents work to be processed by workers
type WorkItem struct {
	Type      WorkType
	Points    []Point
	StartIdx  int
	EndIdx    int
	Iteration int
}

// WorkResult represents the result of worker processing
type WorkResult struct {
	WorkerID       int
	LocalClusters  []*LocalCluster
	Assignments    []int
	Error          error
	ProcessingTime time.Duration
}

// WorkType defines the type of work to be performed
type WorkType int

const (
	AssignmentWork WorkType = iota
	CentroidWork
)

// ClusteringStats contains statistics about the clustering process
type ClusteringStats struct {
	TotalIterations    int
	ConvergenceTime    time.Duration
	TotalPoints        int
	FinalSSE           float64   // Sum of Squared Errors
	Silhouette         float64   // Silhouette coefficient
	WorkerUtilization  []float64 // Per-worker utilization
	IterationTimes     []time.Duration
	MemoryUsage        int64
	ParallelEfficiency float64
	mutex              sync.RWMutex
}

// ClusteringResult contains the final clustering result
type ClusteringResult struct {
	Clusters    []*Cluster
	Assignments []int
	Stats       *ClusteringStats
	Converged   bool
	SSE         float64
	Silhouette  float64
}

// NewKMeansClusterer creates a new parallel K-means clusterer
func NewKMeansClusterer(config ClusteringConfig) *KMeansClusterer {
	if config.K <= 0 {
		config.K = 3
	}
	if config.MaxIterations <= 0 {
		config.MaxIterations = 100
	}
	if config.Tolerance <= 0 {
		config.Tolerance = 1e-6
	}
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 1000
	}
	if config.DistanceMetric == nil {
		config.DistanceMetric = EuclideanDistance
	}
	if config.RandomSeed == 0 {
		config.RandomSeed = time.Now().UnixNano()
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &KMeansClusterer{
		config:      config,
		workQueue:   make(chan WorkItem, config.NumWorkers*2),
		resultQueue: make(chan WorkResult, config.NumWorkers),
		stats:       &ClusteringStats{},
		ctx:         ctx,
		cancel:      cancel,
	}
}

// Fit performs K-means clustering on the given data points
func (km *KMeansClusterer) Fit(points []Point) (*ClusteringResult, error) {
	if len(points) == 0 {
		return nil, errors.New("no data points provided")
	}

	if len(points) < km.config.K {
		return nil, fmt.Errorf("number of points (%d) is less than K (%d)", len(points), km.config.K)
	}

	start := time.Now()
	km.points = points
	km.dimensions = len(points[0])
	km.stats.TotalPoints = len(points)

	// Validate point dimensions
	for i, point := range points {
		if len(point) != km.dimensions {
			return nil, fmt.Errorf("point %d has dimension %d, expected %d", i, len(point), km.dimensions)
		}
	}

	// Initialize clusters
	if err := km.initializeClusters(); err != nil {
		return nil, fmt.Errorf("failed to initialize clusters: %v", err)
	}

	// Start workers
	km.startWorkers()
	defer km.stopWorkers()

	// Run clustering iterations
	if err := km.runIterations(); err != nil {
		return nil, fmt.Errorf("clustering failed: %v", err)
	}

	// Calculate final statistics
	km.calculateFinalStats()
	km.stats.ConvergenceTime = time.Since(start)

	// Create result
	assignments := make([]int, len(points))
	for i, point := range points {
		assignments[i] = km.findClosestCluster(point)
	}

	result := &ClusteringResult{
		Clusters:    km.clusters,
		Assignments: assignments,
		Stats:       km.stats,
		Converged:   km.converged,
		SSE:         km.stats.FinalSSE,
		Silhouette:  km.stats.Silhouette,
	}

	return result, nil
}

// FitPredict performs clustering and returns cluster assignments
func (km *KMeansClusterer) FitPredict(points []Point) ([]int, error) {
	result, err := km.Fit(points)
	if err != nil {
		return nil, err
	}
	return result.Assignments, nil
}

// Predict assigns new points to existing clusters
func (km *KMeansClusterer) Predict(points []Point) ([]int, error) {
	if km.clusters == nil {
		return nil, errors.New("model not fitted yet")
	}

	assignments := make([]int, len(points))
	for i, point := range points {
		assignments[i] = km.findClosestCluster(point)
	}

	return assignments, nil
}

// GetCentroids returns the current cluster centroids
func (km *KMeansClusterer) GetCentroids() []Point {
	km.mutex.RLock()
	defer km.mutex.RUnlock()

	centroids := make([]Point, len(km.clusters))
	for i, cluster := range km.clusters {
		centroids[i] = make(Point, len(cluster.Centroid))
		copy(centroids[i], cluster.Centroid)
	}
	return centroids
}

// GetStats returns clustering statistics
func (km *KMeansClusterer) GetStats() *ClusteringStats {
	km.stats.mutex.RLock()
	defer km.stats.mutex.RUnlock()

	// Return a copy of stats
	stats := &ClusteringStats{
		TotalIterations:    km.stats.TotalIterations,
		ConvergenceTime:    km.stats.ConvergenceTime,
		TotalPoints:        km.stats.TotalPoints,
		FinalSSE:           km.stats.FinalSSE,
		Silhouette:         km.stats.Silhouette,
		ParallelEfficiency: km.stats.ParallelEfficiency,
		MemoryUsage:        km.stats.MemoryUsage,
	}

	stats.WorkerUtilization = make([]float64, len(km.stats.WorkerUtilization))
	copy(stats.WorkerUtilization, km.stats.WorkerUtilization)

	stats.IterationTimes = make([]time.Duration, len(km.stats.IterationTimes))
	copy(stats.IterationTimes, km.stats.IterationTimes)

	return stats
}

// initializeClusters initializes cluster centroids based on the chosen method
func (km *KMeansClusterer) initializeClusters() error {
	km.clusters = make([]*Cluster, km.config.K)
	rand.Seed(km.config.RandomSeed)

	switch km.config.InitMethod {
	case RandomInit:
		return km.initializeRandom()
	case KMeansPlusPlusInit:
		return km.initializeKMeansPlusPlus()
	case ForgyInit:
		return km.initializeForgy()
	case RandomPartitionInit:
		return km.initializeRandomPartition()
	default:
		return km.initializeRandom()
	}
}

// initializeRandom randomly selects K points as initial centroids
func (km *KMeansClusterer) initializeRandom() error {
	selected := make(map[int]bool)
	
	for i := 0; i < km.config.K; i++ {
		var idx int
		for {
			idx = rand.Intn(len(km.points))
			if !selected[idx] {
				break
			}
		}
		selected[idx] = true

		centroid := make(Point, km.dimensions)
		copy(centroid, km.points[idx])

		km.clusters[i] = &Cluster{
			ID:       i,
			Centroid: centroid,
			Points:   make([]Point, 0),
		}
	}
	return nil
}

// initializeKMeansPlusPlus implements K-means++ initialization
func (km *KMeansClusterer) initializeKMeansPlusPlus() error {
	// Choose first centroid randomly
	firstIdx := rand.Intn(len(km.points))
	centroid := make(Point, km.dimensions)
	copy(centroid, km.points[firstIdx])

	km.clusters[0] = &Cluster{
		ID:       0,
		Centroid: centroid,
		Points:   make([]Point, 0),
	}

	// Choose remaining centroids
	for k := 1; k < km.config.K; k++ {
		distances := make([]float64, len(km.points))
		totalDistance := 0.0

		// Calculate distances to nearest existing centroid
		for i, point := range km.points {
			minDist := math.Inf(1)
			for j := 0; j < k; j++ {
				dist := km.config.DistanceMetric(point, km.clusters[j].Centroid)
				if dist < minDist {
					minDist = dist
				}
			}
			distances[i] = minDist * minDist
			totalDistance += distances[i]
		}

		// Choose next centroid with probability proportional to squared distance
		target := rand.Float64() * totalDistance
		cumulative := 0.0
		selectedIdx := 0

		for i, dist := range distances {
			cumulative += dist
			if cumulative >= target {
				selectedIdx = i
				break
			}
		}

		centroid = make(Point, km.dimensions)
		copy(centroid, km.points[selectedIdx])

		km.clusters[k] = &Cluster{
			ID:       k,
			Centroid: centroid,
			Points:   make([]Point, 0),
		}
	}

	return nil
}

// initializeForgy randomly selects K points as initial centroids (same as random)
func (km *KMeansClusterer) initializeForgy() error {
	return km.initializeRandom()
}

// initializeRandomPartition randomly partitions points and calculates centroids
func (km *KMeansClusterer) initializeRandomPartition() error {
	// Create temporary clusters
	tempClusters := make([]*LocalCluster, km.config.K)
	for i := 0; i < km.config.K; i++ {
		tempClusters[i] = &LocalCluster{
			ID:  i,
			Sum: make(Point, km.dimensions),
		}
	}

	// Randomly assign points
	for _, point := range km.points {
		clusterIdx := rand.Intn(km.config.K)
		tempClusters[clusterIdx].Count++
		for j, val := range point {
			tempClusters[clusterIdx].Sum[j] += val
		}
	}

	// Calculate centroids
	for i, tempCluster := range tempClusters {
		if tempCluster.Count == 0 {
			// Fallback to random point if cluster is empty
			randomIdx := rand.Intn(len(km.points))
			centroid := make(Point, km.dimensions)
			copy(centroid, km.points[randomIdx])
			km.clusters[i] = &Cluster{
				ID:       i,
				Centroid: centroid,
				Points:   make([]Point, 0),
			}
		} else {
			centroid := make(Point, km.dimensions)
			for j := 0; j < km.dimensions; j++ {
				centroid[j] = tempCluster.Sum[j] / float64(tempCluster.Count)
			}
			km.clusters[i] = &Cluster{
				ID:       i,
				Centroid: centroid,
				Points:   make([]Point, 0),
			}
		}
	}

	return nil
}

// startWorkers starts the parallel workers
func (km *KMeansClusterer) startWorkers() {
	km.workers = make([]*Worker, km.config.NumWorkers)
	
	for i := 0; i < km.config.NumWorkers; i++ {
		worker := &Worker{
			ID:            i,
			clusterer:     km,
			localClusters: make([]*LocalCluster, km.config.K),
		}

		// Initialize local clusters
		for j := 0; j < km.config.K; j++ {
			worker.localClusters[j] = &LocalCluster{
				ID:  j,
				Sum: make(Point, km.dimensions),
			}
		}

		km.workers[i] = worker
		km.wg.Add(1)
		go worker.run()
	}
}

// stopWorkers stops all workers
func (km *KMeansClusterer) stopWorkers() {
	km.cancel()
	close(km.workQueue)
	km.wg.Wait()
	close(km.resultQueue)
}

// runIterations performs the main clustering iterations
func (km *KMeansClusterer) runIterations() error {
	for km.iteration = 0; km.iteration < km.config.MaxIterations; km.iteration++ {
		iterStart := time.Now()

		// Assignment phase
		if err := km.assignmentPhase(); err != nil {
			return err
		}

		// Update centroids
		oldCentroids := km.copyCentroids()
		if err := km.updateCentroids(); err != nil {
			return err
		}

		iterTime := time.Since(iterStart)
		km.stats.IterationTimes = append(km.stats.IterationTimes, iterTime)

		if km.config.Verbose {
			fmt.Printf("Iteration %d completed in %v\n", km.iteration+1, iterTime)
		}

		// Check convergence
		if km.config.ConvergenceCheck && km.hasConverged(oldCentroids) {
			km.converged = true
			if km.config.Verbose {
				fmt.Printf("Converged after %d iterations\n", km.iteration+1)
			}
			break
		}

		// Check context cancellation
		select {
		case <-km.ctx.Done():
			return km.ctx.Err()
		default:
		}
	}

	km.stats.TotalIterations = km.iteration + 1
	return nil
}

// assignmentPhase assigns points to clusters in parallel
func (km *KMeansClusterer) assignmentPhase() error {
	numBatches := (len(km.points) + km.config.BatchSize - 1) / km.config.BatchSize

	// Send work items
	for i := 0; i < numBatches; i++ {
		startIdx := i * km.config.BatchSize
		endIdx := min(startIdx+km.config.BatchSize, len(km.points))

		workItem := WorkItem{
			Type:      AssignmentWork,
			Points:    km.points[startIdx:endIdx],
			StartIdx:  startIdx,
			EndIdx:    endIdx,
			Iteration: km.iteration,
		}

		select {
		case km.workQueue <- workItem:
		case <-km.ctx.Done():
			return km.ctx.Err()
		}
	}

	// Collect results
	for i := 0; i < numBatches; i++ {
		select {
		case result := <-km.resultQueue:
			if result.Error != nil {
				return result.Error
			}
		case <-km.ctx.Done():
			return km.ctx.Err()
		}
	}

	return nil
}

// updateCentroids updates cluster centroids based on assigned points
func (km *KMeansClusterer) updateCentroids() error {
	// Reset local clusters in workers
	for _, worker := range km.workers {
		for j := 0; j < km.config.K; j++ {
			worker.localClusters[j].Count = 0
			for k := 0; k < km.dimensions; k++ {
				worker.localClusters[j].Sum[k] = 0
			}
		}
	}

	// Send centroid update work
	numBatches := (len(km.points) + km.config.BatchSize - 1) / km.config.BatchSize

	for i := 0; i < numBatches; i++ {
		startIdx := i * km.config.BatchSize
		endIdx := min(startIdx+km.config.BatchSize, len(km.points))

		workItem := WorkItem{
			Type:      CentroidWork,
			Points:    km.points[startIdx:endIdx],
			StartIdx:  startIdx,
			EndIdx:    endIdx,
			Iteration: km.iteration,
		}

		select {
		case km.workQueue <- workItem:
		case <-km.ctx.Done():
			return km.ctx.Err()
		}
	}

	// Collect results and aggregate
	globalSums := make([]Point, km.config.K)
	globalCounts := make([]int, km.config.K)

	for i := 0; i < km.config.K; i++ {
		globalSums[i] = make(Point, km.dimensions)
	}

	for i := 0; i < numBatches; i++ {
		select {
		case result := <-km.resultQueue:
			if result.Error != nil {
				return result.Error
			}

			// Aggregate local results
			for j, localCluster := range result.LocalClusters {
				globalCounts[j] += localCluster.Count
				for k := 0; k < km.dimensions; k++ {
					globalSums[j][k] += localCluster.Sum[k]
				}
			}
		case <-km.ctx.Done():
			return km.ctx.Err()
		}
	}

	// Update centroids
	km.mutex.Lock()
	for i := 0; i < km.config.K; i++ {
		if globalCounts[i] > 0 {
			for j := 0; j < km.dimensions; j++ {
				km.clusters[i].Centroid[j] = globalSums[i][j] / float64(globalCounts[i])
			}
		}
		// If cluster is empty, keep the old centroid or reinitialize
	}
	km.mutex.Unlock()

	return nil
}

// Worker methods

// run starts the worker's main processing loop
func (w *Worker) run() {
	defer w.clusterer.wg.Done()

	for {
		select {
		case workItem, ok := <-w.clusterer.workQueue:
			if !ok {
				return
			}

			startTime := time.Now()
			result := w.processWork(workItem)
			result.ProcessingTime = time.Since(startTime)
			result.WorkerID = w.ID

			atomic.AddInt64(&w.processedWork, 1)

			select {
			case w.clusterer.resultQueue <- result:
			case <-w.clusterer.ctx.Done():
				return
			}

		case <-w.clusterer.ctx.Done():
			return
		}
	}
}

// processWork processes a work item
func (w *Worker) processWork(item WorkItem) WorkResult {
	switch item.Type {
	case AssignmentWork:
		return w.processAssignment(item)
	case CentroidWork:
		return w.processCentroidUpdate(item)
	default:
		return WorkResult{Error: fmt.Errorf("unknown work type: %d", item.Type)}
	}
}

// processAssignment assigns points to closest clusters
func (w *Worker) processAssignment(item WorkItem) WorkResult {
	assignments := make([]int, len(item.Points))

	for i, point := range item.Points {
		assignments[i] = w.clusterer.findClosestCluster(point)
	}

	return WorkResult{
		Assignments: assignments,
	}
}

// processCentroidUpdate accumulates points for centroid calculation
func (w *Worker) processCentroidUpdate(item WorkItem) WorkResult {
	// Reset local clusters
	for j := 0; j < w.clusterer.config.K; j++ {
		w.localClusters[j].Count = 0
		for k := 0; k < w.clusterer.dimensions; k++ {
			w.localClusters[j].Sum[k] = 0
		}
	}

	// Accumulate points
	for _, point := range item.Points {
		clusterIdx := w.clusterer.findClosestCluster(point)
		w.localClusters[clusterIdx].Count++
		for j, val := range point {
			w.localClusters[clusterIdx].Sum[j] += val
		}
	}

	// Create copies for result
	resultClusters := make([]*LocalCluster, w.clusterer.config.K)
	for i := 0; i < w.clusterer.config.K; i++ {
		resultClusters[i] = &LocalCluster{
			ID:    i,
			Count: w.localClusters[i].Count,
			Sum:   make(Point, w.clusterer.dimensions),
		}
		copy(resultClusters[i].Sum, w.localClusters[i].Sum)
	}

	return WorkResult{
		LocalClusters: resultClusters,
	}
}

// Utility methods

// findClosestCluster finds the closest cluster for a given point
func (km *KMeansClusterer) findClosestCluster(point Point) int {
	km.mutex.RLock()
	defer km.mutex.RUnlock()

	minDistance := math.Inf(1)
	closestCluster := 0

	for i, cluster := range km.clusters {
		distance := km.config.DistanceMetric(point, cluster.Centroid)
		if distance < minDistance {
			minDistance = distance
			closestCluster = i
		}
	}

	return closestCluster
}

// copyCentroids creates a copy of current centroids
func (km *KMeansClusterer) copyCentroids() []Point {
	km.mutex.RLock()
	defer km.mutex.RUnlock()

	centroids := make([]Point, len(km.clusters))
	for i, cluster := range km.clusters {
		centroids[i] = make(Point, len(cluster.Centroid))
		copy(centroids[i], cluster.Centroid)
	}
	return centroids
}

// hasConverged checks if the algorithm has converged
func (km *KMeansClusterer) hasConverged(oldCentroids []Point) bool {
	km.mutex.RLock()
	defer km.mutex.RUnlock()

	for i, cluster := range km.clusters {
		distance := km.config.DistanceMetric(cluster.Centroid, oldCentroids[i])
		if distance > km.config.Tolerance {
			return false
		}
	}
	return true
}

// calculateFinalStats calculates final clustering statistics
func (km *KMeansClusterer) calculateFinalStats() {
	km.stats.mutex.Lock()
	defer km.stats.mutex.Unlock()

	// Calculate SSE (Sum of Squared Errors)
	sse := 0.0
	for _, point := range km.points {
		clusterIdx := km.findClosestCluster(point)
		distance := km.config.DistanceMetric(point, km.clusters[clusterIdx].Centroid)
		sse += distance * distance
	}
	km.stats.FinalSSE = sse

	// Calculate Silhouette coefficient (simplified version)
	km.stats.Silhouette = km.calculateSilhouette()

	// Calculate worker utilization
	km.stats.WorkerUtilization = make([]float64, len(km.workers))
	totalWork := int64(0)
	for i, worker := range km.workers {
		km.stats.WorkerUtilization[i] = float64(atomic.LoadInt64(&worker.processedWork))
		totalWork += atomic.LoadInt64(&worker.processedWork)
	}

	// Calculate parallel efficiency
	if totalWork > 0 {
		idealWork := float64(totalWork) / float64(len(km.workers))
		efficiency := 0.0
		for _, util := range km.stats.WorkerUtilization {
			efficiency += util / idealWork
		}
		km.stats.ParallelEfficiency = efficiency / float64(len(km.workers))
	}
}

// calculateSilhouette calculates simplified silhouette coefficient
func (km *KMeansClusterer) calculateSilhouette() float64 {
	if len(km.points) < 2 || km.config.K <= 1 {
		return 0.0
	}

	// Sample-based calculation for large datasets
	sampleSize := min(1000, len(km.points))
	step := len(km.points) / sampleSize
	if step == 0 {
		step = 1
	}

	totalSilhouette := 0.0
	validPoints := 0

	for i := 0; i < len(km.points); i += step {
		point := km.points[i]
		clusterIdx := km.findClosestCluster(point)

		// Calculate a(i) - average distance to points in same cluster
		a := km.averageDistanceToCluster(point, clusterIdx)

		// Calculate b(i) - minimum average distance to points in other clusters
		b := math.Inf(1)
		for j := 0; j < km.config.K; j++ {
			if j != clusterIdx {
				avgDist := km.averageDistanceToCluster(point, j)
				if avgDist < b {
					b = avgDist
				}
			}
		}

		// Silhouette for this point
		if a == 0 && b == 0 {
			continue // Skip if both distances are 0
		}
		s := (b - a) / math.Max(a, b)
		totalSilhouette += s
		validPoints++
	}

	if validPoints == 0 {
		return 0.0
	}
	return totalSilhouette / float64(validPoints)
}

// averageDistanceToCluster calculates average distance from point to cluster
func (km *KMeansClusterer) averageDistanceToCluster(point Point, clusterIdx int) float64 {
	// Simplified: use distance to centroid
	return km.config.DistanceMetric(point, km.clusters[clusterIdx].Centroid)
}

// Distance functions

// EuclideanDistance calculates Euclidean distance between two points
func EuclideanDistance(p1, p2 Point) float64 {
	if len(p1) != len(p2) {
		return math.Inf(1)
	}

	sum := 0.0
	for i := 0; i < len(p1); i++ {
		diff := p1[i] - p2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// ManhattanDistance calculates Manhattan distance between two points
func ManhattanDistance(p1, p2 Point) float64 {
	if len(p1) != len(p2) {
		return math.Inf(1)
	}

	sum := 0.0
	for i := 0; i < len(p1); i++ {
		sum += math.Abs(p1[i] - p2[i])
	}
	return sum
}

// CosineDistance calculates cosine distance between two points
func CosineDistance(p1, p2 Point) float64 {
	if len(p1) != len(p2) {
		return math.Inf(1)
	}

	dotProduct := 0.0
	normP1 := 0.0
	normP2 := 0.0

	for i := 0; i < len(p1); i++ {
		dotProduct += p1[i] * p2[i]
		normP1 += p1[i] * p1[i]
		normP2 += p2[i] * p2[i]
	}

	normP1 = math.Sqrt(normP1)
	normP2 = math.Sqrt(normP2)

	if normP1 == 0 || normP2 == 0 {
		return 1.0
	}

	cosine := dotProduct / (normP1 * normP2)
	return 1.0 - cosine
}

// Utility functions

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// GenerateRandomPoints generates random points for testing
func GenerateRandomPoints(numPoints, dimensions int, seed int64) []Point {
	rand.Seed(seed)
	points := make([]Point, numPoints)

	for i := 0; i < numPoints; i++ {
		point := make(Point, dimensions)
		for j := 0; j < dimensions; j++ {
			point[j] = rand.Float64() * 100 // Scale as needed
		}
		points[i] = point
	}

	return points
}

// GenerateClusteredData generates clustered data for testing
func GenerateClusteredData(numClusters, pointsPerCluster, dimensions int, spread float64, seed int64) []Point {
	rand.Seed(seed)
	points := make([]Point, 0, numClusters*pointsPerCluster)

	// Generate cluster centers
	centers := make([]Point, numClusters)
	for i := 0; i < numClusters; i++ {
		center := make(Point, dimensions)
		for j := 0; j < dimensions; j++ {
			center[j] = rand.Float64() * 100
		}
		centers[i] = center
	}

	// Generate points around each center
	for i := 0; i < numClusters; i++ {
		for j := 0; j < pointsPerCluster; j++ {
			point := make(Point, dimensions)
			for k := 0; k < dimensions; k++ {
				point[k] = centers[i][k] + (rand.Float64()-0.5)*spread
			}
			points = append(points, point)
		}
	}

	// Shuffle points
	for i := len(points) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		points[i], points[j] = points[j], points[i]
	}

	return points
}

// EvaluateClustering evaluates clustering quality
func EvaluateClustering(points []Point, assignments []int, centroids []Point, distanceFunc DistanceFunc) map[string]float64 {
	if len(points) != len(assignments) {
		return nil
	}

	metrics := make(map[string]float64)

	// Calculate Within-Cluster Sum of Squares (WCSS)
	wcss := 0.0
	clusterCounts := make(map[int]int)

	for i, point := range points {
		clusterIdx := assignments[i]
		clusterCounts[clusterIdx]++
		if clusterIdx >= 0 && clusterIdx < len(centroids) {
			distance := distanceFunc(point, centroids[clusterIdx])
			wcss += distance * distance
		}
	}

	metrics["WCSS"] = wcss
	metrics["AverageWCSS"] = wcss / float64(len(points))

	// Calculate number of clusters found
	metrics["NumClusters"] = float64(len(clusterCounts))

	// Calculate cluster balance (standard deviation of cluster sizes)
	if len(clusterCounts) > 1 {
		mean := float64(len(points)) / float64(len(clusterCounts))
		variance := 0.0
		for _, count := range clusterCounts {
			diff := float64(count) - mean
			variance += diff * diff
		}
		variance /= float64(len(clusterCounts))
		metrics["ClusterBalance"] = math.Sqrt(variance)
	}

	return metrics
}

// PrintClusteringResult prints a summary of clustering results
func PrintClusteringResult(result *ClusteringResult) {
	fmt.Printf("Clustering Results:\n")
	fmt.Printf("==================\n")
	fmt.Printf("Converged: %v\n", result.Converged)
	fmt.Printf("SSE: %.4f\n", result.SSE)
	fmt.Printf("Silhouette: %.4f\n", result.Silhouette)
	fmt.Printf("Total Iterations: %d\n", result.Stats.TotalIterations)
	fmt.Printf("Convergence Time: %v\n", result.Stats.ConvergenceTime)
	fmt.Printf("Parallel Efficiency: %.2f%%\n", result.Stats.ParallelEfficiency*100)

	fmt.Printf("\nCluster Summary:\n")
	for i, cluster := range result.Clusters {
		count := 0
		for _, assignment := range result.Assignments {
			if assignment == i {
				count++
			}
		}
		fmt.Printf("Cluster %d: %d points, centroid: %v\n", 
			cluster.ID, count, cluster.Centroid)
	}

	if len(result.Stats.IterationTimes) > 0 {
		fmt.Printf("\nIteration Times:\n")
		totalTime := time.Duration(0)
		for i, iterTime := range result.Stats.IterationTimes {
			fmt.Printf("  Iteration %d: %v\n", i+1, iterTime)
			totalTime += iterTime
		}
		avgTime := totalTime / time.Duration(len(result.Stats.IterationTimes))
		fmt.Printf("  Average: %v\n", avgTime)
	}
}