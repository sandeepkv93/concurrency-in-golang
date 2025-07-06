package concurrentgameoflife

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

type Cell struct {
	Alive bool
	X, Y  int
}

type Grid struct {
	width   int
	height  int
	cells   [][]bool
	mutex   sync.RWMutex
	buffer  [][]bool
}

type GameOfLife struct {
	grid         *Grid
	generation   int
	running      bool
	paused       bool
	speed        time.Duration
	numWorkers   int
	stats        *Statistics
	observers    []Observer
	patterns     map[string]Pattern
	mutex        sync.RWMutex
	stopChan     chan struct{}
	pauseChan    chan struct{}
	resumeChan   chan struct{}
}

type Statistics struct {
	Generation    int
	AliveCells    int
	DeadCells     int
	BirthCount    int
	DeathCount    int
	StablePattern bool
	PopulationHistory []int
	mutex         sync.RWMutex
}

type Observer interface {
	OnGenerationUpdate(generation int, grid *Grid, stats *Statistics)
}

type Pattern struct {
	Name   string
	Width  int
	Height int
	Cells  [][]bool
}

type Region struct {
	startX, endX int
	startY, endY int
}

func NewGrid(width, height int) *Grid {
	cells := make([][]bool, height)
	buffer := make([][]bool, height)
	for i := range cells {
		cells[i] = make([]bool, width)
		buffer[i] = make([]bool, width)
	}
	
	return &Grid{
		width:  width,
		height: height,
		cells:  cells,
		buffer: buffer,
	}
}

func (g *Grid) GetCell(x, y int) bool {
	g.mutex.RLock()
	defer g.mutex.RUnlock()
	
	if x < 0 || x >= g.width || y < 0 || y >= g.height {
		return false
	}
	return g.cells[y][x]
}

func (g *Grid) SetCell(x, y int, alive bool) {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	
	if x >= 0 && x < g.width && y >= 0 && y < g.height {
		g.cells[y][x] = alive
	}
}

func (g *Grid) CountNeighbors(x, y int) int {
	count := 0
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			if dx == 0 && dy == 0 {
				continue
			}
			nx, ny := x+dx, y+dy
			if g.GetCell(nx, ny) {
				count++
			}
		}
	}
	return count
}

func (g *Grid) GetAliveCells() []Cell {
	g.mutex.RLock()
	defer g.mutex.RUnlock()
	
	var cells []Cell
	for y := 0; y < g.height; y++ {
		for x := 0; x < g.width; x++ {
			if g.cells[y][x] {
				cells = append(cells, Cell{Alive: true, X: x, Y: y})
			}
		}
	}
	return cells
}

func (g *Grid) CountAliveCells() int {
	g.mutex.RLock()
	defer g.mutex.RUnlock()
	
	count := 0
	for y := 0; y < g.height; y++ {
		for x := 0; x < g.width; x++ {
			if g.cells[y][x] {
				count++
			}
		}
	}
	return count
}

func (g *Grid) Clear() {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	
	for y := 0; y < g.height; y++ {
		for x := 0; x < g.width; x++ {
			g.cells[y][x] = false
		}
	}
}

func (g *Grid) Copy() *Grid {
	g.mutex.RLock()
	defer g.mutex.RUnlock()
	
	newGrid := NewGrid(g.width, g.height)
	for y := 0; y < g.height; y++ {
		for x := 0; x < g.width; x++ {
			newGrid.cells[y][x] = g.cells[y][x]
		}
	}
	return newGrid
}

func (g *Grid) Equals(other *Grid) bool {
	if g.width != other.width || g.height != other.height {
		return false
	}
	
	g.mutex.RLock()
	other.mutex.RLock()
	defer g.mutex.RUnlock()
	defer other.mutex.RUnlock()
	
	for y := 0; y < g.height; y++ {
		for x := 0; x < g.width; x++ {
			if g.cells[y][x] != other.cells[y][x] {
				return false
			}
		}
	}
	return true
}

func NewGameOfLife(width, height int, numWorkers int) *GameOfLife {
	return &GameOfLife{
		grid:       NewGrid(width, height),
		generation: 0,
		running:    false,
		paused:     false,
		speed:      100 * time.Millisecond,
		numWorkers: numWorkers,
		stats:      &Statistics{PopulationHistory: make([]int, 0)},
		observers:  make([]Observer, 0),
		patterns:   getBuiltInPatterns(),
		stopChan:   make(chan struct{}),
		pauseChan:  make(chan struct{}),
		resumeChan: make(chan struct{}),
	}
}

func (gol *GameOfLife) SetSpeed(speed time.Duration) {
	gol.mutex.Lock()
	defer gol.mutex.Unlock()
	gol.speed = speed
}

func (gol *GameOfLife) AddObserver(observer Observer) {
	gol.mutex.Lock()
	defer gol.mutex.Unlock()
	gol.observers = append(gol.observers, observer)
}

func (gol *GameOfLife) RemoveObserver(observer Observer) {
	gol.mutex.Lock()
	defer gol.mutex.Unlock()
	
	for i, obs := range gol.observers {
		if obs == observer {
			gol.observers = append(gol.observers[:i], gol.observers[i+1:]...)
			break
		}
	}
}

func (gol *GameOfLife) notifyObservers() {
	gol.mutex.RLock()
	observers := make([]Observer, len(gol.observers))
	copy(observers, gol.observers)
	gol.mutex.RUnlock()
	
	for _, observer := range observers {
		observer.OnGenerationUpdate(gol.generation, gol.grid, gol.stats)
	}
}

func (gol *GameOfLife) RandomizeGrid(density float64) {
	rand.Seed(time.Now().UnixNano())
	
	for y := 0; y < gol.grid.height; y++ {
		for x := 0; x < gol.grid.width; x++ {
			gol.grid.SetCell(x, y, rand.Float64() < density)
		}
	}
	
	gol.updateStatistics()
}

func (gol *GameOfLife) LoadPattern(patternName string, startX, startY int) error {
	pattern, exists := gol.patterns[patternName]
	if !exists {
		return fmt.Errorf("pattern '%s' not found", patternName)
	}
	
	for y := 0; y < pattern.Height; y++ {
		for x := 0; x < pattern.Width; x++ {
			cellX, cellY := startX+x, startY+y
			if cellX >= 0 && cellX < gol.grid.width && cellY >= 0 && cellY < gol.grid.height {
				gol.grid.SetCell(cellX, cellY, pattern.Cells[y][x])
			}
		}
	}
	
	gol.updateStatistics()
	return nil
}

func (gol *GameOfLife) NextGeneration(ctx context.Context) {
	regions := gol.divideIntoRegions()
	var wg sync.WaitGroup
	
	births := make(chan Cell, 100)
	deaths := make(chan Cell, 100)
	
	for _, region := range regions {
		wg.Add(1)
		go gol.processRegion(ctx, &wg, region, births, deaths)
	}
	
	go func() {
		wg.Wait()
		close(births)
		close(deaths)
	}()
	
	gol.applyChanges(births, deaths)
	gol.generation++
	gol.updateStatistics()
	gol.notifyObservers()
}

func (gol *GameOfLife) processRegion(ctx context.Context, wg *sync.WaitGroup, region Region, births, deaths chan<- Cell) {
	defer wg.Done()
	
	for y := region.startY; y < region.endY; y++ {
		for x := region.startX; x < region.endX; x++ {
			select {
			case <-ctx.Done():
				return
			default:
				neighbors := gol.grid.CountNeighbors(x, y)
				currentlyAlive := gol.grid.GetCell(x, y)
				
				if currentlyAlive {
					if neighbors < 2 || neighbors > 3 {
						deaths <- Cell{Alive: false, X: x, Y: y}
					}
				} else {
					if neighbors == 3 {
						births <- Cell{Alive: true, X: x, Y: y}
					}
				}
			}
		}
	}
}

func (gol *GameOfLife) applyChanges(births, deaths <-chan Cell) {
	var birthCount, deathCount int
	
	for cell := range births {
		gol.grid.SetCell(cell.X, cell.Y, true)
		birthCount++
	}
	
	for cell := range deaths {
		gol.grid.SetCell(cell.X, cell.Y, false)
		deathCount++
	}
	
	gol.stats.mutex.Lock()
	gol.stats.BirthCount = birthCount
	gol.stats.DeathCount = deathCount
	gol.stats.mutex.Unlock()
}

func (gol *GameOfLife) divideIntoRegions() []Region {
	regions := make([]Region, 0, gol.numWorkers)
	rowsPerWorker := gol.grid.height / gol.numWorkers
	
	for i := 0; i < gol.numWorkers; i++ {
		startY := i * rowsPerWorker
		endY := startY + rowsPerWorker
		
		if i == gol.numWorkers-1 {
			endY = gol.grid.height
		}
		
		regions = append(regions, Region{
			startX: 0,
			endX:   gol.grid.width,
			startY: startY,
			endY:   endY,
		})
	}
	
	return regions
}

func (gol *GameOfLife) updateStatistics() {
	aliveCells := gol.grid.CountAliveCells()
	totalCells := gol.grid.width * gol.grid.height
	
	gol.stats.mutex.Lock()
	defer gol.stats.mutex.Unlock()
	
	gol.stats.Generation = gol.generation
	gol.stats.AliveCells = aliveCells
	gol.stats.DeadCells = totalCells - aliveCells
	gol.stats.PopulationHistory = append(gol.stats.PopulationHistory, aliveCells)
	
	if len(gol.stats.PopulationHistory) > 50 {
		gol.stats.PopulationHistory = gol.stats.PopulationHistory[1:]
	}
	
	if len(gol.stats.PopulationHistory) > 10 {
		lastTen := gol.stats.PopulationHistory[len(gol.stats.PopulationHistory)-10:]
		stable := true
		for i := 1; i < len(lastTen); i++ {
			if lastTen[i] != lastTen[0] {
				stable = false
				break
			}
		}
		gol.stats.StablePattern = stable
	}
}

func (gol *GameOfLife) Run(ctx context.Context) {
	gol.mutex.Lock()
	gol.running = true
	gol.mutex.Unlock()
	
	ticker := time.NewTicker(gol.speed)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			gol.mutex.Lock()
			gol.running = false
			gol.mutex.Unlock()
			return
			
		case <-gol.stopChan:
			gol.mutex.Lock()
			gol.running = false
			gol.mutex.Unlock()
			return
			
		case <-gol.pauseChan:
			gol.mutex.Lock()
			gol.paused = true
			gol.mutex.Unlock()
			
			select {
			case <-gol.resumeChan:
				gol.mutex.Lock()
				gol.paused = false
				gol.mutex.Unlock()
			case <-ctx.Done():
				gol.mutex.Lock()
				gol.running = false
				gol.paused = false
				gol.mutex.Unlock()
				return
			case <-gol.stopChan:
				gol.mutex.Lock()
				gol.running = false
				gol.paused = false
				gol.mutex.Unlock()
				return
			}
			
		case <-ticker.C:
			gol.mutex.RLock()
			currentSpeed := gol.speed
			gol.mutex.RUnlock()
			
			if ticker.C != time.NewTicker(currentSpeed).C {
				ticker.Stop()
				ticker = time.NewTicker(currentSpeed)
			}
			
			gol.NextGeneration(ctx)
		}
	}
}

func (gol *GameOfLife) Stop() {
	select {
	case gol.stopChan <- struct{}{}:
	default:
	}
}

func (gol *GameOfLife) Pause() {
	select {
	case gol.pauseChan <- struct{}{}:
	default:
	}
}

func (gol *GameOfLife) Resume() {
	select {
	case gol.resumeChan <- struct{}{}:
	default:
	}
}

func (gol *GameOfLife) GetGrid() *Grid {
	return gol.grid.Copy()
}

func (gol *GameOfLife) GetStatistics() *Statistics {
	gol.stats.mutex.RLock()
	defer gol.stats.mutex.RUnlock()
	
	history := make([]int, len(gol.stats.PopulationHistory))
	copy(history, gol.stats.PopulationHistory)
	
	return &Statistics{
		Generation:        gol.stats.Generation,
		AliveCells:        gol.stats.AliveCells,
		DeadCells:         gol.stats.DeadCells,
		BirthCount:        gol.stats.BirthCount,
		DeathCount:        gol.stats.DeathCount,
		StablePattern:     gol.stats.StablePattern,
		PopulationHistory: history,
	}
}

func (gol *GameOfLife) Reset() {
	gol.mutex.Lock()
	defer gol.mutex.Unlock()
	
	gol.grid.Clear()
	gol.generation = 0
	gol.stats = &Statistics{PopulationHistory: make([]int, 0)}
	gol.updateStatistics()
}

func (gol *GameOfLife) IsRunning() bool {
	gol.mutex.RLock()
	defer gol.mutex.RUnlock()
	return gol.running
}

func (gol *GameOfLife) IsPaused() bool {
	gol.mutex.RLock()
	defer gol.mutex.RUnlock()
	return gol.paused
}

func (gol *GameOfLife) GetGeneration() int {
	gol.mutex.RLock()
	defer gol.mutex.RUnlock()
	return gol.generation
}

func (gol *GameOfLife) GetAvailablePatterns() []string {
	patterns := make([]string, 0, len(gol.patterns))
	for name := range gol.patterns {
		patterns = append(patterns, name)
	}
	return patterns
}

type ConsoleObserver struct {
	showGrid bool
}

func NewConsoleObserver(showGrid bool) *ConsoleObserver {
	return &ConsoleObserver{showGrid: showGrid}
}

func (co *ConsoleObserver) OnGenerationUpdate(generation int, grid *Grid, stats *Statistics) {
	fmt.Printf("Generation %d: Alive=%d, Deaths=%d, Births=%d\n", 
		generation, stats.AliveCells, stats.DeathCount, stats.BirthCount)
	
	if co.showGrid {
		co.printGrid(grid)
	}
	
	if stats.StablePattern {
		fmt.Println("Stable pattern detected!")
	}
}

func (co *ConsoleObserver) printGrid(grid *Grid) {
	grid.mutex.RLock()
	defer grid.mutex.RUnlock()
	
	for y := 0; y < grid.height; y++ {
		for x := 0; x < grid.width; x++ {
			if grid.cells[y][x] {
				fmt.Print("█")
			} else {
				fmt.Print(".")
			}
		}
		fmt.Println()
	}
	fmt.Println()
}

func getBuiltInPatterns() map[string]Pattern {
	patterns := make(map[string]Pattern)
	
	patterns["glider"] = Pattern{
		Name:   "glider",
		Width:  3,
		Height: 3,
		Cells: [][]bool{
			{false, true, false},
			{false, false, true},
			{true, true, true},
		},
	}
	
	patterns["blinker"] = Pattern{
		Name:   "blinker",
		Width:  3,
		Height: 1,
		Cells: [][]bool{
			{true, true, true},
		},
	}
	
	patterns["block"] = Pattern{
		Name:   "block",
		Width:  2,
		Height: 2,
		Cells: [][]bool{
			{true, true},
			{true, true},
		},
	}
	
	patterns["beacon"] = Pattern{
		Name:   "beacon",
		Width:  4,
		Height: 4,
		Cells: [][]bool{
			{true, true, false, false},
			{true, true, false, false},
			{false, false, true, true},
			{false, false, true, true},
		},
	}
	
	patterns["toad"] = Pattern{
		Name:   "toad",
		Width:  4,
		Height: 2,
		Cells: [][]bool{
			{false, true, true, true},
			{true, true, true, false},
		},
	}
	
	patterns["pulsar"] = Pattern{
		Name:   "pulsar",
		Width:  13,
		Height: 13,
		Cells: [][]bool{
			{false, false, true, true, true, false, false, false, true, true, true, false, false},
			{false, false, false, false, false, false, false, false, false, false, false, false, false},
			{true, false, false, false, false, true, false, true, false, false, false, false, true},
			{true, false, false, false, false, true, false, true, false, false, false, false, true},
			{true, false, false, false, false, true, false, true, false, false, false, false, true},
			{false, false, true, true, true, false, false, false, true, true, true, false, false},
			{false, false, false, false, false, false, false, false, false, false, false, false, false},
			{false, false, true, true, true, false, false, false, true, true, true, false, false},
			{true, false, false, false, false, true, false, true, false, false, false, false, true},
			{true, false, false, false, false, true, false, true, false, false, false, false, true},
			{true, false, false, false, false, true, false, true, false, false, false, false, true},
			{false, false, false, false, false, false, false, false, false, false, false, false, false},
			{false, false, true, true, true, false, false, false, true, true, true, false, false},
		},
	}
	
	return patterns
}

type BenchmarkRunner struct {
	width      int
	height     int
	numWorkers int
	generations int
}

func NewBenchmarkRunner(width, height, numWorkers, generations int) *BenchmarkRunner {
	return &BenchmarkRunner{
		width:       width,
		height:      height,
		numWorkers:  numWorkers,
		generations: generations,
	}
}

func (br *BenchmarkRunner) RunBenchmark() time.Duration {
	gol := NewGameOfLife(br.width, br.height, br.numWorkers)
	gol.RandomizeGrid(0.3)
	
	start := time.Now()
	
	ctx := context.Background()
	for i := 0; i < br.generations; i++ {
		gol.NextGeneration(ctx)
	}
	
	return time.Since(start)
}

func ComparePerformance(width, height, generations int, workerCounts []int) map[int]time.Duration {
	results := make(map[int]time.Duration)
	
	for _, workers := range workerCounts {
		br := NewBenchmarkRunner(width, height, workers, generations)
		duration := br.RunBenchmark()
		results[workers] = duration
		fmt.Printf("Workers: %d, Time: %v\n", workers, duration)
	}
	
	return results
}

type ToroidalGrid struct {
	*Grid
}

func NewToroidalGrid(width, height int) *ToroidalGrid {
	return &ToroidalGrid{
		Grid: NewGrid(width, height),
	}
}

func (tg *ToroidalGrid) GetCell(x, y int) bool {
	tg.mutex.RLock()
	defer tg.mutex.RUnlock()
	
	x = ((x % tg.width) + tg.width) % tg.width
	y = ((y % tg.height) + tg.height) % tg.height
	
	return tg.cells[y][x]
}

func (tg *ToroidalGrid) CountNeighbors(x, y int) int {
	count := 0
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			if dx == 0 && dy == 0 {
				continue
			}
			nx, ny := x+dx, y+dy
			if tg.GetCell(nx, ny) {
				count++
			}
		}
	}
	return count
}