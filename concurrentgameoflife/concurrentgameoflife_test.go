package concurrentgameoflife

import (
	"context"
	"testing"
	"time"
)

func TestNewGrid(t *testing.T) {
	width, height := 10, 8
	grid := NewGrid(width, height)
	
	if grid.width != width {
		t.Errorf("Expected width %d, got %d", width, grid.width)
	}
	
	if grid.height != height {
		t.Errorf("Expected height %d, got %d", height, grid.height)
	}
	
	if grid.CountAliveCells() != 0 {
		t.Error("Expected empty grid to have no alive cells")
	}
}

func TestGridSetGetCell(t *testing.T) {
	grid := NewGrid(5, 5)
	
	grid.SetCell(2, 2, true)
	if !grid.GetCell(2, 2) {
		t.Error("Expected cell (2,2) to be alive")
	}
	
	grid.SetCell(2, 2, false)
	if grid.GetCell(2, 2) {
		t.Error("Expected cell (2,2) to be dead")
	}
	
	if grid.GetCell(-1, -1) {
		t.Error("Expected out-of-bounds cell to be dead")
	}
	
	if grid.GetCell(10, 10) {
		t.Error("Expected out-of-bounds cell to be dead")
	}
}

func TestCountNeighbors(t *testing.T) {
	grid := NewGrid(5, 5)
	
	grid.SetCell(1, 1, true)
	grid.SetCell(1, 2, true)
	grid.SetCell(2, 1, true)
	
	neighbors := grid.CountNeighbors(2, 2)
	if neighbors != 3 {
		t.Errorf("Expected 3 neighbors, got %d", neighbors)
	}
	
	neighbors = grid.CountNeighbors(0, 0)
	if neighbors != 1 {
		t.Errorf("Expected 1 neighbor, got %d", neighbors)
	}
}

func TestGridCountAliveCells(t *testing.T) {
	grid := NewGrid(3, 3)
	
	if grid.CountAliveCells() != 0 {
		t.Error("Expected 0 alive cells in empty grid")
	}
	
	grid.SetCell(0, 0, true)
	grid.SetCell(1, 1, true)
	grid.SetCell(2, 2, true)
	
	if grid.CountAliveCells() != 3 {
		t.Error("Expected 3 alive cells")
	}
}

func TestGridClear(t *testing.T) {
	grid := NewGrid(3, 3)
	
	grid.SetCell(0, 0, true)
	grid.SetCell(1, 1, true)
	grid.SetCell(2, 2, true)
	
	grid.Clear()
	
	if grid.CountAliveCells() != 0 {
		t.Error("Expected 0 alive cells after clear")
	}
}

func TestGridCopy(t *testing.T) {
	grid := NewGrid(3, 3)
	grid.SetCell(1, 1, true)
	
	copy := grid.Copy()
	
	if !copy.GetCell(1, 1) {
		t.Error("Expected copied grid to have alive cell at (1,1)")
	}
	
	copy.SetCell(0, 0, true)
	if grid.GetCell(0, 0) {
		t.Error("Expected original grid to be unaffected by copy modification")
	}
}

func TestGridEquals(t *testing.T) {
	grid1 := NewGrid(3, 3)
	grid2 := NewGrid(3, 3)
	
	if !grid1.Equals(grid2) {
		t.Error("Expected empty grids to be equal")
	}
	
	grid1.SetCell(1, 1, true)
	if grid1.Equals(grid2) {
		t.Error("Expected grids with different states to be unequal")
	}
	
	grid2.SetCell(1, 1, true)
	if !grid1.Equals(grid2) {
		t.Error("Expected grids with same states to be equal")
	}
	
	grid3 := NewGrid(4, 4)
	if grid1.Equals(grid3) {
		t.Error("Expected grids with different sizes to be unequal")
	}
}

func TestNewGameOfLife(t *testing.T) {
	gol := NewGameOfLife(10, 10, 4)
	
	if gol.grid.width != 10 || gol.grid.height != 10 {
		t.Error("Expected 10x10 grid")
	}
	
	if gol.numWorkers != 4 {
		t.Error("Expected 4 workers")
	}
	
	if gol.generation != 0 {
		t.Error("Expected initial generation to be 0")
	}
	
	if gol.running {
		t.Error("Expected game to not be running initially")
	}
}

func TestRandomizeGrid(t *testing.T) {
	gol := NewGameOfLife(10, 10, 2)
	
	gol.RandomizeGrid(0.5)
	aliveCells := gol.grid.CountAliveCells()
	
	if aliveCells == 0 {
		t.Error("Expected some alive cells after randomization")
	}
	
	if aliveCells == 100 {
		t.Error("Expected some dead cells after randomization")
	}
}

func TestLoadPattern(t *testing.T) {
	gol := NewGameOfLife(10, 10, 2)
	
	err := gol.LoadPattern("glider", 1, 1)
	if err != nil {
		t.Errorf("Expected to load glider pattern successfully, got error: %v", err)
	}
	
	if !gol.grid.GetCell(2, 1) {
		t.Error("Expected glider pattern to set cell (2,1) alive")
	}
	
	err = gol.LoadPattern("nonexistent", 0, 0)
	if err == nil {
		t.Error("Expected error when loading nonexistent pattern")
	}
}

func TestGetAvailablePatterns(t *testing.T) {
	gol := NewGameOfLife(10, 10, 2)
	patterns := gol.GetAvailablePatterns()
	
	if len(patterns) == 0 {
		t.Error("Expected built-in patterns to be available")
	}
	
	found := false
	for _, pattern := range patterns {
		if pattern == "glider" {
			found = true
			break
		}
	}
	
	if !found {
		t.Error("Expected 'glider' pattern to be available")
	}
}

func TestNextGeneration(t *testing.T) {
	gol := NewGameOfLife(5, 5, 2)
	
	gol.LoadPattern("blinker", 1, 2)
	
	initialGeneration := gol.GetGeneration()
	ctx := context.Background()
	gol.NextGeneration(ctx)
	
	if gol.GetGeneration() != initialGeneration+1 {
		t.Error("Expected generation to increment")
	}
	
	stats := gol.GetStatistics()
	if stats.Generation != 1 {
		t.Error("Expected statistics generation to be 1")
	}
}

func TestBlinkerPattern(t *testing.T) {
	gol := NewGameOfLife(5, 5, 2)
	gol.LoadPattern("blinker", 1, 2)
	
	initialGrid := gol.GetGrid()
	
	ctx := context.Background()
	gol.NextGeneration(ctx)
	gol.NextGeneration(ctx)
	
	finalGrid := gol.GetGrid()
	
	if !initialGrid.Equals(finalGrid) {
		t.Error("Expected blinker to return to original state after 2 generations")
	}
}

func TestGameOfLifeRules(t *testing.T) {
	gol := NewGameOfLife(5, 5, 2)
	
	gol.grid.SetCell(1, 1, true)
	gol.grid.SetCell(1, 2, true)
	gol.grid.SetCell(2, 1, true)
	
	ctx := context.Background()
	gol.NextGeneration(ctx)
	
	if !gol.grid.GetCell(2, 2) {
		t.Error("Expected cell (2,2) to be born (3 neighbors)")
	}
	
	if !gol.grid.GetCell(1, 1) {
		t.Error("Expected cell (1,1) to survive (2 neighbors)")
	}
}

func TestObserver(t *testing.T) {
	gol := NewGameOfLife(5, 5, 2)
	observer := NewConsoleObserver(false)
	gol.AddObserver(observer)
	
	gol.LoadPattern("glider", 0, 0)
	
	ctx := context.Background()
	gol.NextGeneration(ctx)
	
	gol.RemoveObserver(observer)
	gol.NextGeneration(ctx)
}

func TestStatistics(t *testing.T) {
	gol := NewGameOfLife(5, 5, 2)
	gol.LoadPattern("block", 1, 1)
	
	stats := gol.GetStatistics()
	if stats.AliveCells != 4 {
		t.Errorf("Expected 4 alive cells, got %d", stats.AliveCells)
	}
	
	if stats.DeadCells != 21 {
		t.Errorf("Expected 21 dead cells, got %d", stats.DeadCells)
	}
	
	ctx := context.Background()
	gol.NextGeneration(ctx)
	
	stats = gol.GetStatistics()
	if len(stats.PopulationHistory) != 2 {
		t.Error("Expected population history to track generations")
	}
}

func TestStablePatternDetection(t *testing.T) {
	gol := NewGameOfLife(5, 5, 2)
	gol.LoadPattern("block", 1, 1)
	
	ctx := context.Background()
	
	for i := 0; i < 15; i++ {
		gol.NextGeneration(ctx)
	}
	
	stats := gol.GetStatistics()
	if !stats.StablePattern {
		t.Error("Expected stable pattern to be detected for block")
	}
}

func TestGameControl(t *testing.T) {
	gol := NewGameOfLife(10, 10, 2)
	gol.RandomizeGrid(0.3)
	
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	go gol.Run(ctx)
	
	time.Sleep(50 * time.Millisecond)
	
	if !gol.IsRunning() {
		t.Error("Expected game to be running")
	}
	
	gol.Pause()
	time.Sleep(10 * time.Millisecond)
	
	if !gol.IsPaused() {
		t.Error("Expected game to be paused")
	}
	
	gol.Resume()
	time.Sleep(10 * time.Millisecond)
	
	if gol.IsPaused() {
		t.Error("Expected game to be resumed")
	}
	
	gol.Stop()
	time.Sleep(10 * time.Millisecond)
	
	if gol.IsRunning() {
		t.Error("Expected game to be stopped")
	}
}

func TestReset(t *testing.T) {
	gol := NewGameOfLife(5, 5, 2)
	gol.LoadPattern("glider", 0, 0)
	
	ctx := context.Background()
	gol.NextGeneration(ctx)
	gol.NextGeneration(ctx)
	
	gol.Reset()
	
	if gol.GetGeneration() != 0 {
		t.Error("Expected generation to be reset to 0")
	}
	
	if gol.grid.CountAliveCells() != 0 {
		t.Error("Expected grid to be cleared after reset")
	}
	
	stats := gol.GetStatistics()
	if len(stats.PopulationHistory) != 1 {
		t.Error("Expected population history to be reset")
	}
}

func TestSetSpeed(t *testing.T) {
	gol := NewGameOfLife(5, 5, 2)
	
	newSpeed := 50 * time.Millisecond
	gol.SetSpeed(newSpeed)
	
	if gol.speed != newSpeed {
		t.Error("Expected speed to be updated")
	}
}

func TestToroidalGrid(t *testing.T) {
	tg := NewToroidalGrid(3, 3)
	
	tg.SetCell(0, 0, true)
	
	if !tg.GetCell(-3, -3) {
		t.Error("Expected toroidal wrap-around to find cell at (-3,-3)")
	}
	
	if !tg.GetCell(3, 3) {
		t.Error("Expected toroidal wrap-around to find cell at (3,3)")
	}
	
	tg.SetCell(0, 1, true)
	tg.SetCell(1, 0, true)
	
	neighbors := tg.CountNeighbors(2, 2)
	if neighbors != 3 {
		t.Errorf("Expected 3 neighbors with toroidal wrap, got %d", neighbors)
	}
}

func TestConcurrentAccess(t *testing.T) {
	gol := NewGameOfLife(20, 20, 4)
	gol.RandomizeGrid(0.3)
	
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	
	go gol.Run(ctx)
	
	for i := 0; i < 10; i++ {
		go func() {
			gol.GetStatistics()
			gol.GetGrid()
			gol.GetGeneration()
		}()
	}
	
	time.Sleep(200 * time.Millisecond)
	
	gol.LoadPattern("glider", 5, 5)
	
	time.Sleep(200 * time.Millisecond)
	
	gol.Stop()
}

func TestBenchmarkRunner(t *testing.T) {
	br := NewBenchmarkRunner(20, 20, 4, 10)
	duration := br.RunBenchmark()
	
	if duration <= 0 {
		t.Error("Expected positive benchmark duration")
	}
}

func TestComparePerformance(t *testing.T) {
	results := ComparePerformance(10, 10, 5, []int{1, 2})
	
	if len(results) != 2 {
		t.Error("Expected results for 2 worker configurations")
	}
	
	for workers, duration := range results {
		if duration <= 0 {
			t.Errorf("Expected positive duration for %d workers", workers)
		}
	}
}

func TestPatternValidation(t *testing.T) {
	patterns := getBuiltInPatterns()
	
	expectedPatterns := []string{"glider", "blinker", "block", "beacon", "toad", "pulsar"}
	
	for _, expected := range expectedPatterns {
		if _, exists := patterns[expected]; !exists {
			t.Errorf("Expected pattern '%s' to exist", expected)
		}
	}
	
	glider := patterns["glider"]
	if glider.Width != 3 || glider.Height != 3 {
		t.Error("Expected glider to be 3x3")
	}
	
	if !glider.Cells[0][1] {
		t.Error("Expected glider pattern to have correct shape")
	}
}

func TestCellStruct(t *testing.T) {
	cell := Cell{Alive: true, X: 5, Y: 10}
	
	if !cell.Alive {
		t.Error("Expected cell to be alive")
	}
	
	if cell.X != 5 || cell.Y != 10 {
		t.Error("Expected cell coordinates to be set correctly")
	}
}

func TestGetAliveCells(t *testing.T) {
	grid := NewGrid(5, 5)
	grid.SetCell(1, 1, true)
	grid.SetCell(2, 2, true)
	grid.SetCell(3, 3, true)
	
	aliveCells := grid.GetAliveCells()
	
	if len(aliveCells) != 3 {
		t.Errorf("Expected 3 alive cells, got %d", len(aliveCells))
	}
	
	for _, cell := range aliveCells {
		if !cell.Alive {
			t.Error("Expected all returned cells to be alive")
		}
	}
}

func BenchmarkNextGeneration(b *testing.B) {
	gol := NewGameOfLife(50, 50, 4)
	gol.RandomizeGrid(0.3)
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gol.NextGeneration(ctx)
	}
}

func BenchmarkGridOperations(b *testing.B) {
	grid := NewGrid(100, 100)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grid.SetCell(i%100, (i/100)%100, true)
		grid.GetCell(i%100, (i/100)%100)
		grid.CountNeighbors(i%100, (i/100)%100)
	}
}

func BenchmarkConcurrentVsSequential(b *testing.B) {
	sizes := []int{20, 50, 100}
	workers := []int{1, 2, 4, 8}
	
	for _, size := range sizes {
		for _, numWorkers := range workers {
			b.Run(fmt.Sprintf("Size%dx%d_Workers%d", size, size, numWorkers), func(b *testing.B) {
				gol := NewGameOfLife(size, size, numWorkers)
				gol.RandomizeGrid(0.3)
				ctx := context.Background()
				
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					gol.NextGeneration(ctx)
				}
			})
		}
	}
}