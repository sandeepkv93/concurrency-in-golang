package parallelsudokusolver

import (
	"context"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestNewParallelSudokuSolver(t *testing.T) {
	config := SolverConfig{
		Strategy:       BacktrackingStrategy,
		MaxWorkers:     4,
		TimeLimit:      30 * time.Second,
		UseHeuristics:  true,
		EnablePruning:  true,
		RandomSeed:     12345,
	}
	
	solver := NewParallelSudokuSolver(config)
	
	if solver.config.Strategy != BacktrackingStrategy {
		t.Errorf("Expected strategy %v, got %v", BacktrackingStrategy, solver.config.Strategy)
	}
	
	if solver.config.MaxWorkers != 4 {
		t.Errorf("Expected 4 workers, got %d", solver.config.MaxWorkers)
	}
	
	if solver.config.TimeLimit != 30*time.Second {
		t.Errorf("Expected 30s timeout, got %v", solver.config.TimeLimit)
	}
}

func TestDefaultConfiguration(t *testing.T) {
	config := SolverConfig{}
	solver := NewParallelSudokuSolver(config)
	
	if solver.config.MaxWorkers != runtime.NumCPU() {
		t.Errorf("Expected default workers to be %d, got %d", runtime.NumCPU(), solver.config.MaxWorkers)
	}
	
	if solver.config.TimeLimit != 30*time.Second {
		t.Errorf("Expected default timeout 30s, got %v", solver.config.TimeLimit)
	}
}

func TestIsValidBoard(t *testing.T) {
	solver := NewParallelSudokuSolver(SolverConfig{})
	
	// Valid empty board
	var emptyBoard SudokuBoard
	if !solver.IsValidBoard(emptyBoard) {
		t.Error("Empty board should be valid")
	}
	
	// Valid partial board
	validBoard := SudokuBoard{
		{5, 3, 0, 0, 7, 0, 0, 0, 0},
		{6, 0, 0, 1, 9, 5, 0, 0, 0},
		{0, 9, 8, 0, 0, 0, 0, 6, 0},
		{8, 0, 0, 0, 6, 0, 0, 0, 3},
		{4, 0, 0, 8, 0, 3, 0, 0, 1},
		{7, 0, 0, 0, 2, 0, 0, 0, 6},
		{0, 6, 0, 0, 0, 0, 2, 8, 0},
		{0, 0, 0, 4, 1, 9, 0, 0, 5},
		{0, 0, 0, 0, 8, 0, 0, 7, 9},
	}
	
	if !solver.IsValidBoard(validBoard) {
		t.Error("Valid board should be valid")
	}
	
	// Invalid board (duplicate in row)
	invalidBoard := validBoard
	invalidBoard[0][2] = 5 // Duplicate 5 in first row
	
	if solver.IsValidBoard(invalidBoard) {
		t.Error("Board with duplicate in row should be invalid")
	}
	
	// Invalid board (duplicate in column)
	invalidBoard = validBoard
	invalidBoard[1][0] = 5 // Duplicate 5 in first column
	
	if solver.IsValidBoard(invalidBoard) {
		t.Error("Board with duplicate in column should be invalid")
	}
	
	// Invalid board (duplicate in box)
	invalidBoard = validBoard
	invalidBoard[1][1] = 5 // Duplicate 5 in top-left box
	
	if solver.IsValidBoard(invalidBoard) {
		t.Error("Board with duplicate in box should be invalid")
	}
}

func TestSolveSimplePuzzle(t *testing.T) {
	config := SolverConfig{
		Strategy:   BacktrackingStrategy,
		MaxWorkers: 2,
		TimeLimit:  10 * time.Second,
	}
	
	solver := NewParallelSudokuSolver(config)
	
	// Simple puzzle (missing only a few numbers)
	puzzle := SudokuBoard{
		{5, 3, 4, 6, 7, 8, 9, 1, 2},
		{6, 7, 2, 1, 9, 5, 3, 4, 8},
		{1, 9, 8, 3, 4, 2, 5, 6, 7},
		{8, 5, 9, 7, 6, 1, 4, 2, 3},
		{4, 2, 6, 8, 5, 3, 7, 9, 1},
		{7, 1, 3, 9, 2, 4, 8, 5, 6},
		{9, 6, 1, 5, 3, 7, 2, 8, 4},
		{2, 8, 7, 4, 1, 9, 6, 3, 5},
		{3, 4, 5, 2, 8, 6, 1, 7, 0}, // Missing last number
	}
	
	ctx := context.Background()
	solution, stats, err := solver.Solve(ctx, puzzle)
	
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}
	
	if solution == nil {
		t.Fatal("Expected solution, got nil")
	}
	
	if !solver.isSolved(*solution) {
		t.Error("Solution is not valid")
	}
	
	if stats.TotalAttempts != 1 {
		t.Errorf("Expected 1 attempt, got %d", stats.TotalAttempts)
	}
	
	if stats.SuccessfulSolves != 1 {
		t.Errorf("Expected 1 successful solve, got %d", stats.SuccessfulSolves)
	}
	
	// Check that the missing number is 9
	if (*solution)[8][8] != 9 {
		t.Errorf("Expected last cell to be 9, got %d", (*solution)[8][8])
	}
}

func TestSolveComplexPuzzle(t *testing.T) {
	config := SolverConfig{
		Strategy:      BacktrackingStrategy,
		MaxWorkers:    4,
		TimeLimit:     30 * time.Second,
		UseHeuristics: true,
	}
	
	solver := NewParallelSudokuSolver(config)
	
	// More complex puzzle
	puzzle := SudokuBoard{
		{5, 3, 0, 0, 7, 0, 0, 0, 0},
		{6, 0, 0, 1, 9, 5, 0, 0, 0},
		{0, 9, 8, 0, 0, 0, 0, 6, 0},
		{8, 0, 0, 0, 6, 0, 0, 0, 3},
		{4, 0, 0, 8, 0, 3, 0, 0, 1},
		{7, 0, 0, 0, 2, 0, 0, 0, 6},
		{0, 6, 0, 0, 0, 0, 2, 8, 0},
		{0, 0, 0, 4, 1, 9, 0, 0, 5},
		{0, 0, 0, 0, 8, 0, 0, 7, 9},
	}
	
	ctx := context.Background()
	solution, stats, err := solver.Solve(ctx, puzzle)
	
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}
	
	if solution == nil {
		t.Fatal("Expected solution, got nil")
	}
	
	if !solver.isSolved(*solution) {
		t.Error("Solution is not valid")
	}
	
	if stats.TotalAttempts == 0 {
		t.Error("Expected at least one attempt")
	}
	
	t.Logf("Solved puzzle in %v with %d backtrack operations", 
		stats.AverageSolveTime, stats.BacktrackCount)
}

func TestDifferentStrategies(t *testing.T) {
	puzzle := SudokuBoard{
		{5, 3, 0, 0, 7, 0, 0, 0, 0},
		{6, 0, 0, 1, 9, 5, 0, 0, 0},
		{0, 9, 8, 0, 0, 0, 0, 6, 0},
		{8, 0, 0, 0, 6, 0, 0, 0, 3},
		{4, 0, 0, 8, 0, 3, 0, 0, 1},
		{7, 0, 0, 0, 2, 0, 0, 0, 6},
		{0, 6, 0, 0, 0, 0, 2, 8, 0},
		{0, 0, 0, 4, 1, 9, 0, 0, 5},
		{0, 0, 0, 0, 8, 0, 0, 7, 9},
	}
	
	strategies := []SolverStrategy{
		BacktrackingStrategy,
		ConstraintPropagationStrategy,
		HybridStrategy,
	}
	
	for _, strategy := range strategies {
		t.Run(strategyToString(strategy), func(t *testing.T) {
			config := SolverConfig{
				Strategy:   strategy,
				MaxWorkers: 2,
				TimeLimit:  15 * time.Second,
			}
			
			solver := NewParallelSudokuSolver(config)
			
			ctx := context.Background()
			solution, stats, err := solver.Solve(ctx, puzzle)
			
			if err != nil {
				t.Fatalf("Solve failed with strategy %v: %v", strategy, err)
			}
			
			if solution == nil {
				t.Fatalf("Expected solution with strategy %v, got nil", strategy)
			}
			
			if !solver.isSolved(*solution) {
				t.Errorf("Solution is not valid for strategy %v", strategy)
			}
			
			if stats.Strategy != strategy {
				t.Errorf("Expected strategy %v in stats, got %v", strategy, stats.Strategy)
			}
		})
	}
}

func TestConcurrentSolving(t *testing.T) {
	puzzle := SudokuBoard{
		{5, 3, 0, 0, 7, 0, 0, 0, 0},
		{6, 0, 0, 1, 9, 5, 0, 0, 0},
		{0, 9, 8, 0, 0, 0, 0, 6, 0},
		{8, 0, 0, 0, 6, 0, 0, 0, 3},
		{4, 0, 0, 8, 0, 3, 0, 0, 1},
		{7, 0, 0, 0, 2, 0, 0, 0, 6},
		{0, 6, 0, 0, 0, 0, 2, 8, 0},
		{0, 0, 0, 4, 1, 9, 0, 0, 5},
		{0, 0, 0, 0, 8, 0, 0, 7, 9},
	}
	
	config := SolverConfig{
		Strategy:   BacktrackingStrategy,
		MaxWorkers: 4,
		TimeLimit:  10 * time.Second,
	}
	
	numGoroutines := 10
	var wg sync.WaitGroup
	var successCount int32
	var failCount int32
	
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			solver := NewParallelSudokuSolver(config)
			ctx := context.Background()
			
			solution, _, err := solver.Solve(ctx, puzzle)
			
			if err != nil || solution == nil || !solver.isSolved(*solution) {
				failCount++
			} else {
				successCount++
			}
		}()
	}
	
	wg.Wait()
	
	if successCount == 0 {
		t.Error("Expected at least one successful concurrent solve")
	}
	
	t.Logf("Concurrent solving: %d successes, %d failures", successCount, failCount)
}

func TestSolverTimeout(t *testing.T) {
	// Create a very difficult puzzle or use short timeout
	puzzle := SudokuBoard{
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	
	config := SolverConfig{
		Strategy:   BacktrackingStrategy,
		MaxWorkers: 2,
		TimeLimit:  100 * time.Millisecond, // Very short timeout
	}
	
	solver := NewParallelSudokuSolver(config)
	
	ctx := context.Background()
	start := time.Now()
	solution, _, err := solver.Solve(ctx, puzzle)
	duration := time.Since(start)
	
	// Should timeout or complete quickly
	if duration > 5*time.Second {
		t.Errorf("Solver took too long: %v", duration)
	}
	
	// Either should timeout (error) or find solution quickly
	if err == nil && solution != nil {
		if !solver.isSolved(*solution) {
			t.Error("Solution is invalid")
		}
		t.Log("Solver found solution quickly")
	} else {
		t.Log("Solver timed out as expected")
	}
}

func TestContextCancellation(t *testing.T) {
	puzzle := SudokuBoard{
		{5, 3, 0, 0, 7, 0, 0, 0, 0},
		{6, 0, 0, 1, 9, 5, 0, 0, 0},
		{0, 9, 8, 0, 0, 0, 0, 6, 0},
		{8, 0, 0, 0, 6, 0, 0, 0, 3},
		{4, 0, 0, 8, 0, 3, 0, 0, 1},
		{7, 0, 0, 0, 2, 0, 0, 0, 6},
		{0, 6, 0, 0, 0, 0, 2, 8, 0},
		{0, 0, 0, 4, 1, 9, 0, 0, 5},
		{0, 0, 0, 0, 8, 0, 0, 7, 9},
	}
	
	config := SolverConfig{
		Strategy:   BacktrackingStrategy,
		MaxWorkers: 4,
		TimeLimit:  30 * time.Second,
	}
	
	solver := NewParallelSudokuSolver(config)
	
	ctx, cancel := context.WithCancel(context.Background())
	
	// Start solving in a goroutine
	var solution *SudokuBoard
	var err error
	done := make(chan struct{})
	
	go func() {
		solution, _, err = solver.Solve(ctx, puzzle)
		close(done)
	}()
	
	// Cancel after a short delay
	time.Sleep(50 * time.Millisecond)
	cancel()
	
	// Wait for completion
	select {
	case <-done:
		// Should have been cancelled
		if err != context.Canceled && solution != nil {
			// Sometimes the solver completes before cancellation
			t.Log("Solver completed before cancellation")
		}
	case <-time.After(5 * time.Second):
		t.Error("Solver did not respond to cancellation")
	}
}

func TestStatisticsTracking(t *testing.T) {
	config := SolverConfig{
		Strategy:      BacktrackingStrategy,
		MaxWorkers:    2,
		TimeLimit:     10 * time.Second,
		UseHeuristics: true,
	}
	
	solver := NewParallelSudokuSolver(config)
	
	puzzle := SudokuBoard{
		{5, 3, 0, 0, 7, 0, 0, 0, 0},
		{6, 0, 0, 1, 9, 5, 0, 0, 0},
		{0, 9, 8, 0, 0, 0, 0, 6, 0},
		{8, 0, 0, 0, 6, 0, 0, 0, 3},
		{4, 0, 0, 8, 0, 3, 0, 0, 1},
		{7, 0, 0, 0, 2, 0, 0, 0, 6},
		{0, 6, 0, 0, 0, 0, 2, 8, 0},
		{0, 0, 0, 4, 1, 9, 0, 0, 5},
		{0, 0, 0, 0, 8, 0, 0, 7, 9},
	}
	
	ctx := context.Background()
	_, stats, err := solver.Solve(ctx, puzzle)
	
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}
	
	if stats.TotalAttempts == 0 {
		t.Error("Expected at least one attempt recorded")
	}
	
	if stats.WorkersUsed != 2 {
		t.Errorf("Expected 2 workers used, got %d", stats.WorkersUsed)
	}
	
	if stats.AverageSolveTime <= 0 {
		t.Error("Expected positive solve time")
	}
	
	// Test statistics accumulation
	initialStats := solver.GetStats()
	
	_, _, err = solver.Solve(ctx, puzzle)
	if err != nil {
		t.Fatalf("Second solve failed: %v", err)
	}
	
	finalStats := solver.GetStats()
	
	if finalStats.TotalAttempts <= initialStats.TotalAttempts {
		t.Error("Expected total attempts to increase")
	}
}

func TestPuzzleGeneration(t *testing.T) {
	difficulties := []DifficultyLevel{Easy, Medium, Hard, Expert}
	
	for _, difficulty := range difficulties {
		t.Run(fmt.Sprintf("Difficulty_%d", difficulty), func(t *testing.T) {
			puzzle := GeneratePuzzle(difficulty, 12345)
			
			solver := NewParallelSudokuSolver(SolverConfig{
				Strategy:   HybridStrategy,
				MaxWorkers: 2,
				TimeLimit:  30 * time.Second,
			})
			
			if !solver.IsValidBoard(puzzle) {
				t.Error("Generated puzzle is not valid")
			}
			
			// Count empty cells
			emptyCells := 0
			for i := 0; i < 9; i++ {
				for j := 0; j < 9; j++ {
					if puzzle[i][j] == 0 {
						emptyCells++
					}
				}
			}
			
			expectedEmpty := getCellsToRemove(difficulty)
			if emptyCells < expectedEmpty-10 || emptyCells > expectedEmpty+10 {
				t.Errorf("Expected around %d empty cells for difficulty %d, got %d",
					expectedEmpty, difficulty, emptyCells)
			}
			
			// Try to solve the generated puzzle
			ctx := context.Background()
			solution, _, err := solver.Solve(ctx, puzzle)
			
			if err != nil {
				t.Errorf("Failed to solve generated puzzle: %v", err)
			}
			
			if solution == nil {
				t.Error("Expected solution for generated puzzle")
			}
			
			if solution != nil && !solver.isSolved(*solution) {
				t.Error("Generated puzzle solution is invalid")
			}
		})
	}
}

func TestBoardStringRepresentation(t *testing.T) {
	board := SudokuBoard{
		{5, 3, 4, 6, 7, 8, 9, 1, 2},
		{6, 7, 2, 1, 9, 5, 3, 4, 8},
		{1, 9, 8, 3, 4, 2, 5, 6, 7},
		{8, 5, 9, 7, 6, 1, 4, 2, 3},
		{4, 2, 6, 8, 5, 3, 7, 9, 1},
		{7, 1, 3, 9, 2, 4, 8, 5, 6},
		{9, 6, 1, 5, 3, 7, 2, 8, 4},
		{2, 8, 7, 4, 1, 9, 6, 3, 5},
		{3, 4, 5, 2, 8, 6, 1, 7, 0},
	}
	
	str := board.String()
	if len(str) == 0 {
		t.Error("Expected non-empty string representation")
	}
	
	prettyStr := board.PrettyString()
	if len(prettyStr) == 0 {
		t.Error("Expected non-empty pretty string representation")
	}
	
	// Check that string contains expected characters
	if !contains(str, ".") { // Should contain dots for zeros
		t.Error("String representation should contain dots for empty cells")
	}
	
	if !contains(prettyStr, "┌") { // Should contain box drawing characters
		t.Error("Pretty string should contain box drawing characters")
	}
}

func TestConstraintPropagation(t *testing.T) {
	solver := NewParallelSudokuSolver(SolverConfig{UseHeuristics: true})
	
	board := SudokuBoard{
		{5, 3, 4, 6, 7, 8, 9, 1, 2},
		{6, 7, 2, 1, 9, 5, 3, 4, 8},
		{1, 9, 8, 3, 4, 2, 5, 6, 7},
		{8, 5, 9, 7, 6, 1, 4, 2, 3},
		{4, 2, 6, 8, 5, 3, 7, 9, 1},
		{7, 1, 3, 9, 2, 4, 8, 5, 6},
		{9, 6, 1, 5, 3, 7, 2, 8, 4},
		{2, 8, 7, 4, 1, 9, 6, 3, 5},
		{3, 4, 5, 2, 8, 6, 1, 7, 0}, // Only one cell missing
	}
	
	propagatedBoard := solver.applyConstraintPropagation(board)
	
	// Should have filled the missing cell
	if propagatedBoard[8][8] != 9 {
		t.Errorf("Expected constraint propagation to fill cell with 9, got %d", propagatedBoard[8][8])
	}
}

func TestPossibleValues(t *testing.T) {
	solver := NewParallelSudokuSolver(SolverConfig{})
	
	board := SudokuBoard{
		{5, 3, 4, 6, 7, 8, 9, 1, 2},
		{6, 7, 2, 1, 9, 5, 3, 4, 8},
		{1, 9, 8, 3, 4, 2, 5, 6, 7},
		{8, 5, 9, 7, 6, 1, 4, 2, 3},
		{4, 2, 6, 8, 5, 3, 7, 9, 1},
		{7, 1, 3, 9, 2, 4, 8, 5, 6},
		{9, 6, 1, 5, 3, 7, 2, 8, 4},
		{2, 8, 7, 4, 1, 9, 6, 3, 5},
		{3, 4, 5, 2, 8, 6, 1, 7, 0}, // Only 9 is possible
	}
	
	possibleValues := solver.getPossibleValues(board, 8, 8)
	
	if len(possibleValues) != 1 || possibleValues[0] != 9 {
		t.Errorf("Expected only value 9 possible, got %v", possibleValues)
	}
	
	// Test filled cell
	possibleValues = solver.getPossibleValues(board, 0, 0)
	if len(possibleValues) != 1 || possibleValues[0] != 5 {
		t.Errorf("Expected filled cell to return its value, got %v", possibleValues)
	}
}

func TestSolverReset(t *testing.T) {
	config := SolverConfig{
		Strategy:   BacktrackingStrategy,
		MaxWorkers: 2,
		TimeLimit:  10 * time.Second,
	}
	
	solver := NewParallelSudokuSolver(config)
	
	puzzle := SudokuBoard{
		{5, 3, 4, 6, 7, 8, 9, 1, 2},
		{6, 7, 2, 1, 9, 5, 3, 4, 8},
		{1, 9, 8, 3, 4, 2, 5, 6, 7},
		{8, 5, 9, 7, 6, 1, 4, 2, 3},
		{4, 2, 6, 8, 5, 3, 7, 9, 1},
		{7, 1, 3, 9, 2, 4, 8, 5, 6},
		{9, 6, 1, 5, 3, 7, 2, 8, 4},
		{2, 8, 7, 4, 1, 9, 6, 3, 5},
		{3, 4, 5, 2, 8, 6, 1, 7, 0},
	}
	
	ctx := context.Background()
	_, _, err := solver.Solve(ctx, puzzle)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}
	
	statsBeforeReset := solver.GetStats()
	if statsBeforeReset.TotalAttempts == 0 {
		t.Error("Expected some attempts before reset")
	}
	
	solver.Reset()
	
	statsAfterReset := solver.GetStats()
	if statsAfterReset.TotalAttempts != 0 {
		t.Error("Expected stats to be reset")
	}
	
	if statsAfterReset.Strategy != config.Strategy {
		t.Error("Expected strategy to be preserved after reset")
	}
}

func TestWorkerCounts(t *testing.T) {
	puzzle := SudokuBoard{
		{5, 3, 0, 0, 7, 0, 0, 0, 0},
		{6, 0, 0, 1, 9, 5, 0, 0, 0},
		{0, 9, 8, 0, 0, 0, 0, 6, 0},
		{8, 0, 0, 0, 6, 0, 0, 0, 3},
		{4, 0, 0, 8, 0, 3, 0, 0, 1},
		{7, 0, 0, 0, 2, 0, 0, 0, 6},
		{0, 6, 0, 0, 0, 0, 2, 8, 0},
		{0, 0, 0, 4, 1, 9, 0, 0, 5},
		{0, 0, 0, 0, 8, 0, 0, 7, 9},
	}
	
	workerCounts := []int{1, 2, 4, 8}
	
	for _, workers := range workerCounts {
		t.Run(fmt.Sprintf("Workers_%d", workers), func(t *testing.T) {
			config := SolverConfig{
				Strategy:   BacktrackingStrategy,
				MaxWorkers: workers,
				TimeLimit:  15 * time.Second,
			}
			
			solver := NewParallelSudokuSolver(config)
			
			ctx := context.Background()
			solution, stats, err := solver.Solve(ctx, puzzle)
			
			if err != nil {
				t.Fatalf("Solve failed with %d workers: %v", workers, err)
			}
			
			if solution == nil {
				t.Fatalf("Expected solution with %d workers", workers)
			}
			
			if !solver.isSolved(*solution) {
				t.Errorf("Solution is invalid with %d workers", workers)
			}
			
			if stats.WorkersUsed != workers {
				t.Errorf("Expected %d workers in stats, got %d", workers, stats.WorkersUsed)
			}
		})
	}
}

func BenchmarkSolveSimple(b *testing.B) {
	puzzle := SudokuBoard{
		{5, 3, 4, 6, 7, 8, 9, 1, 2},
		{6, 7, 2, 1, 9, 5, 3, 4, 8},
		{1, 9, 8, 3, 4, 2, 5, 6, 7},
		{8, 5, 9, 7, 6, 1, 4, 2, 3},
		{4, 2, 6, 8, 5, 3, 7, 9, 1},
		{7, 1, 3, 9, 2, 4, 8, 5, 6},
		{9, 6, 1, 5, 3, 7, 2, 8, 4},
		{2, 8, 7, 4, 1, 9, 6, 3, 5},
		{3, 4, 5, 2, 8, 6, 1, 7, 0},
	}
	
	config := SolverConfig{
		Strategy:   BacktrackingStrategy,
		MaxWorkers: runtime.NumCPU(),
		TimeLimit:  30 * time.Second,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		solver := NewParallelSudokuSolver(config)
		ctx := context.Background()
		
		solution, _, err := solver.Solve(ctx, puzzle)
		if err != nil || solution == nil {
			b.Fatalf("Benchmark solve failed: %v", err)
		}
	}
}

func BenchmarkSolveComplex(b *testing.B) {
	puzzle := SudokuBoard{
		{5, 3, 0, 0, 7, 0, 0, 0, 0},
		{6, 0, 0, 1, 9, 5, 0, 0, 0},
		{0, 9, 8, 0, 0, 0, 0, 6, 0},
		{8, 0, 0, 0, 6, 0, 0, 0, 3},
		{4, 0, 0, 8, 0, 3, 0, 0, 1},
		{7, 0, 0, 0, 2, 0, 0, 0, 6},
		{0, 6, 0, 0, 0, 0, 2, 8, 0},
		{0, 0, 0, 4, 1, 9, 0, 0, 5},
		{0, 0, 0, 0, 8, 0, 0, 7, 9},
	}
	
	config := SolverConfig{
		Strategy:      HybridStrategy,
		MaxWorkers:    runtime.NumCPU(),
		TimeLimit:     30 * time.Second,
		UseHeuristics: true,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		solver := NewParallelSudokuSolver(config)
		ctx := context.Background()
		
		solver.Solve(ctx, puzzle)
		// Note: Some puzzles might not solve within time limit in benchmark mode
	}
}

func BenchmarkConstraintPropagation(b *testing.B) {
	solver := NewParallelSudokuSolver(SolverConfig{UseHeuristics: true})
	
	board := SudokuBoard{
		{5, 3, 0, 0, 7, 0, 0, 0, 0},
		{6, 0, 0, 1, 9, 5, 0, 0, 0},
		{0, 9, 8, 0, 0, 0, 0, 6, 0},
		{8, 0, 0, 0, 6, 0, 0, 0, 3},
		{4, 0, 0, 8, 0, 3, 0, 0, 1},
		{7, 0, 0, 0, 2, 0, 0, 0, 6},
		{0, 6, 0, 0, 0, 0, 2, 8, 0},
		{0, 0, 0, 4, 1, 9, 0, 0, 5},
		{0, 0, 0, 0, 8, 0, 0, 7, 9},
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		solver.applyConstraintPropagation(board)
	}
}

// Helper function
func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		(s == substr || len(substr) == 0 || 
		(len(s) > len(substr) && (s[:len(substr)] == substr || 
		s[len(s)-len(substr):] == substr || 
		func() bool {
			for i := 1; i <= len(s)-len(substr); i++ {
				if s[i:i+len(substr)] == substr {
					return true
				}
			}
			return false
		}())))
}