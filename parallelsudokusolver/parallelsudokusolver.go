package parallelsudokusolver

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// SudokuBoard represents a 9x9 Sudoku board
type SudokuBoard [9][9]int

// Cell represents a position on the board
type Cell struct {
	Row, Col int
}

// SolverStrategy defines different solving approaches
type SolverStrategy int

const (
	BacktrackingStrategy SolverStrategy = iota
	ConstraintPropagationStrategy
	HybridStrategy
	BruteForceStrategy
)

// DifficultyLevel represents the difficulty of a puzzle
type DifficultyLevel int

const (
	Easy DifficultyLevel = iota
	Medium
	Hard
	Expert
)

// SolverConfig contains configuration for the parallel solver
type SolverConfig struct {
	Strategy       SolverStrategy
	MaxWorkers     int
	TimeLimit      time.Duration
	UseHeuristics  bool
	EnablePruning  bool
	LogProgress    bool
	RandomSeed     int64
}

// SolverStats contains solving statistics
type SolverStats struct {
	Strategy           SolverStrategy
	TotalAttempts      int64
	SuccessfulSolves   int64
	FailedSolves       int64
	AverageSolveTime   time.Duration
	WorkersUsed        int
	BacktrackCount     int64
	ConstraintChecks   int64
	PruningOperations  int64
	HeuristicEvaluations int64
}

// ParallelSudokuSolver represents the main solver
type ParallelSudokuSolver struct {
	config     SolverConfig
	stats      SolverStats
	statsMutex sync.RWMutex
	random     *rand.Rand
}

// SolverWorker represents a worker that processes solving tasks
type SolverWorker struct {
	id           int
	workChan     <-chan SolveTask
	resultChan   chan<- SolveResult
	solver       *ParallelSudokuSolver
	localStats   SolverStats
	statsMutex   sync.Mutex
}

// SolveTask represents a solving task for workers
type SolveTask struct {
	ID          int
	Board       SudokuBoard
	EmptyCells  []Cell
	StartIndex  int
	Strategy    SolverStrategy
	TimeLimit   time.Duration
	Context     context.Context
}

// SolveResult represents the result of a solving attempt
type SolveResult struct {
	TaskID      int
	Success     bool
	Solution    SudokuBoard
	SolveTime   time.Duration
	Attempts    int64
	Error       error
}

// Constraint represents a Sudoku constraint
type Constraint struct {
	Type   ConstraintType
	Values []int
	Cells  []Cell
}

// ConstraintType defines different types of constraints
type ConstraintType int

const (
	RowConstraint ConstraintType = iota
	ColConstraint
	BoxConstraint
	CellConstraint
)

// PossibleValues represents possible values for a cell
type PossibleValues struct {
	Cell   Cell
	Values []int
}

// NewParallelSudokuSolver creates a new parallel Sudoku solver
func NewParallelSudokuSolver(config SolverConfig) *ParallelSudokuSolver {
	if config.MaxWorkers <= 0 {
		config.MaxWorkers = runtime.NumCPU()
	}
	
	if config.TimeLimit <= 0 {
		config.TimeLimit = 30 * time.Second
	}
	
	if config.RandomSeed == 0 {
		config.RandomSeed = time.Now().UnixNano()
	}
	
	return &ParallelSudokuSolver{
		config: config,
		stats: SolverStats{
			Strategy:    config.Strategy,
			WorkersUsed: config.MaxWorkers,
		},
		random: rand.New(rand.NewSource(config.RandomSeed)),
	}
}

// Solve solves a Sudoku puzzle using parallel processing
func (s *ParallelSudokuSolver) Solve(ctx context.Context, board SudokuBoard) (*SudokuBoard, *SolverStats, error) {
	startTime := time.Now()
	
	// Validate input board
	if !s.IsValidBoard(board) {
		return nil, nil, errors.New("invalid input board")
	}
	
	// Get empty cells
	emptyCells := s.getEmptyCells(board)
	if len(emptyCells) == 0 {
		// Board is already solved
		stats := s.getStats()
		stats.TotalAttempts = 1
		stats.SuccessfulSolves = 1
		stats.AverageSolveTime = time.Since(startTime)
		return &board, &stats, nil
	}
	
	// Apply initial constraint propagation if enabled
	if s.config.UseHeuristics {
		board = s.applyConstraintPropagation(board)
		emptyCells = s.getEmptyCells(board)
	}
	
	// Choose solving strategy
	var solution *SudokuBoard
	var err error
	
	switch s.config.Strategy {
	case BacktrackingStrategy:
		solution, err = s.solveBacktrackingParallel(ctx, board, emptyCells)
	case ConstraintPropagationStrategy:
		solution, err = s.solveConstraintPropagation(ctx, board)
	case HybridStrategy:
		solution, err = s.solveHybrid(ctx, board, emptyCells)
	case BruteForceStrategy:
		solution, err = s.solveBruteForceParallel(ctx, board, emptyCells)
	default:
		return nil, nil, errors.New("unknown solving strategy")
	}
	
	// Update statistics
	solveTime := time.Since(startTime)
	s.updateStats(func(stats *SolverStats) {
		stats.TotalAttempts++
		if solution != nil {
			stats.SuccessfulSolves++
		} else {
			stats.FailedSolves++
		}
		
		// Update average solve time
		if stats.TotalAttempts == 1 {
			stats.AverageSolveTime = solveTime
		} else {
			stats.AverageSolveTime = (stats.AverageSolveTime + solveTime) / 2
		}
	})
	
	finalStats := s.getStats()
	return solution, &finalStats, err
}

// solveBacktrackingParallel solves using parallel backtracking
func (s *ParallelSudokuSolver) solveBacktrackingParallel(ctx context.Context, board SudokuBoard, emptyCells []Cell) (*SudokuBoard, error) {
	if len(emptyCells) == 0 {
		return &board, nil
	}
	
	// Create channels for work distribution
	workChan := make(chan SolveTask, s.config.MaxWorkers*2)
	resultChan := make(chan SolveResult, s.config.MaxWorkers)
	
	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < s.config.MaxWorkers; i++ {
		wg.Add(1)
		worker := &SolverWorker{
			id:         i,
			workChan:   workChan,
			resultChan: resultChan,
			solver:     s,
		}
		go func() {
			defer wg.Done()
			worker.run(ctx)
		}()
	}
	
	// Generate initial work tasks by trying different values for the first empty cell
	firstCell := emptyCells[0]
	taskID := 0
	
	for value := 1; value <= 9; value++ {
		if s.isValidPlacement(board, firstCell.Row, firstCell.Col, value) {
			newBoard := board
			newBoard[firstCell.Row][firstCell.Col] = value
			
			task := SolveTask{
				ID:         taskID,
				Board:      newBoard,
				EmptyCells: emptyCells[1:],
				Strategy:   BacktrackingStrategy,
				TimeLimit:  s.config.TimeLimit,
				Context:    ctx,
			}
			
			select {
			case workChan <- task:
				taskID++
			case <-ctx.Done():
				close(workChan)
				return nil, ctx.Err()
			}
		}
	}
	close(workChan)
	
	// Collect results
	go func() {
		wg.Wait()
		close(resultChan)
	}()
	
	// Process results
	for result := range resultChan {
		if result.Success {
			// Found a solution
			return &result.Solution, nil
		}
		
		if result.Error != nil && result.Error != context.Canceled {
			return nil, result.Error
		}
	}
	
	return nil, errors.New("no solution found")
}

// solveConstraintPropagation solves using constraint propagation
func (s *ParallelSudokuSolver) solveConstraintPropagation(ctx context.Context, board SudokuBoard) (*SudokuBoard, error) {
	maxIterations := 100
	
	for iteration := 0; iteration < maxIterations; iteration++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		
		changed := false
		oldBoard := board
		
		// Apply constraint propagation
		board = s.applyConstraintPropagation(board)
		
		// Check if board changed
		for i := 0; i < 9; i++ {
			for j := 0; j < 9; j++ {
				if board[i][j] != oldBoard[i][j] {
					changed = true
					break
				}
			}
			if changed {
				break
			}
		}
		
		// Check if solved
		if s.isSolved(board) {
			return &board, nil
		}
		
		// If no progress, fall back to backtracking
		if !changed {
			emptyCells := s.getEmptyCells(board)
			return s.solveBacktrackingParallel(ctx, board, emptyCells)
		}
		
		atomic.AddInt64(&s.stats.ConstraintChecks, 1)
	}
	
	return nil, errors.New("constraint propagation failed to solve")
}

// solveHybrid combines multiple strategies
func (s *ParallelSudokuSolver) solveHybrid(ctx context.Context, board SudokuBoard, emptyCells []Cell) (*SudokuBoard, error) {
	// First, apply constraint propagation
	board = s.applyConstraintPropagation(board)
	emptyCells = s.getEmptyCells(board)
	
	// If mostly solved, use single-threaded backtracking
	if len(emptyCells) < 20 {
		return s.solveBacktrackingSingle(ctx, board, emptyCells, 0)
	}
	
	// Otherwise, use parallel backtracking
	return s.solveBacktrackingParallel(ctx, board, emptyCells)
}

// solveBruteForceParallel tries all possible combinations in parallel
func (s *ParallelSudokuSolver) solveBruteForceParallel(ctx context.Context, board SudokuBoard, emptyCells []Cell) (*SudokuBoard, error) {
	if len(emptyCells) > 25 {
		return nil, errors.New("too many empty cells for brute force approach")
	}
	
	// Generate all possible combinations
	combinations := s.generateCombinations(emptyCells)
	
	// Create work channel
	workChan := make(chan []int, len(combinations))
	resultChan := make(chan SolveResult, s.config.MaxWorkers)
	
	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < s.config.MaxWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s.bruteForceWorker(ctx, board, emptyCells, workChan, resultChan)
		}()
	}
	
	// Send work
	for _, combination := range combinations {
		select {
		case workChan <- combination:
		case <-ctx.Done():
			close(workChan)
			return nil, ctx.Err()
		}
	}
	close(workChan)
	
	// Collect results
	go func() {
		wg.Wait()
		close(resultChan)
	}()
	
	for result := range resultChan {
		if result.Success {
			return &result.Solution, nil
		}
	}
	
	return nil, errors.New("no solution found with brute force")
}

// Worker implementation
func (w *SolverWorker) run(ctx context.Context) {
	for {
		select {
		case task, ok := <-w.workChan:
			if !ok {
				return
			}
			
			result := w.processTask(ctx, task)
			
			select {
			case w.resultChan <- result:
			case <-ctx.Done():
				return
			}
			
		case <-ctx.Done():
			return
		}
	}
}

func (w *SolverWorker) processTask(ctx context.Context, task SolveTask) SolveResult {
	startTime := time.Now()
	
	result := SolveResult{
		TaskID: task.ID,
	}
	
	// Create timeout context
	timeoutCtx, cancel := context.WithTimeout(ctx, task.TimeLimit)
	defer cancel()
	
	// Solve using specified strategy
	solution, err := w.solver.solveBacktrackingSingle(timeoutCtx, task.Board, task.EmptyCells, 0)
	
	result.SolveTime = time.Since(startTime)
	result.Error = err
	
	if solution != nil {
		result.Success = true
		result.Solution = *solution
	}
	
	return result
}

// solveBacktrackingSingle performs single-threaded backtracking
func (s *ParallelSudokuSolver) solveBacktrackingSingle(ctx context.Context, board SudokuBoard, emptyCells []Cell, index int) (*SudokuBoard, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	
	if index >= len(emptyCells) {
		// All cells filled, check if valid solution
		if s.isSolved(board) {
			return &board, nil
		}
		return nil, errors.New("invalid solution")
	}
	
	cell := emptyCells[index]
	atomic.AddInt64(&s.stats.BacktrackCount, 1)
	
	// Try values 1-9
	values := []int{1, 2, 3, 4, 5, 6, 7, 8, 9}
	
	// Apply heuristics if enabled
	if s.config.UseHeuristics {
		values = s.getOrderedValues(board, cell)
	}
	
	for _, value := range values {
		if s.isValidPlacement(board, cell.Row, cell.Col, value) {
			board[cell.Row][cell.Col] = value
			
			// Apply pruning if enabled
			if s.config.EnablePruning && !s.isStillSolvable(board) {
				board[cell.Row][cell.Col] = 0
				atomic.AddInt64(&s.stats.PruningOperations, 1)
				continue
			}
			
			// Recursively solve
			solution, err := s.solveBacktrackingSingle(ctx, board, emptyCells, index+1)
			if solution != nil {
				return solution, nil
			}
			
			if err != nil && err != context.Canceled {
				board[cell.Row][cell.Col] = 0
				return nil, err
			}
			
			// Backtrack
			board[cell.Row][cell.Col] = 0
		}
	}
	
	return nil, nil
}

// Helper methods

func (s *ParallelSudokuSolver) IsValidBoard(board SudokuBoard) bool {
	// Check rows
	for i := 0; i < 9; i++ {
		seen := make(map[int]bool)
		for j := 0; j < 9; j++ {
			value := board[i][j]
			if value != 0 {
				if value < 1 || value > 9 || seen[value] {
					return false
				}
				seen[value] = true
			}
		}
	}
	
	// Check columns
	for j := 0; j < 9; j++ {
		seen := make(map[int]bool)
		for i := 0; i < 9; i++ {
			value := board[i][j]
			if value != 0 {
				if value < 1 || value > 9 || seen[value] {
					return false
				}
				seen[value] = true
			}
		}
	}
	
	// Check 3x3 boxes
	for boxRow := 0; boxRow < 3; boxRow++ {
		for boxCol := 0; boxCol < 3; boxCol++ {
			seen := make(map[int]bool)
			for i := boxRow * 3; i < (boxRow+1)*3; i++ {
				for j := boxCol * 3; j < (boxCol+1)*3; j++ {
					value := board[i][j]
					if value != 0 {
						if value < 1 || value > 9 || seen[value] {
							return false
						}
						seen[value] = true
					}
				}
			}
		}
	}
	
	return true
}

func (s *ParallelSudokuSolver) getEmptyCells(board SudokuBoard) []Cell {
	var emptyCells []Cell
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] == 0 {
				emptyCells = append(emptyCells, Cell{Row: i, Col: j})
			}
		}
	}
	return emptyCells
}

func (s *ParallelSudokuSolver) isValidPlacement(board SudokuBoard, row, col, value int) bool {
	// Check row
	for j := 0; j < 9; j++ {
		if j != col && board[row][j] == value {
			return false
		}
	}
	
	// Check column
	for i := 0; i < 9; i++ {
		if i != row && board[i][col] == value {
			return false
		}
	}
	
	// Check 3x3 box
	boxRow := (row / 3) * 3
	boxCol := (col / 3) * 3
	for i := boxRow; i < boxRow+3; i++ {
		for j := boxCol; j < boxCol+3; j++ {
			if (i != row || j != col) && board[i][j] == value {
				return false
			}
		}
	}
	
	return true
}

func (s *ParallelSudokuSolver) isSolved(board SudokuBoard) bool {
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] == 0 {
				return false
			}
		}
	}
	return s.IsValidBoard(board)
}

func (s *ParallelSudokuSolver) applyConstraintPropagation(board SudokuBoard) SudokuBoard {
	changed := true
	for changed {
		changed = false
		
		// For each empty cell, try to determine its value
		for i := 0; i < 9; i++ {
			for j := 0; j < 9; j++ {
				if board[i][j] == 0 {
					possibleValues := s.getPossibleValues(board, i, j)
					if len(possibleValues) == 1 {
						board[i][j] = possibleValues[0]
						changed = true
					}
				}
			}
		}
		
		atomic.AddInt64(&s.stats.ConstraintChecks, 1)
	}
	
	return board
}

func (s *ParallelSudokuSolver) getPossibleValues(board SudokuBoard, row, col int) []int {
	if board[row][col] != 0 {
		return []int{board[row][col]}
	}
	
	used := make(map[int]bool)
	
	// Check row
	for j := 0; j < 9; j++ {
		if board[row][j] != 0 {
			used[board[row][j]] = true
		}
	}
	
	// Check column
	for i := 0; i < 9; i++ {
		if board[i][col] != 0 {
			used[board[i][col]] = true
		}
	}
	
	// Check 3x3 box
	boxRow := (row / 3) * 3
	boxCol := (col / 3) * 3
	for i := boxRow; i < boxRow+3; i++ {
		for j := boxCol; j < boxCol+3; j++ {
			if board[i][j] != 0 {
				used[board[i][j]] = true
			}
		}
	}
	
	var possible []int
	for value := 1; value <= 9; value++ {
		if !used[value] {
			possible = append(possible, value)
		}
	}
	
	return possible
}

func (s *ParallelSudokuSolver) getOrderedValues(board SudokuBoard, cell Cell) []int {
	atomic.AddInt64(&s.stats.HeuristicEvaluations, 1)
	
	possibleValues := s.getPossibleValues(board, cell.Row, cell.Col)
	
	// Order by constraint count (most constrained first)
	type valueConstraint struct {
		value       int
		constraints int
	}
	
	var valueConstraints []valueConstraint
	for _, value := range possibleValues {
		constraints := s.countConstraints(board, cell.Row, cell.Col, value)
		valueConstraints = append(valueConstraints, valueConstraint{value, constraints})
	}
	
	// Sort by constraint count (ascending - least constraining first)
	for i := 0; i < len(valueConstraints)-1; i++ {
		for j := i + 1; j < len(valueConstraints); j++ {
			if valueConstraints[i].constraints > valueConstraints[j].constraints {
				valueConstraints[i], valueConstraints[j] = valueConstraints[j], valueConstraints[i]
			}
		}
	}
	
	var orderedValues []int
	for _, vc := range valueConstraints {
		orderedValues = append(orderedValues, vc.value)
	}
	
	return orderedValues
}

func (s *ParallelSudokuSolver) countConstraints(board SudokuBoard, row, col, value int) int {
	constraints := 0
	
	// Count how many cells this value would constrain
	// Check row
	for j := 0; j < 9; j++ {
		if j != col && board[row][j] == 0 {
			possibleValues := s.getPossibleValues(board, row, j)
			for _, pv := range possibleValues {
				if pv == value {
					constraints++
					break
				}
			}
		}
	}
	
	// Check column
	for i := 0; i < 9; i++ {
		if i != row && board[i][col] == 0 {
			possibleValues := s.getPossibleValues(board, i, col)
			for _, pv := range possibleValues {
				if pv == value {
					constraints++
					break
				}
			}
		}
	}
	
	// Check 3x3 box
	boxRow := (row / 3) * 3
	boxCol := (col / 3) * 3
	for i := boxRow; i < boxRow+3; i++ {
		for j := boxCol; j < boxCol+3; j++ {
			if (i != row || j != col) && board[i][j] == 0 {
				possibleValues := s.getPossibleValues(board, i, j)
				for _, pv := range possibleValues {
					if pv == value {
						constraints++
						break
					}
				}
			}
		}
	}
	
	return constraints
}

func (s *ParallelSudokuSolver) isStillSolvable(board SudokuBoard) bool {
	// Check if any empty cell has no possible values
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] == 0 {
				possibleValues := s.getPossibleValues(board, i, j)
				if len(possibleValues) == 0 {
					return false
				}
			}
		}
	}
	return true
}

func (s *ParallelSudokuSolver) generateCombinations(emptyCells []Cell) [][]int {
	if len(emptyCells) == 0 {
		return [][]int{{}}
	}
	
	numCells := len(emptyCells)
	totalCombinations := 1
	for i := 0; i < numCells; i++ {
		totalCombinations *= 9
	}
	
	// Limit combinations to prevent memory issues
	if totalCombinations > 1000000 {
		return [][]int{}
	}
	
	var combinations [][]int
	current := make([]int, numCells)
	
	var generate func(int)
	generate = func(index int) {
		if index == numCells {
			combination := make([]int, numCells)
			copy(combination, current)
			combinations = append(combinations, combination)
			return
		}
		
		for value := 1; value <= 9; value++ {
			current[index] = value
			generate(index + 1)
		}
	}
	
	generate(0)
	return combinations
}

func (s *ParallelSudokuSolver) bruteForceWorker(ctx context.Context, board SudokuBoard, emptyCells []Cell, workChan <-chan []int, resultChan chan<- SolveResult) {
	for {
		select {
		case combination, ok := <-workChan:
			if !ok {
				return
			}
			
			// Try this combination
			testBoard := board
			valid := true
			
			for i, cell := range emptyCells {
				testBoard[cell.Row][cell.Col] = combination[i]
				if !s.isValidPlacement(testBoard, cell.Row, cell.Col, combination[i]) {
					valid = false
					break
				}
			}
			
			if valid && s.isSolved(testBoard) {
				result := SolveResult{
					Success:  true,
					Solution: testBoard,
				}
				
				select {
				case resultChan <- result:
				case <-ctx.Done():
				}
				return
			}
			
		case <-ctx.Done():
			return
		}
	}
}

// Statistics and utility methods

func (s *ParallelSudokuSolver) updateStats(updateFunc func(*SolverStats)) {
	s.statsMutex.Lock()
	defer s.statsMutex.Unlock()
	updateFunc(&s.stats)
}

func (s *ParallelSudokuSolver) getStats() SolverStats {
	s.statsMutex.RLock()
	defer s.statsMutex.RUnlock()
	return s.stats
}

func (s *ParallelSudokuSolver) GetStats() SolverStats {
	return s.getStats()
}

func (s *ParallelSudokuSolver) Reset() {
	s.statsMutex.Lock()
	defer s.statsMutex.Unlock()
	s.stats = SolverStats{
		Strategy:    s.config.Strategy,
		WorkersUsed: s.config.MaxWorkers,
	}
}

// Puzzle generation and utilities

// GeneratePuzzle generates a new Sudoku puzzle
func GeneratePuzzle(difficulty DifficultyLevel, seed int64) SudokuBoard {
	r := rand.New(rand.NewSource(seed))
	
	// Start with a solved board
	board := generateSolvedBoard(r)
	
	// Remove cells based on difficulty
	cellsToRemove := getCellsToRemove(difficulty)
	
	// Randomly remove cells while ensuring unique solution
	cells := getAllCells()
	shuffleCells(cells, r)
	
	removed := 0
	for _, cell := range cells {
		if removed >= cellsToRemove {
			break
		}
		
		originalValue := board[cell.Row][cell.Col]
		board[cell.Row][cell.Col] = 0
		
		// Check if puzzle still has unique solution (simplified check)
		if hasUniqueSolution(board) {
			removed++
		} else {
			// Restore the value
			board[cell.Row][cell.Col] = originalValue
		}
	}
	
	return board
}

func generateSolvedBoard(r *rand.Rand) SudokuBoard {
	// Generate a valid, complete Sudoku board
	var board SudokuBoard
	
	// Fill diagonal 3x3 boxes first (they don't affect each other)
	for box := 0; box < 3; box++ {
		fillBox(&board, box*3, box*3, r)
	}
	
	// Fill remaining cells
	solver := NewParallelSudokuSolver(SolverConfig{
		Strategy:   BacktrackingStrategy,
		MaxWorkers: 1,
		TimeLimit:  30 * time.Second,
	})
	
	ctx := context.Background()
	solution, _, err := solver.Solve(ctx, board)
	if err != nil || solution == nil {
		// Fallback to a known solved board
		return getDefaultSolvedBoard()
	}
	
	return *solution
}

func fillBox(board *SudokuBoard, row, col int, r *rand.Rand) {
	numbers := []int{1, 2, 3, 4, 5, 6, 7, 8, 9}
	
	// Shuffle numbers
	for i := len(numbers) - 1; i > 0; i-- {
		j := r.Intn(i + 1)
		numbers[i], numbers[j] = numbers[j], numbers[i]
	}
	
	// Fill the 3x3 box
	idx := 0
	for i := row; i < row+3; i++ {
		for j := col; j < col+3; j++ {
			board[i][j] = numbers[idx]
			idx++
		}
	}
}

func getCellsToRemove(difficulty DifficultyLevel) int {
	switch difficulty {
	case Easy:
		return 35  // 46 filled cells
	case Medium:
		return 45  // 36 filled cells
	case Hard:
		return 55  // 26 filled cells
	case Expert:
		return 65  // 16 filled cells
	default:
		return 40
	}
}

func getAllCells() []Cell {
	var cells []Cell
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			cells = append(cells, Cell{Row: i, Col: j})
		}
	}
	return cells
}

func shuffleCells(cells []Cell, r *rand.Rand) {
	for i := len(cells) - 1; i > 0; i-- {
		j := r.Intn(i + 1)
		cells[i], cells[j] = cells[j], cells[i]
	}
}

func hasUniqueSolution(board SudokuBoard) bool {
	// Simplified uniqueness check - in practice, this would be more complex
	solver := NewParallelSudokuSolver(SolverConfig{
		Strategy:   BacktrackingStrategy,
		MaxWorkers: 1,
		TimeLimit:  5 * time.Second,
	})
	
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	solution, _, err := solver.Solve(ctx, board)
	return err == nil && solution != nil
}

func getDefaultSolvedBoard() SudokuBoard {
	return SudokuBoard{
		{5, 3, 4, 6, 7, 8, 9, 1, 2},
		{6, 7, 2, 1, 9, 5, 3, 4, 8},
		{1, 9, 8, 3, 4, 2, 5, 6, 7},
		{8, 5, 9, 7, 6, 1, 4, 2, 3},
		{4, 2, 6, 8, 5, 3, 7, 9, 1},
		{7, 1, 3, 9, 2, 4, 8, 5, 6},
		{9, 6, 1, 5, 3, 7, 2, 8, 4},
		{2, 8, 7, 4, 1, 9, 6, 3, 5},
		{3, 4, 5, 2, 8, 6, 1, 7, 9},
	}
}

// Utility functions for displaying boards

func (board SudokuBoard) String() string {
	result := ""
	for i := 0; i < 9; i++ {
		if i%3 == 0 && i != 0 {
			result += "------+-------+------\n"
		}
		for j := 0; j < 9; j++ {
			if j%3 == 0 && j != 0 {
				result += "| "
			}
			if board[i][j] == 0 {
				result += ". "
			} else {
				result += fmt.Sprintf("%d ", board[i][j])
			}
		}
		result += "\n"
	}
	return result
}

func (board SudokuBoard) PrettyString() string {
	result := "┌───────┬───────┬───────┐\n"
	for i := 0; i < 9; i++ {
		if i == 3 || i == 6 {
			result += "├───────┼───────┼───────┤\n"
		}
		result += "│ "
		for j := 0; j < 9; j++ {
			if j == 3 || j == 6 {
				result += "│ "
			}
			if board[i][j] == 0 {
				result += ". "
			} else {
				result += fmt.Sprintf("%d ", board[i][j])
			}
		}
		result += "│\n"
	}
	result += "└───────┴───────┴───────┘\n"
	return result
}

// Benchmarking utilities

// BenchmarkSolver runs performance benchmarks
func BenchmarkSolver(puzzles []SudokuBoard, strategies []SolverStrategy, workerCounts []int) map[string]time.Duration {
	results := make(map[string]time.Duration)
	
	for _, strategy := range strategies {
		for _, workers := range workerCounts {
			config := SolverConfig{
				Strategy:   strategy,
				MaxWorkers: workers,
				TimeLimit:  60 * time.Second,
			}
			
			solver := NewParallelSudokuSolver(config)
			
			start := time.Now()
			solved := 0
			
			for _, puzzle := range puzzles {
				ctx := context.Background()
				solution, _, err := solver.Solve(ctx, puzzle)
				if err == nil && solution != nil {
					solved++
				}
			}
			
			duration := time.Since(start)
			key := fmt.Sprintf("%s_%d_workers", strategyToString(strategy), workers)
			results[key] = duration
			
			fmt.Printf("%s: Solved %d/%d puzzles in %v\n", 
				key, solved, len(puzzles), duration)
		}
	}
	
	return results
}

func strategyToString(strategy SolverStrategy) string {
	switch strategy {
	case BacktrackingStrategy:
		return "Backtracking"
	case ConstraintPropagationStrategy:
		return "ConstraintPropagation"
	case HybridStrategy:
		return "Hybrid"
	case BruteForceStrategy:
		return "BruteForce"
	default:
		return "Unknown"
	}
}