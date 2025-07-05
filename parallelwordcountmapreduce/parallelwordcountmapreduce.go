package parallelwordcountmapreduce

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// MapReduceFramework implements parallel word count using MapReduce pattern
type MapReduceFramework struct {
	numMappers     int
	numReducers    int
	chunkSize      int
	intermediateDir string
	cleanupTemp    bool
}

// MapReduceConfig holds configuration for MapReduce operations
type MapReduceConfig struct {
	NumMappers     int
	NumReducers    int
	ChunkSize      int
	IntermediateDir string
	CleanupTemp    bool
	CaseSensitive  bool
	MinWordLength  int
	StopWords      []string
	WordPattern    string
}

// WordCountJob represents a word count MapReduce job
type WordCountJob struct {
	JobID       string
	InputFiles  []string
	OutputPath  string
	Config      MapReduceConfig
	Framework   *MapReduceFramework
	StartTime   time.Time
	EndTime     time.Time
	Result      *WordCountResult
	Progress    *JobProgress
}

// WordCountResult contains the results of word counting
type WordCountResult struct {
	TotalWords      int64
	UniqueWords     int64
	TotalFiles      int
	ProcessingTime  time.Duration
	TopWords        []WordFrequency
	WordFrequencies map[string]int64
	FileStats       map[string]FileStats
}

// WordFrequency represents a word and its frequency
type WordFrequency struct {
	Word      string
	Frequency int64
}

// FileStats contains statistics for a processed file
type FileStats struct {
	FileName   string
	WordCount  int64
	UniqueWords int64
	Size       int64
	ProcessingTime time.Duration
}

// JobProgress tracks job execution progress
type JobProgress struct {
	Phase         Phase
	FilesProcessed int64
	TotalFiles    int64
	WordsProcessed int64
	MapTasksComplete    int64
	ReduceTasksComplete int64
	StartTime     time.Time
	mu            sync.RWMutex
}

// Phase represents the current phase of MapReduce execution
type Phase int

const (
	PhaseInitialization Phase = iota
	PhaseMapping
	PhaseShuffling
	PhaseReducing
	PhaseCompleted
	PhaseFailed
)

// MapTask represents a mapping task
type MapTask struct {
	TaskID    int
	InputFile string
	StartByte int64
	EndByte   int64
	Config    MapReduceConfig
}

// ReduceTask represents a reduce task
type ReduceTask struct {
	TaskID        int
	PartitionID   int
	InputFiles    []string
	OutputFile    string
	Config        MapReduceConfig
}

// KeyValue represents a key-value pair
type KeyValue struct {
	Key   string
	Value string
}

// IntermediateData represents intermediate data between map and reduce
type IntermediateData struct {
	Partition int
	Data      []KeyValue
}

// Mapper interface for map operations
type Mapper interface {
	Map(input string, emitFunc func(key, value string)) error
}

// Reducer interface for reduce operations
type Reducer interface {
	Reduce(key string, values []string, emitFunc func(key, value string)) error
}

// WordCountMapper implements word counting mapper
type WordCountMapper struct {
	config MapReduceConfig
	wordRegex *regexp.Regexp
	stopWords map[string]bool
}

// WordCountReducer implements word counting reducer
type WordCountReducer struct {
	config MapReduceConfig
}

// NewMapReduceFramework creates a new MapReduce framework
func NewMapReduceFramework(config MapReduceConfig) *MapReduceFramework {
	if config.NumMappers <= 0 {
		config.NumMappers = runtime.NumCPU()
	}
	if config.NumReducers <= 0 {
		config.NumReducers = runtime.NumCPU()
	}
	if config.ChunkSize <= 0 {
		config.ChunkSize = 64 * 1024 * 1024 // 64MB default
	}
	if config.IntermediateDir == "" {
		config.IntermediateDir = os.TempDir()
	}
	if config.WordPattern == "" {
		config.WordPattern = `\b[a-zA-Z]+\b`
	}
	if config.MinWordLength <= 0 {
		config.MinWordLength = 1
	}

	return &MapReduceFramework{
		numMappers:     config.NumMappers,
		numReducers:    config.NumReducers,
		chunkSize:      config.ChunkSize,
		intermediateDir: config.IntermediateDir,
		cleanupTemp:    config.CleanupTemp,
	}
}

// ProcessWordCount executes word count using MapReduce
func (mrf *MapReduceFramework) ProcessWordCount(inputFiles []string, outputPath string, config MapReduceConfig) (*WordCountResult, error) {
	job := &WordCountJob{
		JobID:      fmt.Sprintf("wordcount_%d", time.Now().UnixNano()),
		InputFiles: inputFiles,
		OutputPath: outputPath,
		Config:     config,
		Framework:  mrf,
		StartTime:  time.Now(),
		Progress: &JobProgress{
			Phase:      PhaseInitialization,
			TotalFiles: int64(len(inputFiles)),
			StartTime:  time.Now(),
		},
	}

	// Execute MapReduce phases
	result, err := mrf.executeMapReduce(job)
	if err != nil {
		job.Progress.Phase = PhaseFailed
		return nil, err
	}

	job.EndTime = time.Now()
	job.Progress.Phase = PhaseCompleted
	job.Result = result

	return result, nil
}

func (mrf *MapReduceFramework) executeMapReduce(job *WordCountJob) (*WordCountResult, error) {
	// Phase 1: Map
	job.Progress.Phase = PhaseMapping
	intermediateFiles, err := mrf.mapPhase(job)
	if err != nil {
		return nil, fmt.Errorf("map phase failed: %w", err)
	}

	// Phase 2: Shuffle (group by key)
	job.Progress.Phase = PhaseShuffling
	groupedFiles, err := mrf.shufflePhase(intermediateFiles, job)
	if err != nil {
		return nil, fmt.Errorf("shuffle phase failed: %w", err)
	}

	// Phase 3: Reduce
	job.Progress.Phase = PhaseReducing
	result, err := mrf.reducePhase(groupedFiles, job)
	if err != nil {
		return nil, fmt.Errorf("reduce phase failed: %w", err)
	}

	// Cleanup temporary files
	if mrf.cleanupTemp {
		mrf.cleanup(intermediateFiles)
		mrf.cleanup(groupedFiles)
	}

	return result, nil
}

func (mrf *MapReduceFramework) mapPhase(job *WordCountJob) ([]string, error) {
	// Create map tasks
	tasks, err := mrf.createMapTasks(job)
	if err != nil {
		return nil, err
	}

	// Create mapper
	mapper := NewWordCountMapper(job.Config)

	// Process tasks in parallel
	var wg sync.WaitGroup
	taskChan := make(chan MapTask, len(tasks))
	resultChan := make(chan []string, len(tasks))
	errorChan := make(chan error, len(tasks))

	// Start map workers
	for i := 0; i < mrf.numMappers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			mrf.mapWorker(taskChan, resultChan, errorChan, mapper, job)
		}()
	}

	// Send tasks
	for _, task := range tasks {
		taskChan <- task
	}
	close(taskChan)

	// Wait for completion
	wg.Wait()
	close(resultChan)
	close(errorChan)

	// Collect results and errors
	var allIntermediateFiles []string
	for files := range resultChan {
		allIntermediateFiles = append(allIntermediateFiles, files...)
	}

	var errors []error
	for err := range errorChan {
		if err != nil {
			errors = append(errors, err)
		}
	}

	if len(errors) > 0 {
		return nil, fmt.Errorf("map phase errors: %v", errors)
	}

	return allIntermediateFiles, nil
}

func (mrf *MapReduceFramework) createMapTasks(job *WordCountJob) ([]MapTask, error) {
	var tasks []MapTask
	taskID := 0

	for _, inputFile := range job.InputFiles {
		fileInfo, err := os.Stat(inputFile)
		if err != nil {
			return nil, err
		}

		fileSize := fileInfo.Size()
		chunkSize := int64(mrf.chunkSize)

		// Create chunks for large files
		for offset := int64(0); offset < fileSize; offset += chunkSize {
			endByte := offset + chunkSize
			if endByte > fileSize {
				endByte = fileSize
			}

			tasks = append(tasks, MapTask{
				TaskID:    taskID,
				InputFile: inputFile,
				StartByte: offset,
				EndByte:   endByte,
				Config:    job.Config,
			})
			taskID++
		}
	}

	return tasks, nil
}

func (mrf *MapReduceFramework) mapWorker(taskChan <-chan MapTask, resultChan chan<- []string, errorChan chan<- error, mapper *WordCountMapper, job *WordCountJob) {
	for task := range taskChan {
		files, err := mrf.executeMapTask(task, mapper, job)
		if err != nil {
			errorChan <- err
		} else {
			resultChan <- files
			atomic.AddInt64(&job.Progress.MapTasksComplete, 1)
		}
	}
}

func (mrf *MapReduceFramework) executeMapTask(task MapTask, mapper *WordCountMapper, job *WordCountJob) ([]string, error) {
	// Read input chunk
	content, err := mrf.readFileChunk(task.InputFile, task.StartByte, task.EndByte)
	if err != nil {
		return nil, err
	}

	// Create intermediate files for each reducer
	var intermediateFiles []string
	var writers []*os.File

	for i := 0; i < mrf.numReducers; i++ {
		filename := filepath.Join(mrf.intermediateDir, fmt.Sprintf("map_%d_reduce_%d.txt", task.TaskID, i))
		file, err := os.Create(filename)
		if err != nil {
			return nil, err
		}
		writers = append(writers, file)
		intermediateFiles = append(intermediateFiles, filename)
	}

	defer func() {
		for _, writer := range writers {
			writer.Close()
		}
	}()

	// Create emit function
	emitFunc := func(key, value string) {
		// Hash key to determine which reducer gets this key-value pair
		reducerID := mrf.hashKey(key) % mrf.numReducers
		writers[reducerID].WriteString(fmt.Sprintf("%s\t%s\n", key, value))
	}

	// Execute map operation
	err = mapper.Map(content, emitFunc)
	if err != nil {
		return nil, err
	}

	return intermediateFiles, nil
}

func (mrf *MapReduceFramework) readFileChunk(filename string, start, end int64) (string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return "", err
	}
	defer file.Close()

	// Seek to start position
	if _, err := file.Seek(start, 0); err != nil {
		return "", err
	}

	// Read chunk
	chunkSize := end - start
	buffer := make([]byte, chunkSize)
	n, err := file.Read(buffer)
	if err != nil && err != io.EOF {
		return "", err
	}

	content := string(buffer[:n])

	// Adjust chunk boundaries to word boundaries (except for first chunk)
	if start > 0 {
		// Find first complete word
		firstSpace := strings.Index(content, " ")
		if firstSpace != -1 {
			content = content[firstSpace+1:]
		}
	}

	if end < getFileSize(filename) {
		// Find last complete word
		lastSpace := strings.LastIndex(content, " ")
		if lastSpace != -1 {
			content = content[:lastSpace]
		}
	}

	return content, nil
}

func (mrf *MapReduceFramework) shufflePhase(intermediateFiles []string, job *WordCountJob) ([]string, error) {
	// Group intermediate files by reducer
	reducerFiles := make([][]string, mrf.numReducers)

	for _, file := range intermediateFiles {
		// Extract reducer ID from filename
		parts := strings.Split(filepath.Base(file), "_")
		if len(parts) >= 4 {
			reducerID := 0
			fmt.Sscanf(parts[3], "%d.txt", &reducerID)
			if reducerID < mrf.numReducers {
				reducerFiles[reducerID] = append(reducerFiles[reducerID], file)
			}
		}
	}

	// Create merged files for each reducer
	var mergedFiles []string
	var wg sync.WaitGroup
	resultChan := make(chan string, mrf.numReducers)
	errorChan := make(chan error, mrf.numReducers)

	for reducerID, files := range reducerFiles {
		wg.Add(1)
		go func(id int, inputFiles []string) {
			defer wg.Done()

			outputFile := filepath.Join(mrf.intermediateDir, fmt.Sprintf("shuffled_%d.txt", id))
			err := mrf.mergeAndSort(inputFiles, outputFile)
			if err != nil {
				errorChan <- err
			} else {
				resultChan <- outputFile
			}
		}(reducerID, files)
	}

	wg.Wait()
	close(resultChan)
	close(errorChan)

	// Collect results
	for file := range resultChan {
		mergedFiles = append(mergedFiles, file)
	}

	// Check for errors
	for err := range errorChan {
		if err != nil {
			return nil, err
		}
	}

	return mergedFiles, nil
}

func (mrf *MapReduceFramework) mergeAndSort(inputFiles []string, outputFile string) error {
	// Collect all key-value pairs
	var keyValues []KeyValue

	for _, inputFile := range inputFiles {
		file, err := os.Open(inputFile)
		if err != nil {
			return err
		}

		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			parts := strings.SplitN(line, "\t", 2)
			if len(parts) == 2 {
				keyValues = append(keyValues, KeyValue{
					Key:   parts[0],
					Value: parts[1],
				})
			}
		}
		file.Close()
	}

	// Sort by key
	sort.Slice(keyValues, func(i, j int) bool {
		return keyValues[i].Key < keyValues[j].Key
	})

	// Write sorted data
	outFile, err := os.Create(outputFile)
	if err != nil {
		return err
	}
	defer outFile.Close()

	for _, kv := range keyValues {
		outFile.WriteString(fmt.Sprintf("%s\t%s\n", kv.Key, kv.Value))
	}

	return nil
}

func (mrf *MapReduceFramework) reducePhase(shuffledFiles []string, job *WordCountJob) (*WordCountResult, error) {
	reducer := NewWordCountReducer(job.Config)

	var wg sync.WaitGroup
	resultChan := make(chan map[string]int64, len(shuffledFiles))
	errorChan := make(chan error, len(shuffledFiles))

	// Process each shuffled file
	for _, file := range shuffledFiles {
		wg.Add(1)
		go func(inputFile string) {
			defer wg.Done()

			result, err := mrf.executeReduceTask(inputFile, reducer, job)
			if err != nil {
				errorChan <- err
			} else {
				resultChan <- result
				atomic.AddInt64(&job.Progress.ReduceTasksComplete, 1)
			}
		}(file)
	}

	wg.Wait()
	close(resultChan)
	close(errorChan)

	// Check for errors
	for err := range errorChan {
		if err != nil {
			return nil, err
		}
	}

	// Merge results from all reducers
	finalResult := make(map[string]int64)
	for partialResult := range resultChan {
		for word, count := range partialResult {
			finalResult[word] += count
		}
	}

	// Create final result
	return mrf.createWordCountResult(finalResult, job), nil
}

func (mrf *MapReduceFramework) executeReduceTask(inputFile string, reducer *WordCountReducer, job *WordCountJob) (map[string]int64, error) {
	result := make(map[string]int64)

	file, err := os.Open(inputFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	currentKey := ""
	var values []string

	emitFunc := func(key, value string) {
		if count, err := fmt.Sscanf(value, "%d", new(int)); err == nil {
			result[key] += int64(count)
		}
	}

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) != 2 {
			continue
		}

		key, value := parts[0], parts[1]

		if key != currentKey {
			// Process previous key
			if currentKey != "" && len(values) > 0 {
				reducer.Reduce(currentKey, values, emitFunc)
			}
			// Start new key
			currentKey = key
			values = []string{value}
		} else {
			values = append(values, value)
		}

		atomic.AddInt64(&job.Progress.WordsProcessed, 1)
	}

	// Process last key
	if currentKey != "" && len(values) > 0 {
		reducer.Reduce(currentKey, values, emitFunc)
	}

	return result, scanner.Err()
}

func (mrf *MapReduceFramework) createWordCountResult(wordCounts map[string]int64, job *WordCountJob) *WordCountResult {
	var totalWords int64
	uniqueWords := int64(len(wordCounts))

	// Calculate total words and create top words list
	var topWords []WordFrequency
	for word, count := range wordCounts {
		totalWords += count
		topWords = append(topWords, WordFrequency{
			Word:      word,
			Frequency: count,
		})
	}

	// Sort top words by frequency
	sort.Slice(topWords, func(i, j int) bool {
		return topWords[i].Frequency > topWords[j].Frequency
	})

	// Keep only top 100
	if len(topWords) > 100 {
		topWords = topWords[:100]
	}

	processingTime := time.Since(job.StartTime)

	return &WordCountResult{
		TotalWords:      totalWords,
		UniqueWords:     uniqueWords,
		TotalFiles:      len(job.InputFiles),
		ProcessingTime:  processingTime,
		TopWords:        topWords,
		WordFrequencies: wordCounts,
		FileStats:       make(map[string]FileStats),
	}
}

// WordCountMapper implementation

func NewWordCountMapper(config MapReduceConfig) *WordCountMapper {
	wordRegex := regexp.MustCompile(config.WordPattern)
	stopWords := make(map[string]bool)
	for _, word := range config.StopWords {
		if !config.CaseSensitive {
			word = strings.ToLower(word)
		}
		stopWords[word] = true
	}

	return &WordCountMapper{
		config:    config,
		wordRegex: wordRegex,
		stopWords: stopWords,
	}
}

func (wcm *WordCountMapper) Map(input string, emitFunc func(key, value string)) error {
	words := wcm.wordRegex.FindAllString(input, -1)

	for _, word := range words {
		// Apply case sensitivity
		if !wcm.config.CaseSensitive {
			word = strings.ToLower(word)
		}

		// Filter by length
		if len(word) < wcm.config.MinWordLength {
			continue
		}

		// Filter stop words
		if wcm.stopWords[word] {
			continue
		}

		emitFunc(word, "1")
	}

	return nil
}

// WordCountReducer implementation

func NewWordCountReducer(config MapReduceConfig) *WordCountReducer {
	return &WordCountReducer{
		config: config,
	}
}

func (wcr *WordCountReducer) Reduce(key string, values []string, emitFunc func(key, value string)) error {
	count := int64(len(values)) // Each value is "1"
	emitFunc(key, fmt.Sprintf("%d", count))
	return nil
}

// Utility functions

func (mrf *MapReduceFramework) hashKey(key string) int {
	hash := 0
	for _, c := range key {
		hash = 31*hash + int(c)
	}
	if hash < 0 {
		hash = -hash
	}
	return hash
}

func (mrf *MapReduceFramework) cleanup(files []string) {
	for _, file := range files {
		os.Remove(file)
	}
}

func getFileSize(filename string) int64 {
	if info, err := os.Stat(filename); err == nil {
		return info.Size()
	}
	return 0
}

// Progress tracking

func (jp *JobProgress) GetProgress() (Phase, int64, int64, int64, time.Duration) {
	jp.mu.RLock()
	defer jp.mu.RUnlock()

	elapsed := time.Since(jp.StartTime)
	return jp.Phase, jp.FilesProcessed, jp.TotalFiles, jp.WordsProcessed, elapsed
}

func (jp *JobProgress) UpdateFilesProcessed(count int64) {
	jp.mu.Lock()
	defer jp.mu.Unlock()
	jp.FilesProcessed += count
}

func (jp *JobProgress) UpdateWordsProcessed(count int64) {
	jp.mu.Lock()
	defer jp.mu.Unlock()
	jp.WordsProcessed += count
}

// SaveResults saves word count results to file
func (wcr *WordCountResult) SaveResults(outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// Write header
	writer.WriteString(fmt.Sprintf("Total Words: %d\n", wcr.TotalWords))
	writer.WriteString(fmt.Sprintf("Unique Words: %d\n", wcr.UniqueWords))
	writer.WriteString(fmt.Sprintf("Total Files: %d\n", wcr.TotalFiles))
	writer.WriteString(fmt.Sprintf("Processing Time: %v\n", wcr.ProcessingTime))
	writer.WriteString("\nTop Words:\n")

	// Write top words
	for i, word := range wcr.TopWords {
		writer.WriteString(fmt.Sprintf("%d. %s: %d\n", i+1, word.Word, word.Frequency))
		if i >= 49 { // Top 50
			break
		}
	}

	return nil
}

// ProcessDirectory processes all text files in a directory
func ProcessDirectory(dirPath string, outputPath string, config MapReduceConfig) (*WordCountResult, error) {
	var inputFiles []string

	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() && (strings.HasSuffix(path, ".txt") || strings.HasSuffix(path, ".log")) {
			inputFiles = append(inputFiles, path)
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	framework := NewMapReduceFramework(config)
	return framework.ProcessWordCount(inputFiles, outputPath, config)
}

// ProcessWithContext processes word count with context for cancellation
func (mrf *MapReduceFramework) ProcessWithContext(ctx context.Context, inputFiles []string, outputPath string, config MapReduceConfig) (*WordCountResult, error) {
	resultChan := make(chan *WordCountResult, 1)
	errorChan := make(chan error, 1)

	go func() {
		result, err := mrf.ProcessWordCount(inputFiles, outputPath, config)
		if err != nil {
			errorChan <- err
		} else {
			resultChan <- result
		}
	}()

	select {
	case result := <-resultChan:
		return result, nil
	case err := <-errorChan:
		return nil, err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Example demonstrates parallel word count MapReduce
func Example() {
	fmt.Println("=== Parallel Word Count MapReduce Example ===")

	// Create sample text files
	sampleTexts := []string{
		"The quick brown fox jumps over the lazy dog. The dog was very lazy.",
		"MapReduce is a programming model for processing large data sets. MapReduce simplifies parallel processing.",
		"Concurrent programming with Go makes parallel processing easier. Go routines are lightweight threads.",
	}

	var inputFiles []string
	for i, text := range sampleTexts {
		filename := fmt.Sprintf("sample_%d.txt", i)
		if err := os.WriteFile(filename, []byte(text), 0644); err != nil {
			fmt.Printf("Failed to create sample file: %v\n", err)
			return
		}
		inputFiles = append(inputFiles, filename)
		defer os.Remove(filename) // Cleanup
	}

	// Configure MapReduce
	config := MapReduceConfig{
		NumMappers:    2,
		NumReducers:   2,
		ChunkSize:     1024,
		CaseSensitive: false,
		MinWordLength: 2,
		StopWords:     []string{"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "was", "are", "were"},
		CleanupTemp:   true,
	}

	// Create framework and process
	framework := NewMapReduceFramework(config)
	result, err := framework.ProcessWordCount(inputFiles, "word_count_results.txt", config)

	if err != nil {
		fmt.Printf("Word count failed: %v\n", err)
		return
	}

	// Display results
	fmt.Printf("Results:\n")
	fmt.Printf("  Total Words: %d\n", result.TotalWords)
	fmt.Printf("  Unique Words: %d\n", result.UniqueWords)
	fmt.Printf("  Processing Time: %v\n", result.ProcessingTime)
	fmt.Printf("  Files Processed: %d\n", result.TotalFiles)

	fmt.Printf("\nTop 10 Words:\n")
	for i, word := range result.TopWords {
		if i >= 10 {
			break
		}
		fmt.Printf("  %d. %s: %d\n", i+1, word.Word, word.Frequency)
	}

	// Save results
	if err := result.SaveResults("word_count_results.txt"); err != nil {
		fmt.Printf("Failed to save results: %v\n", err)
	} else {
		fmt.Println("\nResults saved to word_count_results.txt")
	}

	// Cleanup
	os.Remove("word_count_results.txt")
}