package parallelwordcountmapreduce

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func createTestFiles(t *testing.T) ([]string, func()) {
	testData := []string{
		"the quick brown fox jumps over the lazy dog",
		"hello world hello universe hello golang",
		"mapreduce is a programming model for processing large data sets",
		"concurrent programming with go makes parallel processing easier",
		"the dog was lazy but the fox was quick",
	}

	var files []string
	tempDir := t.TempDir()

	for i, data := range testData {
		filename := filepath.Join(tempDir, fmt.Sprintf("test_%d.txt", i))
		if err := os.WriteFile(filename, []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
		files = append(files, filename)
	}

	cleanup := func() {
		os.RemoveAll(tempDir)
	}

	return files, cleanup
}

func TestMapReduceFramework(t *testing.T) {
	files, cleanup := createTestFiles(t)
	defer cleanup()

	config := MapReduceConfig{
		NumMappers:    2,
		NumReducers:   2,
		ChunkSize:     1024,
		CaseSensitive: false,
		MinWordLength: 1,
		CleanupTemp:   true,
	}

	framework := NewMapReduceFramework(config)
	result, err := framework.ProcessWordCount(files, "", config)

	if err != nil {
		t.Fatalf("Word count failed: %v", err)
	}

	// Verify basic results
	if result.TotalWords == 0 {
		t.Error("Total words should be greater than 0")
	}

	if result.UniqueWords == 0 {
		t.Error("Unique words should be greater than 0")
	}

	if result.TotalFiles != len(files) {
		t.Errorf("Expected %d files, got %d", len(files), result.TotalFiles)
	}

	if result.ProcessingTime <= 0 {
		t.Error("Processing time should be positive")
	}

	// Check for expected words
	expectedWords := []string{"hello", "the", "programming", "go"}
	for _, word := range expectedWords {
		if _, exists := result.WordFrequencies[word]; !exists {
			t.Errorf("Expected word '%s' not found in results", word)
		}
	}

	// Verify top words are sorted by frequency
	for i := 1; i < len(result.TopWords); i++ {
		if result.TopWords[i-1].Frequency < result.TopWords[i].Frequency {
			t.Error("Top words should be sorted by frequency in descending order")
			break
		}
	}

	t.Logf("Total words: %d, Unique words: %d, Processing time: %v",
		result.TotalWords, result.UniqueWords, result.ProcessingTime)
}

func TestWordCountMapper(t *testing.T) {
	config := MapReduceConfig{
		CaseSensitive: false,
		MinWordLength: 2,
		WordPattern:   `\b[a-zA-Z]+\b`,
		StopWords:     []string{"the", "and", "or"},
	}

	mapper := NewWordCountMapper(config)

	var emittedPairs []struct {
		key   string
		value string
	}

	emitFunc := func(key, value string) {
		emittedPairs = append(emittedPairs, struct {
			key   string
			value string
		}{key, value})
	}

	input := "The quick brown fox and the lazy dog"
	err := mapper.Map(input, emitFunc)

	if err != nil {
		t.Fatalf("Mapper failed: %v", err)
	}

	// Check that words were emitted
	if len(emittedPairs) == 0 {
		t.Error("No words were emitted")
	}

	// Check that stop words were filtered
	for _, pair := range emittedPairs {
		if pair.key == "the" || pair.key == "and" {
			t.Errorf("Stop word '%s' should have been filtered", pair.key)
		}
	}

	// Check that short words were filtered (< 2 characters)
	for _, pair := range emittedPairs {
		if len(pair.key) < 2 {
			t.Errorf("Short word '%s' should have been filtered", pair.key)
		}
	}

	// All values should be "1"
	for _, pair := range emittedPairs {
		if pair.value != "1" {
			t.Errorf("Expected value '1', got '%s'", pair.value)
		}
	}

	t.Logf("Emitted %d word pairs", len(emittedPairs))
}

func TestWordCountReducer(t *testing.T) {
	config := MapReduceConfig{}
	reducer := NewWordCountReducer(config)

	var result []struct {
		key   string
		value string
	}

	emitFunc := func(key, value string) {
		result = append(result, struct {
			key   string
			value string
		}{key, value})
	}

	// Test reducing multiple occurrences of same word
	key := "hello"
	values := []string{"1", "1", "1", "1", "1"} // 5 occurrences

	err := reducer.Reduce(key, values, emitFunc)
	if err != nil {
		t.Fatalf("Reducer failed: %v", err)
	}

	if len(result) != 1 {
		t.Errorf("Expected 1 result, got %d", len(result))
	}

	if result[0].key != key {
		t.Errorf("Expected key '%s', got '%s'", key, result[0].key)
	}

	if result[0].value != "5" {
		t.Errorf("Expected count '5', got '%s'", result[0].value)
	}
}

func TestCaseSensitivity(t *testing.T) {
	tempDir := t.TempDir()
	testFile := filepath.Join(tempDir, "case_test.txt")
	
	content := "Hello HELLO hello HeLLo"
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	// Test case-sensitive
	config := MapReduceConfig{
		NumMappers:    1,
		NumReducers:   1,
		CaseSensitive: true,
		MinWordLength: 1,
		CleanupTemp:   true,
	}

	framework := NewMapReduceFramework(config)
	result, err := framework.ProcessWordCount([]string{testFile}, "", config)
	if err != nil {
		t.Fatalf("Case-sensitive test failed: %v", err)
	}

	// Should have different entries for different cases
	caseVariants := 0
	for word := range result.WordFrequencies {
		if strings.ToLower(word) == "hello" {
			caseVariants++
		}
	}

	if caseVariants < 2 {
		t.Errorf("Expected multiple case variants, got %d", caseVariants)
	}

	// Test case-insensitive
	config.CaseSensitive = false
	result, err = framework.ProcessWordCount([]string{testFile}, "", config)
	if err != nil {
		t.Fatalf("Case-insensitive test failed: %v", err)
	}

	// Should have only one entry for "hello"
	if count, exists := result.WordFrequencies["hello"]; !exists {
		t.Error("Expected 'hello' entry in case-insensitive mode")
	} else if count != 4 {
		t.Errorf("Expected count 4 for 'hello', got %d", count)
	}
}

func TestStopWords(t *testing.T) {
	tempDir := t.TempDir()
	testFile := filepath.Join(tempDir, "stop_test.txt")
	
	content := "the quick brown fox and the lazy dog"
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	config := MapReduceConfig{
		NumMappers:    1,
		NumReducers:   1,
		CaseSensitive: false,
		MinWordLength: 1,
		StopWords:     []string{"the", "and"},
		CleanupTemp:   true,
	}

	framework := NewMapReduceFramework(config)
	result, err := framework.ProcessWordCount([]string{testFile}, "", config)
	if err != nil {
		t.Fatalf("Stop words test failed: %v", err)
	}

	// Check that stop words are not in results
	for _, stopWord := range config.StopWords {
		if _, exists := result.WordFrequencies[stopWord]; exists {
			t.Errorf("Stop word '%s' should not be in results", stopWord)
		}
	}

	// Check that non-stop words are present
	expectedWords := []string{"quick", "brown", "fox", "lazy", "dog"}
	for _, word := range expectedWords {
		if _, exists := result.WordFrequencies[word]; !exists {
			t.Errorf("Expected word '%s' not found", word)
		}
	}
}

func TestMinWordLength(t *testing.T) {
	tempDir := t.TempDir()
	testFile := filepath.Join(tempDir, "length_test.txt")
	
	content := "a bb ccc dddd eeeee"
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	config := MapReduceConfig{
		NumMappers:    1,
		NumReducers:   1,
		MinWordLength: 3,
		CleanupTemp:   true,
	}

	framework := NewMapReduceFramework(config)
	result, err := framework.ProcessWordCount([]string{testFile}, "", config)
	if err != nil {
		t.Fatalf("Min word length test failed: %v", err)
	}

	// Only words with 3+ characters should be present
	expectedWords := []string{"ccc", "dddd", "eeeee"}
	unexpectedWords := []string{"a", "bb"}

	for _, word := range expectedWords {
		if _, exists := result.WordFrequencies[word]; !exists {
			t.Errorf("Expected word '%s' not found", word)
		}
	}

	for _, word := range unexpectedWords {
		if _, exists := result.WordFrequencies[word]; exists {
			t.Errorf("Word '%s' should have been filtered by length", word)
		}
	}
}

func TestLargeFile(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large file test in short mode")
	}

	tempDir := t.TempDir()
	testFile := filepath.Join(tempDir, "large_test.txt")

	// Create a large file with repeated words
	var content strings.Builder
	words := []string{"apple", "banana", "cherry", "date", "elderberry"}
	
	for i := 0; i < 10000; i++ {
		word := words[i%len(words)]
		content.WriteString(word + " ")
	}

	if err := os.WriteFile(testFile, []byte(content.String()), 0644); err != nil {
		t.Fatal(err)
	}

	config := MapReduceConfig{
		NumMappers:    4,
		NumReducers:   4,
		ChunkSize:     1024, // Small chunks to test chunking
		CaseSensitive: false,
		MinWordLength: 1,
		CleanupTemp:   true,
	}

	framework := NewMapReduceFramework(config)
	start := time.Now()
	result, err := framework.ProcessWordCount([]string{testFile}, "", config)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Large file test failed: %v", err)
	}

	// Each word should appear 2000 times
	expectedCount := int64(2000)
	for _, word := range words {
		if count, exists := result.WordFrequencies[word]; !exists {
			t.Errorf("Expected word '%s' not found", word)
		} else if count != expectedCount {
			t.Errorf("Word '%s': expected count %d, got %d", word, expectedCount, count)
		}
	}

	totalExpected := int64(len(words)) * expectedCount
	if result.TotalWords != totalExpected {
		t.Errorf("Expected total words %d, got %d", totalExpected, result.TotalWords)
	}

	t.Logf("Large file processed in %v", duration)
}

func TestFileChunking(t *testing.T) {
	tempDir := t.TempDir()
	testFile := filepath.Join(tempDir, "chunk_test.txt")

	// Create content that will be split across chunks
	content := strings.Repeat("word ", 1000) // 5000 bytes approximately
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	config := MapReduceConfig{
		NumMappers:    2,
		NumReducers:   2,
		ChunkSize:     2000, // Will create multiple chunks
		CaseSensitive: false,
		MinWordLength: 1,
		CleanupTemp:   true,
	}

	framework := NewMapReduceFramework(config)
	result, err := framework.ProcessWordCount([]string{testFile}, "", config)
	if err != nil {
		t.Fatalf("Chunking test failed: %v", err)
	}

	// Should have exactly 1000 occurrences of "word"
	if count, exists := result.WordFrequencies["word"]; !exists {
		t.Error("Expected word 'word' not found")
	} else if count != 1000 {
		t.Errorf("Expected count 1000, got %d", count)
	}
}

func TestConcurrentProcessing(t *testing.T) {
	files, cleanup := createTestFiles(t)
	defer cleanup()

	config := MapReduceConfig{
		NumMappers:    4,
		NumReducers:   4,
		ChunkSize:     1024,
		CaseSensitive: false,
		MinWordLength: 1,
		CleanupTemp:   true,
	}

	framework := NewMapReduceFramework(config)

	// Run multiple word counts concurrently
	numConcurrent := 3
	results := make([]*WordCountResult, numConcurrent)
	errors := make([]error, numConcurrent)

	var wg sync.WaitGroup
	for i := 0; i < numConcurrent; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx], errors[idx] = framework.ProcessWordCount(files, "", config)
		}(i)
	}

	wg.Wait()

	// Check that all runs completed successfully
	for i, err := range errors {
		if err != nil {
			t.Errorf("Concurrent run %d failed: %v", i, err)
		}
	}

	// Results should be identical
	for i := 1; i < len(results); i++ {
		if results[0].TotalWords != results[i].TotalWords {
			t.Errorf("Total words mismatch between runs: %d vs %d", 
				results[0].TotalWords, results[i].TotalWords)
		}
		if results[0].UniqueWords != results[i].UniqueWords {
			t.Errorf("Unique words mismatch between runs: %d vs %d", 
				results[0].UniqueWords, results[i].UniqueWords)
		}
	}
}

func TestProcessWithContext(t *testing.T) {
	files, cleanup := createTestFiles(t)
	defer cleanup()

	config := MapReduceConfig{
		NumMappers:  2,
		NumReducers: 2,
		CleanupTemp: true,
	}

	framework := NewMapReduceFramework(config)

	// Test successful processing with context
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	result, err := framework.ProcessWithContext(ctx, files, "", config)
	if err != nil {
		t.Fatalf("Context processing failed: %v", err)
	}

	if result.TotalWords == 0 {
		t.Error("Expected some words to be processed")
	}

	// Test cancellation
	ctx2, cancel2 := context.WithCancel(context.Background())
	cancel2() // Cancel immediately

	_, err = framework.ProcessWithContext(ctx2, files, "", config)
	if err == nil {
		t.Error("Expected cancellation error")
	}
}

func TestSaveResults(t *testing.T) {
	result := &WordCountResult{
		TotalWords:     100,
		UniqueWords:    50,
		TotalFiles:     3,
		ProcessingTime: time.Second,
		TopWords: []WordFrequency{
			{Word: "test", Frequency: 10},
			{Word: "word", Frequency: 8},
			{Word: "example", Frequency: 6},
		},
	}

	tempDir := t.TempDir()
	outputFile := filepath.Join(tempDir, "results.txt")

	err := result.SaveResults(outputFile)
	if err != nil {
		t.Fatalf("Failed to save results: %v", err)
	}

	// Verify file was created and has content
	content, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("Failed to read results file: %v", err)
	}

	contentStr := string(content)
	expectedStrings := []string{
		"Total Words: 100",
		"Unique Words: 50",
		"Total Files: 3",
		"test: 10",
		"word: 8",
		"example: 6",
	}

	for _, expected := range expectedStrings {
		if !strings.Contains(contentStr, expected) {
			t.Errorf("Expected '%s' in results file", expected)
		}
	}
}

func TestProcessDirectory(t *testing.T) {
	tempDir := t.TempDir()

	// Create multiple text files in directory
	files := []string{
		"file1.txt",
		"file2.txt", 
		"file3.log",
		"file4.bin", // Should be ignored
	}

	contents := []string{
		"hello world from file one",
		"hello universe from file two", 
		"hello golang from file three",
		"binary content",
	}

	for i, filename := range files {
		path := filepath.Join(tempDir, filename)
		if err := os.WriteFile(path, []byte(contents[i]), 0644); err != nil {
			t.Fatal(err)
		}
	}

	config := MapReduceConfig{
		NumMappers:  2,
		NumReducers: 2,
		CleanupTemp: true,
	}

	result, err := ProcessDirectory(tempDir, "", config)
	if err != nil {
		t.Fatalf("Process directory failed: %v", err)
	}

	// Should process 3 files (.txt and .log), ignore .bin
	if result.TotalFiles != 3 {
		t.Errorf("Expected 3 files processed, got %d", result.TotalFiles)
	}

	// Should find "hello" 3 times
	if count, exists := result.WordFrequencies["hello"]; !exists {
		t.Error("Expected 'hello' in results")
	} else if count != 3 {
		t.Errorf("Expected 'hello' count 3, got %d", count)
	}
}

func BenchmarkWordCount(b *testing.B) {
	// Create test file
	tempDir := b.TempDir()
	testFile := filepath.Join(tempDir, "bench.txt")
	
	content := strings.Repeat("the quick brown fox jumps over the lazy dog ", 1000)
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		b.Fatal(err)
	}

	config := MapReduceConfig{
		NumMappers:  runtime.NumCPU(),
		NumReducers: runtime.NumCPU(),
		CleanupTemp: true,
	}

	framework := NewMapReduceFramework(config)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := framework.ProcessWordCount([]string{testFile}, "", config)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMapperWorkers(b *testing.B) {
	tempDir := b.TempDir()
	testFile := filepath.Join(tempDir, "bench.txt")
	
	content := strings.Repeat("benchmark test word counting performance ", 10000)
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		b.Fatal(err)
	}

	workerCounts := []int{1, 2, 4, 8}

	for _, workers := range workerCounts {
		b.Run(fmt.Sprintf("Mappers%d", workers), func(b *testing.B) {
			config := MapReduceConfig{
				NumMappers:  workers,
				NumReducers: workers,
				CleanupTemp: true,
			}

			framework := NewMapReduceFramework(config)

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, err := framework.ProcessWordCount([]string{testFile}, "", config)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}