package paralleltextsearch

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func createTestFiles(t *testing.T) (string, []string) {
	// Create temporary directory
	dir, err := ioutil.TempDir("", "textsearch_test")
	if err != nil {
		t.Fatal(err)
	}
	
	files := []string{
		"file1.txt",
		"file2.txt",
		"file3.go",
		"file4.md",
		"file5.txt",
	}
	
	contents := []string{
		"The quick brown fox jumps over the lazy dog.\nThis is a test file with some text.\nFox is clever.",
		"Another file with different content.\nThe fox appears here too.\nGolang is great for concurrent programming.",
		"package main\n\nfunc main() {\n\t// Fox in a comment\n\tprintln(\"Hello, fox!\")\n}",
		"# Markdown file\n\nThe **fox** is a beautiful animal.\n\n## Section about foxes\n\nFoxes are clever.",
		"No matches here.\nJust some random text.\nNothing to see.",
	}
	
	createdFiles := make([]string, len(files))
	
	for i, file := range files {
		path := filepath.Join(dir, file)
		err := ioutil.WriteFile(path, []byte(contents[i]), 0644)
		if err != nil {
			t.Fatal(err)
		}
		createdFiles[i] = path
	}
	
	return dir, createdFiles
}

func TestBasicSearch(t *testing.T) {
	dir, files := createTestFiles(t)
	defer os.RemoveAll(dir)
	
	options := SearchOptions{
		Pattern:       "fox",
		CaseSensitive: false,
		MaxWorkers:    2,
	}
	
	searcher, err := NewTextSearcher(options)
	if err != nil {
		t.Fatal(err)
	}
	
	result := searcher.SearchFiles(files)
	
	if len(result.Matches) == 0 {
		t.Error("Expected to find matches")
	}
	
	// Count matches per file
	matchCount := make(map[string]int)
	for _, match := range result.Matches {
		matchCount[match.File]++
	}
	
	// file1.txt should have 2 matches
	if matchCount[files[0]] != 2 {
		t.Errorf("Expected 2 matches in file1.txt, got %d", matchCount[files[0]])
	}
	
	// file5.txt should have 0 matches
	if matchCount[files[4]] != 0 {
		t.Errorf("Expected 0 matches in file5.txt, got %d", matchCount[files[4]])
	}
}

func TestCaseSensitive(t *testing.T) {
	dir, files := createTestFiles(t)
	defer os.RemoveAll(dir)
	
	// Case sensitive search
	options := SearchOptions{
		Pattern:       "Fox",
		CaseSensitive: true,
		MaxWorkers:    2,
	}
	
	searcher, err := NewTextSearcher(options)
	if err != nil {
		t.Fatal(err)
	}
	
	result := searcher.SearchFiles(files)
	
	// Should only match "Fox" with capital F
	for _, match := range result.Matches {
		if !strings.Contains(match.Text, "Fox") {
			t.Errorf("Case sensitive search returned incorrect match: %s", match.Text)
		}
	}
}

func TestWholeWord(t *testing.T) {
	dir, _ := createTestFiles(t)
	defer os.RemoveAll(dir)
	
	// Create a file with partial matches
	testFile := filepath.Join(dir, "whole_word_test.txt")
	content := "foxes foxy fox prefox foxsuffix fox."
	err := ioutil.WriteFile(testFile, []byte(content), 0644)
	if err != nil {
		t.Fatal(err)
	}
	
	options := SearchOptions{
		Pattern:       "fox",
		WholeWord:     true,
		CaseSensitive: false,
		MaxWorkers:    1,
	}
	
	searcher, err := NewTextSearcher(options)
	if err != nil {
		t.Fatal(err)
	}
	
	result := searcher.SearchFiles([]string{testFile})
	
	// Should match "fox" but not "foxes", "foxy", etc.
	if len(result.Matches) != 2 { // "fox" appears twice as whole word
		t.Errorf("Expected 2 whole word matches, got %d", len(result.Matches))
	}
}

func TestRegexSearch(t *testing.T) {
	dir, files := createTestFiles(t)
	defer os.RemoveAll(dir)
	
	options := SearchOptions{
		Pattern:    `[Tt]he\s+\w+\s+fox`,
		Regex:      true,
		MaxWorkers: 2,
	}
	
	searcher, err := NewTextSearcher(options)
	if err != nil {
		t.Fatal(err)
	}
	
	result := searcher.SearchFiles(files)
	
	if len(result.Matches) == 0 {
		t.Error("Expected regex matches")
	}
	
	// Verify matches follow the pattern
	for _, match := range result.Matches {
		text := match.Text[match.Column-1:]
		if !strings.HasPrefix(strings.ToLower(text), "the ") {
			t.Errorf("Regex match doesn't start with 'the': %s", text)
		}
	}
}

func TestContextLines(t *testing.T) {
	dir, _ := createTestFiles(t)
	defer os.RemoveAll(dir)
	
	// Create a file with multiple lines
	testFile := filepath.Join(dir, "context_test.txt")
	content := "Line 1\nLine 2\nThe fox is here\nLine 4\nLine 5"
	err := ioutil.WriteFile(testFile, []byte(content), 0644)
	if err != nil {
		t.Fatal(err)
	}
	
	options := SearchOptions{
		Pattern:      "fox",
		ContextLines: 2,
		MaxWorkers:   1,
	}
	
	searcher, err := NewTextSearcher(options)
	if err != nil {
		t.Fatal(err)
	}
	
	result := searcher.SearchFiles([]string{testFile})
	
	if len(result.Matches) != 1 {
		t.Fatal("Expected exactly one match")
	}
	
	match := result.Matches[0]
	
	// Should have 5 context lines (2 before, match line, 2 after)
	if len(match.Context) != 5 {
		t.Errorf("Expected 5 context lines, got %d", len(match.Context))
	}
	
	// Verify context content
	expectedContext := []string{"Line 1", "Line 2", "The fox is here", "Line 4", "Line 5"}
	for i, line := range match.Context {
		if line != expectedContext[i] {
			t.Errorf("Context line %d: expected '%s', got '%s'", i, expectedContext[i], line)
		}
	}
}

func TestFileExtensions(t *testing.T) {
	dir, files := createTestFiles(t)
	defer os.RemoveAll(dir)
	
	// Search only in .txt files
	options := SearchOptions{
		Pattern:           "fox",
		IncludeExtensions: []string{".txt"},
		MaxWorkers:        2,
	}
	
	searcher, err := NewTextSearcher(options)
	if err != nil {
		t.Fatal(err)
	}
	
	result := searcher.SearchDirectory(dir)
	
	// Verify all matches are from .txt files
	for _, match := range result.Matches {
		if !strings.HasSuffix(match.File, ".txt") {
			t.Errorf("Found match in non-.txt file: %s", match.File)
		}
	}
	
	// Search excluding .txt files
	options2 := SearchOptions{
		Pattern:           "fox",
		ExcludeExtensions: []string{".txt"},
		MaxWorkers:        2,
	}
	
	searcher2, _ := NewTextSearcher(options2)
	result2 := searcher2.SearchDirectory(dir)
	
	// Verify no matches are from .txt files
	for _, match := range result2.Matches {
		if strings.HasSuffix(match.File, ".txt") {
			t.Errorf("Found match in excluded .txt file: %s", match.File)
		}
	}
}

func TestLargeFile(t *testing.T) {
	dir, _ := createTestFiles(t)
	defer os.RemoveAll(dir)
	
	// Create a large file
	largeFile := filepath.Join(dir, "large.txt")
	var content strings.Builder
	for i := 0; i < 10000; i++ {
		if i == 5000 {
			content.WriteString("The fox appears in the middle\n")
		} else {
			content.WriteString(fmt.Sprintf("Line %d with no matches\n", i))
		}
	}
	
	err := ioutil.WriteFile(largeFile, []byte(content.String()), 0644)
	if err != nil {
		t.Fatal(err)
	}
	
	options := SearchOptions{
		Pattern:    "fox",
		MaxWorkers: 4,
		BufferSize: 128 * 1024,
	}
	
	searcher, err := NewTextSearcher(options)
	if err != nil {
		t.Fatal(err)
	}
	
	start := time.Now()
	result := searcher.SearchFiles([]string{largeFile})
	elapsed := time.Since(start)
	
	if len(result.Matches) != 1 {
		t.Errorf("Expected 1 match in large file, got %d", len(result.Matches))
	}
	
	if result.Matches[0].Line != 5001 {
		t.Errorf("Expected match at line 5001, got %d", result.Matches[0].Line)
	}
	
	t.Logf("Large file search took %v", elapsed)
}

func TestStreamingSearch(t *testing.T) {
	dir, files := createTestFiles(t)
	defer os.RemoveAll(dir)
	
	options := SearchOptions{
		Pattern:    "fox",
		MaxWorkers: 2,
	}
	
	streamer, err := NewStreamingSearcher(options)
	if err != nil {
		t.Fatal(err)
	}
	
	matchChan := streamer.Start(files)
	
	matches := make([]Match, 0)
	for match := range matchChan {
		matches = append(matches, match)
	}
	
	if len(matches) == 0 {
		t.Error("Expected streaming search to find matches")
	}
}

func TestIndexedSearch(t *testing.T) {
	dir, files := createTestFiles(t)
	defer os.RemoveAll(dir)
	
	indexer := NewIndexedSearcher()
	
	// Build index
	err := indexer.BuildIndex(files, 2)
	if err != nil {
		t.Fatal(err)
	}
	
	// Search for a word
	results := indexer.Search("fox")
	
	if len(results) == 0 {
		t.Error("Expected indexed search to find results")
	}
	
	// Verify results
	for _, entry := range results {
		found := false
		for _, file := range files {
			if entry.File == file {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Indexed search returned unknown file: %s", entry.File)
		}
	}
}

func TestConcurrentSearches(t *testing.T) {
	dir, files := createTestFiles(t)
	defer os.RemoveAll(dir)
	
	// Run multiple searches concurrently
	patterns := []string{"fox", "the", "golang", "text"}
	results := make(chan *SearchResult, len(patterns))
	
	for _, pattern := range patterns {
		go func(p string) {
			options := SearchOptions{
				Pattern:    p,
				MaxWorkers: 2,
			}
			searcher, _ := NewTextSearcher(options)
			results <- searcher.SearchFiles(files)
		}(pattern)
	}
	
	// Collect results
	for i := 0; i < len(patterns); i++ {
		result := <-results
		if result.FilesSearched != len(files) {
			t.Errorf("Expected to search %d files, searched %d", len(files), result.FilesSearched)
		}
	}
}

func BenchmarkTextSearch(b *testing.B) {
	dir, files := createTestFiles(b)
	defer os.RemoveAll(dir)
	
	// Create more files for benchmarking
	for i := 0; i < 50; i++ {
		file := filepath.Join(dir, fmt.Sprintf("bench%d.txt", i))
		content := strings.Repeat("The quick brown fox jumps over the lazy dog.\n", 100)
		ioutil.WriteFile(file, []byte(content), 0644)
		files = append(files, file)
	}
	
	workerCounts := []int{1, 2, 4, 8}
	
	for _, workers := range workerCounts {
		b.Run(fmt.Sprintf("Workers_%d", workers), func(b *testing.B) {
			options := SearchOptions{
				Pattern:    "fox",
				MaxWorkers: workers,
			}
			
			searcher, _ := NewTextSearcher(options)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				searcher.SearchFiles(files)
			}
		})
	}
}

func BenchmarkRegexVsPlainText(b *testing.B) {
	dir, _ := createTestFiles(b)
	defer os.RemoveAll(dir)
	
	// Create test file
	file := filepath.Join(dir, "benchmark.txt")
	content := strings.Repeat("The quick brown fox jumps over the lazy dog.\n", 1000)
	ioutil.WriteFile(file, []byte(content), 0644)
	
	files := []string{file}
	
	b.Run("PlainText", func(b *testing.B) {
		options := SearchOptions{
			Pattern:    "fox",
			MaxWorkers: 1,
		}
		searcher, _ := NewTextSearcher(options)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			searcher.SearchFiles(files)
		}
	})
	
	b.Run("Regex", func(b *testing.B) {
		options := SearchOptions{
			Pattern:    `\bfox\b`,
			Regex:      true,
			MaxWorkers: 1,
		}
		searcher, _ := NewTextSearcher(options)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			searcher.SearchFiles(files)
		}
	})
}