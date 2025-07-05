package paralleltextsearch

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Match represents a search match
type Match struct {
	File       string
	Line       int
	Column     int
	Text       string
	Context    []string // Lines around the match for context
}

// SearchResult represents the result of a search
type SearchResult struct {
	Matches      []Match
	FilesSearched int
	TimeElapsed  time.Duration
	Errors       []error
}

// SearchOptions configures the search behavior
type SearchOptions struct {
	Pattern          string
	Regex            bool
	CaseSensitive    bool
	WholeWord        bool
	MaxWorkers       int
	BufferSize       int
	ContextLines     int
	IncludeExtensions []string
	ExcludeExtensions []string
	MaxFileSize      int64
}

// TextSearcher performs parallel text searches
type TextSearcher struct {
	options SearchOptions
	pattern *regexp.Regexp
}

// NewTextSearcher creates a new text searcher
func NewTextSearcher(options SearchOptions) (*TextSearcher, error) {
	if options.MaxWorkers <= 0 {
		options.MaxWorkers = 4
	}
	if options.BufferSize <= 0 {
		options.BufferSize = 64 * 1024 // 64KB
	}
	
	searcher := &TextSearcher{
		options: options,
	}
	
	// Compile pattern
	patternStr := options.Pattern
	if !options.Regex {
		patternStr = regexp.QuoteMeta(patternStr)
	}
	if options.WholeWord {
		patternStr = `\b` + patternStr + `\b`
	}
	
	flags := ""
	if !options.CaseSensitive {
		flags = "(?i)"
	}
	
	pattern, err := regexp.Compile(flags + patternStr)
	if err != nil {
		return nil, fmt.Errorf("invalid pattern: %w", err)
	}
	
	searcher.pattern = pattern
	return searcher, nil
}

// SearchFiles searches for pattern in multiple files
func (ts *TextSearcher) SearchFiles(files []string) *SearchResult {
	start := time.Now()
	result := &SearchResult{
		Matches: make([]Match, 0),
		Errors:  make([]error, 0),
	}
	
	// Channels for work distribution
	fileChan := make(chan string, len(files))
	matchChan := make(chan []Match, len(files))
	errorChan := make(chan error, len(files))
	
	var wg sync.WaitGroup
	var filesSearched int32
	
	// Start workers
	for i := 0; i < ts.options.MaxWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ts.worker(fileChan, matchChan, errorChan, &filesSearched)
		}()
	}
	
	// Send files to workers
	for _, file := range files {
		fileChan <- file
	}
	close(fileChan)
	
	// Collect results
	go func() {
		wg.Wait()
		close(matchChan)
		close(errorChan)
	}()
	
	// Gather matches
	for matches := range matchChan {
		result.Matches = append(result.Matches, matches...)
	}
	
	// Gather errors
	for err := range errorChan {
		if err != nil {
			result.Errors = append(result.Errors, err)
		}
	}
	
	result.FilesSearched = int(atomic.LoadInt32(&filesSearched))
	result.TimeElapsed = time.Since(start)
	
	return result
}

// SearchDirectory searches recursively in a directory
func (ts *TextSearcher) SearchDirectory(dir string) *SearchResult {
	files := make([]string, 0)
	
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip files with errors
		}
		
		if info.IsDir() {
			return nil
		}
		
		// Check file size
		if ts.options.MaxFileSize > 0 && info.Size() > ts.options.MaxFileSize {
			return nil
		}
		
		// Check extensions
		ext := strings.ToLower(filepath.Ext(path))
		
		if len(ts.options.IncludeExtensions) > 0 {
			included := false
			for _, inc := range ts.options.IncludeExtensions {
				if ext == inc {
					included = true
					break
				}
			}
			if !included {
				return nil
			}
		}
		
		if len(ts.options.ExcludeExtensions) > 0 {
			for _, exc := range ts.options.ExcludeExtensions {
				if ext == exc {
					return nil
				}
			}
		}
		
		files = append(files, path)
		return nil
	})
	
	result := ts.SearchFiles(files)
	
	if err != nil {
		result.Errors = append(result.Errors, err)
	}
	
	return result
}

// worker processes files from the file channel
func (ts *TextSearcher) worker(fileChan <-chan string, matchChan chan<- []Match, 
	errorChan chan<- error, filesSearched *int32) {
	
	for file := range fileChan {
		matches, err := ts.searchFile(file)
		if err != nil {
			errorChan <- fmt.Errorf("error searching %s: %w", file, err)
		} else {
			if len(matches) > 0 {
				matchChan <- matches
			}
			atomic.AddInt32(filesSearched, 1)
		}
	}
}

// searchFile searches for pattern in a single file
func (ts *TextSearcher) searchFile(filename string) ([]Match, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	matches := make([]Match, 0)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, ts.options.BufferSize), ts.options.BufferSize)
	
	lineNum := 0
	var lines []string
	
	// If context is requested, maintain a sliding window of lines
	if ts.options.ContextLines > 0 {
		lines = make([]string, 0, ts.options.ContextLines*2+1)
	}
	
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		
		// Maintain context lines
		if ts.options.ContextLines > 0 {
			lines = append(lines, line)
			if len(lines) > ts.options.ContextLines*2+1 {
				lines = lines[1:]
			}
		}
		
		// Find matches in line
		indices := ts.pattern.FindAllStringIndex(line, -1)
		for _, loc := range indices {
			match := Match{
				File:   filename,
				Line:   lineNum,
				Column: loc[0] + 1,
				Text:   line,
			}
			
			// Add context if requested
			if ts.options.ContextLines > 0 && len(lines) > 0 {
				contextStart := len(lines) - ts.options.ContextLines - 1
				if contextStart < 0 {
					contextStart = 0
				}
				match.Context = make([]string, len(lines[contextStart:]))
				copy(match.Context, lines[contextStart:])
			}
			
			matches = append(matches, match)
		}
	}
	
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	
	return matches, nil
}

// StreamingSearcher performs streaming search with results sent via channel
type StreamingSearcher struct {
	*TextSearcher
	matchChan chan Match
	doneChan  chan bool
	wg        sync.WaitGroup
}

// NewStreamingSearcher creates a new streaming searcher
func NewStreamingSearcher(options SearchOptions) (*StreamingSearcher, error) {
	ts, err := NewTextSearcher(options)
	if err != nil {
		return nil, err
	}
	
	return &StreamingSearcher{
		TextSearcher: ts,
		matchChan:    make(chan Match, 100),
		doneChan:     make(chan bool),
	}, nil
}

// Start begins the streaming search
func (ss *StreamingSearcher) Start(files []string) <-chan Match {
	go func() {
		fileChan := make(chan string, len(files))
		
		// Start workers
		for i := 0; i < ss.options.MaxWorkers; i++ {
			ss.wg.Add(1)
			go ss.streamingWorker(fileChan)
		}
		
		// Send files
		for _, file := range files {
			fileChan <- file
		}
		close(fileChan)
		
		// Wait and cleanup
		ss.wg.Wait()
		close(ss.matchChan)
	}()
	
	return ss.matchChan
}

func (ss *StreamingSearcher) streamingWorker(fileChan <-chan string) {
	defer ss.wg.Done()
	
	for file := range fileChan {
		select {
		case <-ss.doneChan:
			return
		default:
			matches, err := ss.searchFile(file)
			if err == nil {
				for _, match := range matches {
					select {
					case ss.matchChan <- match:
					case <-ss.doneChan:
						return
					}
				}
			}
		}
	}
}

// Stop stops the streaming search
func (ss *StreamingSearcher) Stop() {
	close(ss.doneChan)
}

// IndexedSearcher uses an index for faster searches
type IndexedSearcher struct {
	index      map[string][]IndexEntry
	indexMutex sync.RWMutex
}

// IndexEntry represents an entry in the search index
type IndexEntry struct {
	File   string
	Line   int
	Column int
}

// NewIndexedSearcher creates a new indexed searcher
func NewIndexedSearcher() *IndexedSearcher {
	return &IndexedSearcher{
		index: make(map[string][]IndexEntry),
	}
}

// BuildIndex builds an index from files
func (is *IndexedSearcher) BuildIndex(files []string, numWorkers int) error {
	if numWorkers <= 0 {
		numWorkers = 4
	}
	
	fileChan := make(chan string, len(files))
	indexChan := make(chan map[string][]IndexEntry, len(files))
	var wg sync.WaitGroup
	
	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			is.indexWorker(fileChan, indexChan)
		}()
	}
	
	// Send files
	for _, file := range files {
		fileChan <- file
	}
	close(fileChan)
	
	// Collect results
	go func() {
		wg.Wait()
		close(indexChan)
	}()
	
	// Merge indices
	for partialIndex := range indexChan {
		is.indexMutex.Lock()
		for word, entries := range partialIndex {
			is.index[word] = append(is.index[word], entries...)
		}
		is.indexMutex.Unlock()
	}
	
	return nil
}

func (is *IndexedSearcher) indexWorker(fileChan <-chan string, indexChan chan<- map[string][]IndexEntry) {
	for file := range fileChan {
		localIndex := make(map[string][]IndexEntry)
		
		f, err := os.Open(file)
		if err != nil {
			continue
		}
		
		scanner := bufio.NewScanner(f)
		lineNum := 0
		
		for scanner.Scan() {
			lineNum++
			line := scanner.Text()
			words := strings.Fields(line)
			
			for col, word := range words {
				word = strings.ToLower(word)
				localIndex[word] = append(localIndex[word], IndexEntry{
					File:   file,
					Line:   lineNum,
					Column: col,
				})
			}
		}
		
		f.Close()
		
		if len(localIndex) > 0 {
			indexChan <- localIndex
		}
	}
}

// Search searches using the index
func (is *IndexedSearcher) Search(word string) []IndexEntry {
	is.indexMutex.RLock()
	defer is.indexMutex.RUnlock()
	
	word = strings.ToLower(word)
	return is.index[word]
}

// Example demonstrates parallel text search
func Example() {
	fmt.Println("=== Parallel Text Search Example ===")
	
	// Create sample files for demonstration
	files := []string{
		"sample1.txt",
		"sample2.txt",
		"sample3.txt",
	}
	
	// Create sample content
	contents := []string{
		"The quick brown fox jumps over the lazy dog.\nGolang is great for concurrent programming.",
		"Concurrent programming in Go is powerful.\nThe fox is quick and brown.",
		"Parallel text search can be very fast.\nThe dog is lazy but friendly.",
	}
	
	// Write sample files
	for i, file := range files {
		err := os.WriteFile(file, []byte(contents[i]), 0644)
		if err != nil {
			fmt.Printf("Error creating sample file: %v\n", err)
			return
		}
		defer os.Remove(file)
	}
	
	// Basic search
	options := SearchOptions{
		Pattern:       "fox",
		CaseSensitive: false,
		MaxWorkers:    2,
		ContextLines:  1,
	}
	
	searcher, err := NewTextSearcher(options)
	if err != nil {
		fmt.Printf("Error creating searcher: %v\n", err)
		return
	}
	
	result := searcher.SearchFiles(files)
	
	fmt.Printf("Found %d matches in %d files (took %v)\n", 
		len(result.Matches), result.FilesSearched, result.TimeElapsed)
	
	for _, match := range result.Matches {
		fmt.Printf("\n%s:%d:%d: %s\n", match.File, match.Line, match.Column, match.Text)
		if len(match.Context) > 0 {
			fmt.Println("Context:")
			for i, line := range match.Context {
				prefix := "  "
				if i == len(match.Context)/2 {
					prefix = "> "
				}
				fmt.Printf("%s%s\n", prefix, line)
			}
		}
	}
	
	// Regex search
	fmt.Println("\n--- Regex Search ---")
	regexOptions := SearchOptions{
		Pattern:    `\b[Tt]he\s+\w+\s+fox\b`,
		Regex:      true,
		MaxWorkers: 2,
	}
	
	regexSearcher, _ := NewTextSearcher(regexOptions)
	regexResult := regexSearcher.SearchFiles(files)
	
	for _, match := range regexResult.Matches {
		fmt.Printf("%s:%d: %s\n", match.File, match.Line, match.Text)
	}
}