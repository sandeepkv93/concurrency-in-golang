package parallellogprocessor

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// LogEntry represents a parsed log entry
type LogEntry struct {
	Timestamp time.Time
	Level     string
	Source    string
	Message   string
	Fields    map[string]interface{}
	Raw       string
}

// LogParser interface for different log formats
type LogParser interface {
	Parse(line string) (*LogEntry, error)
	GetPattern() string
}

// LogProcessor handles parallel log processing
type LogProcessor struct {
	parser      LogParser
	filters     []LogFilter
	aggregators []LogAggregator
	numWorkers  int
	bufferSize  int
}

// LogFilter filters log entries
type LogFilter interface {
	Match(entry *LogEntry) bool
}

// LogAggregator aggregates log data
type LogAggregator interface {
	Process(entry *LogEntry)
	GetResult() interface{}
	Reset()
}

// ProcessingResult contains processing results
type ProcessingResult struct {
	TotalLines      int64
	ProcessedLines  int64
	FilteredLines   int64
	ErrorLines      int64
	ProcessingTime  time.Duration
	AggregateResults map[string]interface{}
}

// NewLogProcessor creates a new log processor
func NewLogProcessor(parser LogParser, numWorkers int) *LogProcessor {
	if numWorkers <= 0 {
		numWorkers = 4
	}
	
	return &LogProcessor{
		parser:      parser,
		filters:     make([]LogFilter, 0),
		aggregators: make([]LogAggregator, 0),
		numWorkers:  numWorkers,
		bufferSize:  1024 * 1024, // 1MB buffer
	}
}

// AddFilter adds a filter to the processor
func (lp *LogProcessor) AddFilter(filter LogFilter) {
	lp.filters = append(lp.filters, filter)
}

// AddAggregator adds an aggregator to the processor
func (lp *LogProcessor) AddAggregator(name string, aggregator LogAggregator) {
	lp.aggregators = append(lp.aggregators, aggregator)
}

// ProcessFile processes a single log file
func (lp *LogProcessor) ProcessFile(filename string) (*ProcessingResult, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	var reader io.Reader = file
	
	// Handle compressed files
	if strings.HasSuffix(filename, ".gz") {
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return nil, err
		}
		defer gzReader.Close()
		reader = gzReader
	}
	
	return lp.processReader(reader)
}

// ProcessFiles processes multiple log files in parallel
func (lp *LogProcessor) ProcessFiles(filenames []string) (*ProcessingResult, error) {
	start := time.Now()
	
	// Aggregate results
	totalResult := &ProcessingResult{
		AggregateResults: make(map[string]interface{}),
	}
	
	// Process files concurrently
	fileChan := make(chan string, len(filenames))
	resultChan := make(chan *ProcessingResult, len(filenames))
	errorChan := make(chan error, len(filenames))
	
	var wg sync.WaitGroup
	
	// Start file processors
	numFileWorkers := min(lp.numWorkers, len(filenames))
	for i := 0; i < numFileWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			for filename := range fileChan {
				result, err := lp.ProcessFile(filename)
				if err != nil {
					errorChan <- fmt.Errorf("error processing %s: %w", filename, err)
				} else {
					resultChan <- result
				}
			}
		}()
	}
	
	// Send files to process
	for _, filename := range filenames {
		fileChan <- filename
	}
	close(fileChan)
	
	wg.Wait()
	close(resultChan)
	close(errorChan)
	
	// Collect results
	for result := range resultChan {
		atomic.AddInt64(&totalResult.TotalLines, result.TotalLines)
		atomic.AddInt64(&totalResult.ProcessedLines, result.ProcessedLines)
		atomic.AddInt64(&totalResult.FilteredLines, result.FilteredLines)
		atomic.AddInt64(&totalResult.ErrorLines, result.ErrorLines)
	}
	
	// Collect errors
	var errors []error
	for err := range errorChan {
		errors = append(errors, err)
	}
	
	totalResult.ProcessingTime = time.Since(start)
	
	if len(errors) > 0 {
		return totalResult, fmt.Errorf("processed with %d errors", len(errors))
	}
	
	return totalResult, nil
}

func (lp *LogProcessor) processReader(reader io.Reader) (*ProcessingResult, error) {
	start := time.Now()
	
	result := &ProcessingResult{
		AggregateResults: make(map[string]interface{}),
	}
	
	// Create channels
	lineChan := make(chan string, 1000)
	entryChan := make(chan *LogEntry, 1000)
	
	var wg sync.WaitGroup
	
	// Start line reader
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(lineChan)
		
		scanner := bufio.NewScanner(reader)
		scanner.Buffer(make([]byte, lp.bufferSize), lp.bufferSize)
		
		for scanner.Scan() {
			atomic.AddInt64(&result.TotalLines, 1)
			lineChan <- scanner.Text()
		}
	}()
	
	// Start parsers
	for i := 0; i < lp.numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			for line := range lineChan {
				entry, err := lp.parser.Parse(line)
				if err != nil {
					atomic.AddInt64(&result.ErrorLines, 1)
					continue
				}
				
				// Apply filters
				if lp.passesFilters(entry) {
					atomic.AddInt64(&result.ProcessedLines, 1)
					entryChan <- entry
				} else {
					atomic.AddInt64(&result.FilteredLines, 1)
				}
			}
		}()
	}
	
	// Start aggregators
	aggregatorResults := make([]interface{}, len(lp.aggregators))
	var aggWg sync.WaitGroup
	
	for i, aggregator := range lp.aggregators {
		aggWg.Add(1)
		go func(idx int, agg LogAggregator) {
			defer aggWg.Done()
			
			for entry := range entryChan {
				agg.Process(entry)
			}
			
			aggregatorResults[idx] = agg.GetResult()
		}(i, aggregator)
	}
	
	// Wait for parsing to complete
	wg.Wait()
	close(entryChan)
	
	// Wait for aggregation to complete
	aggWg.Wait()
	
	// Collect aggregator results
	for i, aggResult := range aggregatorResults {
		result.AggregateResults[fmt.Sprintf("aggregator_%d", i)] = aggResult
	}
	
	result.ProcessingTime = time.Since(start)
	return result, nil
}

func (lp *LogProcessor) passesFilters(entry *LogEntry) bool {
	for _, filter := range lp.filters {
		if !filter.Match(entry) {
			return false
		}
	}
	return true
}

// Common Log Parsers

// CommonLogParser parses Apache Common Log Format
type CommonLogParser struct {
	pattern *regexp.Regexp
}

func NewCommonLogParser() *CommonLogParser {
	pattern := regexp.MustCompile(
		`^(\S+) \S+ \S+ \[([^\]]+)\] "([^"]+)" (\d+) (\d+|-)`)
	
	return &CommonLogParser{
		pattern: pattern,
	}
}

func (p *CommonLogParser) Parse(line string) (*LogEntry, error) {
	matches := p.pattern.FindStringSubmatch(line)
	if len(matches) < 6 {
		return nil, fmt.Errorf("invalid common log format")
	}
	
	// Parse timestamp
	timestamp, err := time.Parse("02/Jan/2006:15:04:05 -0700", matches[2])
	if err != nil {
		return nil, err
	}
	
	return &LogEntry{
		Timestamp: timestamp,
		Level:     "INFO",
		Source:    matches[1],
		Message:   matches[3],
		Fields: map[string]interface{}{
			"status": matches[4],
			"size":   matches[5],
		},
		Raw: line,
	}, nil
}

func (p *CommonLogParser) GetPattern() string {
	return "common"
}

// JSONLogParser parses JSON formatted logs
type JSONLogParser struct{}

func NewJSONLogParser() *JSONLogParser {
	return &JSONLogParser{}
}

func (p *JSONLogParser) Parse(line string) (*LogEntry, error) {
	var data map[string]interface{}
	err := json.Unmarshal([]byte(line), &data)
	if err != nil {
		return nil, err
	}
	
	entry := &LogEntry{
		Fields: make(map[string]interface{}),
		Raw:    line,
	}
	
	// Extract common fields
	if ts, ok := data["timestamp"].(string); ok {
		if t, err := time.Parse(time.RFC3339, ts); err == nil {
			entry.Timestamp = t
		}
	}
	
	if level, ok := data["level"].(string); ok {
		entry.Level = strings.ToUpper(level)
	}
	
	if source, ok := data["source"].(string); ok {
		entry.Source = source
	}
	
	if msg, ok := data["message"].(string); ok {
		entry.Message = msg
	}
	
	// Copy remaining fields
	for k, v := range data {
		if k != "timestamp" && k != "level" && k != "source" && k != "message" {
			entry.Fields[k] = v
		}
	}
	
	return entry, nil
}

func (p *JSONLogParser) GetPattern() string {
	return "json"
}

// SyslogParser parses syslog format
type SyslogParser struct {
	pattern *regexp.Regexp
}

func NewSyslogParser() *SyslogParser {
	pattern := regexp.MustCompile(
		`^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)(?:\[(\d+)\])?: (.*)$`)
	
	return &SyslogParser{
		pattern: pattern,
	}
}

func (p *SyslogParser) Parse(line string) (*LogEntry, error) {
	matches := p.pattern.FindStringSubmatch(line)
	if len(matches) < 6 {
		return nil, fmt.Errorf("invalid syslog format")
	}
	
	// Parse timestamp (add current year)
	year := time.Now().Year()
	tsStr := fmt.Sprintf("%d %s", year, matches[1])
	timestamp, err := time.Parse("2006 Jan 2 15:04:05", tsStr)
	if err != nil {
		return nil, err
	}
	
	entry := &LogEntry{
		Timestamp: timestamp,
		Source:    matches[3],
		Message:   matches[5],
		Fields:    make(map[string]interface{}),
		Raw:       line,
	}
	
	// Extract level from message if present
	levelPattern := regexp.MustCompile(`^\[(\w+)\]\s*(.*)`)
	if levelMatches := levelPattern.FindStringSubmatch(entry.Message); len(levelMatches) > 2 {
		entry.Level = strings.ToUpper(levelMatches[1])
		entry.Message = levelMatches[2]
	}
	
	// Add PID if present
	if matches[4] != "" {
		entry.Fields["pid"] = matches[4]
	}
	
	entry.Fields["host"] = matches[2]
	
	return entry, nil
}

func (p *SyslogParser) GetPattern() string {
	return "syslog"
}

// Common Filters

// LevelFilter filters by log level
type LevelFilter struct {
	MinLevel string
	levels   map[string]int
}

func NewLevelFilter(minLevel string) *LevelFilter {
	return &LevelFilter{
		MinLevel: strings.ToUpper(minLevel),
		levels: map[string]int{
			"TRACE": 0,
			"DEBUG": 1,
			"INFO":  2,
			"WARN":  3,
			"ERROR": 4,
			"FATAL": 5,
		},
	}
}

func (f *LevelFilter) Match(entry *LogEntry) bool {
	entryLevel, ok1 := f.levels[entry.Level]
	minLevel, ok2 := f.levels[f.MinLevel]
	
	if !ok1 || !ok2 {
		return true // Pass through if levels unknown
	}
	
	return entryLevel >= minLevel
}

// TimeRangeFilter filters by time range
type TimeRangeFilter struct {
	Start time.Time
	End   time.Time
}

func NewTimeRangeFilter(start, end time.Time) *TimeRangeFilter {
	return &TimeRangeFilter{
		Start: start,
		End:   end,
	}
}

func (f *TimeRangeFilter) Match(entry *LogEntry) bool {
	return !entry.Timestamp.Before(f.Start) && !entry.Timestamp.After(f.End)
}

// RegexFilter filters by regex pattern
type RegexFilter struct {
	Pattern *regexp.Regexp
	Field   string
}

func NewRegexFilter(pattern string, field string) (*RegexFilter, error) {
	regex, err := regexp.Compile(pattern)
	if err != nil {
		return nil, err
	}
	
	return &RegexFilter{
		Pattern: regex,
		Field:   field,
	}, nil
}

func (f *RegexFilter) Match(entry *LogEntry) bool {
	var value string
	
	switch f.Field {
	case "message":
		value = entry.Message
	case "source":
		value = entry.Source
	case "level":
		value = entry.Level
	case "raw":
		value = entry.Raw
	default:
		if v, ok := entry.Fields[f.Field]; ok {
			value = fmt.Sprintf("%v", v)
		}
	}
	
	return f.Pattern.MatchString(value)
}

// Common Aggregators

// CountAggregator counts log entries
type CountAggregator struct {
	count int64
	mutex sync.Mutex
}

func NewCountAggregator() *CountAggregator {
	return &CountAggregator{}
}

func (a *CountAggregator) Process(entry *LogEntry) {
	atomic.AddInt64(&a.count, 1)
}

func (a *CountAggregator) GetResult() interface{} {
	return atomic.LoadInt64(&a.count)
}

func (a *CountAggregator) Reset() {
	atomic.StoreInt64(&a.count, 0)
}

// GroupCountAggregator counts by grouping field
type GroupCountAggregator struct {
	field  string
	counts sync.Map
}

func NewGroupCountAggregator(field string) *GroupCountAggregator {
	return &GroupCountAggregator{
		field: field,
	}
}

func (a *GroupCountAggregator) Process(entry *LogEntry) {
	var key string
	
	switch a.field {
	case "level":
		key = entry.Level
	case "source":
		key = entry.Source
	default:
		if v, ok := entry.Fields[a.field]; ok {
			key = fmt.Sprintf("%v", v)
		} else {
			key = "unknown"
		}
	}
	
	count, _ := a.counts.LoadOrStore(key, new(int64))
	atomic.AddInt64(count.(*int64), 1)
}

func (a *GroupCountAggregator) GetResult() interface{} {
	result := make(map[string]int64)
	
	a.counts.Range(func(key, value interface{}) bool {
		result[key.(string)] = atomic.LoadInt64(value.(*int64))
		return true
	})
	
	return result
}

func (a *GroupCountAggregator) Reset() {
	a.counts = sync.Map{}
}

// TimeSeriesAggregator aggregates by time buckets
type TimeSeriesAggregator struct {
	bucketSize time.Duration
	buckets    sync.Map
}

func NewTimeSeriesAggregator(bucketSize time.Duration) *TimeSeriesAggregator {
	return &TimeSeriesAggregator{
		bucketSize: bucketSize,
	}
}

func (a *TimeSeriesAggregator) Process(entry *LogEntry) {
	bucket := entry.Timestamp.Truncate(a.bucketSize)
	count, _ := a.buckets.LoadOrStore(bucket, new(int64))
	atomic.AddInt64(count.(*int64), 1)
}

func (a *TimeSeriesAggregator) GetResult() interface{} {
	result := make(map[time.Time]int64)
	
	a.buckets.Range(func(key, value interface{}) bool {
		result[key.(time.Time)] = atomic.LoadInt64(value.(*int64))
		return true
	})
	
	return result
}

func (a *TimeSeriesAggregator) Reset() {
	a.buckets = sync.Map{}
}

// StreamProcessor processes logs in streaming fashion
type StreamProcessor struct {
	processor *LogProcessor
	output    io.Writer
}

func NewStreamProcessor(processor *LogProcessor, output io.Writer) *StreamProcessor {
	return &StreamProcessor{
		processor: processor,
		output:    output,
	}
}

func (sp *StreamProcessor) ProcessStream(reader io.Reader) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, sp.processor.bufferSize), sp.processor.bufferSize)
	
	for scanner.Scan() {
		line := scanner.Text()
		
		entry, err := sp.processor.parser.Parse(line)
		if err != nil {
			continue
		}
		
		if sp.processor.passesFilters(entry) {
			// Format and output
			output := fmt.Sprintf("[%s] %s %s: %s\n",
				entry.Timestamp.Format(time.RFC3339),
				entry.Level,
				entry.Source,
				entry.Message,
			)
			
			sp.output.Write([]byte(output))
		}
	}
	
	return scanner.Err()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Example demonstrates log processing
func Example() {
	fmt.Println("=== Parallel Log Processor Example ===")
	
	// Create processor with JSON parser
	parser := NewJSONLogParser()
	processor := NewLogProcessor(parser, 4)
	
	// Add filters
	levelFilter := NewLevelFilter("WARN")
	processor.AddFilter(levelFilter)
	
	// Add aggregators
	countAgg := NewCountAggregator()
	processor.AddAggregator("total_count", countAgg)
	
	levelAgg := NewGroupCountAggregator("level")
	processor.AddAggregator("by_level", levelAgg)
	
	// Process sample logs
	sampleLogs := []string{
		`{"timestamp":"2023-01-01T10:00:00Z","level":"info","source":"app","message":"Application started"}`,
		`{"timestamp":"2023-01-01T10:00:01Z","level":"warn","source":"app","message":"Low memory warning"}`,
		`{"timestamp":"2023-01-01T10:00:02Z","level":"error","source":"db","message":"Connection failed"}`,
		`{"timestamp":"2023-01-01T10:00:03Z","level":"info","source":"app","message":"Request processed"}`,
		`{"timestamp":"2023-01-01T10:00:04Z","level":"error","source":"app","message":"Invalid input"}`,
	}
	
	// Create temporary file
	tmpFile, _ := os.CreateTemp("", "logs*.json")
	defer os.Remove(tmpFile.Name())
	
	for _, log := range sampleLogs {
		tmpFile.WriteString(log + "\n")
	}
	tmpFile.Close()
	
	// Process file
	result, err := processor.ProcessFile(tmpFile.Name())
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	fmt.Printf("Processing Results:\n")
	fmt.Printf("  Total lines: %d\n", result.TotalLines)
	fmt.Printf("  Processed: %d\n", result.ProcessedLines)
	fmt.Printf("  Filtered: %d\n", result.FilteredLines)
	fmt.Printf("  Errors: %d\n", result.ErrorLines)
	fmt.Printf("  Time: %v\n", result.ProcessingTime)
}