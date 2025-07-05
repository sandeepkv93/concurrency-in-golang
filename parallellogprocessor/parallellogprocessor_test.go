package parallellogprocessor

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestLogProcessor(t *testing.T) {
	parser := NewJSONLogParser()
	processor := NewLogProcessor(parser, 4)

	// Create test log
	logs := []string{
		`{"timestamp":"2023-01-01T10:00:00Z","level":"info","source":"app","message":"Started"}`,
		`{"timestamp":"2023-01-01T10:00:01Z","level":"error","source":"db","message":"Connection failed"}`,
		`{"timestamp":"2023-01-01T10:00:02Z","level":"warn","source":"app","message":"High memory"}`,
	}

	tmpFile, err := os.CreateTemp("", "test*.log")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	for _, log := range logs {
		tmpFile.WriteString(log + "\n")
	}
	tmpFile.Close()

	result, err := processor.ProcessFile(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	if result.TotalLines != 3 {
		t.Errorf("Expected 3 total lines, got %d", result.TotalLines)
	}

	if result.ProcessedLines != 3 {
		t.Errorf("Expected 3 processed lines, got %d", result.ProcessedLines)
	}

	if result.ErrorLines != 0 {
		t.Errorf("Expected 0 error lines, got %d", result.ErrorLines)
	}
}

func TestJSONLogParser(t *testing.T) {
	parser := NewJSONLogParser()

	tests := []struct {
		name     string
		input    string
		expected LogEntry
		wantErr  bool
	}{
		{
			name:  "Valid JSON log",
			input: `{"timestamp":"2023-01-01T10:00:00Z","level":"info","source":"app","message":"Test message","request_id":"123"}`,
			expected: LogEntry{
				Timestamp: time.Date(2023, 1, 1, 10, 0, 0, 0, time.UTC),
				Level:     "INFO",
				Source:    "app",
				Message:   "Test message",
				Fields:    map[string]interface{}{"request_id": "123"},
			},
		},
		{
			name:    "Invalid JSON",
			input:   `{invalid json}`,
			wantErr: true,
		},
		{
			name:  "Missing fields",
			input: `{"message":"Test"}`,
			expected: LogEntry{
				Message: "Test",
				Fields:  map[string]interface{}{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry, err := parser.Parse(tt.input)

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if entry.Message != tt.expected.Message {
				t.Errorf("Message: got %q, want %q", entry.Message, tt.expected.Message)
			}

			if entry.Level != tt.expected.Level {
				t.Errorf("Level: got %q, want %q", entry.Level, tt.expected.Level)
			}

			if entry.Source != tt.expected.Source {
				t.Errorf("Source: got %q, want %q", entry.Source, tt.expected.Source)
			}

			if !entry.Timestamp.IsZero() && !entry.Timestamp.Equal(tt.expected.Timestamp) {
				t.Errorf("Timestamp: got %v, want %v", entry.Timestamp, tt.expected.Timestamp)
			}
		})
	}
}

func TestCommonLogParser(t *testing.T) {
	parser := NewCommonLogParser()

	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{
			name:    "Valid common log",
			input:   `127.0.0.1 - - [10/Oct/2023:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326`,
			wantErr: false,
		},
		{
			name:    "Invalid format",
			input:   `invalid log format`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry, err := parser.Parse(tt.input)

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if entry.Source != "127.0.0.1" {
				t.Errorf("Source: got %q, want %q", entry.Source, "127.0.0.1")
			}

			if entry.Fields["status"] != "200" {
				t.Errorf("Status: got %v, want %v", entry.Fields["status"], "200")
			}
		})
	}
}

func TestSyslogParser(t *testing.T) {
	parser := NewSyslogParser()

	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{
			name:    "Valid syslog",
			input:   `Jan 10 10:15:35 hostname sshd[1234]: [INFO] Accepted password for user`,
			wantErr: false,
		},
		{
			name:    "Invalid format",
			input:   `invalid syslog`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry, err := parser.Parse(tt.input)

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if entry.Source != "sshd" {
				t.Errorf("Source: got %q, want %q", entry.Source, "sshd")
			}

			if entry.Fields["pid"] != "1234" {
				t.Errorf("PID: got %v, want %v", entry.Fields["pid"], "1234")
			}
		})
	}
}

func TestLevelFilter(t *testing.T) {
	filter := NewLevelFilter("WARN")

	tests := []struct {
		level   string
		allowed bool
	}{
		{"TRACE", false},
		{"DEBUG", false},
		{"INFO", false},
		{"WARN", true},
		{"ERROR", true},
		{"FATAL", true},
		{"UNKNOWN", true}, // Unknown levels pass through
	}

	for _, tt := range tests {
		t.Run(tt.level, func(t *testing.T) {
			entry := &LogEntry{Level: tt.level}
			result := filter.Match(entry)
			if result != tt.allowed {
				t.Errorf("Level %s: got %v, want %v", tt.level, result, tt.allowed)
			}
		})
	}
}

func TestTimeRangeFilter(t *testing.T) {
	start := time.Date(2023, 1, 1, 10, 0, 0, 0, time.UTC)
	end := time.Date(2023, 1, 1, 11, 0, 0, 0, time.UTC)
	filter := NewTimeRangeFilter(start, end)

	tests := []struct {
		name      string
		timestamp time.Time
		allowed   bool
	}{
		{
			name:      "Before range",
			timestamp: time.Date(2023, 1, 1, 9, 30, 0, 0, time.UTC),
			allowed:   false,
		},
		{
			name:      "Within range",
			timestamp: time.Date(2023, 1, 1, 10, 30, 0, 0, time.UTC),
			allowed:   true,
		},
		{
			name:      "After range",
			timestamp: time.Date(2023, 1, 1, 11, 30, 0, 0, time.UTC),
			allowed:   false,
		},
		{
			name:      "At start",
			timestamp: start,
			allowed:   true,
		},
		{
			name:      "At end",
			timestamp: end,
			allowed:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry := &LogEntry{Timestamp: tt.timestamp}
			result := filter.Match(entry)
			if result != tt.allowed {
				t.Errorf("Got %v, want %v", result, tt.allowed)
			}
		})
	}
}

func TestRegexFilter(t *testing.T) {
	filter, err := NewRegexFilter(`error|fail`, "message")
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		message string
		match   bool
	}{
		{"Connection error occurred", true},
		{"Failed to connect", true},
		{"Everything is fine", false},
		{"System failure detected", true},
	}

	for _, tt := range tests {
		t.Run(tt.message, func(t *testing.T) {
			entry := &LogEntry{Message: tt.message}
			result := filter.Match(entry)
			if result != tt.match {
				t.Errorf("Message %q: got %v, want %v", tt.message, result, tt.match)
			}
		})
	}
}

func TestCountAggregator(t *testing.T) {
	agg := NewCountAggregator()

	// Process entries
	for i := 0; i < 100; i++ {
		agg.Process(&LogEntry{})
	}

	result := agg.GetResult().(int64)
	if result != 100 {
		t.Errorf("Expected count 100, got %d", result)
	}

	// Test reset
	agg.Reset()
	result = agg.GetResult().(int64)
	if result != 0 {
		t.Errorf("Expected count 0 after reset, got %d", result)
	}
}

func TestGroupCountAggregator(t *testing.T) {
	agg := NewGroupCountAggregator("level")

	levels := []string{"INFO", "WARN", "ERROR", "INFO", "ERROR", "ERROR"}
	for _, level := range levels {
		agg.Process(&LogEntry{Level: level})
	}

	result := agg.GetResult().(map[string]int64)

	expected := map[string]int64{
		"INFO":  2,
		"WARN":  1,
		"ERROR": 3,
	}

	for level, count := range expected {
		if result[level] != count {
			t.Errorf("Level %s: expected %d, got %d", level, count, result[level])
		}
	}
}

func TestTimeSeriesAggregator(t *testing.T) {
	agg := NewTimeSeriesAggregator(time.Hour)

	base := time.Date(2023, 1, 1, 10, 0, 0, 0, time.UTC)
	timestamps := []time.Time{
		base,
		base.Add(30 * time.Minute),
		base.Add(45 * time.Minute),
		base.Add(90 * time.Minute),
	}

	for _, ts := range timestamps {
		agg.Process(&LogEntry{Timestamp: ts})
	}

	result := agg.GetResult().(map[time.Time]int64)

	// Should have 2 buckets
	if len(result) != 2 {
		t.Errorf("Expected 2 time buckets, got %d", len(result))
	}

	// First hour should have 3 entries
	if result[base] != 3 {
		t.Errorf("First bucket: expected 3, got %d", result[base])
	}

	// Second hour should have 1 entry
	secondBucket := base.Add(time.Hour)
	if result[secondBucket] != 1 {
		t.Errorf("Second bucket: expected 1, got %d", result[secondBucket])
	}
}

func TestParallelProcessing(t *testing.T) {
	// Create multiple log files
	numFiles := 5
	files := make([]string, numFiles)

	for i := 0; i < numFiles; i++ {
		tmpFile, err := os.CreateTemp("", fmt.Sprintf("test%d*.log", i))
		if err != nil {
			t.Fatal(err)
		}
		defer os.Remove(tmpFile.Name())

		// Write test logs
		for j := 0; j < 100; j++ {
			log := fmt.Sprintf(`{"timestamp":"2023-01-01T10:%02d:00Z","level":"info","source":"app%d","message":"Message %d"}`,
				j%60, i, j)
			tmpFile.WriteString(log + "\n")
		}
		tmpFile.Close()
		files[i] = tmpFile.Name()
	}

	parser := NewJSONLogParser()
	processor := NewLogProcessor(parser, 4)

	result, err := processor.ProcessFiles(files)
	if err != nil && !strings.Contains(err.Error(), "processed with") {
		t.Fatal(err)
	}

	expectedTotal := int64(numFiles * 100)
	if result.TotalLines != expectedTotal {
		t.Errorf("Expected %d total lines, got %d", expectedTotal, result.TotalLines)
	}

	if result.ProcessedLines != expectedTotal {
		t.Errorf("Expected %d processed lines, got %d", expectedTotal, result.ProcessedLines)
	}
}

func TestCompressedFiles(t *testing.T) {
	// Create gzipped log file
	tmpFile, err := os.CreateTemp("", "test*.log.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	gzWriter := gzip.NewWriter(tmpFile)
	for i := 0; i < 10; i++ {
		log := fmt.Sprintf(`{"timestamp":"2023-01-01T10:%02d:00Z","level":"info","message":"Message %d"}`, i, i)
		gzWriter.Write([]byte(log + "\n"))
	}
	gzWriter.Close()
	tmpFile.Close()

	parser := NewJSONLogParser()
	processor := NewLogProcessor(parser, 2)

	result, err := processor.ProcessFile(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	if result.TotalLines != 10 {
		t.Errorf("Expected 10 lines from gzipped file, got %d", result.TotalLines)
	}
}

func TestStreamProcessor(t *testing.T) {
	parser := NewJSONLogParser()
	processor := NewLogProcessor(parser, 2)

	// Add filter
	levelFilter := NewLevelFilter("WARN")
	processor.AddFilter(levelFilter)

	// Create output buffer
	var output strings.Builder
	streamProcessor := NewStreamProcessor(processor, &output)

	// Create input
	logs := []string{
		`{"timestamp":"2023-01-01T10:00:00Z","level":"info","source":"app","message":"Info message"}`,
		`{"timestamp":"2023-01-01T10:00:01Z","level":"warn","source":"app","message":"Warning message"}`,
		`{"timestamp":"2023-01-01T10:00:02Z","level":"error","source":"app","message":"Error message"}`,
	}

	reader := strings.NewReader(strings.Join(logs, "\n"))
	err := streamProcessor.ProcessStream(reader)
	if err != nil {
		t.Fatal(err)
	}

	outputStr := output.String()
	// Should only have WARN and ERROR messages
	if strings.Contains(outputStr, "Info message") {
		t.Error("Stream processor should have filtered out INFO messages")
	}

	if !strings.Contains(outputStr, "Warning message") {
		t.Error("Stream processor should include WARN messages")
	}

	if !strings.Contains(outputStr, "Error message") {
		t.Error("Stream processor should include ERROR messages")
	}
}

func TestConcurrentAggregation(t *testing.T) {
	parser := NewJSONLogParser()
	processor := NewLogProcessor(parser, 4)

	// Add multiple aggregators
	countAgg := NewCountAggregator()
	processor.AddAggregator("count", countAgg)

	levelAgg := NewGroupCountAggregator("level")
	processor.AddAggregator("levels", levelAgg)

	// Create large log file
	tmpFile, err := os.CreateTemp("", "test*.log")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	levels := []string{"info", "warn", "error"}
	for i := 0; i < 1000; i++ {
		level := levels[i%3]
		log := fmt.Sprintf(`{"timestamp":"2023-01-01T10:00:00Z","level":"%s","message":"Message %d"}`, level, i)
		tmpFile.WriteString(log + "\n")
	}
	tmpFile.Close()

	result, err := processor.ProcessFile(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	if result.TotalLines != 1000 {
		t.Errorf("Expected 1000 total lines, got %d", result.TotalLines)
	}
}

func TestEdgeCases(t *testing.T) {
	parser := NewJSONLogParser()
	processor := NewLogProcessor(parser, 2)

	// Test empty file
	tmpFile, err := os.CreateTemp("", "empty*.log")
	if err != nil {
		t.Fatal(err)
	}
	tmpFile.Close()
	defer os.Remove(tmpFile.Name())

	result, err := processor.ProcessFile(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	if result.TotalLines != 0 {
		t.Errorf("Expected 0 lines for empty file, got %d", result.TotalLines)
	}

	// Test non-existent file
	_, err = processor.ProcessFile("/non/existent/file.log")
	if err == nil {
		t.Error("Expected error for non-existent file")
	}
}

func BenchmarkLogParsing(b *testing.B) {
	parsers := []struct {
		name   string
		parser LogParser
		sample string
	}{
		{
			"JSON",
			NewJSONLogParser(),
			`{"timestamp":"2023-01-01T10:00:00Z","level":"info","source":"app","message":"Test message"}`,
		},
		{
			"CommonLog",
			NewCommonLogParser(),
			`127.0.0.1 - - [10/Oct/2023:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326`,
		},
		{
			"Syslog",
			NewSyslogParser(),
			`Jan 10 10:15:35 hostname sshd[1234]: [INFO] Accepted password for user`,
		},
	}

	for _, p := range parsers {
		b.Run(p.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.parser.Parse(p.sample)
			}
		})
	}
}

func BenchmarkParallelProcessing(b *testing.B) {
	// Create test file
	tmpFile, _ := os.CreateTemp("", "bench*.log")
	defer os.Remove(tmpFile.Name())

	// Write many log lines
	writer := bufio.NewWriter(tmpFile)
	for i := 0; i < 10000; i++ {
		log := fmt.Sprintf(`{"timestamp":"2023-01-01T10:00:00Z","level":"info","message":"Message %d"}`, i)
		writer.WriteString(log + "\n")
	}
	writer.Flush()
	tmpFile.Close()

	parser := NewJSONLogParser()

	workerCounts := []int{1, 2, 4, 8}
	for _, workers := range workerCounts {
		b.Run(fmt.Sprintf("Workers%d", workers), func(b *testing.B) {
			processor := NewLogProcessor(parser, workers)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				processor.ProcessFile(tmpFile.Name())
			}
		})
	}
}

func BenchmarkAggregators(b *testing.B) {
	aggregators := []struct {
		name string
		agg  LogAggregator
	}{
		{"Count", NewCountAggregator()},
		{"GroupCount", NewGroupCountAggregator("level")},
		{"TimeSeries", NewTimeSeriesAggregator(time.Hour)},
	}

	entry := &LogEntry{
		Timestamp: time.Now(),
		Level:     "INFO",
		Source:    "app",
		Message:   "Test message",
		Fields:    map[string]interface{}{"request_id": "123"},
	}

	for _, a := range aggregators {
		b.Run(a.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				a.agg.Process(entry)
			}
		})
	}
}

func TestConcurrentFiltering(t *testing.T) {
	parser := NewJSONLogParser()
	processor := NewLogProcessor(parser, 4)

	// Add multiple filters
	levelFilter := NewLevelFilter("WARN")
	processor.AddFilter(levelFilter)

	regexFilter, _ := NewRegexFilter(`error|critical`, "message")
	processor.AddFilter(regexFilter)

	// Create test data
	tmpFile, err := os.CreateTemp("", "filter*.log")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	logs := []struct {
		level   string
		message string
		expect  bool
	}{
		{"info", "Normal operation", false},
		{"warn", "Warning message", false},
		{"error", "Error occurred", true},
		{"error", "Normal error", true},
		{"warn", "Critical warning", true},
	}

	for _, log := range logs {
		line := fmt.Sprintf(`{"level":"%s","message":"%s"}`, log.level, log.message)
		tmpFile.WriteString(line + "\n")
	}
	tmpFile.Close()

	result, err := processor.ProcessFile(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	expectedProcessed := int64(3) // Only entries matching both filters
	if result.ProcessedLines != expectedProcessed {
		t.Errorf("Expected %d processed lines, got %d", expectedProcessed, result.ProcessedLines)
	}
}

func TestProcessorScaling(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping scaling test in short mode")
	}

	// Create large log file
	tmpFile, err := os.CreateTemp("", "scale*.log")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	// Write many entries
	numEntries := 100000
	writer := bufio.NewWriter(tmpFile)
	for i := 0; i < numEntries; i++ {
		level := []string{"info", "warn", "error"}[i%3]
		log := fmt.Sprintf(`{"timestamp":"2023-01-01T%02d:%02d:%02d","level":"%s","message":"Message %d"}`,
			i/3600, (i/60)%60, i%60, level, i)
		writer.WriteString(log + "\n")
	}
	writer.Flush()
	tmpFile.Close()

	parser := NewJSONLogParser()

	// Test scaling
	for workers := 1; workers <= 8; workers *= 2 {
		processor := NewLogProcessor(parser, workers)
		
		// Add aggregators to increase work
		processor.AddAggregator("count", NewCountAggregator())
		processor.AddAggregator("levels", NewGroupCountAggregator("level"))
		processor.AddAggregator("timeseries", NewTimeSeriesAggregator(time.Minute))

		start := time.Now()
		result, err := processor.ProcessFile(tmpFile.Name())
		elapsed := time.Since(start)

		if err != nil {
			t.Fatal(err)
		}

		t.Logf("Workers: %d, Time: %v, Lines/sec: %.0f",
			workers, elapsed, float64(result.TotalLines)/elapsed.Seconds())
	}
}

func TestRaceConditions(t *testing.T) {
	parser := NewJSONLogParser()
	processor := NewLogProcessor(parser, 8)

	// Add aggregators that might have race conditions
	var aggregators []LogAggregator
	for i := 0; i < 10; i++ {
		agg := NewGroupCountAggregator("level")
		processor.AddAggregator(fmt.Sprintf("agg%d", i), agg)
		aggregators = append(aggregators, agg)
	}

	// Create test file
	tmpFile, err := os.CreateTemp("", "race*.log")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	// Write concurrent logs
	var wg sync.WaitGroup
	writer := bufio.NewWriter(tmpFile)
	mu := &sync.Mutex{}

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				log := fmt.Sprintf(`{"level":"info","source":"worker%d","message":"Message %d"}`, id, j)
				mu.Lock()
				writer.WriteString(log + "\n")
				mu.Unlock()
			}
		}(i)
	}

	wg.Wait()
	writer.Flush()
	tmpFile.Close()

	// Process with race detector enabled
	result, err := processor.ProcessFile(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	if result.TotalLines != 1000 {
		t.Errorf("Expected 1000 lines, got %d", result.TotalLines)
	}
}