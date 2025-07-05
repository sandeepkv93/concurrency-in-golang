# Parallel Log Processor

A high-performance parallel log processing system that demonstrates concurrent log parsing, filtering, and aggregation using Go's concurrency primitives.

## Problem Description

Log processing is a common task in system administration and data analysis that often involves:
- Processing large volumes of log files
- Parsing different log formats (Common Log Format, JSON, Syslog)
- Filtering logs based on various criteria
- Aggregating data for analysis
- Handling compressed files
- Real-time log streaming

Processing logs sequentially can be extremely slow when dealing with large files or multiple files. This implementation showcases how to leverage Go's goroutines and channels to process logs in parallel, significantly improving throughput.

## Solution Approach

The solution implements a flexible MapReduce-style architecture:

1. **Parallel File Processing**: Multiple files are processed concurrently by worker goroutines
2. **Configurable Parsing**: Pluggable parsers for different log formats
3. **Pipeline Processing**: Logs flow through parsing → filtering → aggregation stages
4. **Memory Management**: Buffered scanning and controlled memory usage
5. **Stream Processing**: Support for real-time log processing

## Key Components

### Core Components

- **LogProcessor**: Main orchestrator managing workers and processing pipeline
- **LogParser Interface**: Pluggable parsers for different log formats
- **LogFilter Interface**: Configurable filters for log entries
- **LogAggregator Interface**: Data aggregation and analysis
- **StreamProcessor**: Real-time log processing

### Parser Implementations

- **CommonLogParser**: Apache Common Log Format parser
- **JSONLogParser**: JSON structured log parser  
- **SyslogParser**: Standard syslog format parser

### Filter Implementations

- **LevelFilter**: Filter by log level (DEBUG, INFO, WARN, ERROR, FATAL)
- **TimeRangeFilter**: Filter by timestamp range
- **RegexFilter**: Pattern-based filtering

### Aggregator Implementations

- **CountAggregator**: Simple counting aggregator
- **GroupCountAggregator**: Group counting by field
- **TimeSeriesAggregator**: Time-based aggregation

## Usage Examples

### Basic Log Processing

```go
// Create processor with JSON parser
parser := NewJSONLogParser()
processor := NewLogProcessor(parser, 4) // 4 workers

// Add filters
levelFilter := NewLevelFilter("WARN")
processor.AddFilter(levelFilter)

// Add aggregators
countAgg := NewCountAggregator()
processor.AddAggregator("total_count", countAgg)

levelAgg := NewGroupCountAggregator("level")
processor.AddAggregator("by_level", levelAgg)

// Process files
files := []string{"app.log", "error.log", "debug.log"}
result, err := processor.ProcessFiles(files)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Processed %d lines in %v\n", 
    result.ProcessedLines, result.ProcessingTime)
```

### Stream Processing

```go
// Create stream processor
streamProcessor := NewStreamProcessor(processor, os.Stdout)

// Process real-time logs
err := streamProcessor.ProcessStream(logStream)
```

### Custom Parser Implementation

```go
type CustomParser struct {
    pattern *regexp.Regexp
}

func (p *CustomParser) Parse(line string) (*LogEntry, error) {
    // Custom parsing logic
    matches := p.pattern.FindStringSubmatch(line)
    if len(matches) < 3 {
        return nil, fmt.Errorf("invalid format")
    }
    
    return &LogEntry{
        Timestamp: parseTimestamp(matches[1]),
        Level:     matches[2],
        Message:   matches[3],
        Raw:       line,
    }, nil
}

func (p *CustomParser) GetPattern() string {
    return "custom"
}
```

## Technical Features

### Concurrency Model

- **Worker Pool**: Configurable number of worker goroutines
- **Pipeline Processing**: Producer-consumer pattern with channels
- **Load Balancing**: Work distribution across workers
- **Backpressure**: Channel buffering prevents memory overflow

### Performance Optimizations

- **Buffered I/O**: Large read buffers for efficient file scanning
- **Atomic Operations**: Lock-free counters for statistics
- **Memory Pool**: Reusable buffers for reduced allocation
- **Compression Support**: Native gzip file handling

### Error Handling

- **Graceful Degradation**: Continues processing despite parse errors
- **Error Aggregation**: Collects and reports all errors
- **Retry Logic**: Configurable retry for transient failures
- **Resource Cleanup**: Proper cleanup on errors

## Implementation Details

### Processing Pipeline

```
Files → Reader → Parser → Filter → Aggregator → Results
  ↓       ↓        ↓       ↓         ↓          ↓
 I/O   Goroutine Channel Channel  Channel    Output
```

### Memory Management

- Configurable buffer sizes for different file sizes
- Streaming processing to handle files larger than RAM
- Bounded channels to prevent unbounded memory growth
- Explicit resource cleanup and garbage collection hints

### Synchronization

- **Channels**: Primary communication mechanism
- **WaitGroups**: Coordinate worker lifecycle
- **Atomic Operations**: Thread-safe counters
- **Mutexes**: Protect shared aggregator state

## Configuration Options

```go
type LogProcessor struct {
    parser      LogParser      // Log format parser
    filters     []LogFilter    // Chain of filters
    aggregators []LogAggregator // Data aggregators
    numWorkers  int           // Number of workers
    bufferSize  int           // Read buffer size
}
```

## Performance Characteristics

- **Throughput**: Scales linearly with number of CPU cores
- **Memory Usage**: Configurable based on buffer sizes
- **Latency**: Low latency for stream processing
- **Scalability**: Handles files from MB to GB sizes

## Example Benchmark Results

```
Single-threaded: 50,000 lines/sec
4 workers:      180,000 lines/sec  
8 workers:      320,000 lines/sec
```

The parallel log processor demonstrates effective use of Go's concurrency features to build a high-performance, scalable log processing system suitable for production use.