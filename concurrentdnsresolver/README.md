# Concurrent DNS Resolver

A high-performance, concurrent DNS resolver implementation in Go that can handle multiple DNS queries simultaneously using goroutines and channels.

## Features

### Core DNS Resolution
- **Single Domain Resolution**: Resolve individual domains with caching
- **Multiple Domain Resolution**: Concurrent resolution of multiple domains using worker pools
- **DNS Record Types**: Support for A, AAAA, MX, TXT, and CNAME records
- **Timeout Management**: Configurable timeouts for DNS queries
- **Error Handling**: Comprehensive error handling for network issues and invalid domains

### Advanced Concurrency Features
- **Worker Pool**: Configurable number of concurrent workers for optimal performance
- **Racing Queries**: Race multiple DNS servers to get the fastest response
- **Async Resolution**: Asynchronous DNS resolution with query queuing
- **Batch Processing**: Efficient bulk domain resolution with configurable batch sizes
- **Retry Logic**: Automatic retry mechanism with exponential backoff

### Performance Optimizations
- **Caching**: In-memory caching with TTL support to reduce redundant queries
- **Context Support**: Full context support for cancellation and timeout management
- **Concurrent Access**: Thread-safe operations with proper synchronization
- **Statistics**: Built-in metrics for monitoring resolver performance

## Usage Examples

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "time"
    
    "github.com/yourusername/concurrency-in-golang/concurrentdnsresolver"
)

func main() {
    resolver := concurrentdnsresolver.NewDNSResolver(10, 5*time.Second)
    
    // Single domain resolution
    record := resolver.ResolveSingle(context.Background(), "google.com")
    if record.Error != nil {
        fmt.Printf("Error: %v\n", record.Error)
    } else {
        fmt.Printf("Domain: %s, IP: %s, Type: %s, Latency: %v\n", 
            record.Domain, record.IP, record.Type, record.Latency)
    }
}
```

### Multiple Domain Resolution

```go
domains := []string{"google.com", "github.com", "stackoverflow.com"}
results := resolver.ResolveMultiple(context.Background(), domains)

for _, result := range results {
    if result.Error != nil {
        fmt.Printf("Error resolving %s: %v\n", result.Domain, result.Error)
    } else {
        fmt.Printf("Resolved %s -> %s (%v latency)\n", 
            result.Domain, result.IP, result.Latency)
    }
}
```

### DNS Server Racing

```go
servers := []string{"8.8.8.8", "8.8.4.4", "1.1.1.1"}
record := resolver.ResolveWithRace(context.Background(), "example.com", servers)

fmt.Printf("Fastest response: %s -> %s (latency: %v)\n", 
    record.Domain, record.IP, record.Latency)
```

### Async Resolution

```go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

queryQueue, resultChan := resolver.StartAsyncResolver(ctx)

// Send queries asynchronously
go func() {
    queryQueue <- concurrentdnsresolver.DNSQuery{
        Domain:    "google.com",
        UseRacing: false,
    }
    queryQueue <- concurrentdnsresolver.DNSQuery{
        Domain:    "github.com",
        Servers:   []string{"8.8.8.8", "1.1.1.1"},
        UseRacing: true,
    }
}()

// Process results
for i := 0; i < 2; i++ {
    result := <-resultChan
    fmt.Printf("Async result: %s -> %s\n", result.Domain, result.IP)
}
```

### Bulk Resolution with Batching

```go
largeDomainList := []string{
    "google.com", "github.com", "stackoverflow.com",
    "reddit.com", "twitter.com", "facebook.com",
    // ... many more domains
}

results := resolver.BulkResolveWithBatching(context.Background(), largeDomainList, 5)
fmt.Printf("Resolved %d domains\n", len(results))
```

### Special Record Types

```go
// MX Records
mxRecords, err := resolver.ResolveMX(context.Background(), "google.com")
if err == nil {
    for _, mx := range mxRecords {
        fmt.Printf("MX: %s (priority: %d)\n", mx.Host, mx.Pref)
    }
}

// TXT Records
txtRecords, err := resolver.ResolveTXT(context.Background(), "google.com")
if err == nil {
    for _, txt := range txtRecords {
        fmt.Printf("TXT: %s\n", txt)
    }
}

// CNAME Records
cname, err := resolver.ResolveCNAME(context.Background(), "www.github.com")
if err == nil {
    fmt.Printf("CNAME: %s\n", cname)
}
```

### Retry Logic

```go
record := resolver.ResolveWithRetry(context.Background(), "unreliable-domain.com", 3)
if record.Error != nil {
    fmt.Printf("Failed after 3 retries: %v\n", record.Error)
} else {
    fmt.Printf("Successfully resolved: %s -> %s\n", record.Domain, record.IP)
}
```

## Architecture

### Core Components

1. **DNSResolver**: Main resolver struct with configurable workers and timeout
2. **DNSRecord**: Represents a DNS resolution result with metadata
3. **Worker Pool**: Concurrent workers for processing DNS queries
4. **Cache**: In-memory cache with TTL support
5. **Query Queue**: Async query processing system

### Synchronization

- **RWMutex**: Protects the DNS cache for concurrent read/write access
- **Channels**: Used for job distribution and result collection
- **Context**: Provides cancellation and timeout capabilities
- **WaitGroups**: Coordinates batch processing operations

### Performance Characteristics

- **Concurrent Workers**: 1-50 workers (configurable)
- **Query Timeout**: 1-30 seconds (configurable)
- **Cache TTL**: 1 hour (configurable)
- **Batch Size**: 1-100 domains per batch (configurable)
- **Memory Usage**: ~1KB per cached domain

## Testing

Run the comprehensive test suite:

```bash
go test -v ./concurrentdnsresolver/
```

Run benchmarks:

```bash
go test -bench=. ./concurrentdnsresolver/
```

### Test Coverage

- Single and multiple domain resolution
- DNS server racing functionality
- Caching behavior and TTL expiration
- Async resolution with query queuing
- Batch processing with different batch sizes
- Error handling and timeout scenarios
- Concurrent access safety
- Context cancellation
- Special record types (MX, TXT, CNAME)
- Retry mechanisms
- Performance benchmarks

## Configuration

```go
type Config struct {
    MaxWorkers    int           // Number of concurrent workers
    Timeout       time.Duration // Query timeout
    CacheTTL      time.Duration // Cache entry TTL
    RetryAttempts int           // Number of retry attempts
    BatchSize     int           // Batch size for bulk operations
}
```

## Error Handling

The resolver handles various error conditions:

- **Network Timeouts**: Configurable timeout for DNS queries
- **Invalid Domains**: Proper error reporting for malformed domains
- **DNS Server Errors**: Handling of DNS server failures
- **Context Cancellation**: Graceful handling of cancelled operations
- **Cache Misses**: Transparent cache miss handling

## Performance Considerations

- **Worker Pool Size**: Tune based on network latency and CPU cores
- **Batch Size**: Optimize for memory usage vs. processing efficiency
- **Cache Size**: Monitor memory usage for large-scale deployments
- **Timeout Values**: Balance between responsiveness and reliability
- **Retry Logic**: Configure retry attempts based on network reliability

## Use Cases

1. **Web Crawlers**: Resolve domains for large-scale web crawling
2. **Network Monitoring**: Monitor DNS resolution times and failures
3. **Load Balancers**: Resolve backend server addresses
4. **Security Tools**: Bulk domain validation and threat intelligence
5. **CDN Systems**: Geographic DNS resolution optimization
6. **API Gateways**: Dynamic service discovery via DNS