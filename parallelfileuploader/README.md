# Parallel File Uploader

A concurrent file upload simulation in Go that demonstrates efficient parallel processing patterns for I/O-bound operations using goroutines, WaitGroups, and various optimization techniques for handling large-scale file uploads.

## Problem Description

File upload operations present several challenges that benefit from parallel processing:

- **I/O Bound Operations**: File uploads are limited by network bandwidth, not CPU
- **Sequential Bottleneck**: Uploading files one by one is inefficient and slow
- **Resource Management**: Need to control concurrent uploads to avoid overwhelming the server
- **Progress Tracking**: Monitoring upload progress across multiple concurrent operations
- **Error Handling**: Managing failures in individual uploads without affecting others
- **Bandwidth Optimization**: Balancing concurrency with available network resources

## Solution Approach

This implementation demonstrates several concurrent upload patterns:

1. **Basic Parallel Upload**: Simple goroutine-per-file approach
2. **Worker Pool Pattern**: Limited concurrency with bounded workers
3. **Rate Limited Upload**: Controlling upload rate to respect server limits
4. **Progress Tracking**: Real-time monitoring of upload progress
5. **Error Resilience**: Graceful handling of upload failures

## Key Components

### Core Structures

- **File**: Represents a file with name and size
- **UploadResult**: Contains upload outcome and metrics
- **UploadManager**: Orchestrates concurrent uploads with various strategies

### Upload Strategies

- **Unlimited Concurrency**: Maximum parallelism (basic example)
- **Worker Pool**: Bounded concurrency for resource control
- **Rate Limited**: Throttled uploads respecting bandwidth limits
- **Progressive**: Adaptive concurrency based on performance

## Technical Features

### Concurrency Patterns

1. **Fan-Out Pattern**: Distributing upload tasks across multiple goroutines
2. **Worker Pool**: Fixed number of workers processing upload queue
3. **Rate Limiting**: Controlling request frequency with time-based throttling
4. **Synchronization**: Using WaitGroups for coordinated completion
5. **Context Cancellation**: Graceful shutdown of upload operations

### Advanced Features

- **Progress Monitoring**: Real-time upload progress tracking
- **Retry Logic**: Automatic retry on transient failures
- **Bandwidth Throttling**: Respecting server and network limits
- **Metrics Collection**: Upload statistics and performance monitoring
- **Memory Efficiency**: Handling large file lists without excessive memory

## Usage Examples

### Basic Parallel Upload

```go
package main

import (
    "fmt"
    "math/rand"
    "sync"
    "time"
)

type File struct {
    Name string
    Size int
}

func Upload(file File, wg *sync.WaitGroup) error {
    defer wg.Done()
    
    // Simulate upload time based on file size
    time.Sleep(time.Duration(file.Size) * time.Millisecond)
    fmt.Printf("Uploaded file %s (%d bytes)\n", file.Name, file.Size)
    return nil
}

func main() {
    // Create sample files
    files := make([]File, 100)
    for i := range files {
        files[i] = File{
            Name: fmt.Sprintf("file-%d.txt", i),
            Size: rand.Intn(1000) + 100, // 100-1100 bytes
        }
    }
    
    // Upload files concurrently
    var wg sync.WaitGroup
    wg.Add(len(files))
    
    for _, file := range files {
        go Upload(file, &wg)
    }
    
    wg.Wait()
    fmt.Println("All files uploaded successfully!")
}
```

### Worker Pool Pattern

```go
type UploadManager struct {
    workerCount int
    fileQueue   chan File
    results     chan UploadResult
    wg          sync.WaitGroup
}

type UploadResult struct {
    File     File
    Success  bool
    Error    error
    Duration time.Duration
}

func NewUploadManager(workerCount int) *UploadManager {
    return &UploadManager{
        workerCount: workerCount,
        fileQueue:   make(chan File, 100),
        results:     make(chan UploadResult, 100),
    }
}

func (um *UploadManager) Start() {
    // Start worker goroutines
    for i := 0; i < um.workerCount; i++ {
        um.wg.Add(1)
        go um.worker(i)
    }
}

func (um *UploadManager) worker(id int) {
    defer um.wg.Done()
    
    for file := range um.fileQueue {
        start := time.Now()
        err := um.uploadFile(file)
        duration := time.Since(start)
        
        result := UploadResult{
            File:     file,
            Success:  err == nil,
            Error:    err,
            Duration: duration,
        }
        
        um.results <- result
    }
}

func (um *UploadManager) uploadFile(file File) error {
    // Simulate upload operation
    time.Sleep(time.Duration(file.Size) * time.Millisecond)
    
    // Simulate occasional failures
    if rand.Float32() < 0.05 { // 5% failure rate
        return fmt.Errorf("upload failed for %s", file.Name)
    }
    
    return nil
}

func (um *UploadManager) UploadFiles(files []File) []UploadResult {
    um.Start()
    
    // Send files to workers
    go func() {
        for _, file := range files {
            um.fileQueue <- file
        }
        close(um.fileQueue)
    }()
    
    // Collect results
    results := make([]UploadResult, 0, len(files))
    for i := 0; i < len(files); i++ {
        result := <-um.results
        results = append(results, result)
    }
    
    um.wg.Wait()
    return results
}

// Usage
manager := NewUploadManager(10) // 10 concurrent workers
results := manager.UploadFiles(files)

successCount := 0
for _, result := range results {
    if result.Success {
        successCount++
    } else {
        fmt.Printf("Failed to upload %s: %v\n", result.File.Name, result.Error)
    }
}

fmt.Printf("Successfully uploaded %d/%d files\n", successCount, len(files))
```

### Rate Limited Upload

```go
type RateLimitedUploader struct {
    maxUploadsPerSecond int
    rateLimiter         chan struct{}
    ticker              *time.Ticker
}

func NewRateLimitedUploader(maxUploadsPerSecond int) *RateLimitedUploader {
    uploader := &RateLimitedUploader{
        maxUploadsPerSecond: maxUploadsPerSecond,
        rateLimiter:         make(chan struct{}, maxUploadsPerSecond),
        ticker:              time.NewTicker(time.Second),
    }
    
    // Fill rate limiter initially
    for i := 0; i < maxUploadsPerSecond; i++ {
        uploader.rateLimiter <- struct{}{}
    }
    
    // Refill rate limiter every second
    go func() {
        for range uploader.ticker.C {
            select {
            case uploader.rateLimiter <- struct{}{}:
            default:
                // Channel full, skip
            }
        }
    }()
    
    return uploader
}

func (rl *RateLimitedUploader) Upload(file File) error {
    // Wait for rate limit permission
    <-rl.rateLimiter
    
    // Simulate upload
    time.Sleep(time.Duration(file.Size) * time.Millisecond)
    fmt.Printf("Rate-limited upload: %s\n", file.Name)
    return nil
}

func (rl *RateLimitedUploader) UploadFiles(files []File) {
    var wg sync.WaitGroup
    
    for _, file := range files {
        wg.Add(1)
        go func(f File) {
            defer wg.Done()
            rl.Upload(f)
        }(file)
    }
    
    wg.Wait()
}

func (rl *RateLimitedUploader) Close() {
    rl.ticker.Stop()
}

// Usage
uploader := NewRateLimitedUploader(5) // Max 5 uploads per second
defer uploader.Close()

uploader.UploadFiles(files)
```

### Progress Tracking

```go
type ProgressTracker struct {
    totalFiles    int
    uploadedFiles int64
    totalSize     int64
    uploadedSize  int64
    mutex         sync.RWMutex
}

func NewProgressTracker(files []File) *ProgressTracker {
    totalSize := int64(0)
    for _, file := range files {
        totalSize += int64(file.Size)
    }
    
    return &ProgressTracker{
        totalFiles: len(files),
        totalSize:  totalSize,
    }
}

func (pt *ProgressTracker) UpdateProgress(fileSize int) {
    pt.mutex.Lock()
    defer pt.mutex.Unlock()
    
    pt.uploadedFiles++
    pt.uploadedSize += int64(fileSize)
}

func (pt *ProgressTracker) GetProgress() (float64, float64) {
    pt.mutex.RLock()
    defer pt.mutex.RUnlock()
    
    fileProgress := float64(pt.uploadedFiles) / float64(pt.totalFiles) * 100
    sizeProgress := float64(pt.uploadedSize) / float64(pt.totalSize) * 100
    
    return fileProgress, sizeProgress
}

func (pt *ProgressTracker) StartProgressMonitor() chan struct{} {
    stop := make(chan struct{})
    
    go func() {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()
        
        for {
            select {
            case <-ticker.C:
                fileProgress, sizeProgress := pt.GetProgress()
                fmt.Printf("Progress: %.1f%% files, %.1f%% bytes\n", 
                    fileProgress, sizeProgress)
            case <-stop:
                return
            }
        }
    }()
    
    return stop
}

func UploadWithProgress(file File, tracker *ProgressTracker, wg *sync.WaitGroup) {
    defer wg.Done()
    
    // Simulate upload
    time.Sleep(time.Duration(file.Size) * time.Millisecond)
    
    // Update progress
    tracker.UpdateProgress(file.Size)
}

// Usage
tracker := NewProgressTracker(files)
stopMonitor := tracker.StartProgressMonitor()
defer close(stopMonitor)

var wg sync.WaitGroup
wg.Add(len(files))

for _, file := range files {
    go UploadWithProgress(file, tracker, &wg)
}

wg.Wait()
```

### Retry Logic with Exponential Backoff

```go
type RetryableUploader struct {
    maxRetries int
    baseDelay  time.Duration
}

func NewRetryableUploader(maxRetries int, baseDelay time.Duration) *RetryableUploader {
    return &RetryableUploader{
        maxRetries: maxRetries,
        baseDelay:  baseDelay,
    }
}

func (ru *RetryableUploader) UploadWithRetry(file File) error {
    var lastError error
    
    for attempt := 0; attempt <= ru.maxRetries; attempt++ {
        err := ru.attemptUpload(file)
        if err == nil {
            return nil // Success
        }
        
        lastError = err
        
        if attempt < ru.maxRetries {
            delay := ru.baseDelay * time.Duration(1<<attempt) // Exponential backoff
            fmt.Printf("Upload failed for %s (attempt %d/%d), retrying in %v: %v\n", 
                file.Name, attempt+1, ru.maxRetries+1, delay, err)
            time.Sleep(delay)
        }
    }
    
    return fmt.Errorf("upload failed after %d attempts: %w", ru.maxRetries+1, lastError)
}

func (ru *RetryableUploader) attemptUpload(file File) error {
    // Simulate upload with 30% failure rate
    if rand.Float32() < 0.3 {
        return fmt.Errorf("network error")
    }
    
    time.Sleep(time.Duration(file.Size) * time.Millisecond)
    return nil
}

// Usage
uploader := NewRetryableUploader(3, 100*time.Millisecond)

var wg sync.WaitGroup
for _, file := range files {
    wg.Add(1)
    go func(f File) {
        defer wg.Done()
        if err := uploader.UploadWithRetry(f); err != nil {
            fmt.Printf("Final failure: %v\n", err)
        }
    }(file)
}
wg.Wait()
```

## Implementation Details

### Basic Parallel Pattern

The fundamental pattern uses goroutines with WaitGroup synchronization:

```go
func Upload(file File, wg *sync.WaitGroup) error {
    defer wg.Done()
    
    // Simulate upload duration based on file size
    time.Sleep(time.Duration(file.Size) * time.Millisecond)
    fmt.Printf("Uploaded file %s (%d bytes)\n", file.Name, file.Size)
    return nil
}

func main() {
    var wg sync.WaitGroup
    wg.Add(len(files))
    
    for _, file := range files {
        go Upload(file, &wg)
    }
    
    wg.Wait()
}
```

### Resource Management

For large file sets, controlling concurrency prevents resource exhaustion:

```go
func UploadWithWorkerPool(files []File, maxWorkers int) {
    semaphore := make(chan struct{}, maxWorkers)
    var wg sync.WaitGroup
    
    for _, file := range files {
        wg.Add(1)
        go func(f File) {
            defer wg.Done()
            
            semaphore <- struct{}{} // Acquire
            defer func() { <-semaphore }() // Release
            
            Upload(f, &sync.WaitGroup{})
        }(file)
    }
    
    wg.Wait()
}
```

## Testing

The package includes tests for:

- **Concurrent Upload Correctness**: All files upload successfully
- **Worker Pool Behavior**: Proper resource management
- **Rate Limiting**: Respect for bandwidth constraints
- **Progress Tracking**: Accurate progress reporting
- **Error Handling**: Graceful failure management

Run the tests:

```bash
go test -v ./parallelfileuploader
go test -race ./parallelfileuploader  # Race condition detection
```

## Performance Considerations

1. **Optimal Worker Count**: Usually 2-4x CPU cores for I/O bound operations
2. **Memory Usage**: Each goroutine uses ~2KB of memory
3. **Network Bandwidth**: Balance concurrency with available bandwidth
4. **Server Limits**: Respect server connection limits
5. **File Size Distribution**: Larger files may need different strategies

### Performance Tuning

```go
// Dynamic worker count based on system resources
func OptimalWorkerCount() int {
    return runtime.NumCPU() * 4 // I/O bound operations
}

// Adaptive rate limiting based on response times
func AdaptiveRateLimit(baseRate int, avgResponseTime time.Duration) int {
    if avgResponseTime > 5*time.Second {
        return baseRate / 2
    } else if avgResponseTime < 1*time.Second {
        return baseRate * 2
    }
    return baseRate
}
```

## Real-World Applications

This parallel file upload pattern is applicable for:

- **Cloud Storage**: Uploading files to AWS S3, Google Cloud Storage
- **Content Management**: Bulk media uploads to CDNs
- **Backup Systems**: Concurrent backup of multiple files
- **CI/CD Pipelines**: Parallel artifact uploads
- **Media Processing**: Batch upload of processed content
- **Data Migration**: Moving files between storage systems

## Advanced Features

### Chunked Upload for Large Files

```go
type ChunkedUploader struct {
    chunkSize int
    maxChunks int
}

func (cu *ChunkedUploader) UploadLargeFile(file LargeFile) error {
    chunks := cu.splitIntoChunks(file)
    
    var wg sync.WaitGroup
    semaphore := make(chan struct{}, cu.maxChunks)
    
    for i, chunk := range chunks {
        wg.Add(1)
        go func(chunkIndex int, data []byte) {
            defer wg.Done()
            
            semaphore <- struct{}{}
            defer func() { <-semaphore }()
            
            cu.uploadChunk(file.ID, chunkIndex, data)
        }(i, chunk)
    }
    
    wg.Wait()
    return cu.finalizeUpload(file.ID)
}
```

### Bandwidth-Aware Upload

```go
type BandwidthMonitor struct {
    totalBytes int64
    startTime  time.Time
    mutex      sync.Mutex
}

func (bm *BandwidthMonitor) RecordUpload(bytes int) {
    bm.mutex.Lock()
    defer bm.mutex.Unlock()
    
    bm.totalBytes += int64(bytes)
}

func (bm *BandwidthMonitor) GetBandwidth() float64 {
    bm.mutex.Lock()
    defer bm.mutex.Unlock()
    
    duration := time.Since(bm.startTime).Seconds()
    return float64(bm.totalBytes) / duration / 1024 / 1024 // MB/s
}
```

The implementation demonstrates efficient concurrent programming patterns for I/O-bound operations, providing a foundation for building scalable file upload systems with proper resource management and error handling.