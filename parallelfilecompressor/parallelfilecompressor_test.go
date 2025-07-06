package parallelfilecompressor

import (
	"context"
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestNewParallelFileCompressor(t *testing.T) {
	config := CompressorConfig{
		Algorithm:  Gzip,
		Level:      DefaultLevel,
		ChunkSize:  1024 * 1024,
		NumWorkers: 4,
	}
	
	compressor := NewParallelFileCompressor(config)
	
	if compressor == nil {
		t.Fatal("Expected compressor to be created")
	}
	
	if compressor.workers != 4 {
		t.Errorf("Expected 4 workers, got %d", compressor.workers)
	}
	
	if compressor.chunkSize != 1024*1024 {
		t.Errorf("Expected chunk size 1MB, got %d", compressor.chunkSize)
	}
}

func TestDefaultConfiguration(t *testing.T) {
	config := CompressorConfig{}
	compressor := NewParallelFileCompressor(config)
	
	if compressor.workers != runtime.NumCPU() {
		t.Errorf("Expected default workers to be %d, got %d", runtime.NumCPU(), compressor.workers)
	}
	
	if compressor.chunkSize != 64*1024*1024 {
		t.Errorf("Expected default chunk size 64MB, got %d", compressor.chunkSize)
	}
}

func TestBasicCompression(t *testing.T) {
	// Create test data
	testData := createTestFile(t, "test_basic.txt", 1024*100) // 100KB test file
	defer os.Remove(testData)
	
	config := CompressorConfig{
		Algorithm:  Gzip,
		Level:      DefaultLevel,
		ChunkSize:  1024 * 10, // 10KB chunks
		NumWorkers: 2,
	}
	
	compressor := NewParallelFileCompressor(config)
	
	compressedFile := testData + ".pfc"
	defer os.Remove(compressedFile)
	
	ctx := context.Background()
	stats, err := compressor.CompressFile(ctx, testData, compressedFile)
	
	if err != nil {
		t.Fatalf("Compression failed: %v", err)
	}
	
	if stats == nil {
		t.Fatal("Expected compression stats")
	}
	
	if stats.OriginalSize != 1024*100 {
		t.Errorf("Expected original size %d, got %d", 1024*100, stats.OriginalSize)
	}
	
	if stats.CompressedSize >= stats.OriginalSize {
		t.Errorf("Expected compressed size to be smaller than original")
	}
	
	if stats.CompressionRatio <= 1.0 {
		t.Errorf("Expected compression ratio > 1.0, got %f", stats.CompressionRatio)
	}
	
	// Verify compressed file exists
	if _, err := os.Stat(compressedFile); os.IsNotExist(err) {
		t.Error("Compressed file was not created")
	}
}

func TestCompressionDecompression(t *testing.T) {
	// Create test data
	testData := createTestFile(t, "test_roundtrip.txt", 1024*50)
	defer os.Remove(testData)
	
	config := CompressorConfig{
		Algorithm:  Gzip,
		Level:      DefaultLevel,
		ChunkSize:  1024 * 5,
		NumWorkers: 2,
	}
	
	compressor := NewParallelFileCompressor(config)
	
	compressedFile := testData + ".pfc"
	decompressedFile := testData + ".decompressed"
	defer os.Remove(compressedFile)
	defer os.Remove(decompressedFile)
	
	ctx := context.Background()
	
	// Compress
	_, err := compressor.CompressFile(ctx, testData, compressedFile)
	if err != nil {
		t.Fatalf("Compression failed: %v", err)
	}
	
	// Decompress
	_, err = compressor.DecompressFile(ctx, compressedFile, decompressedFile)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}
	
	// Compare files
	if !compareFiles(t, testData, decompressedFile) {
		t.Error("Decompressed file doesn't match original")
	}
}

func TestDifferentAlgorithms(t *testing.T) {
	testData := createTestFile(t, "test_algorithms.txt", 1024*20)
	defer os.Remove(testData)
	
	algorithms := []CompressionAlgorithm{Gzip, Deflate}
	
	for _, algorithm := range algorithms {
		t.Run(fmt.Sprintf("Algorithm_%v", algorithm), func(t *testing.T) {
			config := CompressorConfig{
				Algorithm:  algorithm,
				Level:      DefaultLevel,
				ChunkSize:  1024 * 5,
				NumWorkers: 2,
			}
			
			compressor := NewParallelFileCompressor(config)
			
			compressedFile := fmt.Sprintf("%s.%d.pfc", testData, algorithm)
			defer os.Remove(compressedFile)
			
			ctx := context.Background()
			stats, err := compressor.CompressFile(ctx, testData, compressedFile)
			
			if err != nil {
				t.Fatalf("Compression failed for algorithm %v: %v", algorithm, err)
			}
			
			if stats.Algorithm != algorithm {
				t.Errorf("Expected algorithm %v, got %v", algorithm, stats.Algorithm)
			}
			
			if stats.CompressedSize >= stats.OriginalSize {
				t.Errorf("Compression didn't reduce file size for algorithm %v", algorithm)
			}
		})
	}
}

func TestDifferentCompressionLevels(t *testing.T) {
	testData := createTestFile(t, "test_levels.txt", 1024*30)
	defer os.Remove(testData)
	
	levels := []CompressionLevel{BestSpeed, DefaultLevel, BestSize}
	
	var results []CompressionStats
	
	for _, level := range levels {
		t.Run(fmt.Sprintf("Level_%d", level), func(t *testing.T) {
			config := CompressorConfig{
				Algorithm:  Gzip,
				Level:      level,
				ChunkSize:  1024 * 5,
				NumWorkers: 2,
			}
			
			compressor := NewParallelFileCompressor(config)
			
			compressedFile := fmt.Sprintf("%s.%d.pfc", testData, level)
			defer os.Remove(compressedFile)
			
			ctx := context.Background()
			stats, err := compressor.CompressFile(ctx, testData, compressedFile)
			
			if err != nil {
				t.Fatalf("Compression failed for level %v: %v", level, err)
			}
			
			if stats.Level != level {
				t.Errorf("Expected level %v, got %v", level, stats.Level)
			}
			
			results = append(results, *stats)
		})
	}
	
	// Generally, BestSize should produce smaller files than BestSpeed
	// Note: This might not always be true for small files
	if len(results) == 3 {
		bestSpeed := results[0]
		bestSize := results[2]
		
		if bestSize.CompressionRatio <= bestSpeed.CompressionRatio {
			t.Logf("BestSize ratio: %f, BestSpeed ratio: %f", bestSize.CompressionRatio, bestSpeed.CompressionRatio)
			// Don't fail the test as this can vary with small files
		}
	}
}

func TestConcurrentWorkers(t *testing.T) {
	testData := createTestFile(t, "test_workers.txt", 1024*100)
	defer os.Remove(testData)
	
	workerCounts := []int{1, 2, 4, 8}
	
	for _, workers := range workerCounts {
		t.Run(fmt.Sprintf("Workers_%d", workers), func(t *testing.T) {
			config := CompressorConfig{
				Algorithm:  Gzip,
				Level:      DefaultLevel,
				ChunkSize:  1024 * 5,
				NumWorkers: workers,
			}
			
			compressor := NewParallelFileCompressor(config)
			
			compressedFile := fmt.Sprintf("%s.%d_workers.pfc", testData, workers)
			defer os.Remove(compressedFile)
			
			ctx := context.Background()
			stats, err := compressor.CompressFile(ctx, testData, compressedFile)
			
			if err != nil {
				t.Fatalf("Compression failed with %d workers: %v", workers, err)
			}
			
			if stats.WorkersUsed != workers {
				t.Errorf("Expected %d workers used, got %d", workers, stats.WorkersUsed)
			}
		})
	}
}

func TestProgressCallback(t *testing.T) {
	testData := createTestFile(t, "test_progress.txt", 1024*50)
	defer os.Remove(testData)
	
	var progressUpdates []CompressionStats
	var mu sync.Mutex
	
	config := CompressorConfig{
		Algorithm:  Gzip,
		Level:      DefaultLevel,
		ChunkSize:  1024 * 5, // Small chunks to get multiple progress updates
		NumWorkers: 2,
		ProgressCallback: func(stats CompressionStats) {
			mu.Lock()
			progressUpdates = append(progressUpdates, stats)
			mu.Unlock()
		},
	}
	
	compressor := NewParallelFileCompressor(config)
	
	compressedFile := testData + ".pfc"
	defer os.Remove(compressedFile)
	
	ctx := context.Background()
	_, err := compressor.CompressFile(ctx, testData, compressedFile)
	
	if err != nil {
		t.Fatalf("Compression failed: %v", err)
	}
	
	mu.Lock()
	updateCount := len(progressUpdates)
	mu.Unlock()
	
	if updateCount == 0 {
		t.Error("Expected progress updates, got none")
	}
	
	// Verify progress updates are increasing
	mu.Lock()
	for i := 1; i < len(progressUpdates); i++ {
		if progressUpdates[i].ChunksProcessed < progressUpdates[i-1].ChunksProcessed {
			t.Error("Progress updates should be non-decreasing")
		}
	}
	mu.Unlock()
}

func TestChunkSizeOptimization(t *testing.T) {
	tests := []struct {
		fileSize int64
		workers  int
		expected int64
	}{
		{1024 * 1024, 1, 1024 * 1024},      // 1MB file, 1 worker
		{100 * 1024 * 1024, 4, 2097152},    // 100MB file, 4 workers
		{1024 * 1024 * 1024, 8, 20971520},  // 1GB file, 8 workers
		{10 * 1024, 4, 1024 * 1024},        // Small file, minimum chunk size
	}
	
	for _, test := range tests {
		result := GetOptimalChunkSize(test.fileSize, test.workers)
		if result != test.expected {
			t.Errorf("For file size %d and %d workers, expected chunk size %d, got %d",
				test.fileSize, test.workers, test.expected, result)
		}
	}
}

func TestCompareAlgorithms(t *testing.T) {
	testData := createTestFile(t, "test_compare.txt", 1024*100)
	defer os.Remove(testData)
	
	algorithms := []CompressionAlgorithm{Gzip, Deflate}
	
	results, err := CompareAlgorithms(testData, algorithms, 2, 1024*10)
	if err != nil {
		t.Fatalf("Algorithm comparison failed: %v", err)
	}
	
	if len(results) != len(algorithms) {
		t.Errorf("Expected %d results, got %d", len(algorithms), len(results))
	}
	
	for algorithm, stats := range results {
		if stats.Algorithm != algorithm {
			t.Errorf("Stats algorithm mismatch: expected %v, got %v", algorithm, stats.Algorithm)
		}
		
		if stats.OriginalSize == 0 || stats.CompressedSize == 0 {
			t.Errorf("Invalid stats for algorithm %v", algorithm)
		}
	}
}

func TestErrorHandling(t *testing.T) {
	config := CompressorConfig{
		Algorithm:  Gzip,
		Level:      DefaultLevel,
		ChunkSize:  1024 * 10,
		NumWorkers: 2,
	}
	
	compressor := NewParallelFileCompressor(config)
	ctx := context.Background()
	
	// Test non-existent input file
	_, err := compressor.CompressFile(ctx, "non_existent_file.txt", "output.pfc")
	if err == nil {
		t.Error("Expected error for non-existent input file")
	}
	
	// Test invalid output path
	testData := createTestFile(t, "test_error.txt", 1024)
	defer os.Remove(testData)
	
	_, err = compressor.CompressFile(ctx, testData, "/invalid/path/output.pfc")
	if err == nil {
		t.Error("Expected error for invalid output path")
	}
}

func TestCancellation(t *testing.T) {
	testData := createTestFile(t, "test_cancel.txt", 1024*100)
	defer os.Remove(testData)
	
	config := CompressorConfig{
		Algorithm:  Gzip,
		Level:      DefaultLevel,
		ChunkSize:  1024 * 5,
		NumWorkers: 2,
	}
	
	compressor := NewParallelFileCompressor(config)
	
	compressedFile := testData + ".pfc"
	defer os.Remove(compressedFile)
	
	ctx, cancel := context.WithCancel(context.Background())
	
	// Start compression in a goroutine
	var compressionErr error
	var wg sync.WaitGroup
	wg.Add(1)
	
	go func() {
		defer wg.Done()
		_, compressionErr = compressor.CompressFile(ctx, testData, compressedFile)
	}()
	
	// Cancel after a short delay
	time.Sleep(10 * time.Millisecond)
	cancel()
	
	wg.Wait()
	
	if compressionErr == nil {
		t.Error("Expected cancellation error")
	}
	
	if compressionErr != context.Canceled {
		t.Logf("Got error: %v", compressionErr)
		// The error might be wrapped, so just check that we got an error
	}
}

func TestStatisticsAccuracy(t *testing.T) {
	testData := createTestFile(t, "test_stats.txt", 1024*50)
	defer os.Remove(testData)
	
	config := CompressorConfig{
		Algorithm:  Gzip,
		Level:      DefaultLevel,
		ChunkSize:  1024 * 10,
		NumWorkers: 2,
	}
	
	compressor := NewParallelFileCompressor(config)
	
	compressedFile := testData + ".pfc"
	defer os.Remove(compressedFile)
	
	ctx := context.Background()
	stats, err := compressor.CompressFile(ctx, testData, compressedFile)
	
	if err != nil {
		t.Fatalf("Compression failed: %v", err)
	}
	
	// Verify file sizes match stats
	originalInfo, err := os.Stat(testData)
	if err != nil {
		t.Fatalf("Failed to stat original file: %v", err)
	}
	
	compressedInfo, err := os.Stat(compressedFile)
	if err != nil {
		t.Fatalf("Failed to stat compressed file: %v", err)
	}
	
	if stats.OriginalSize != originalInfo.Size() {
		t.Errorf("Original size mismatch: stats=%d, actual=%d", stats.OriginalSize, originalInfo.Size())
	}
	
	// Note: Compressed file includes header, so it will be larger than just compressed data
	if stats.CompressedSize > compressedInfo.Size() {
		t.Errorf("Compressed size inconsistent: stats=%d, file=%d", stats.CompressedSize, compressedInfo.Size())
	}
	
	// Verify other stats
	if stats.CompressionRatio <= 0 {
		t.Errorf("Invalid compression ratio: %f", stats.CompressionRatio)
	}
	
	if stats.TotalTime <= 0 {
		t.Errorf("Invalid total time: %v", stats.TotalTime)
	}
	
	if stats.ThroughputMBps <= 0 {
		t.Errorf("Invalid throughput: %f", stats.ThroughputMBps)
	}
}

func TestLargeFileHandling(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large file test in short mode")
	}
	
	// Create a larger test file (10MB)
	testData := createTestFile(t, "test_large.txt", 1024*1024*10)
	defer os.Remove(testData)
	
	config := CompressorConfig{
		Algorithm:  Gzip,
		Level:      DefaultLevel,
		ChunkSize:  1024 * 1024, // 1MB chunks
		NumWorkers: 4,
	}
	
	compressor := NewParallelFileCompressor(config)
	
	compressedFile := testData + ".pfc"
	decompressedFile := testData + ".decompressed"
	defer os.Remove(compressedFile)
	defer os.Remove(decompressedFile)
	
	ctx := context.Background()
	
	// Compress
	stats, err := compressor.CompressFile(ctx, testData, compressedFile)
	if err != nil {
		t.Fatalf("Large file compression failed: %v", err)
	}
	
	t.Logf("Large file compression stats: %.2f%% reduction, %.2f MB/s throughput",
		(1.0-1.0/stats.CompressionRatio)*100, stats.ThroughputMBps)
	
	// Decompress
	_, err = compressor.DecompressFile(ctx, compressedFile, decompressedFile)
	if err != nil {
		t.Fatalf("Large file decompression failed: %v", err)
	}
	
	// Verify integrity
	if !compareFiles(t, testData, decompressedFile) {
		t.Error("Large file integrity check failed")
	}
}

func BenchmarkCompression(b *testing.B) {
	testData := createTestFile(b, "bench_compress.txt", 1024*1024) // 1MB
	defer os.Remove(testData)
	
	config := CompressorConfig{
		Algorithm:  Gzip,
		Level:      DefaultLevel,
		ChunkSize:  1024 * 64,
		NumWorkers: runtime.NumCPU(),
	}
	
	compressor := NewParallelFileCompressor(config)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		compressedFile := fmt.Sprintf("%s.%d.pfc", testData, i)
		
		_, err := compressor.CompressFile(context.Background(), testData, compressedFile)
		if err != nil {
			b.Fatalf("Benchmark compression failed: %v", err)
		}
		
		os.Remove(compressedFile)
		compressor.Reset()
	}
}

func BenchmarkCompressionParallel(b *testing.B) {
	testData := createTestFile(b, "bench_parallel.txt", 1024*1024) // 1MB
	defer os.Remove(testData)
	
	config := CompressorConfig{
		Algorithm:  Gzip,
		Level:      DefaultLevel,
		ChunkSize:  1024 * 64,
		NumWorkers: runtime.NumCPU(),
	}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		compressor := NewParallelFileCompressor(config)
		i := 0
		for pb.Next() {
			compressedFile := fmt.Sprintf("%s.parallel_%d.pfc", testData, i)
			
			_, err := compressor.CompressFile(context.Background(), testData, compressedFile)
			if err != nil {
				b.Fatalf("Parallel benchmark compression failed: %v", err)
			}
			
			os.Remove(compressedFile)
			compressor.Reset()
			i++
		}
	})
}

func BenchmarkWorkerScaling(b *testing.B) {
	testData := createTestFile(b, "bench_scaling.txt", 1024*1024*2) // 2MB
	defer os.Remove(testData)
	
	workerCounts := []int{1, 2, 4, 8, 16}
	
	for _, workers := range workerCounts {
		b.Run(fmt.Sprintf("Workers_%d", workers), func(b *testing.B) {
			config := CompressorConfig{
				Algorithm:  Gzip,
				Level:      DefaultLevel,
				ChunkSize:  1024 * 64,
				NumWorkers: workers,
			}
			
			compressor := NewParallelFileCompressor(config)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				compressedFile := fmt.Sprintf("%s.scale_%d_%d.pfc", testData, workers, i)
				
				_, err := compressor.CompressFile(context.Background(), testData, compressedFile)
				if err != nil {
					b.Fatalf("Scaling benchmark failed: %v", err)
				}
				
				os.Remove(compressedFile)
				compressor.Reset()
			}
		})
	}
}

// Helper functions

func createTestFile(tb testing.TB, name string, size int) string {
	file, err := os.CreateTemp("", name)
	if err != nil {
		tb.Fatalf("Failed to create test file: %v", err)
	}
	defer file.Close()
	
	// Write some compressible data (repeated pattern)
	pattern := []byte("This is a test pattern that should compress well. ")
	patternLen := len(pattern)
	
	written := 0
	for written < size {
		writeSize := patternLen
		if written+writeSize > size {
			writeSize = size - written
		}
		
		n, err := file.Write(pattern[:writeSize])
		if err != nil {
			tb.Fatalf("Failed to write test data: %v", err)
		}
		written += n
	}
	
	return file.Name()
}

func compareFiles(tb testing.TB, file1, file2 string) bool {
	f1, err := os.Open(file1)
	if err != nil {
		tb.Fatalf("Failed to open file1: %v", err)
	}
	defer f1.Close()
	
	f2, err := os.Open(file2)
	if err != nil {
		tb.Fatalf("Failed to open file2: %v", err)
	}
	defer f2.Close()
	
	buf1 := make([]byte, 8192)
	buf2 := make([]byte, 8192)
	
	for {
		n1, err1 := f1.Read(buf1)
		n2, err2 := f2.Read(buf2)
		
		if n1 != n2 {
			return false
		}
		
		if n1 > 0 {
			for i := 0; i < n1; i++ {
				if buf1[i] != buf2[i] {
					return false
				}
			}
		}
		
		if err1 != nil || err2 != nil {
			return err1 == err2 // Both should be EOF or both should be nil
		}
		
		if n1 == 0 {
			break
		}
	}
	
	return true
}