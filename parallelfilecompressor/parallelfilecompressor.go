package parallelfilecompressor

import (
	"bufio"
	"bytes"
	"compress/flate"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

type CompressionAlgorithm int

const (
	Gzip CompressionAlgorithm = iota
	Deflate
	LZ4
	Brotli
)

type CompressionLevel int

const (
	BestSpeed    CompressionLevel = 1
	BestSize     CompressionLevel = 9
	DefaultLevel CompressionLevel = 6
)

type ChunkInfo struct {
	Index       int
	Offset      int64
	Size        int64
	OriginalCRC uint32
}

type CompressedChunk struct {
	ChunkInfo
	Data           []byte
	CompressedSize int64
	CompressionCRC uint32
	Error          error
}

type CompressionStats struct {
	OriginalSize       int64
	CompressedSize     int64
	CompressionRatio   float64
	TotalTime          time.Duration
	ThroughputMBps     float64
	ChunksProcessed    int32
	TotalChunks        int32
	WorkersUsed        int
	Algorithm          CompressionAlgorithm
	Level              CompressionLevel
	ChunkSize          int64
}

type ProgressCallback func(stats CompressionStats)

type CompressorConfig struct {
	Algorithm        CompressionAlgorithm
	Level            CompressionLevel
	ChunkSize        int64
	NumWorkers       int
	ProgressCallback ProgressCallback
	BufferSize       int
	TempDir          string
	MemoryLimit      int64
}

type ParallelFileCompressor struct {
	config     CompressorConfig
	workers    int
	chunkSize  int64
	tempDir    string
	memLimit   int64
	stats      CompressionStats
	statsMutex sync.RWMutex
}

type CompressorWorker struct {
	id        int
	workChan  <-chan ChunkInfo
	resultChan chan<- CompressedChunk
	algorithm CompressionAlgorithm
	level     CompressionLevel
	inputFile *os.File
	stats     *CompressionStats
	statsMutex *sync.RWMutex
}

type FileHeader struct {
	Magic           [4]byte
	Version         uint16
	Algorithm       CompressionAlgorithm
	Level           CompressionLevel
	ChunkSize       int64
	TotalChunks     int32
	OriginalSize    int64
	OriginalCRC     uint32
	Created         int64
	ChunkInfos      []ChunkInfo
}

func NewParallelFileCompressor(config CompressorConfig) *ParallelFileCompressor {
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	
	if config.ChunkSize <= 0 {
		config.ChunkSize = 64 * 1024 * 1024 // 64MB default
	}
	
	if config.BufferSize <= 0 {
		config.BufferSize = 32 * 1024 // 32KB buffer
	}
	
	if config.TempDir == "" {
		config.TempDir = os.TempDir()
	}
	
	if config.MemoryLimit <= 0 {
		config.MemoryLimit = 1024 * 1024 * 1024 // 1GB default
	}
	
	return &ParallelFileCompressor{
		config:    config,
		workers:   config.NumWorkers,
		chunkSize: config.ChunkSize,
		tempDir:   config.TempDir,
		memLimit:  config.MemoryLimit,
	}
}

func (pfc *ParallelFileCompressor) CompressFile(ctx context.Context, inputPath, outputPath string) (*CompressionStats, error) {
	startTime := time.Now()
	
	// Reset stats
	pfc.statsMutex.Lock()
	pfc.stats = CompressionStats{
		Algorithm:   pfc.config.Algorithm,
		Level:       pfc.config.Level,
		WorkersUsed: pfc.workers,
		ChunkSize:   pfc.chunkSize,
	}
	pfc.statsMutex.Unlock()
	
	// Open input file
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open input file: %w", err)
	}
	defer inputFile.Close()
	
	// Get file info
	fileInfo, err := inputFile.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %w", err)
	}
	
	originalSize := fileInfo.Size()
	pfc.updateStats(func(s *CompressionStats) {
		s.OriginalSize = originalSize
		s.TotalChunks = int32((originalSize + pfc.chunkSize - 1) / pfc.chunkSize)
	})
	
	// Create chunks
	chunks, err := pfc.createChunks(inputFile, originalSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create chunks: %w", err)
	}
	
	// Compress chunks in parallel
	compressedChunks, err := pfc.compressChunks(ctx, inputFile, chunks)
	if err != nil {
		return nil, fmt.Errorf("failed to compress chunks: %w", err)
	}
	
	// Write compressed file
	compressedSize, err := pfc.writeCompressedFile(outputPath, chunks, compressedChunks)
	if err != nil {
		return nil, fmt.Errorf("failed to write compressed file: %w", err)
	}
	
	// Update final stats
	duration := time.Since(startTime)
	pfc.updateStats(func(s *CompressionStats) {
		s.CompressedSize = compressedSize
		s.CompressionRatio = float64(originalSize) / float64(compressedSize)
		s.TotalTime = duration
		if duration.Seconds() > 0 {
			s.ThroughputMBps = float64(originalSize) / (1024 * 1024) / duration.Seconds()
		}
	})
	
	finalStats := pfc.getStats()
	return &finalStats, nil
}

func (pfc *ParallelFileCompressor) DecompressFile(ctx context.Context, inputPath, outputPath string) (*CompressionStats, error) {
	startTime := time.Now()
	
	// Open compressed file
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open compressed file: %w", err)
	}
	defer inputFile.Close()
	
	// Read file header
	header, err := pfc.readFileHeader(inputFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read file header: %w", err)
	}
	
	// Validate header
	if err := pfc.validateHeader(header); err != nil {
		return nil, fmt.Errorf("invalid file header: %w", err)
	}
	
	// Create output file
	outputFile, err := os.Create(outputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create output file: %w", err)
	}
	defer outputFile.Close()
	
	// Decompress chunks in parallel
	err = pfc.decompressChunks(ctx, inputFile, outputFile, header)
	if err != nil {
		return nil, fmt.Errorf("failed to decompress chunks: %w", err)
	}
	
	// Calculate stats
	duration := time.Since(startTime)
	stats := CompressionStats{
		OriginalSize:     header.OriginalSize,
		CompressedSize:   0, // Will be calculated from file size
		CompressionRatio: 0,
		TotalTime:        duration,
		ThroughputMBps:   float64(header.OriginalSize) / (1024 * 1024) / duration.Seconds(),
		ChunksProcessed:  header.TotalChunks,
		TotalChunks:      header.TotalChunks,
		WorkersUsed:      pfc.workers,
		Algorithm:        header.Algorithm,
		Level:            header.Level,
		ChunkSize:        header.ChunkSize,
	}
	
	return &stats, nil
}

func (pfc *ParallelFileCompressor) createChunks(file *os.File, fileSize int64) ([]ChunkInfo, error) {
	var chunks []ChunkInfo
	
	for offset := int64(0); offset < fileSize; offset += pfc.chunkSize {
		size := pfc.chunkSize
		if offset+size > fileSize {
			size = fileSize - offset
		}
		
		chunk := ChunkInfo{
			Index:  len(chunks),
			Offset: offset,
			Size:   size,
		}
		
		// Calculate CRC for this chunk
		crc, err := pfc.calculateChunkCRC(file, offset, size)
		if err != nil {
			return nil, fmt.Errorf("failed to calculate CRC for chunk %d: %w", chunk.Index, err)
		}
		chunk.OriginalCRC = crc
		
		chunks = append(chunks, chunk)
	}
	
	return chunks, nil
}

func (pfc *ParallelFileCompressor) calculateChunkCRC(file *os.File, offset, size int64) (uint32, error) {
	// Simplified CRC calculation (in real implementation, use proper CRC32)
	_, err := file.Seek(offset, io.SeekStart)
	if err != nil {
		return 0, err
	}
	
	buffer := make([]byte, size)
	_, err = io.ReadFull(file, buffer)
	if err != nil {
		return 0, err
	}
	
	var crc uint32
	for _, b := range buffer {
		crc = crc*31 + uint32(b)
	}
	
	return crc, nil
}

func (pfc *ParallelFileCompressor) compressChunks(ctx context.Context, inputFile *os.File, chunks []ChunkInfo) ([]CompressedChunk, error) {
	workChan := make(chan ChunkInfo, len(chunks))
	resultChan := make(chan CompressedChunk, len(chunks))
	
	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < pfc.workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			worker := &CompressorWorker{
				id:         workerID,
				workChan:   workChan,
				resultChan: resultChan,
				algorithm:  pfc.config.Algorithm,
				level:      pfc.config.Level,
				inputFile:  inputFile,
				stats:      &pfc.stats,
				statsMutex: &pfc.statsMutex,
			}
			
			worker.run(ctx)
		}(i)
	}
	
	// Send work
	for _, chunk := range chunks {
		select {
		case workChan <- chunk:
		case <-ctx.Done():
			close(workChan)
			return nil, ctx.Err()
		}
	}
	close(workChan)
	
	// Collect results
	var results []CompressedChunk
	go func() {
		wg.Wait()
		close(resultChan)
	}()
	
	for result := range resultChan {
		if result.Error != nil {
			return nil, fmt.Errorf("compression error in chunk %d: %w", result.Index, result.Error)
		}
		results = append(results, result)
		
		// Update progress
		atomic.AddInt32(&pfc.stats.ChunksProcessed, 1)
		if pfc.config.ProgressCallback != nil {
			pfc.config.ProgressCallback(pfc.getStats())
		}
	}
	
	// Sort results by chunk index
	sortedResults := make([]CompressedChunk, len(chunks))
	for _, result := range results {
		sortedResults[result.Index] = result
	}
	
	return sortedResults, nil
}

func (w *CompressorWorker) run(ctx context.Context) {
	for {
		select {
		case chunk, ok := <-w.workChan:
			if !ok {
				return
			}
			
			result := w.compressChunk(chunk)
			
			select {
			case w.resultChan <- result:
			case <-ctx.Done():
				return
			}
			
		case <-ctx.Done():
			return
		}
	}
}

func (w *CompressorWorker) compressChunk(chunk ChunkInfo) CompressedChunk {
	result := CompressedChunk{
		ChunkInfo: chunk,
	}
	
	// Read chunk data
	data := make([]byte, chunk.Size)
	_, err := w.inputFile.ReadAt(data, chunk.Offset)
	if err != nil {
		result.Error = fmt.Errorf("failed to read chunk data: %w", err)
		return result
	}
	
	// Compress data
	var compressedData []byte
	switch w.algorithm {
	case Gzip:
		compressedData, err = w.compressWithGzip(data, int(w.level))
	case Deflate:
		compressedData, err = w.compressWithDeflate(data, int(w.level))
	default:
		err = fmt.Errorf("unsupported compression algorithm: %v", w.algorithm)
	}
	
	if err != nil {
		result.Error = fmt.Errorf("compression failed: %w", err)
		return result
	}
	
	result.Data = compressedData
	result.CompressedSize = int64(len(compressedData))
	
	// Calculate compressed data CRC
	var crc uint32
	for _, b := range compressedData {
		crc = crc*31 + uint32(b)
	}
	result.CompressionCRC = crc
	
	return result
}

func (w *CompressorWorker) compressWithGzip(data []byte, level int) ([]byte, error) {
	var buf bytes.Buffer
	
	gzipWriter, err := gzip.NewWriterLevel(&buf, level)
	if err != nil {
		return nil, err
	}
	
	_, err = gzipWriter.Write(data)
	if err != nil {
		gzipWriter.Close()
		return nil, err
	}
	
	err = gzipWriter.Close()
	if err != nil {
		return nil, err
	}
	
	return buf.Bytes(), nil
}

func (w *CompressorWorker) compressWithDeflate(data []byte, level int) ([]byte, error) {
	var buf bytes.Buffer
	
	deflateWriter, err := flate.NewWriter(&buf, level)
	if err != nil {
		return nil, err
	}
	
	_, err = deflateWriter.Write(data)
	if err != nil {
		deflateWriter.Close()
		return nil, err
	}
	
	err = deflateWriter.Close()
	if err != nil {
		return nil, err
	}
	
	return buf.Bytes(), nil
}

func (pfc *ParallelFileCompressor) writeCompressedFile(outputPath string, chunks []ChunkInfo, compressedChunks []CompressedChunk) (int64, error) {
	outputFile, err := os.Create(outputPath)
	if err != nil {
		return 0, err
	}
	defer outputFile.Close()
	
	writer := bufio.NewWriter(outputFile)
	defer writer.Flush()
	
	// Write file header
	header := FileHeader{
		Magic:        [4]byte{'P', 'F', 'C', '1'},
		Version:      1,
		Algorithm:    pfc.config.Algorithm,
		Level:        pfc.config.Level,
		ChunkSize:    pfc.chunkSize,
		TotalChunks:  int32(len(chunks)),
		OriginalSize: pfc.stats.OriginalSize,
		Created:      time.Now().Unix(),
		ChunkInfos:   chunks,
	}
	
	headerBytes, err := pfc.encodeHeader(header)
	if err != nil {
		return 0, fmt.Errorf("failed to encode header: %w", err)
	}
	
	written, err := writer.Write(headerBytes)
	if err != nil {
		return 0, fmt.Errorf("failed to write header: %w", err)
	}
	
	totalWritten := int64(written)
	
	// Write compressed chunks
	for _, compressedChunk := range compressedChunks {
		chunkHeader := make([]byte, 16)
		// Write chunk index (4 bytes)
		chunkHeader[0] = byte(compressedChunk.Index)
		chunkHeader[1] = byte(compressedChunk.Index >> 8)
		chunkHeader[2] = byte(compressedChunk.Index >> 16)
		chunkHeader[3] = byte(compressedChunk.Index >> 24)
		
		// Write compressed size (8 bytes)
		size := compressedChunk.CompressedSize
		for i := 0; i < 8; i++ {
			chunkHeader[4+i] = byte(size >> (i * 8))
		}
		
		// Write CRC (4 bytes)
		crc := compressedChunk.CompressionCRC
		chunkHeader[12] = byte(crc)
		chunkHeader[13] = byte(crc >> 8)
		chunkHeader[14] = byte(crc >> 16)
		chunkHeader[15] = byte(crc >> 24)
		
		written, err := writer.Write(chunkHeader)
		if err != nil {
			return 0, fmt.Errorf("failed to write chunk header: %w", err)
		}
		totalWritten += int64(written)
		
		written, err = writer.Write(compressedChunk.Data)
		if err != nil {
			return 0, fmt.Errorf("failed to write chunk data: %w", err)
		}
		totalWritten += int64(written)
	}
	
	return totalWritten, nil
}

func (pfc *ParallelFileCompressor) encodeHeader(header FileHeader) ([]byte, error) {
	var buf bytes.Buffer
	
	// Write magic and version
	buf.Write(header.Magic[:])
	buf.WriteByte(byte(header.Version))
	buf.WriteByte(byte(header.Version >> 8))
	
	// Write algorithm and level
	buf.WriteByte(byte(header.Algorithm))
	buf.WriteByte(byte(header.Level))
	
	// Write chunk size (8 bytes)
	for i := 0; i < 8; i++ {
		buf.WriteByte(byte(header.ChunkSize >> (i * 8)))
	}
	
	// Write total chunks (4 bytes)
	buf.WriteByte(byte(header.TotalChunks))
	buf.WriteByte(byte(header.TotalChunks >> 8))
	buf.WriteByte(byte(header.TotalChunks >> 16))
	buf.WriteByte(byte(header.TotalChunks >> 24))
	
	// Write original size (8 bytes)
	for i := 0; i < 8; i++ {
		buf.WriteByte(byte(header.OriginalSize >> (i * 8)))
	}
	
	// Write created timestamp (8 bytes)
	for i := 0; i < 8; i++ {
		buf.WriteByte(byte(header.Created >> (i * 8)))
	}
	
	// Write chunk infos
	for _, chunk := range header.ChunkInfos {
		// Index (4 bytes)
		buf.WriteByte(byte(chunk.Index))
		buf.WriteByte(byte(chunk.Index >> 8))
		buf.WriteByte(byte(chunk.Index >> 16))
		buf.WriteByte(byte(chunk.Index >> 24))
		
		// Offset (8 bytes)
		for i := 0; i < 8; i++ {
			buf.WriteByte(byte(chunk.Offset >> (i * 8)))
		}
		
		// Size (8 bytes)
		for i := 0; i < 8; i++ {
			buf.WriteByte(byte(chunk.Size >> (i * 8)))
		}
		
		// CRC (4 bytes)
		buf.WriteByte(byte(chunk.OriginalCRC))
		buf.WriteByte(byte(chunk.OriginalCRC >> 8))
		buf.WriteByte(byte(chunk.OriginalCRC >> 16))
		buf.WriteByte(byte(chunk.OriginalCRC >> 24))
	}
	
	return buf.Bytes(), nil
}

func (pfc *ParallelFileCompressor) readFileHeader(file *os.File) (*FileHeader, error) {
	// Read basic header first
	basicHeader := make([]byte, 40) // Magic(4) + Version(2) + Algorithm(1) + Level(1) + ChunkSize(8) + TotalChunks(4) + OriginalSize(8) + Created(8) + OriginalCRC(4)
	_, err := io.ReadFull(file, basicHeader[:36]) // Read without OriginalCRC for now
	if err != nil {
		return nil, err
	}
	
	header := &FileHeader{}
	copy(header.Magic[:], basicHeader[0:4])
	header.Version = uint16(basicHeader[4]) | uint16(basicHeader[5])<<8
	header.Algorithm = CompressionAlgorithm(basicHeader[6])
	header.Level = CompressionLevel(basicHeader[7])
	
	// Read chunk size
	header.ChunkSize = 0
	for i := 0; i < 8; i++ {
		header.ChunkSize |= int64(basicHeader[8+i]) << (i * 8)
	}
	
	// Read total chunks
	header.TotalChunks = 0
	for i := 0; i < 4; i++ {
		header.TotalChunks |= int32(basicHeader[16+i]) << (i * 8)
	}
	
	// Read original size
	header.OriginalSize = 0
	for i := 0; i < 8; i++ {
		header.OriginalSize |= int64(basicHeader[20+i]) << (i * 8)
	}
	
	// Read created timestamp
	header.Created = 0
	for i := 0; i < 8; i++ {
		header.Created |= int64(basicHeader[28+i]) << (i * 8)
	}
	
	// Read chunk infos
	header.ChunkInfos = make([]ChunkInfo, header.TotalChunks)
	for i := int32(0); i < header.TotalChunks; i++ {
		chunkData := make([]byte, 24) // Index(4) + Offset(8) + Size(8) + CRC(4)
		_, err = io.ReadFull(file, chunkData)
		if err != nil {
			return nil, err
		}
		
		chunk := ChunkInfo{}
		chunk.Index = int(chunkData[0]) | int(chunkData[1])<<8 | int(chunkData[2])<<16 | int(chunkData[3])<<24
		
		for j := 0; j < 8; j++ {
			chunk.Offset |= int64(chunkData[4+j]) << (j * 8)
		}
		
		for j := 0; j < 8; j++ {
			chunk.Size |= int64(chunkData[12+j]) << (j * 8)
		}
		
		chunk.OriginalCRC = uint32(chunkData[20]) | uint32(chunkData[21])<<8 | uint32(chunkData[22])<<16 | uint32(chunkData[23])<<24
		
		header.ChunkInfos[i] = chunk
	}
	
	return header, nil
}

func (pfc *ParallelFileCompressor) validateHeader(header *FileHeader) error {
	if header.Magic != [4]byte{'P', 'F', 'C', '1'} {
		return errors.New("invalid magic number")
	}
	
	if header.Version != 1 {
		return fmt.Errorf("unsupported version: %d", header.Version)
	}
	
	if header.Algorithm != Gzip && header.Algorithm != Deflate {
		return fmt.Errorf("unsupported algorithm: %v", header.Algorithm)
	}
	
	return nil
}

func (pfc *ParallelFileCompressor) decompressChunks(ctx context.Context, inputFile *os.File, outputFile *os.File, header *FileHeader) error {
	// Implementation would be similar to compression but in reverse
	// For brevity, this is a simplified version
	
	for _, chunk := range header.ChunkInfos {
		// Read chunk header from input file
		chunkHeader := make([]byte, 16)
		_, err := io.ReadFull(inputFile, chunkHeader)
		if err != nil {
			return fmt.Errorf("failed to read chunk header: %w", err)
		}
		
		// Extract compressed size
		compressedSize := int64(0)
		for i := 0; i < 8; i++ {
			compressedSize |= int64(chunkHeader[4+i]) << (i * 8)
		}
		
		// Read compressed data
		compressedData := make([]byte, compressedSize)
		_, err = io.ReadFull(inputFile, compressedData)
		if err != nil {
			return fmt.Errorf("failed to read compressed data: %w", err)
		}
		
		// Decompress data
		var decompressedData []byte
		switch header.Algorithm {
		case Gzip:
			decompressedData, err = pfc.decompressWithGzip(compressedData)
		case Deflate:
			decompressedData, err = pfc.decompressWithDeflate(compressedData)
		default:
			return fmt.Errorf("unsupported algorithm: %v", header.Algorithm)
		}
		
		if err != nil {
			return fmt.Errorf("decompression failed for chunk %d: %w", chunk.Index, err)
		}
		
		// Write decompressed data to output file
		_, err = outputFile.WriteAt(decompressedData, chunk.Offset)
		if err != nil {
			return fmt.Errorf("failed to write decompressed data: %w", err)
		}
	}
	
	return nil
}

func (pfc *ParallelFileCompressor) decompressWithGzip(data []byte) ([]byte, error) {
	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	
	var buf bytes.Buffer
	_, err = io.Copy(&buf, reader)
	if err != nil {
		return nil, err
	}
	
	return buf.Bytes(), nil
}

func (pfc *ParallelFileCompressor) decompressWithDeflate(data []byte) ([]byte, error) {
	reader := flate.NewReader(bytes.NewReader(data))
	defer reader.Close()
	
	var buf bytes.Buffer
	_, err := io.Copy(&buf, reader)
	if err != nil {
		return nil, err
	}
	
	return buf.Bytes(), nil
}

func (pfc *ParallelFileCompressor) updateStats(updateFunc func(*CompressionStats)) {
	pfc.statsMutex.Lock()
	defer pfc.statsMutex.Unlock()
	updateFunc(&pfc.stats)
}

func (pfc *ParallelFileCompressor) getStats() CompressionStats {
	pfc.statsMutex.RLock()
	defer pfc.statsMutex.RUnlock()
	return pfc.stats
}

func (pfc *ParallelFileCompressor) GetProgress() CompressionStats {
	return pfc.getStats()
}

func (pfc *ParallelFileCompressor) Reset() {
	pfc.statsMutex.Lock()
	defer pfc.statsMutex.Unlock()
	pfc.stats = CompressionStats{
		Algorithm:   pfc.config.Algorithm,
		Level:       pfc.config.Level,
		WorkersUsed: pfc.workers,
		ChunkSize:   pfc.chunkSize,
	}
}

// Utility functions

func CompareAlgorithms(inputPath string, algorithms []CompressionAlgorithm, workers int, chunkSize int64) (map[CompressionAlgorithm]*CompressionStats, error) {
	results := make(map[CompressionAlgorithm]*CompressionStats)
	
	for _, algorithm := range algorithms {
		config := CompressorConfig{
			Algorithm: algorithm,
			Level:     DefaultLevel,
			ChunkSize: chunkSize,
			NumWorkers: workers,
		}
		
		compressor := NewParallelFileCompressor(config)
		
		outputPath := fmt.Sprintf("%s.%s.pfc", inputPath, algorithmToString(algorithm))
		stats, err := compressor.CompressFile(context.Background(), inputPath, outputPath)
		if err != nil {
			return nil, fmt.Errorf("compression failed for algorithm %v: %w", algorithm, err)
		}
		
		results[algorithm] = stats
		
		// Clean up
		os.Remove(outputPath)
	}
	
	return results, nil
}

func BenchmarkCompression(inputPath string, algorithm CompressionAlgorithm, level CompressionLevel, workerCounts []int, chunkSize int64) (map[int]time.Duration, error) {
	results := make(map[int]time.Duration)
	
	for _, workers := range workerCounts {
		config := CompressorConfig{
			Algorithm:  algorithm,
			Level:      level,
			ChunkSize:  chunkSize,
			NumWorkers: workers,
		}
		
		compressor := NewParallelFileCompressor(config)
		
		outputPath := fmt.Sprintf("%s.bench_%d.pfc", inputPath, workers)
		start := time.Now()
		
		_, err := compressor.CompressFile(context.Background(), inputPath, outputPath)
		if err != nil {
			return nil, fmt.Errorf("compression failed for %d workers: %w", workers, err)
		}
		
		results[workers] = time.Since(start)
		
		// Clean up
		os.Remove(outputPath)
	}
	
	return results, nil
}

func algorithmToString(algorithm CompressionAlgorithm) string {
	switch algorithm {
	case Gzip:
		return "gzip"
	case Deflate:
		return "deflate"
	case LZ4:
		return "lz4"
	case Brotli:
		return "brotli"
	default:
		return "unknown"
	}
}

func GetOptimalChunkSize(fileSize int64, workers int) int64 {
	// Rule of thumb: aim for 10-100 chunks per worker
	targetChunks := int64(workers * 50)
	chunkSize := fileSize / targetChunks
	
	// Ensure minimum chunk size of 1MB
	const minChunkSize = 1024 * 1024
	if chunkSize < minChunkSize {
		chunkSize = minChunkSize
	}
	
	// Ensure maximum chunk size of 256MB
	const maxChunkSize = 256 * 1024 * 1024
	if chunkSize > maxChunkSize {
		chunkSize = maxChunkSize
	}
	
	return chunkSize
}

func EstimateCompressionTime(fileSize int64, algorithm CompressionAlgorithm, workers int) time.Duration {
	// Rough estimates based on typical performance
	var throughputMBps float64
	
	switch algorithm {
	case Gzip:
		throughputMBps = 20.0 * float64(workers) // Scales with workers
	case Deflate:
		throughputMBps = 30.0 * float64(workers)
	case LZ4:
		throughputMBps = 100.0 * float64(workers)
	default:
		throughputMBps = 20.0 * float64(workers)
	}
	
	// Account for diminishing returns with more workers
	if workers > runtime.NumCPU() {
		throughputMBps *= 0.8
	}
	
	fileSizeMB := float64(fileSize) / (1024 * 1024)
	estimatedSeconds := fileSizeMB / throughputMBps
	
	return time.Duration(estimatedSeconds * float64(time.Second))
}