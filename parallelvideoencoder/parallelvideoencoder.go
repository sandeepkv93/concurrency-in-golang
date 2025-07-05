package parallelvideoencoder

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// VideoEncoder represents a parallel video encoder
type VideoEncoder struct {
	config         EncoderConfig
	workers        []*EncoderWorker
	frameQueue     chan *Frame
	segmentQueue   chan *Segment
	outputQueue    chan *EncodedSegment
	progressTracker *ProgressTracker
	stats          *EncodingStats
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup
	mu             sync.RWMutex
}

// EncoderConfig holds video encoding configuration
type EncoderConfig struct {
	InputFile        string
	OutputFile       string
	NumWorkers       int
	SegmentDuration  time.Duration
	OutputFormat     VideoFormat
	Quality          QualityLevel
	Resolution       Resolution
	Bitrate          int64
	FrameRate        float64
	AudioCodec       AudioCodec
	VideoCodec       VideoCodec
	MaxMemoryUsage   int64
	ChunkSize        int
	EnableGPU        bool
	PresetName       string
	ThreadsPerWorker int
}

// VideoFormat represents video output format
type VideoFormat int

const (
	FormatMP4 VideoFormat = iota
	FormatAVI
	FormatMOV
	FormatWEBM
	FormatMKV
)

// QualityLevel represents encoding quality
type QualityLevel int

const (
	QualityLow QualityLevel = iota
	QualityMedium
	QualityHigh
	QualityUltra
	QualityLossless
)

// Resolution represents video resolution
type Resolution struct {
	Width  int
	Height int
}

// AudioCodec represents audio encoding codec
type AudioCodec int

const (
	AudioAAC AudioCodec = iota
	AudioMP3
	AudioOGG
	AudioFLAC
	AudioOpus
)

// VideoCodec represents video encoding codec
type VideoCodec int

const (
	VideoH264 VideoCodec = iota
	VideoH265
	VideoVP8
	VideoVP9
	VideoAV1
)

// Frame represents a video frame
type Frame struct {
	Index       int64
	Timestamp   time.Duration
	Data        []byte
	Width       int
	Height      int
	Format      PixelFormat
	Size        int64
	IsKeyFrame  bool
	QualityHint QualityLevel
}

// PixelFormat represents pixel format
type PixelFormat int

const (
	FormatRGB24 PixelFormat = iota
	FormatYUV420P
	FormatYUV422P
	FormatYUV444P
	FormatRGBA
)

// Segment represents a video segment for encoding
type Segment struct {
	ID            int
	StartFrame    int64
	EndFrame      int64
	Frames        []*Frame
	Duration      time.Duration
	ExpectedSize  int64
	Priority      int
	RetryCount    int
	WorkerID      int
}

// EncodedSegment represents an encoded video segment
type EncodedSegment struct {
	ID           int
	Data         []byte
	Size         int64
	Duration     time.Duration
	Bitrate      int64
	Quality      float64
	WorkerID     int
	ProcessTime  time.Duration
	FrameCount   int64
	IsComplete   bool
	Metadata     SegmentMetadata
}

// SegmentMetadata contains segment encoding metadata
type SegmentMetadata struct {
	StartTimestamp time.Duration
	EndTimestamp   time.Duration
	KeyFrames      []int64
	AverageBitrate int64
	PeakBitrate    int64
	CompressionRatio float64
	EncodingMode   string
}

// EncoderWorker represents a video encoding worker
type EncoderWorker struct {
	ID              int
	encoder         *VideoEncoder
	isActive        bool
	currentSegment  *Segment
	processedFrames int64
	processedBytes  int64
	processingTime  time.Duration
	errors          []error
	capabilities    WorkerCapabilities
	mu              sync.RWMutex
}

// WorkerCapabilities represents worker capabilities
type WorkerCapabilities struct {
	SupportsGPU     bool
	MaxResolution   Resolution
	SupportedCodecs []VideoCodec
	ThreadCount     int
	MemoryLimit     int64
}

// ProgressTracker tracks encoding progress
type ProgressTracker struct {
	totalFrames       int64
	processedFrames   int64
	totalSegments     int64
	processedSegments int64
	totalSize         int64
	processedSize     int64
	startTime         time.Time
	estimatedEndTime  time.Time
	currentPhase      ProcessingPhase
	mu                sync.RWMutex
}

// ProcessingPhase represents current processing phase
type ProcessingPhase int

const (
	PhaseAnalyzing ProcessingPhase = iota
	PhaseSegmenting
	PhaseEncoding
	PhaseMuxing
	PhaseComplete
	PhaseError
)

// EncodingStats contains encoding statistics
type EncodingStats struct {
	TotalFrames      int64
	ProcessedFrames  int64
	TotalDuration    time.Duration
	ProcessedTime    time.Duration
	AverageFPS       float64
	PeakFPS          float64
	InputSize        int64
	OutputSize       int64
	CompressionRatio float64
	AverageBitrate   int64
	PeakBitrate      int64
	QualityScore     float64
	ErrorCount       int64
	WorkerStats      map[int]*WorkerStats
	StartTime        time.Time
	EndTime          time.Time
}

// WorkerStats contains per-worker statistics
type WorkerStats struct {
	WorkerID        int
	FramesProcessed int64
	BytesProcessed  int64
	ProcessingTime  time.Duration
	AverageFPS      float64
	ErrorCount      int64
	Efficiency      float64
}

// EncodingJob represents a complete encoding job
type EncodingJob struct {
	ID          string
	Config      EncoderConfig
	Status      JobStatus
	Progress    *ProgressTracker
	Stats       *EncodingStats
	Result      *EncodingResult
	CreatedAt   time.Time
	StartedAt   time.Time
	CompletedAt time.Time
	Error       error
}

// JobStatus represents job status
type JobStatus int

const (
	StatusPending JobStatus = iota
	StatusRunning
	StatusCompleted
	StatusFailed
	StatusCancelled
)

// EncodingResult contains final encoding results
type EncodingResult struct {
	OutputFile       string
	OutputSize       int64
	Duration         time.Duration
	ProcessingTime   time.Duration
	CompressionRatio float64
	QualityMetrics   QualityMetrics
	Bitrate          int64
	FrameRate        float64
	Resolution       Resolution
	Metadata         map[string]interface{}
}

// QualityMetrics contains quality assessment metrics
type QualityMetrics struct {
	PSNR     float64 // Peak Signal-to-Noise Ratio
	SSIM     float64 // Structural Similarity Index
	VMAF     float64 // Video Multimethod Assessment Fusion
	Bitrate  int64
	Filesize int64
}

// NewVideoEncoder creates a new parallel video encoder
func NewVideoEncoder(config EncoderConfig) *VideoEncoder {
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	if config.SegmentDuration <= 0 {
		config.SegmentDuration = 10 * time.Second
	}
	if config.ChunkSize <= 0 {
		config.ChunkSize = 1024 * 1024 // 1MB
	}
	if config.ThreadsPerWorker <= 0 {
		config.ThreadsPerWorker = 2
	}
	if config.MaxMemoryUsage <= 0 {
		config.MaxMemoryUsage = 1024 * 1024 * 1024 // 1GB
	}

	ctx, cancel := context.WithCancel(context.Background())

	encoder := &VideoEncoder{
		config:          config,
		frameQueue:      make(chan *Frame, config.NumWorkers*2),
		segmentQueue:    make(chan *Segment, config.NumWorkers),
		outputQueue:     make(chan *EncodedSegment, config.NumWorkers),
		progressTracker: &ProgressTracker{startTime: time.Now()},
		stats:           &EncodingStats{WorkerStats: make(map[int]*WorkerStats), StartTime: time.Now()},
		ctx:             ctx,
		cancel:          cancel,
	}

	// Initialize workers
	encoder.initializeWorkers()

	return encoder
}

func (ve *VideoEncoder) initializeWorkers() {
	ve.workers = make([]*EncoderWorker, ve.config.NumWorkers)
	for i := 0; i < ve.config.NumWorkers; i++ {
		ve.workers[i] = &EncoderWorker{
			ID:      i,
			encoder: ve,
			capabilities: WorkerCapabilities{
				SupportsGPU:     ve.config.EnableGPU,
				MaxResolution:   Resolution{Width: 4096, Height: 2160},
				SupportedCodecs: []VideoCodec{VideoH264, VideoH265, VideoVP9},
				ThreadCount:     ve.config.ThreadsPerWorker,
				MemoryLimit:     ve.config.MaxMemoryUsage / int64(ve.config.NumWorkers),
			},
		}
		ve.stats.WorkerStats[i] = &WorkerStats{WorkerID: i}
	}
}

// EncodeVideo encodes a video file using parallel processing
func (ve *VideoEncoder) EncodeVideo() (*EncodingResult, error) {
	ve.mu.Lock()
	defer ve.mu.Unlock()

	// Phase 1: Analyze input video
	ve.progressTracker.currentPhase = PhaseAnalyzing
	videoInfo, err := ve.analyzeInput()
	if err != nil {
		return nil, fmt.Errorf("input analysis failed: %w", err)
	}

	// Phase 2: Create segments
	ve.progressTracker.currentPhase = PhaseSegmenting
	segments, err := ve.createSegments(videoInfo)
	if err != nil {
		return nil, fmt.Errorf("segmentation failed: %w", err)
	}

	// Phase 3: Start workers
	ve.startWorkers()

	// Phase 4: Process segments
	ve.progressTracker.currentPhase = PhaseEncoding
	encodedSegments, err := ve.processSegments(segments)
	if err != nil {
		ve.stopWorkers()
		return nil, fmt.Errorf("encoding failed: %w", err)
	}

	// Phase 5: Mux segments into final output
	ve.progressTracker.currentPhase = PhaseMuxing
	result, err := ve.muxSegments(encodedSegments, videoInfo)
	if err != nil {
		ve.stopWorkers()
		return nil, fmt.Errorf("muxing failed: %w", err)
	}

	ve.stopWorkers()
	ve.progressTracker.currentPhase = PhaseComplete
	ve.stats.EndTime = time.Now()

	return result, nil
}

// VideoInfo contains information about input video
type VideoInfo struct {
	Duration     time.Duration
	FrameRate    float64
	TotalFrames  int64
	Resolution   Resolution
	Bitrate      int64
	AudioTracks  int
	VideoTracks  int
	Format       string
	Size         int64
	HasAudio     bool
	HasVideo     bool
}

func (ve *VideoEncoder) analyzeInput() (*VideoInfo, error) {
	file, err := os.Open(ve.config.InputFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	stat, err := file.Stat()
	if err != nil {
		return nil, err
	}

	// Simulate video analysis (in real implementation, would use FFmpeg/similar)
	info := &VideoInfo{
		Duration:     120 * time.Second, // Mock 2-minute video
		FrameRate:    30.0,
		TotalFrames:  3600, // 120 seconds * 30 fps
		Resolution:   ve.config.Resolution,
		Bitrate:      2000000, // 2 Mbps
		AudioTracks:  1,
		VideoTracks:  1,
		Format:       "mp4",
		Size:         stat.Size(),
		HasAudio:     true,
		HasVideo:     true,
	}

	ve.progressTracker.totalFrames = info.TotalFrames
	ve.stats.TotalFrames = info.TotalFrames
	ve.stats.TotalDuration = info.Duration
	ve.stats.InputSize = info.Size

	return info, nil
}

func (ve *VideoEncoder) createSegments(info *VideoInfo) ([]*Segment, error) {
	segmentFrames := int64(float64(ve.config.SegmentDuration.Seconds()) * info.FrameRate)
	numSegments := int((info.TotalFrames + segmentFrames - 1) / segmentFrames)

	segments := make([]*Segment, numSegments)
	for i := 0; i < numSegments; i++ {
		startFrame := int64(i) * segmentFrames
		endFrame := startFrame + segmentFrames
		if endFrame > info.TotalFrames {
			endFrame = info.TotalFrames
		}

		segments[i] = &Segment{
			ID:           i,
			StartFrame:   startFrame,
			EndFrame:     endFrame,
			Duration:     time.Duration(float64(endFrame-startFrame)/info.FrameRate) * time.Second,
			ExpectedSize: (info.Size / info.TotalFrames) * (endFrame - startFrame),
			Priority:     calculatePriority(i, numSegments),
		}
	}

	ve.progressTracker.totalSegments = int64(numSegments)
	sort.Slice(segments, func(i, j int) bool {
		return segments[i].Priority > segments[j].Priority
	})

	return segments, nil
}

func calculatePriority(segmentIndex, totalSegments int) int {
	// Higher priority for beginning and end segments
	if segmentIndex < totalSegments/10 || segmentIndex > totalSegments*9/10 {
		return 3
	}
	if segmentIndex < totalSegments/4 || segmentIndex > totalSegments*3/4 {
		return 2
	}
	return 1
}

func (ve *VideoEncoder) startWorkers() {
	for _, worker := range ve.workers {
		ve.wg.Add(1)
		go ve.workerLoop(worker)
	}
}

func (ve *VideoEncoder) stopWorkers() {
	ve.cancel()
	close(ve.segmentQueue)
	ve.wg.Wait()
}

func (ve *VideoEncoder) workerLoop(worker *EncoderWorker) {
	defer ve.wg.Done()

	for {
		select {
		case segment, ok := <-ve.segmentQueue:
			if !ok {
				return
			}
			ve.processSegmentByWorker(worker, segment)

		case <-ve.ctx.Done():
			return
		}
	}
}

func (ve *VideoEncoder) processSegmentByWorker(worker *EncoderWorker, segment *Segment) {
	worker.mu.Lock()
	worker.isActive = true
	worker.currentSegment = segment
	worker.mu.Unlock()

	startTime := time.Now()
	
	// Simulate frame extraction and encoding
	encodedSegment, err := ve.encodeSegment(worker, segment)
	if err != nil {
		worker.mu.Lock()
		worker.errors = append(worker.errors, err)
		worker.mu.Unlock()
		
		// Retry logic
		if segment.RetryCount < 3 {
			segment.RetryCount++
			segment.WorkerID = -1 // Allow different worker
			ve.segmentQueue <- segment
			return
		}
		
		atomic.AddInt64(&ve.stats.ErrorCount, 1)
		return
	}

	processingTime := time.Since(startTime)
	encodedSegment.ProcessTime = processingTime
	encodedSegment.WorkerID = worker.ID

	// Update worker stats
	worker.mu.Lock()
	worker.processedFrames += encodedSegment.FrameCount
	worker.processedBytes += encodedSegment.Size
	worker.processingTime += processingTime
	worker.isActive = false
	worker.currentSegment = nil
	worker.mu.Unlock()

	// Update global stats
	workerStats := ve.stats.WorkerStats[worker.ID]
	workerStats.FramesProcessed += encodedSegment.FrameCount
	workerStats.BytesProcessed += encodedSegment.Size
	workerStats.ProcessingTime += processingTime

	if processingTime > 0 {
		fps := float64(encodedSegment.FrameCount) / processingTime.Seconds()
		workerStats.AverageFPS = (workerStats.AverageFPS + fps) / 2
	}

	ve.outputQueue <- encodedSegment

	// Update progress
	atomic.AddInt64(&ve.progressTracker.processedSegments, 1)
	atomic.AddInt64(&ve.progressTracker.processedFrames, encodedSegment.FrameCount)
	atomic.AddInt64(&ve.progressTracker.processedSize, encodedSegment.Size)
}

func (ve *VideoEncoder) encodeSegment(worker *EncoderWorker, segment *Segment) (*EncodedSegment, error) {
	// Simulate encoding process
	frameCount := segment.EndFrame - segment.StartFrame
	
	// Mock encoding parameters based on quality and codec
	var bitrate int64
	var compressionRatio float64
	
	switch ve.config.Quality {
	case QualityLow:
		bitrate = 500000
		compressionRatio = 0.1
	case QualityMedium:
		bitrate = 1000000
		compressionRatio = 0.2
	case QualityHigh:
		bitrate = 2000000
		compressionRatio = 0.3
	case QualityUltra:
		bitrate = 4000000
		compressionRatio = 0.4
	case QualityLossless:
		bitrate = 10000000
		compressionRatio = 0.8
	}

	// Simulate processing time based on complexity
	processingDuration := time.Duration(frameCount) * time.Millisecond * 10
	if ve.config.EnableGPU {
		processingDuration /= 4 // GPU acceleration
	}
	
	// Simulate actual processing
	time.Sleep(processingDuration)

	encodedSize := int64(float64(segment.ExpectedSize) * compressionRatio)
	quality := calculateQualityScore(ve.config.Quality, bitrate)

	return &EncodedSegment{
		ID:          segment.ID,
		Data:        make([]byte, encodedSize), // Mock encoded data
		Size:        encodedSize,
		Duration:    segment.Duration,
		Bitrate:     bitrate,
		Quality:     quality,
		FrameCount:  frameCount,
		IsComplete:  true,
		Metadata: SegmentMetadata{
			StartTimestamp:   time.Duration(segment.StartFrame) * time.Second / 30,
			EndTimestamp:     time.Duration(segment.EndFrame) * time.Second / 30,
			KeyFrames:        []int64{segment.StartFrame},
			AverageBitrate:   bitrate,
			PeakBitrate:      bitrate * 2,
			CompressionRatio: compressionRatio,
			EncodingMode:     getEncodingMode(ve.config.VideoCodec),
		},
	}, nil
}

func calculateQualityScore(quality QualityLevel, bitrate int64) float64 {
	baseScore := 0.5
	switch quality {
	case QualityLow:
		baseScore = 0.5
	case QualityMedium:
		baseScore = 0.65
	case QualityHigh:
		baseScore = 0.8
	case QualityUltra:
		baseScore = 0.9
	case QualityLossless:
		baseScore = 0.95
	}
	
	// Adjust based on bitrate (small bonus to avoid capping)
	bitrateBonus := math.Min(float64(bitrate)/20000000.0, 0.05)
	return math.Min(baseScore+bitrateBonus, 1.0)
}

func getEncodingMode(codec VideoCodec) string {
	switch codec {
	case VideoH264:
		return "H.264/AVC"
	case VideoH265:
		return "H.265/HEVC"
	case VideoVP8:
		return "VP8"
	case VideoVP9:
		return "VP9"
	case VideoAV1:
		return "AV1"
	default:
		return "Unknown"
	}
}

func (ve *VideoEncoder) processSegments(segments []*Segment) ([]*EncodedSegment, error) {
	// Send segments to workers
	go func() {
		for _, segment := range segments {
			select {
			case ve.segmentQueue <- segment:
			case <-ve.ctx.Done():
				return
			}
		}
	}()

	// Collect encoded segments
	var encodedSegments []*EncodedSegment
	expectedSegments := len(segments)

	for len(encodedSegments) < expectedSegments {
		select {
		case encodedSegment := <-ve.outputQueue:
			encodedSegments = append(encodedSegments, encodedSegment)

		case <-ve.ctx.Done():
			return nil, ve.ctx.Err()

		case <-time.After(30 * time.Second):
			return nil, fmt.Errorf("encoding timeout")
		}
	}

	// Sort segments by ID to maintain order
	sort.Slice(encodedSegments, func(i, j int) bool {
		return encodedSegments[i].ID < encodedSegments[j].ID
	})

	return encodedSegments, nil
}

func (ve *VideoEncoder) muxSegments(segments []*EncodedSegment, info *VideoInfo) (*EncodingResult, error) {
	outputFile, err := os.Create(ve.config.OutputFile)
	if err != nil {
		return nil, err
	}
	defer outputFile.Close()

	writer := bufio.NewWriter(outputFile)
	defer writer.Flush()

	var totalSize int64
	var totalBitrate int64
	var avgQuality float64

	// Write mock video header
	header := fmt.Sprintf("# Encoded Video File\n# Format: %s\n# Resolution: %dx%d\n# Frame Rate: %.2f\n\n",
		getFormatString(ve.config.OutputFormat),
		ve.config.Resolution.Width,
		ve.config.Resolution.Height,
		ve.config.FrameRate)
	writer.WriteString(header)

	// Mux segments in order
	for i, segment := range segments {
		// Write segment data (in real implementation, would properly mux video/audio)
		writer.WriteString(fmt.Sprintf("# Segment %d (%.2fs - %.2fs)\n",
			segment.ID,
			segment.Metadata.StartTimestamp.Seconds(),
			segment.Metadata.EndTimestamp.Seconds()))
		
		// Simulate writing encoded data
		writer.Write(segment.Data[:min(len(segment.Data), 1024)]) // Write sample of data
		writer.WriteString(fmt.Sprintf("\n# End Segment %d\n\n", segment.ID))

		totalSize += segment.Size
		totalBitrate += segment.Bitrate
		avgQuality += segment.Quality

		// Update progress
		progress := float64(i+1) / float64(len(segments))
		ve.updateProgress(progress)
	}

	avgQuality /= float64(len(segments))
	avgBitrate := totalBitrate / int64(len(segments))
	compressionRatio := float64(totalSize) / float64(info.Size)

	processingTime := time.Since(ve.stats.StartTime)

	result := &EncodingResult{
		OutputFile:       ve.config.OutputFile,
		OutputSize:       totalSize,
		Duration:         info.Duration,
		ProcessingTime:   processingTime,
		CompressionRatio: compressionRatio,
		QualityMetrics: QualityMetrics{
			PSNR:     30.0 + (avgQuality * 20), // Mock PSNR
			SSIM:     0.8 + (avgQuality * 0.2), // Mock SSIM
			VMAF:     60.0 + (avgQuality * 40), // Mock VMAF
			Bitrate:  avgBitrate,
			Filesize: totalSize,
		},
		Bitrate:    avgBitrate,
		FrameRate:  ve.config.FrameRate,
		Resolution: ve.config.Resolution,
		Metadata: map[string]interface{}{
			"encoder":         "Parallel Video Encoder",
			"workers":         ve.config.NumWorkers,
			"segments":        len(segments),
			"processing_time": processingTime.String(),
			"gpu_acceleration": ve.config.EnableGPU,
		},
	}

	ve.stats.OutputSize = totalSize
	ve.stats.CompressionRatio = compressionRatio
	ve.stats.AverageBitrate = avgBitrate
	ve.stats.QualityScore = avgQuality

	return result, nil
}

func getFormatString(format VideoFormat) string {
	switch format {
	case FormatMP4:
		return "MP4"
	case FormatAVI:
		return "AVI"
	case FormatMOV:
		return "MOV"
	case FormatWEBM:
		return "WebM"
	case FormatMKV:
		return "MKV"
	default:
		return "Unknown"
	}
}

func (ve *VideoEncoder) updateProgress(progress float64) {
	ve.progressTracker.mu.Lock()
	defer ve.progressTracker.mu.Unlock()

	if progress > 0 {
		elapsed := time.Since(ve.progressTracker.startTime)
		estimated := time.Duration(float64(elapsed) / progress)
		ve.progressTracker.estimatedEndTime = ve.progressTracker.startTime.Add(estimated)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// GetProgress returns current encoding progress
func (ve *VideoEncoder) GetProgress() (float64, ProcessingPhase, time.Duration) {
	ve.progressTracker.mu.RLock()
	defer ve.progressTracker.mu.RUnlock()

	var progress float64
	if ve.progressTracker.totalFrames > 0 {
		progress = float64(ve.progressTracker.processedFrames) / float64(ve.progressTracker.totalFrames)
	}

	elapsed := time.Since(ve.progressTracker.startTime)
	return progress, ve.progressTracker.currentPhase, elapsed
}

// GetStats returns current encoding statistics
func (ve *VideoEncoder) GetStats() *EncodingStats {
	ve.mu.RLock()
	defer ve.mu.RUnlock()

	// Create a deep copy
	statsCopy := EncodingStats{
		TotalFrames:      ve.stats.TotalFrames,
		ProcessedFrames:  ve.stats.ProcessedFrames,
		TotalDuration:    ve.stats.TotalDuration,
		ProcessedTime:    ve.stats.ProcessedTime,
		AverageFPS:       ve.stats.AverageFPS,
		PeakFPS:          ve.stats.PeakFPS,
		InputSize:        ve.stats.InputSize,
		OutputSize:       ve.stats.OutputSize,
		CompressionRatio: ve.stats.CompressionRatio,
		AverageBitrate:   ve.stats.AverageBitrate,
		PeakBitrate:      ve.stats.PeakBitrate,
		QualityScore:     ve.stats.QualityScore,
		ErrorCount:       ve.stats.ErrorCount,
		StartTime:        ve.stats.StartTime,
		EndTime:          ve.stats.EndTime,
		WorkerStats:      make(map[int]*WorkerStats),
	}
	
	for k, v := range ve.stats.WorkerStats {
		workerStatsCopy := &WorkerStats{
			WorkerID:        v.WorkerID,
			FramesProcessed: v.FramesProcessed,
			BytesProcessed:  v.BytesProcessed,
			ProcessingTime:  v.ProcessingTime,
			AverageFPS:      v.AverageFPS,
			ErrorCount:      v.ErrorCount,
			Efficiency:      v.Efficiency,
		}
		statsCopy.WorkerStats[k] = workerStatsCopy
	}

	return &statsCopy
}

// Cancel cancels the encoding process
func (ve *VideoEncoder) Cancel() {
	ve.cancel()
}

// GetWorkerStatus returns status of all workers
func (ve *VideoEncoder) GetWorkerStatus() []WorkerStatus {
	var status []WorkerStatus
	for _, worker := range ve.workers {
		worker.mu.RLock()
		ws := WorkerStatus{
			ID:              worker.ID,
			IsActive:        worker.isActive,
			ProcessedFrames: worker.processedFrames,
			ProcessedBytes:  worker.processedBytes,
			ProcessingTime:  worker.processingTime,
			ErrorCount:      int64(len(worker.errors)),
		}
		if worker.currentSegment != nil {
			ws.CurrentSegment = &worker.currentSegment.ID
		}
		worker.mu.RUnlock()
		status = append(status, ws)
	}
	return status
}

// WorkerStatus represents worker status information
type WorkerStatus struct {
	ID              int
	IsActive        bool
	CurrentSegment  *int
	ProcessedFrames int64
	ProcessedBytes  int64
	ProcessingTime  time.Duration
	ErrorCount      int64
}

// CreateEncodingPreset creates a predefined encoding preset
func CreateEncodingPreset(preset string) EncoderConfig {
	base := EncoderConfig{
		NumWorkers:       runtime.NumCPU(),
		SegmentDuration:  10 * time.Second,
		FrameRate:        30.0,
		AudioCodec:       AudioAAC,
		VideoCodec:       VideoH264,
		ThreadsPerWorker: 2,
		MaxMemoryUsage:   1024 * 1024 * 1024,
	}

	switch strings.ToLower(preset) {
	case "fast":
		base.Quality = QualityMedium
		base.OutputFormat = FormatMP4
		base.Resolution = Resolution{Width: 1280, Height: 720}
		base.Bitrate = 1000000
		base.PresetName = "Fast"

	case "balanced":
		base.Quality = QualityHigh
		base.OutputFormat = FormatMP4
		base.Resolution = Resolution{Width: 1920, Height: 1080}
		base.Bitrate = 2000000
		base.PresetName = "Balanced"

	case "quality":
		base.Quality = QualityUltra
		base.OutputFormat = FormatMP4
		base.Resolution = Resolution{Width: 1920, Height: 1080}
		base.Bitrate = 4000000
		base.VideoCodec = VideoH265
		base.PresetName = "Quality"

	case "4k":
		base.Quality = QualityUltra
		base.OutputFormat = FormatMP4
		base.Resolution = Resolution{Width: 3840, Height: 2160}
		base.Bitrate = 8000000
		base.VideoCodec = VideoH265
		base.PresetName = "4K"

	default:
		base.Quality = QualityMedium
		base.OutputFormat = FormatMP4
		base.Resolution = Resolution{Width: 1280, Height: 720}
		base.Bitrate = 1000000
		base.PresetName = "Default"
	}

	return base
}

// Example demonstrates parallel video encoding
func Example() {
	fmt.Println("=== Parallel Video Encoder Example ===")

	// Create sample input file (mock)
	inputFile := "sample_video.mp4"
	if err := createSampleVideoFile(inputFile); err != nil {
		fmt.Printf("Failed to create sample video: %v\n", err)
		return
	}
	defer os.Remove(inputFile)

	// Create encoding configuration
	config := CreateEncodingPreset("balanced")
	config.InputFile = inputFile
	config.OutputFile = "encoded_output.mp4"
	config.NumWorkers = 4
	config.EnableGPU = false // Disable for demo

	fmt.Printf("Configuration:\n")
	fmt.Printf("  Workers: %d\n", config.NumWorkers)
	fmt.Printf("  Quality: %v\n", config.Quality)
	fmt.Printf("  Resolution: %dx%d\n", config.Resolution.Width, config.Resolution.Height)
	fmt.Printf("  Codec: %v\n", config.VideoCodec)
	fmt.Printf("  Target Bitrate: %d bps\n", config.Bitrate)

	// Create encoder and start encoding
	encoder := NewVideoEncoder(config)

	fmt.Println("\nStarting parallel video encoding...")

	// Monitor progress in separate goroutine
	done := make(chan bool)
	go func() {
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				progress, phase, elapsed := encoder.GetProgress()
				fmt.Printf("\rProgress: %.1f%% | Phase: %v | Time: %v",
					progress*100, getPhaseString(phase), elapsed.Truncate(time.Second))

			case <-done:
				return
			}
		}
	}()

	// Perform encoding
	result, err := encoder.EncodeVideo()
	close(done)

	if err != nil {
		fmt.Printf("\nEncoding failed: %v\n", err)
		return
	}

	// Display results
	fmt.Printf("\n\nEncoding completed successfully!\n")
	fmt.Printf("Results:\n")
	fmt.Printf("  Output File: %s\n", result.OutputFile)
	fmt.Printf("  Output Size: %s\n", formatBytes(result.OutputSize))
	fmt.Printf("  Compression Ratio: %.2f%%\n", result.CompressionRatio*100)
	fmt.Printf("  Processing Time: %v\n", result.ProcessingTime.Truncate(time.Millisecond))
	fmt.Printf("  Average Bitrate: %s\n", formatBitrate(result.Bitrate))
	fmt.Printf("  Frame Rate: %.2f fps\n", result.FrameRate)

	fmt.Printf("\nQuality Metrics:\n")
	fmt.Printf("  PSNR: %.2f dB\n", result.QualityMetrics.PSNR)
	fmt.Printf("  SSIM: %.3f\n", result.QualityMetrics.SSIM)
	fmt.Printf("  VMAF: %.1f\n", result.QualityMetrics.VMAF)

	// Display worker statistics
	stats := encoder.GetStats()
	fmt.Printf("\nWorker Statistics:\n")
	for id, workerStats := range stats.WorkerStats {
		fmt.Printf("  Worker %d: %d frames, %.1f fps avg, %v time\n",
			id, workerStats.FramesProcessed, workerStats.AverageFPS, 
			workerStats.ProcessingTime.Truncate(time.Millisecond))
	}

	// Cleanup
	os.Remove(result.OutputFile)
	fmt.Println("\nDemo completed!")
}

func createSampleVideoFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Create a mock video file with metadata
	content := `# Mock Video File
# This represents a sample video file for encoding demonstration
# Duration: 120 seconds
# Resolution: 1920x1080
# Frame Rate: 30 fps
# Original Bitrate: 2 Mbps

[VIDEO_DATA]
Mock video data would be here in a real video file.
This file simulates a 2-minute video at 1080p resolution.
` + strings.Repeat("Frame data simulation...\n", 1000)

	_, err = file.WriteString(content)
	return err
}

func getPhaseString(phase ProcessingPhase) string {
	switch phase {
	case PhaseAnalyzing:
		return "Analyzing"
	case PhaseSegmenting:
		return "Segmenting"
	case PhaseEncoding:
		return "Encoding"
	case PhaseMuxing:
		return "Muxing"
	case PhaseComplete:
		return "Complete"
	case PhaseError:
		return "Error"
	default:
		return "Unknown"
	}
}

func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

func formatBitrate(bitrate int64) string {
	if bitrate < 1000 {
		return fmt.Sprintf("%d bps", bitrate)
	} else if bitrate < 1000000 {
		return fmt.Sprintf("%.1f Kbps", float64(bitrate)/1000)
	} else {
		return fmt.Sprintf("%.1f Mbps", float64(bitrate)/1000000)
	}
}