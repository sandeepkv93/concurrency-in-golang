package parallelvideoencoder

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

func createTestVideoFile(t *testing.T, filename string, size int) {
	content := `# Test Video File
# Duration: 60 seconds
# Resolution: 1280x720
# Frame Rate: 30 fps

[VIDEO_HEADER]
Mock video metadata here
`
	// Add mock video data to reach desired size
	mockData := strings.Repeat("Frame data chunk...\n", size/20)
	content += mockData

	if err := os.WriteFile(filename, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
}

func TestVideoEncoderCreation(t *testing.T) {
	config := EncoderConfig{
		InputFile:    "test_input.mp4",
		OutputFile:   "test_output.mp4",
		NumWorkers:   4,
		Quality:      QualityMedium,
		Resolution:   Resolution{Width: 1280, Height: 720},
		VideoCodec:   VideoH264,
		AudioCodec:   AudioAAC,
		OutputFormat: FormatMP4,
	}

	encoder := NewVideoEncoder(config)

	if encoder == nil {
		t.Fatal("Failed to create video encoder")
	}

	if len(encoder.workers) != config.NumWorkers {
		t.Errorf("Expected %d workers, got %d", config.NumWorkers, len(encoder.workers))
	}

	if encoder.config.NumWorkers != config.NumWorkers {
		t.Errorf("Expected %d workers in config, got %d", config.NumWorkers, encoder.config.NumWorkers)
	}

	// Check default values
	if encoder.config.SegmentDuration <= 0 {
		t.Error("Segment duration should be set to default")
	}

	if encoder.config.ChunkSize <= 0 {
		t.Error("Chunk size should be set to default")
	}
}

func TestVideoEncoderDefaults(t *testing.T) {
	config := EncoderConfig{
		InputFile:  "test.mp4",
		OutputFile: "output.mp4",
	}

	encoder := NewVideoEncoder(config)

	if encoder.config.NumWorkers <= 0 {
		t.Error("NumWorkers should default to positive value")
	}

	if encoder.config.SegmentDuration <= 0 {
		t.Error("SegmentDuration should have default value")
	}

	if encoder.config.ChunkSize <= 0 {
		t.Error("ChunkSize should have default value")
	}

	if encoder.config.ThreadsPerWorker <= 0 {
		t.Error("ThreadsPerWorker should have default value")
	}

	if encoder.config.MaxMemoryUsage <= 0 {
		t.Error("MaxMemoryUsage should have default value")
	}
}

func TestVideoAnalysis(t *testing.T) {
	tempDir := t.TempDir()
	testFile := filepath.Join(tempDir, "test_video.mp4")
	createTestVideoFile(t, testFile, 1024*100) // 100KB test file

	config := EncoderConfig{
		InputFile:  testFile,
		OutputFile: filepath.Join(tempDir, "output.mp4"),
		Resolution: Resolution{Width: 1920, Height: 1080},
	}

	encoder := NewVideoEncoder(config)
	info, err := encoder.analyzeInput()

	if err != nil {
		t.Fatalf("Failed to analyze input: %v", err)
	}

	if info.Duration <= 0 {
		t.Error("Duration should be positive")
	}

	if info.TotalFrames <= 0 {
		t.Error("Total frames should be positive")
	}

	if info.Size <= 0 {
		t.Error("File size should be positive")
	}

	if info.Resolution.Width != config.Resolution.Width {
		t.Errorf("Expected width %d, got %d", config.Resolution.Width, info.Resolution.Width)
	}

	if info.Resolution.Height != config.Resolution.Height {
		t.Errorf("Expected height %d, got %d", config.Resolution.Height, info.Resolution.Height)
	}
}

func TestSegmentCreation(t *testing.T) {
	config := EncoderConfig{
		SegmentDuration: 5 * time.Second,
	}

	encoder := NewVideoEncoder(config)
	
	info := &VideoInfo{
		Duration:    60 * time.Second,
		FrameRate:   30.0,
		TotalFrames: 1800, // 60 seconds * 30 fps
		Size:        1024 * 1024, // 1MB
	}

	segments, err := encoder.createSegments(info)
	if err != nil {
		t.Fatalf("Failed to create segments: %v", err)
	}

	expectedSegments := 12 // 60 seconds / 5 seconds per segment
	if len(segments) != expectedSegments {
		t.Errorf("Expected %d segments, got %d", expectedSegments, len(segments))
	}

	// Sort segments by ID for checking continuity
	sort.Slice(segments, func(i, j int) bool {
		return segments[i].ID < segments[j].ID
	})

	// Check segment continuity
	for i, segment := range segments {
		if i > 0 {
			prevSegment := segments[i-1]
			if segment.StartFrame != prevSegment.EndFrame {
				t.Errorf("Segment %d start frame (%d) doesn't match previous end frame (%d)",
					i, segment.StartFrame, prevSegment.EndFrame)
			}
		}

		if segment.StartFrame >= segment.EndFrame {
			t.Errorf("Segment %d has invalid frame range: %d to %d",
				i, segment.StartFrame, segment.EndFrame)
		}

		if segment.Duration <= 0 {
			t.Errorf("Segment %d has invalid duration: %v", i, segment.Duration)
		}
	}

	// Check total frames coverage
	totalFramesCovered := segments[len(segments)-1].EndFrame
	if totalFramesCovered != info.TotalFrames {
		t.Errorf("Segments don't cover all frames: %d vs %d", totalFramesCovered, info.TotalFrames)
	}
}

func TestPriorityCalculation(t *testing.T) {
	tests := []struct {
		segmentIndex  int
		totalSegments int
		expectedMin   int
		expectedMax   int
	}{
		{0, 100, 2, 3},    // Beginning segment
		{5, 100, 2, 3},    // Early segment
		{50, 100, 1, 2},   // Middle segment
		{95, 100, 2, 3},   // End segment
		{99, 100, 2, 3},   // Last segment
	}

	for _, test := range tests {
		priority := calculatePriority(test.segmentIndex, test.totalSegments)
		if priority < test.expectedMin || priority > test.expectedMax {
			t.Errorf("Priority for segment %d/%d: expected %d-%d, got %d",
				test.segmentIndex, test.totalSegments, test.expectedMin, test.expectedMax, priority)
		}
	}
}

func TestWorkerInitialization(t *testing.T) {
	config := EncoderConfig{
		NumWorkers:      4,
		EnableGPU:       true,
		MaxMemoryUsage:  1024 * 1024 * 1024, // 1GB
		ThreadsPerWorker: 3,
	}

	encoder := NewVideoEncoder(config)

	if len(encoder.workers) != config.NumWorkers {
		t.Errorf("Expected %d workers, got %d", config.NumWorkers, len(encoder.workers))
	}

	for i, worker := range encoder.workers {
		if worker.ID != i {
			t.Errorf("Worker %d has incorrect ID: %d", i, worker.ID)
		}

		if worker.encoder != encoder {
			t.Errorf("Worker %d has incorrect encoder reference", i)
		}

		if !worker.capabilities.SupportsGPU {
			t.Errorf("Worker %d should support GPU", i)
		}

		if worker.capabilities.ThreadCount != config.ThreadsPerWorker {
			t.Errorf("Worker %d has incorrect thread count: %d vs %d",
				i, worker.capabilities.ThreadCount, config.ThreadsPerWorker)
		}

		expectedMemory := config.MaxMemoryUsage / int64(config.NumWorkers)
		if worker.capabilities.MemoryLimit != expectedMemory {
			t.Errorf("Worker %d has incorrect memory limit: %d vs %d",
				i, worker.capabilities.MemoryLimit, expectedMemory)
		}
	}
}

func TestSegmentEncoding(t *testing.T) {
	config := EncoderConfig{
		Quality:     QualityMedium,
		VideoCodec:  VideoH264,
		EnableGPU:   false,
	}

	encoder := NewVideoEncoder(config)
	worker := encoder.workers[0]

	segment := &Segment{
		ID:           1,
		StartFrame:   0,
		EndFrame:     150, // 5 seconds at 30fps
		Duration:     5 * time.Second,
		ExpectedSize: 1024 * 1024, // 1MB
		Priority:     2,
	}

	encodedSegment, err := encoder.encodeSegment(worker, segment)
	if err != nil {
		t.Fatalf("Failed to encode segment: %v", err)
	}

	if encodedSegment.ID != segment.ID {
		t.Errorf("Encoded segment ID mismatch: %d vs %d", encodedSegment.ID, segment.ID)
	}

	if encodedSegment.FrameCount != segment.EndFrame-segment.StartFrame {
		t.Errorf("Frame count mismatch: %d vs %d", 
			encodedSegment.FrameCount, segment.EndFrame-segment.StartFrame)
	}

	if encodedSegment.Duration != segment.Duration {
		t.Errorf("Duration mismatch: %v vs %v", encodedSegment.Duration, segment.Duration)
	}

	if encodedSegment.Size <= 0 {
		t.Error("Encoded segment size should be positive")
	}

	if encodedSegment.Bitrate <= 0 {
		t.Error("Encoded segment bitrate should be positive")
	}

	if encodedSegment.Quality <= 0 || encodedSegment.Quality > 1 {
		t.Errorf("Quality should be between 0 and 1, got %f", encodedSegment.Quality)
	}

	if !encodedSegment.IsComplete {
		t.Error("Encoded segment should be marked as complete")
	}
}

func TestQualityLevels(t *testing.T) {
	qualities := []QualityLevel{QualityLow, QualityMedium, QualityHigh, QualityUltra, QualityLossless}
	var prevScore float64

	for i, quality := range qualities {
		score := calculateQualityScore(quality, 2000000) // 2 Mbps
		
		if i > 0 && score <= prevScore {
			t.Errorf("Quality score should increase with quality level: %f vs %f", score, prevScore)
		}
		
		if score <= 0 || score > 1 {
			t.Errorf("Quality score should be between 0 and 1, got %f for %v", score, quality)
		}
		
		prevScore = score
	}
}

func TestEncodingModes(t *testing.T) {
	codecs := []VideoCodec{VideoH264, VideoH265, VideoVP8, VideoVP9, VideoAV1}
	
	for _, codec := range codecs {
		mode := getEncodingMode(codec)
		if mode == "" || mode == "Unknown" {
			t.Errorf("Should have valid encoding mode for codec %v", codec)
		}
	}
}

func TestProgressTracking(t *testing.T) {
	config := EncoderConfig{
		InputFile:  "test.mp4",
		OutputFile: "output.mp4",
	}

	encoder := NewVideoEncoder(config)
	
	// Simulate progress
	encoder.progressTracker.totalFrames = 1000
	encoder.progressTracker.processedFrames = 250

	progress, _, elapsed := encoder.GetProgress()

	expectedProgress := 0.25
	if progress != expectedProgress {
		t.Errorf("Expected progress %f, got %f", expectedProgress, progress)
	}

	if elapsed <= 0 {
		t.Error("Elapsed time should be positive")
	}

	// Test with zero total frames
	encoder.progressTracker.totalFrames = 0
	progress, _, _ = encoder.GetProgress()
	if progress != 0 {
		t.Errorf("Progress should be 0 when total frames is 0, got %f", progress)
	}
}

func TestWorkerStatus(t *testing.T) {
	config := EncoderConfig{
		NumWorkers: 3,
	}

	encoder := NewVideoEncoder(config)
	
	// Simulate some worker activity
	encoder.workers[0].processedFrames = 100
	encoder.workers[0].processedBytes = 1024 * 1024
	encoder.workers[0].processingTime = 5 * time.Second
	encoder.workers[0].isActive = true

	status := encoder.GetWorkerStatus()

	if len(status) != config.NumWorkers {
		t.Errorf("Expected %d worker statuses, got %d", config.NumWorkers, len(status))
	}

	worker0Status := status[0]
	if worker0Status.ID != 0 {
		t.Errorf("Worker 0 should have ID 0, got %d", worker0Status.ID)
	}

	if worker0Status.ProcessedFrames != 100 {
		t.Errorf("Worker 0 should have processed 100 frames, got %d", worker0Status.ProcessedFrames)
	}

	if worker0Status.ProcessedBytes != 1024*1024 {
		t.Errorf("Worker 0 should have processed 1MB, got %d", worker0Status.ProcessedBytes)
	}

	if !worker0Status.IsActive {
		t.Error("Worker 0 should be active")
	}
}

func TestEncodingPresets(t *testing.T) {
	presets := []string{"fast", "balanced", "quality", "4k", "unknown"}
	
	for _, preset := range presets {
		config := CreateEncodingPreset(preset)
		
		if config.NumWorkers <= 0 {
			t.Errorf("Preset %s should have positive worker count", preset)
		}
		
		if config.Resolution.Width <= 0 || config.Resolution.Height <= 0 {
			t.Errorf("Preset %s should have valid resolution", preset)
		}
		
		if config.Bitrate <= 0 {
			t.Errorf("Preset %s should have positive bitrate", preset)
		}
		
		if config.FrameRate <= 0 {
			t.Errorf("Preset %s should have positive frame rate", preset)
		}
	}

	// Test specific preset characteristics
	fastConfig := CreateEncodingPreset("fast")
	if fastConfig.Quality != QualityMedium {
		t.Error("Fast preset should use medium quality")
	}

	qualityConfig := CreateEncodingPreset("quality")
	if qualityConfig.Quality != QualityUltra {
		t.Error("Quality preset should use ultra quality")
	}
	if qualityConfig.VideoCodec != VideoH265 {
		t.Error("Quality preset should use H.265 codec")
	}

	config4k := CreateEncodingPreset("4k")
	if config4k.Resolution.Width != 3840 || config4k.Resolution.Height != 2160 {
		t.Error("4K preset should use 4K resolution")
	}
}

func TestConcurrentEncoding(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping concurrent encoding test in short mode")
	}

	tempDir := t.TempDir()
	
	config := EncoderConfig{
		NumWorkers:   4,
		Quality:      QualityLow, // Use low quality for faster testing
		EnableGPU:    false,
		Resolution:   Resolution{Width: 640, Height: 480},
		OutputFormat: FormatMP4,
	}

	// Test multiple concurrent encodings
	numJobs := 3
	var wg sync.WaitGroup
	results := make(chan error, numJobs)

	for i := 0; i < numJobs; i++ {
		wg.Add(1)
		go func(jobID int) {
			defer wg.Done()
			
			inputFile := filepath.Join(tempDir, fmt.Sprintf("input_%d.mp4", jobID))
			outputFile := filepath.Join(tempDir, fmt.Sprintf("output_%d.mp4", jobID))
			
			createTestVideoFile(t, inputFile, 1024*50) // 50KB
			
			jobConfig := config
			jobConfig.InputFile = inputFile
			jobConfig.OutputFile = outputFile
			
			encoder := NewVideoEncoder(jobConfig)
			_, err := encoder.EncodeVideo()
			results <- err
		}(i)
	}

	wg.Wait()
	close(results)

	// Check results
	for err := range results {
		if err != nil {
			t.Errorf("Concurrent encoding failed: %v", err)
		}
	}
}

func TestContextCancellation(t *testing.T) {
	tempDir := t.TempDir()
	inputFile := filepath.Join(tempDir, "input.mp4")
	outputFile := filepath.Join(tempDir, "output.mp4")

	createTestVideoFile(t, inputFile, 1024*10) // Smaller file for faster test

	config := EncoderConfig{
		InputFile:    inputFile,
		OutputFile:   outputFile,
		NumWorkers:   1, // Use fewer workers for more predictable timing
		Quality:      QualityLow, // Use lower quality for faster processing
		EnableGPU:    false,
	}

	encoder := NewVideoEncoder(config)

	// Cancel immediately
	encoder.Cancel()

	// Start encoding after cancellation
	_, err := encoder.EncodeVideo()
	
	// Should get cancelled error or context error
	if err == nil {
		t.Error("Expected cancellation error")
	}
}

func TestStatsCollection(t *testing.T) {
	config := EncoderConfig{
		NumWorkers: 2,
	}

	encoder := NewVideoEncoder(config)
	
	// Simulate some processing
	encoder.stats.TotalFrames = 1000
	encoder.stats.ProcessedFrames = 500
	encoder.stats.InputSize = 1024 * 1024 * 10  // 10MB
	encoder.stats.OutputSize = 1024 * 1024 * 2  // 2MB
	encoder.stats.AverageBitrate = 2000000      // 2Mbps

	stats := encoder.GetStats()
	
	if stats.TotalFrames != 1000 {
		t.Errorf("Expected 1000 total frames, got %d", stats.TotalFrames)
	}
	
	if stats.ProcessedFrames != 500 {
		t.Errorf("Expected 500 processed frames, got %d", stats.ProcessedFrames)
	}
	
	if stats.InputSize != 1024*1024*10 {
		t.Errorf("Expected 10MB input size, got %d", stats.InputSize)
	}
	
	// Verify stats copy (not reference)
	originalValue := encoder.stats.TotalFrames
	encoder.stats.TotalFrames = 2000
	statsAfter := encoder.GetStats()
	if statsAfter.TotalFrames != 2000 {
		t.Errorf("GetStats should reflect current state: expected 2000, got %d", statsAfter.TotalFrames)
	}
	
	// Modify the returned stats and verify original is unchanged
	statsAfter.TotalFrames = 5000
	finalStats := encoder.GetStats()
	if finalStats.TotalFrames == 5000 {
		t.Error("Modifying returned stats should not affect subsequent calls")
	}
	
	// Restore original value
	encoder.stats.TotalFrames = originalValue
}

func TestFormatUtilities(t *testing.T) {
	tests := []struct {
		bytes    int64
		expected string
	}{
		{512, "512 B"},
		{1024, "1.0 KB"},
		{1536, "1.5 KB"},
		{1024 * 1024, "1.0 MB"},
		{1024 * 1024 * 1024, "1.0 GB"},
	}

	for _, test := range tests {
		result := formatBytes(test.bytes)
		if result != test.expected {
			t.Errorf("formatBytes(%d): expected %s, got %s", test.bytes, test.expected, result)
		}
	}

	bitrateTests := []struct {
		bitrate  int64
		expected string
	}{
		{500, "500 bps"},
		{1500, "1.5 Kbps"},
		{2000000, "2.0 Mbps"},
	}

	for _, test := range bitrateTests {
		result := formatBitrate(test.bitrate)
		if result != test.expected {
			t.Errorf("formatBitrate(%d): expected %s, got %s", test.bitrate, test.expected, result)
		}
	}
}

func TestPhaseStrings(t *testing.T) {
	phases := []ProcessingPhase{
		PhaseAnalyzing, PhaseSegmenting, PhaseEncoding, 
		PhaseMuxing, PhaseComplete, PhaseError,
	}

	for _, phase := range phases {
		str := getPhaseString(phase)
		if str == "" || str == "Unknown" {
			t.Errorf("Should have valid string for phase %v", phase)
		}
	}
}

func BenchmarkSegmentCreation(b *testing.B) {
	config := EncoderConfig{
		SegmentDuration: 10 * time.Second,
	}

	encoder := NewVideoEncoder(config)
	info := &VideoInfo{
		Duration:    600 * time.Second, // 10 minutes
		FrameRate:   30.0,
		TotalFrames: 18000,
		Size:        1024 * 1024 * 100, // 100MB
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := encoder.createSegments(info)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWorkerSimulation(b *testing.B) {
	config := EncoderConfig{
		NumWorkers: runtime.NumCPU(),
		Quality:    QualityMedium,
		EnableGPU:  false,
	}

	encoder := NewVideoEncoder(config)
	worker := encoder.workers[0]

	segment := &Segment{
		ID:           1,
		StartFrame:   0,
		EndFrame:     300, // 10 seconds at 30fps
		Duration:     10 * time.Second,
		ExpectedSize: 1024 * 1024 * 2, // 2MB
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := encoder.encodeSegment(worker, segment)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQualityCalculation(b *testing.B) {
	qualities := []QualityLevel{QualityLow, QualityMedium, QualityHigh, QualityUltra, QualityLossless}
	bitrate := int64(2000000)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		for _, quality := range qualities {
			calculateQualityScore(quality, bitrate)
		}
	}
}