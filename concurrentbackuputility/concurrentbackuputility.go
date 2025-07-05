package concurrentbackuputility

import (
	"archive/tar"
	"compress/gzip"
	"crypto/md5"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// BackupUtility handles concurrent backup operations
type BackupUtility struct {
	numWorkers    int
	bufferSize    int
	compressionLevel int
	enableChecksum   bool
	excludePatterns  []string
	includePatterns  []string
	maxFileSize      int64
}

// BackupConfig holds backup configuration
type BackupConfig struct {
	NumWorkers       int
	BufferSize       int
	CompressionLevel int
	EnableChecksum   bool
	ExcludePatterns  []string
	IncludePatterns  []string
	MaxFileSize      int64
}

// BackupJob represents a backup operation
type BackupJob struct {
	ID            string
	SourcePaths   []string
	DestinationPath string
	Config        BackupConfig
	StartTime     time.Time
	EndTime       time.Time
	Status        JobStatus
	Progress      *BackupProgress
	Result        *BackupResult
}

// JobStatus represents backup job status
type JobStatus int

const (
	StatusPending JobStatus = iota
	StatusRunning
	StatusCompleted
	StatusFailed
	StatusCancelled
)

// BackupProgress tracks backup progress
type BackupProgress struct {
	TotalFiles     int64
	ProcessedFiles int64
	TotalBytes     int64
	ProcessedBytes int64
	CurrentFile    string
	StartTime      time.Time
	mu             sync.RWMutex
}

// BackupResult contains backup operation results
type BackupResult struct {
	TotalFiles       int64
	ProcessedFiles   int64
	SkippedFiles     int64
	ErrorFiles       int64
	TotalBytes       int64
	ProcessedBytes   int64
	CompressionRatio float64
	Duration         time.Duration
	ChecksumMap      map[string]string
	Errors           []string
}

// FileInfo represents file information
type FileInfo struct {
	Path         string
	Size         int64
	ModTime      time.Time
	Mode         os.FileMode
	IsDir        bool
	Checksum     string
	CompressedSize int64
}

// BackupArchive manages archive operations
type BackupArchive struct {
	path       string
	tarWriter  *tar.Writer
	gzipWriter *gzip.Writer
	file       *os.File
	mu         sync.Mutex
}

// NewBackupUtility creates a new backup utility
func NewBackupUtility(config BackupConfig) *BackupUtility {
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	if config.BufferSize <= 0 {
		config.BufferSize = 64 * 1024 // 64KB
	}
	if config.CompressionLevel < 0 || config.CompressionLevel > 9 {
		config.CompressionLevel = 6 // Default compression
	}
	if config.MaxFileSize <= 0 {
		config.MaxFileSize = 100 * 1024 * 1024 // 100MB
	}

	return &BackupUtility{
		numWorkers:       config.NumWorkers,
		bufferSize:       config.BufferSize,
		compressionLevel: config.CompressionLevel,
		enableChecksum:   config.EnableChecksum,
		excludePatterns:  config.ExcludePatterns,
		includePatterns:  config.IncludePatterns,
		maxFileSize:      config.MaxFileSize,
	}
}

// CreateBackup creates a backup archive from source paths
func (bu *BackupUtility) CreateBackup(sourcePaths []string, destPath string) (*BackupResult, error) {
	result := &BackupResult{
		ChecksumMap: make(map[string]string),
		Errors:      make([]string, 0),
	}

	start := time.Now()

	// Create archive
	archive, err := bu.createArchive(destPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create archive: %w", err)
	}
	defer archive.Close()

	// Scan source paths
	fileChan := make(chan string, 1000)
	progress := &BackupProgress{
		StartTime: start,
	}

	// Start file scanner
	var scanWg sync.WaitGroup
	scanWg.Add(1)
	go func() {
		defer scanWg.Done()
		defer close(fileChan)
		
		for _, sourcePath := range sourcePaths {
			if err := bu.scanPath(sourcePath, fileChan, progress); err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("scan error for %s: %v", sourcePath, err))
			}
		}
	}()

	// Start backup workers
	var workWg sync.WaitGroup
	for i := 0; i < bu.numWorkers; i++ {
		workWg.Add(1)
		go func() {
			defer workWg.Done()
			bu.backupWorker(fileChan, archive, progress, result)
		}()
	}

	// Wait for scanning to complete
	scanWg.Wait()

	// Wait for all workers to complete
	workWg.Wait()

	result.Duration = time.Since(start)
	result.TotalFiles = atomic.LoadInt64(&progress.TotalFiles)
	result.ProcessedFiles = atomic.LoadInt64(&progress.ProcessedFiles)
	result.TotalBytes = atomic.LoadInt64(&progress.TotalBytes)
	result.ProcessedBytes = atomic.LoadInt64(&progress.ProcessedBytes)

	// Calculate compression ratio
	if result.TotalBytes > 0 {
		archiveInfo, _ := os.Stat(destPath)
		if archiveInfo != nil {
			result.CompressionRatio = float64(archiveInfo.Size()) / float64(result.TotalBytes)
		}
	}

	return result, nil
}

func (bu *BackupUtility) createArchive(destPath string) (*BackupArchive, error) {
	// Create destination directory if needed
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return nil, err
	}

	file, err := os.Create(destPath)
	if err != nil {
		return nil, err
	}

	gzipWriter, err := gzip.NewWriterLevel(file, bu.compressionLevel)
	if err != nil {
		file.Close()
		return nil, err
	}

	tarWriter := tar.NewWriter(gzipWriter)

	return &BackupArchive{
		path:       destPath,
		file:       file,
		gzipWriter: gzipWriter,
		tarWriter:  tarWriter,
	}, nil
}

func (bu *BackupUtility) scanPath(sourcePath string, fileChan chan<- string, progress *BackupProgress) error {
	return filepath.Walk(sourcePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Check exclusion patterns
		if bu.shouldExclude(path) {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Check inclusion patterns
		if len(bu.includePatterns) > 0 && !bu.shouldInclude(path) {
			if info.IsDir() {
				return nil // Continue scanning directory
			}
			return nil
		}

		// Check file size limit
		if !info.IsDir() && info.Size() > bu.maxFileSize {
			atomic.AddInt64(&progress.TotalFiles, 1)
			return nil // Skip large files
		}

		atomic.AddInt64(&progress.TotalFiles, 1)
		if !info.IsDir() {
			atomic.AddInt64(&progress.TotalBytes, info.Size())
		}

		fileChan <- path
		return nil
	})
}

func (bu *BackupUtility) shouldExclude(path string) bool {
	for _, pattern := range bu.excludePatterns {
		if matched, _ := filepath.Match(pattern, filepath.Base(path)); matched {
			return true
		}
		if strings.Contains(path, pattern) {
			return true
		}
	}
	return false
}

func (bu *BackupUtility) shouldInclude(path string) bool {
	for _, pattern := range bu.includePatterns {
		if matched, _ := filepath.Match(pattern, filepath.Base(path)); matched {
			return true
		}
		if strings.Contains(path, pattern) {
			return true
		}
	}
	return false
}

func (bu *BackupUtility) backupWorker(fileChan <-chan string, archive *BackupArchive, progress *BackupProgress, result *BackupResult) {
	buffer := make([]byte, bu.bufferSize)

	for filePath := range fileChan {
		if err := bu.processFile(filePath, archive, progress, result, buffer); err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("error processing %s: %v", filePath, err))
			atomic.AddInt64(&result.ErrorFiles, 1)
		}
	}
}

func (bu *BackupUtility) processFile(filePath string, archive *BackupArchive, progress *BackupProgress, result *BackupResult, buffer []byte) error {
	info, err := os.Lstat(filePath)
	if err != nil {
		return err
	}

	// Update progress
	progress.mu.Lock()
	progress.CurrentFile = filePath
	progress.mu.Unlock()

	// Create tar header
	header, err := tar.FileInfoHeader(info, "")
	if err != nil {
		return err
	}

	header.Name = strings.TrimPrefix(filePath, string(filepath.Separator))

	// Handle symlinks
	if info.Mode()&os.ModeSymlink != 0 {
		linkTarget, err := os.Readlink(filePath)
		if err != nil {
			return err
		}
		header.Linkname = linkTarget
	}

	// Write header to archive
	archive.mu.Lock()
	if err := archive.tarWriter.WriteHeader(header); err != nil {
		archive.mu.Unlock()
		return err
	}

	// Write file content if it's a regular file
	if info.Mode().IsRegular() {
		file, err := os.Open(filePath)
		if err != nil {
			archive.mu.Unlock()
			return err
		}

		var checksum string
		if bu.enableChecksum {
			checksum, err = bu.writeFileWithChecksum(file, archive.tarWriter, buffer)
		} else {
			_, err = bu.writeFile(file, archive.tarWriter, buffer)
		}

		file.Close()
		archive.mu.Unlock()

		if err != nil {
			return err
		}

		if bu.enableChecksum {
			result.ChecksumMap[filePath] = checksum
		}

		atomic.AddInt64(&progress.ProcessedBytes, info.Size())
	} else {
		archive.mu.Unlock()
	}

	atomic.AddInt64(&progress.ProcessedFiles, 1)
	return nil
}

func (bu *BackupUtility) writeFile(src io.Reader, dst io.Writer, buffer []byte) (int64, error) {
	var written int64
	for {
		n, err := src.Read(buffer)
		if n > 0 {
			if _, writeErr := dst.Write(buffer[:n]); writeErr != nil {
				return written, writeErr
			}
			written += int64(n)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return written, err
		}
	}
	return written, nil
}

func (bu *BackupUtility) writeFileWithChecksum(src io.Reader, dst io.Writer, buffer []byte) (string, error) {
	hasher := sha256.New()
	var written int64

	for {
		n, err := src.Read(buffer)
		if n > 0 {
			// Write to destination
			if _, writeErr := dst.Write(buffer[:n]); writeErr != nil {
				return "", writeErr
			}
			// Update hash
			hasher.Write(buffer[:n])
			written += int64(n)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
	}

	return hex.EncodeToString(hasher.Sum(nil)), nil
}

// ExtractBackup extracts files from a backup archive
func (bu *BackupUtility) ExtractBackup(archivePath, destPath string) (*BackupResult, error) {
	result := &BackupResult{
		ChecksumMap: make(map[string]string),
		Errors:      make([]string, 0),
	}

	start := time.Now()

	// Open archive
	file, err := os.Open(archivePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gzipReader.Close()

	tarReader := tar.NewReader(gzipReader)

	// Create destination directory
	if err := os.MkdirAll(destPath, 0755); err != nil {
		return nil, err
	}

	// Extract files
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		if err := bu.extractFile(tarReader, header, destPath, result); err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("error extracting %s: %v", header.Name, err))
			atomic.AddInt64(&result.ErrorFiles, 1)
		} else {
			atomic.AddInt64(&result.ProcessedFiles, 1)
			atomic.AddInt64(&result.ProcessedBytes, header.Size)
		}
		atomic.AddInt64(&result.TotalFiles, 1)
		atomic.AddInt64(&result.TotalBytes, header.Size)
	}

	result.Duration = time.Since(start)
	return result, nil
}

func (bu *BackupUtility) extractFile(tarReader *tar.Reader, header *tar.Header, destPath string, result *BackupResult) error {
	targetPath := filepath.Join(destPath, header.Name)

	// Ensure target directory exists
	if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
		return err
	}

	switch header.Typeflag {
	case tar.TypeDir:
		return os.MkdirAll(targetPath, header.FileInfo().Mode())

	case tar.TypeReg:
		return bu.extractRegularFile(tarReader, targetPath, header, result)

	case tar.TypeSymlink:
		return os.Symlink(header.Linkname, targetPath)

	case tar.TypeLink:
		return os.Link(filepath.Join(destPath, header.Linkname), targetPath)

	default:
		return fmt.Errorf("unsupported file type: %v", header.Typeflag)
	}
}

func (bu *BackupUtility) extractRegularFile(tarReader *tar.Reader, targetPath string, header *tar.Header, result *BackupResult) error {
	file, err := os.OpenFile(targetPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, header.FileInfo().Mode())
	if err != nil {
		return err
	}
	defer file.Close()

	buffer := make([]byte, bu.bufferSize)
	var checksum string

	if bu.enableChecksum {
		checksum, err = bu.copyWithChecksum(tarReader, file, buffer)
	} else {
		_, err = bu.writeFile(tarReader, file, buffer)
	}

	if err != nil {
		return err
	}

	if bu.enableChecksum {
		result.ChecksumMap[targetPath] = checksum
	}

	// Set file modification time
	return os.Chtimes(targetPath, header.AccessTime, header.ModTime)
}

func (bu *BackupUtility) copyWithChecksum(src io.Reader, dst io.Writer, buffer []byte) (string, error) {
	hasher := sha256.New()
	
	for {
		n, err := src.Read(buffer)
		if n > 0 {
			if _, writeErr := dst.Write(buffer[:n]); writeErr != nil {
				return "", writeErr
			}
			hasher.Write(buffer[:n])
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
	}

	return hex.EncodeToString(hasher.Sum(nil)), nil
}

// VerifyBackup verifies the integrity of a backup archive
func (bu *BackupUtility) VerifyBackup(archivePath string) (*BackupResult, error) {
	result := &BackupResult{
		ChecksumMap: make(map[string]string),
		Errors:      make([]string, 0),
	}

	start := time.Now()

	file, err := os.Open(archivePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gzipReader.Close()

	tarReader := tar.NewReader(gzipReader)

	// Verify archive contents
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		atomic.AddInt64(&result.TotalFiles, 1)
		atomic.AddInt64(&result.TotalBytes, header.Size)

		if header.Typeflag == tar.TypeReg {
			if err := bu.verifyFile(tarReader, header, result); err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("verification failed for %s: %v", header.Name, err))
				atomic.AddInt64(&result.ErrorFiles, 1)
			} else {
				atomic.AddInt64(&result.ProcessedFiles, 1)
				atomic.AddInt64(&result.ProcessedBytes, header.Size)
			}
		}
	}

	result.Duration = time.Since(start)
	return result, nil
}

func (bu *BackupUtility) verifyFile(tarReader *tar.Reader, header *tar.Header, result *BackupResult) error {
	hasher := sha256.New()
	buffer := make([]byte, bu.bufferSize)
	var totalRead int64

	for totalRead < header.Size {
		n, err := tarReader.Read(buffer)
		if n > 0 {
			hasher.Write(buffer[:n])
			totalRead += int64(n)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}

	checksum := hex.EncodeToString(hasher.Sum(nil))
	result.ChecksumMap[header.Name] = checksum

	return nil
}

// ListBackupContents lists the contents of a backup archive
func (bu *BackupUtility) ListBackupContents(archivePath string) ([]*FileInfo, error) {
	file, err := os.Open(archivePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gzipReader.Close()

	tarReader := tar.NewReader(gzipReader)

	var files []*FileInfo

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		fileInfo := &FileInfo{
			Path:    header.Name,
			Size:    header.Size,
			ModTime: header.ModTime,
			Mode:    header.FileInfo().Mode(),
			IsDir:   header.Typeflag == tar.TypeDir,
		}

		files = append(files, fileInfo)
	}

	// Sort by path
	sort.Slice(files, func(i, j int) bool {
		return files[i].Path < files[j].Path
	})

	return files, nil
}

// GetProgress returns current backup progress
func (bp *BackupProgress) GetProgress() (int64, int64, int64, int64, string, time.Duration) {
	bp.mu.RLock()
	defer bp.mu.RUnlock()

	elapsed := time.Since(bp.StartTime)
	return atomic.LoadInt64(&bp.TotalFiles),
		atomic.LoadInt64(&bp.ProcessedFiles),
		atomic.LoadInt64(&bp.TotalBytes),
		atomic.LoadInt64(&bp.ProcessedBytes),
		bp.CurrentFile,
		elapsed
}

// CalculateSpeed calculates backup speed
func (bp *BackupProgress) CalculateSpeed() (float64, float64) {
	_, processedFiles, _, processedBytes, _, elapsed := bp.GetProgress()

	if elapsed.Seconds() == 0 {
		return 0, 0
	}

	filesPerSecond := float64(processedFiles) / elapsed.Seconds()
	bytesPerSecond := float64(processedBytes) / elapsed.Seconds()

	return filesPerSecond, bytesPerSecond
}

// EstimateTimeRemaining estimates time remaining for backup
func (bp *BackupProgress) EstimateTimeRemaining() time.Duration {
	totalFiles, processedFiles, totalBytes, processedBytes, _, elapsed := bp.GetProgress()

	if processedFiles == 0 || processedBytes == 0 {
		return 0
	}

	// Use the more conservative estimate between files and bytes
	fileProgress := float64(processedFiles) / float64(totalFiles)
	byteProgress := float64(processedBytes) / float64(totalBytes)
	
	progress := fileProgress
	if byteProgress < fileProgress {
		progress = byteProgress
	}

	if progress == 0 {
		return 0
	}

	totalTime := time.Duration(float64(elapsed) / progress)
	return totalTime - elapsed
}

// Close closes the backup archive
func (ba *BackupArchive) Close() error {
	ba.mu.Lock()
	defer ba.mu.Unlock()

	var errors []error

	if ba.tarWriter != nil {
		if err := ba.tarWriter.Close(); err != nil {
			errors = append(errors, err)
		}
	}

	if ba.gzipWriter != nil {
		if err := ba.gzipWriter.Close(); err != nil {
			errors = append(errors, err)
		}
	}

	if ba.file != nil {
		if err := ba.file.Close(); err != nil {
			errors = append(errors, err)
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("errors closing archive: %v", errors)
	}

	return nil
}

// IncrementalBackup performs incremental backup based on modification times
func (bu *BackupUtility) IncrementalBackup(sourcePaths []string, destPath string, lastBackupTime time.Time) (*BackupResult, error) {
	// Create a modified backup utility for incremental backup
	tempConfig := BackupConfig{
		NumWorkers:       bu.numWorkers,
		BufferSize:       bu.bufferSize,
		CompressionLevel: bu.compressionLevel,
		EnableChecksum:   bu.enableChecksum,
		ExcludePatterns:  bu.excludePatterns,
		IncludePatterns:  bu.includePatterns,
		MaxFileSize:      bu.maxFileSize,
	}

	incrementalBU := NewBackupUtility(tempConfig)
	
	// Override scanning to include only modified files
	return incrementalBU.createIncrementalBackup(sourcePaths, destPath, lastBackupTime)
}

func (bu *BackupUtility) createIncrementalBackup(sourcePaths []string, destPath string, lastBackupTime time.Time) (*BackupResult, error) {
	result := &BackupResult{
		ChecksumMap: make(map[string]string),
		Errors:      make([]string, 0),
	}

	start := time.Now()

	// Create archive
	archive, err := bu.createArchive(destPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create archive: %w", err)
	}
	defer archive.Close()

	// Scan for modified files
	fileChan := make(chan string, 1000)
	progress := &BackupProgress{
		StartTime: start,
	}

	// Start file scanner for incremental backup
	var scanWg sync.WaitGroup
	scanWg.Add(1)
	go func() {
		defer scanWg.Done()
		defer close(fileChan)
		
		for _, sourcePath := range sourcePaths {
			if err := bu.scanIncrementalPath(sourcePath, fileChan, progress, lastBackupTime); err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("scan error for %s: %v", sourcePath, err))
			}
		}
	}()

	// Start backup workers
	var workWg sync.WaitGroup
	for i := 0; i < bu.numWorkers; i++ {
		workWg.Add(1)
		go func() {
			defer workWg.Done()
			bu.backupWorker(fileChan, archive, progress, result)
		}()
	}

	// Wait for scanning to complete
	scanWg.Wait()

	// Wait for all workers to complete
	workWg.Wait()

	result.Duration = time.Since(start)
	result.TotalFiles = atomic.LoadInt64(&progress.TotalFiles)
	result.ProcessedFiles = atomic.LoadInt64(&progress.ProcessedFiles)
	result.TotalBytes = atomic.LoadInt64(&progress.TotalBytes)
	result.ProcessedBytes = atomic.LoadInt64(&progress.ProcessedBytes)

	return result, nil
}

func (bu *BackupUtility) scanIncrementalPath(sourcePath string, fileChan chan<- string, progress *BackupProgress, lastBackupTime time.Time) error {
	return filepath.Walk(sourcePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Check if file was modified after last backup
		if !info.ModTime().After(lastBackupTime) {
			if info.IsDir() {
				return nil // Continue scanning directory
			}
			return nil // Skip unmodified file
		}

		// Apply same filters as regular backup
		if bu.shouldExclude(path) {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if len(bu.includePatterns) > 0 && !bu.shouldInclude(path) {
			if info.IsDir() {
				return nil
			}
			return nil
		}

		if !info.IsDir() && info.Size() > bu.maxFileSize {
			atomic.AddInt64(&progress.TotalFiles, 1)
			return nil
		}

		atomic.AddInt64(&progress.TotalFiles, 1)
		if !info.IsDir() {
			atomic.AddInt64(&progress.TotalBytes, info.Size())
		}

		fileChan <- path
		return nil
	})
}

// CalculateChecksumMD5 calculates MD5 checksum of a file
func CalculateChecksumMD5(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	hasher := md5.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", err
	}

	return hex.EncodeToString(hasher.Sum(nil)), nil
}

// FormatBytes formats bytes into human readable format
func FormatBytes(bytes int64) string {
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

// Example demonstrates concurrent backup utility
func Example() {
	fmt.Println("=== Concurrent Backup Utility Example ===")

	// Create backup configuration
	config := BackupConfig{
		NumWorkers:       4,
		BufferSize:       64 * 1024,
		CompressionLevel: 6,
		EnableChecksum:   true,
		ExcludePatterns:  []string{"*.tmp", "*.log", ".git"},
		MaxFileSize:      100 * 1024 * 1024, // 100MB
	}

	// Create backup utility
	backupUtil := NewBackupUtility(config)

	// Example: Create backup
	sourcePaths := []string{"."}
	destPath := "./backup.tar.gz"

	fmt.Println("Creating backup...")
	result, err := backupUtil.CreateBackup(sourcePaths, destPath)
	if err != nil {
		fmt.Printf("Backup failed: %v\n", err)
		return
	}

	fmt.Printf("Backup Results:\n")
	fmt.Printf("  Files processed: %d/%d\n", result.ProcessedFiles, result.TotalFiles)
	fmt.Printf("  Bytes processed: %s\n", FormatBytes(result.ProcessedBytes))
	fmt.Printf("  Compression ratio: %.2f\n", result.CompressionRatio)
	fmt.Printf("  Duration: %v\n", result.Duration)
	fmt.Printf("  Errors: %d\n", len(result.Errors))

	// Example: List contents
	fmt.Println("\nListing backup contents...")
	files, err := backupUtil.ListBackupContents(destPath)
	if err != nil {
		fmt.Printf("Failed to list contents: %v\n", err)
		return
	}

	fmt.Printf("Archive contains %d files:\n", len(files))
	for i, file := range files {
		if i < 10 { // Show first 10 files
			fmt.Printf("  %s (%s)\n", file.Path, FormatBytes(file.Size))
		}
	}

	// Example: Verify backup
	fmt.Println("\nVerifying backup...")
	verifyResult, err := backupUtil.VerifyBackup(destPath)
	if err != nil {
		fmt.Printf("Verification failed: %v\n", err)
		return
	}

	fmt.Printf("Verification Results:\n")
	fmt.Printf("  Files verified: %d\n", verifyResult.ProcessedFiles)
	fmt.Printf("  Verification errors: %d\n", len(verifyResult.Errors))

	// Cleanup
	os.Remove(destPath)
}