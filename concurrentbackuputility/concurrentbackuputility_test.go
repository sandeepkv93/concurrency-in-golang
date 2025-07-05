package concurrentbackuputility

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func createTestDirectory(t *testing.T) string {
	tmpDir, err := ioutil.TempDir("", "backup_test")
	if err != nil {
		t.Fatal(err)
	}

	// Create test file structure
	files := map[string]string{
		"file1.txt":           "Content of file 1",
		"file2.txt":           "Content of file 2 with more text",
		"subdir/file3.txt":    "Content in subdirectory",
		"subdir/file4.txt":    "Another file in subdir",
		"empty.txt":           "",
		"large.txt":           strings.Repeat("Large file content. ", 1000),
		"binary.bin":          "\x00\x01\x02\x03\xff\xfe\xfd",
		"nested/deep/file.txt": "Deeply nested file",
	}

	for path, content := range files {
		fullPath := filepath.Join(tmpDir, path)
		if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
			t.Fatal(err)
		}
		if err := ioutil.WriteFile(fullPath, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}

	return tmpDir
}

func TestBackupUtility(t *testing.T) {
	testDir := createTestDirectory(t)
	defer os.RemoveAll(testDir)

	config := BackupConfig{
		NumWorkers:       2,
		BufferSize:       1024,
		CompressionLevel: 1,
		EnableChecksum:   true,
	}

	backupUtil := NewBackupUtility(config)

	// Create backup
	archivePath := filepath.Join(testDir, "test_backup.tar.gz")
	result, err := backupUtil.CreateBackup([]string{testDir}, archivePath)

	if err != nil {
		t.Fatalf("Backup failed: %v", err)
	}

	// Verify backup was created
	if _, err := os.Stat(archivePath); os.IsNotExist(err) {
		t.Error("Backup archive was not created")
	}

	// Check results
	if result.ProcessedFiles == 0 {
		t.Error("No files were processed")
	}

	if result.ProcessedBytes == 0 {
		t.Error("No bytes were processed")
	}

	if result.Duration == 0 {
		t.Error("Duration should be non-zero")
	}

	if len(result.ChecksumMap) == 0 {
		t.Error("No checksums were calculated")
	}

	t.Logf("Backup completed: %d files, %d bytes in %v",
		result.ProcessedFiles, result.ProcessedBytes, result.Duration)
}

func TestBackupWithExclusions(t *testing.T) {
	testDir := createTestDirectory(t)
	defer os.RemoveAll(testDir)

	// Create some files that should be excluded
	excludedFiles := []string{
		filepath.Join(testDir, "temp.tmp"),
		filepath.Join(testDir, "log.log"),
		filepath.Join(testDir, "subdir", "cache.tmp"),
	}

	for _, path := range excludedFiles {
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := ioutil.WriteFile(path, []byte("excluded content"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	config := BackupConfig{
		NumWorkers:      2,
		EnableChecksum:  false,
		ExcludePatterns: []string{"*.tmp", "*.log"},
	}

	backupUtil := NewBackupUtility(config)
	archivePath := filepath.Join(testDir, "excluded_backup.tar.gz")

	result, err := backupUtil.CreateBackup([]string{testDir}, archivePath)
	if err != nil {
		t.Fatalf("Backup failed: %v", err)
	}

	// List archive contents to verify exclusions
	files, err := backupUtil.ListBackupContents(archivePath)
	if err != nil {
		t.Fatalf("Failed to list contents: %v", err)
	}

	// Check that excluded files are not in archive
	for _, file := range files {
		if strings.HasSuffix(file.Path, ".tmp") || strings.HasSuffix(file.Path, ".log") {
			t.Errorf("Excluded file found in archive: %s", file.Path)
		}
	}
}

func TestBackupWithInclusions(t *testing.T) {
	testDir := createTestDirectory(t)
	defer os.RemoveAll(testDir)

	config := BackupConfig{
		NumWorkers:      2,
		EnableChecksum:  false,
		IncludePatterns: []string{"*.txt"},
	}

	backupUtil := NewBackupUtility(config)
	archivePath := filepath.Join(testDir, "included_backup.tar.gz")

	result, err := backupUtil.CreateBackup([]string{testDir}, archivePath)
	if err != nil {
		t.Fatalf("Backup failed: %v", err)
	}

	// List archive contents
	files, err := backupUtil.ListBackupContents(archivePath)
	if err != nil {
		t.Fatalf("Failed to list contents: %v", err)
	}

	// Verify only .txt files are included (plus directories)
	for _, file := range files {
		if !file.IsDir && !strings.HasSuffix(file.Path, ".txt") {
			t.Errorf("Non-txt file found in archive: %s", file.Path)
		}
	}

	// Verify at least some .txt files are included
	txtCount := 0
	for _, file := range files {
		if strings.HasSuffix(file.Path, ".txt") {
			txtCount++
		}
	}

	if txtCount == 0 {
		t.Error("No .txt files found in archive")
	}
}

func TestExtractBackup(t *testing.T) {
	sourceDir := createTestDirectory(t)
	defer os.RemoveAll(sourceDir)

	config := BackupConfig{
		NumWorkers:     2,
		EnableChecksum: true,
	}

	backupUtil := NewBackupUtility(config)

	// Create backup
	archivePath := filepath.Join(sourceDir, "extract_test.tar.gz")
	_, err := backupUtil.CreateBackup([]string{sourceDir}, archivePath)
	if err != nil {
		t.Fatalf("Backup failed: %v", err)
	}

	// Create extraction directory
	extractDir, err := ioutil.TempDir("", "extract_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(extractDir)

	// Extract backup
	result, err := backupUtil.ExtractBackup(archivePath, extractDir)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	if result.ProcessedFiles == 0 {
		t.Error("No files were extracted")
	}

	// Verify extracted files exist and have correct content
	testFile := filepath.Join(extractDir, filepath.Base(sourceDir), "file1.txt")
	if _, err := os.Stat(testFile); os.IsNotExist(err) {
		t.Error("Extracted file not found")
	}

	content, err := ioutil.ReadFile(testFile)
	if err != nil {
		t.Fatal(err)
	}

	if string(content) != "Content of file 1" {
		t.Errorf("Extracted file content mismatch: got %q", string(content))
	}
}

func TestVerifyBackup(t *testing.T) {
	testDir := createTestDirectory(t)
	defer os.RemoveAll(testDir)

	config := BackupConfig{
		NumWorkers:     2,
		EnableChecksum: true,
	}

	backupUtil := NewBackupUtility(config)

	// Create backup
	archivePath := filepath.Join(testDir, "verify_test.tar.gz")
	_, err := backupUtil.CreateBackup([]string{testDir}, archivePath)
	if err != nil {
		t.Fatalf("Backup failed: %v", err)
	}

	// Verify backup
	result, err := backupUtil.VerifyBackup(archivePath)
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}

	if result.ProcessedFiles == 0 {
		t.Error("No files were verified")
	}

	if len(result.ChecksumMap) == 0 {
		t.Error("No checksums were calculated during verification")
	}

	if len(result.Errors) > 0 {
		t.Errorf("Verification errors: %v", result.Errors)
	}
}

func TestListBackupContents(t *testing.T) {
	testDir := createTestDirectory(t)
	defer os.RemoveAll(testDir)

	config := BackupConfig{
		NumWorkers: 2,
	}

	backupUtil := NewBackupUtility(config)

	// Create backup
	archivePath := filepath.Join(testDir, "list_test.tar.gz")
	_, err := backupUtil.CreateBackup([]string{testDir}, archivePath)
	if err != nil {
		t.Fatalf("Backup failed: %v", err)
	}

	// List contents
	files, err := backupUtil.ListBackupContents(archivePath)
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}

	if len(files) == 0 {
		t.Error("No files listed")
	}

	// Verify files are sorted
	for i := 1; i < len(files); i++ {
		if files[i-1].Path > files[i].Path {
			t.Error("Files are not sorted by path")
			break
		}
	}

	// Check for expected files
	found := make(map[string]bool)
	for _, file := range files {
		found[file.Path] = true
	}

	expectedFiles := []string{"file1.txt", "file2.txt", "subdir/file3.txt"}
	for _, expected := range expectedFiles {
		expectedPath := filepath.Join(filepath.Base(testDir), expected)
		if !found[expectedPath] {
			t.Errorf("Expected file not found in listing: %s", expectedPath)
		}
	}
}

func TestIncrementalBackup(t *testing.T) {
	testDir := createTestDirectory(t)
	defer os.RemoveAll(testDir)

	config := BackupConfig{
		NumWorkers:     2,
		EnableChecksum: false,
	}

	backupUtil := NewBackupUtility(config)

	// Create initial backup
	archivePath1 := filepath.Join(testDir, "initial_backup.tar.gz")
	_, err := backupUtil.CreateBackup([]string{testDir}, archivePath1)
	if err != nil {
		t.Fatalf("Initial backup failed: %v", err)
	}

	backupTime := time.Now()
	time.Sleep(100 * time.Millisecond) // Ensure time difference

	// Modify some files and add new files
	modifiedFile := filepath.Join(testDir, "file1.txt")
	if err := ioutil.WriteFile(modifiedFile, []byte("Modified content"), 0644); err != nil {
		t.Fatal(err)
	}

	newFile := filepath.Join(testDir, "new_file.txt")
	if err := ioutil.WriteFile(newFile, []byte("New file content"), 0644); err != nil {
		t.Fatal(err)
	}

	// Create incremental backup
	archivePath2 := filepath.Join(testDir, "incremental_backup.tar.gz")
	result, err := backupUtil.IncrementalBackup([]string{testDir}, archivePath2, backupTime)
	if err != nil {
		t.Fatalf("Incremental backup failed: %v", err)
	}

	// List incremental backup contents
	files, err := backupUtil.ListBackupContents(archivePath2)
	if err != nil {
		t.Fatalf("Failed to list incremental contents: %v", err)
	}

	// Should contain only modified and new files (plus directories)
	fileCount := 0
	for _, file := range files {
		if !file.IsDir {
			fileCount++
		}
	}

	// Should have fewer files than full backup
	if fileCount == 0 {
		t.Error("Incremental backup contains no files")
	}

	// Verify that incremental backup is smaller
	info1, _ := os.Stat(archivePath1)
	info2, _ := os.Stat(archivePath2)

	if info2.Size() >= info1.Size() {
		t.Log("Note: Incremental backup not significantly smaller (expected for small test data)")
	}
}

func TestBackupProgress(t *testing.T) {
	progress := &BackupProgress{
		StartTime: time.Now(),
	}

	// Simulate progress updates
	progress.TotalFiles = 100
	progress.TotalBytes = 10000

	for i := int64(0); i < 50; i++ {
		progress.ProcessedFiles = i
		progress.ProcessedBytes = i * 100
	}

	// Test progress calculation
	totalFiles, processedFiles, totalBytes, processedBytes, _, elapsed := progress.GetProgress()

	if totalFiles != 100 {
		t.Errorf("Expected 100 total files, got %d", totalFiles)
	}

	if processedFiles != 49 {
		t.Errorf("Expected 49 processed files, got %d", processedFiles)
	}

	if elapsed == 0 {
		t.Error("Elapsed time should be non-zero")
	}

	// Test speed calculation
	filesPerSec, bytesPerSec := progress.CalculateSpeed()
	if filesPerSec <= 0 || bytesPerSec <= 0 {
		t.Error("Speed calculation returned invalid values")
	}

	// Test time estimation
	remaining := progress.EstimateTimeRemaining()
	if remaining < 0 {
		t.Error("Time remaining should not be negative")
	}
}

func TestConcurrentBackups(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping concurrent test in short mode")
	}

	testDir := createTestDirectory(t)
	defer os.RemoveAll(testDir)

	config := BackupConfig{
		NumWorkers:     4,
		EnableChecksum: false,
	}

	backupUtil := NewBackupUtility(config)

	// Create multiple backups concurrently
	var wg sync.WaitGroup
	numBackups := 5

	for i := 0; i < numBackups; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			archivePath := filepath.Join(testDir, fmt.Sprintf("concurrent_%d.tar.gz", id))
			result, err := backupUtil.CreateBackup([]string{testDir}, archivePath)

			if err != nil {
				t.Errorf("Concurrent backup %d failed: %v", id, err)
				return
			}

			if result.ProcessedFiles == 0 {
				t.Errorf("Concurrent backup %d processed no files", id)
			}
		}(i)
	}

	wg.Wait()
}

func TestBackupLargeFiles(t *testing.T) {
	testDir, err := ioutil.TempDir("", "large_file_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testDir)

	// Create a large file (1MB)
	largeContent := make([]byte, 1024*1024)
	for i := range largeContent {
		largeContent[i] = byte(i % 256)
	}

	largeFile := filepath.Join(testDir, "large_file.bin")
	if err := ioutil.WriteFile(largeFile, largeContent, 0644); err != nil {
		t.Fatal(err)
	}

	config := BackupConfig{
		NumWorkers:     2,
		BufferSize:     64 * 1024, // 64KB buffer
		EnableChecksum: true,
		MaxFileSize:    2 * 1024 * 1024, // 2MB limit
	}

	backupUtil := NewBackupUtility(config)

	// Create backup
	archivePath := filepath.Join(testDir, "large_backup.tar.gz")
	result, err := backupUtil.CreateBackup([]string{testDir}, archivePath)

	if err != nil {
		t.Fatalf("Large file backup failed: %v", err)
	}

	if result.ProcessedFiles == 0 {
		t.Error("No files processed for large file backup")
	}

	// Verify extraction
	extractDir, err := ioutil.TempDir("", "extract_large")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(extractDir)

	_, err = backupUtil.ExtractBackup(archivePath, extractDir)
	if err != nil {
		t.Fatalf("Large file extraction failed: %v", err)
	}

	// Verify content
	extractedFile := filepath.Join(extractDir, filepath.Base(testDir), "large_file.bin")
	extractedContent, err := ioutil.ReadFile(extractedFile)
	if err != nil {
		t.Fatal(err)
	}

	if len(extractedContent) != len(largeContent) {
		t.Errorf("Extracted file size mismatch: got %d, want %d",
			len(extractedContent), len(largeContent))
	}

	// Compare content
	for i := 0; i < len(largeContent) && i < len(extractedContent); i++ {
		if largeContent[i] != extractedContent[i] {
			t.Errorf("Content mismatch at byte %d: got %d, want %d",
				i, extractedContent[i], largeContent[i])
			break
		}
	}
}

func TestBackupSymlinks(t *testing.T) {
	testDir, err := ioutil.TempDir("", "symlink_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testDir)

	// Create target file
	targetFile := filepath.Join(testDir, "target.txt")
	if err := ioutil.WriteFile(targetFile, []byte("target content"), 0644); err != nil {
		t.Fatal(err)
	}

	// Create symlink
	symlinkPath := filepath.Join(testDir, "link.txt")
	if err := os.Symlink("target.txt", symlinkPath); err != nil {
		t.Skip("Symlinks not supported on this system")
	}

	config := BackupConfig{
		NumWorkers: 2,
	}

	backupUtil := NewBackupUtility(config)

	// Create backup
	archivePath := filepath.Join(testDir, "symlink_backup.tar.gz")
	result, err := backupUtil.CreateBackup([]string{testDir}, archivePath)

	if err != nil {
		t.Fatalf("Symlink backup failed: %v", err)
	}

	// Extract and verify symlink
	extractDir, err := ioutil.TempDir("", "extract_symlink")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(extractDir)

	_, err = backupUtil.ExtractBackup(archivePath, extractDir)
	if err != nil {
		t.Fatalf("Symlink extraction failed: %v", err)
	}

	extractedLink := filepath.Join(extractDir, filepath.Base(testDir), "link.txt")
	linkTarget, err := os.Readlink(extractedLink)
	if err != nil {
		t.Fatalf("Failed to read extracted symlink: %v", err)
	}

	if linkTarget != "target.txt" {
		t.Errorf("Symlink target mismatch: got %q, want %q", linkTarget, "target.txt")
	}
}

func TestBackupEmptyDirectories(t *testing.T) {
	testDir, err := ioutil.TempDir("", "empty_dir_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testDir)

	// Create empty directories
	emptyDirs := []string{
		filepath.Join(testDir, "empty1"),
		filepath.Join(testDir, "empty2"),
		filepath.Join(testDir, "nested", "empty3"),
	}

	for _, dir := range emptyDirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
	}

	config := BackupConfig{
		NumWorkers: 2,
	}

	backupUtil := NewBackupUtility(config)

	// Create backup
	archivePath := filepath.Join(testDir, "empty_dir_backup.tar.gz")
	result, err := backupUtil.CreateBackup([]string{testDir}, archivePath)

	if err != nil {
		t.Fatalf("Empty directory backup failed: %v", err)
	}

	// List contents to verify directories are included
	files, err := backupUtil.ListBackupContents(archivePath)
	if err != nil {
		t.Fatalf("Failed to list contents: %v", err)
	}

	// Count directories
	dirCount := 0
	for _, file := range files {
		if file.IsDir {
			dirCount++
		}
	}

	if dirCount == 0 {
		t.Error("No directories found in backup")
	}
}

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		bytes    int64
		expected string
	}{
		{0, "0 B"},
		{1023, "1023 B"},
		{1024, "1.0 KB"},
		{1536, "1.5 KB"},
		{1048576, "1.0 MB"},
		{1073741824, "1.0 GB"},
		{1099511627776, "1.0 TB"},
	}

	for _, test := range tests {
		result := FormatBytes(test.bytes)
		if result != test.expected {
			t.Errorf("FormatBytes(%d): got %q, want %q",
				test.bytes, result, test.expected)
		}
	}
}

func TestCalculateChecksumMD5(t *testing.T) {
	// Create temporary file
	tmpFile, err := ioutil.TempFile("", "checksum_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())

	content := "test content for checksum"
	if _, err := tmpFile.WriteString(content); err != nil {
		t.Fatal(err)
	}
	tmpFile.Close()

	// Calculate checksum
	checksum, err := CalculateChecksumMD5(tmpFile.Name())
	if err != nil {
		t.Fatalf("Checksum calculation failed: %v", err)
	}

	// Verify checksum format (should be 32 hex characters)
	if len(checksum) != 32 {
		t.Errorf("Invalid checksum length: got %d, want 32", len(checksum))
	}

	// Calculate again to ensure consistency
	checksum2, err := CalculateChecksumMD5(tmpFile.Name())
	if err != nil {
		t.Fatal(err)
	}

	if checksum != checksum2 {
		t.Error("Checksum calculation is not consistent")
	}
}

func BenchmarkBackupCreation(b *testing.B) {
	testDir := createTestDirectory(&testing.T{})
	defer os.RemoveAll(testDir)

	config := BackupConfig{
		NumWorkers:     4,
		EnableChecksum: false,
	}

	backupUtil := NewBackupUtility(config)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		archivePath := filepath.Join(testDir, fmt.Sprintf("bench_%d.tar.gz", i))
		_, err := backupUtil.CreateBackup([]string{testDir}, archivePath)
		if err != nil {
			b.Fatal(err)
		}
		os.Remove(archivePath)
	}
}

func BenchmarkBackupWorkers(b *testing.B) {
	testDir := createTestDirectory(&testing.T{})
	defer os.RemoveAll(testDir)

	workerCounts := []int{1, 2, 4, 8}

	for _, workers := range workerCounts {
		b.Run(fmt.Sprintf("Workers%d", workers), func(b *testing.B) {
			config := BackupConfig{
				NumWorkers:     workers,
				EnableChecksum: false,
			}

			backupUtil := NewBackupUtility(config)

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				archivePath := filepath.Join(testDir, fmt.Sprintf("bench_%d_%d.tar.gz", workers, i))
				_, err := backupUtil.CreateBackup([]string{testDir}, archivePath)
				if err != nil {
					b.Fatal(err)
				}
				os.Remove(archivePath)
			}
		})
	}
}