package concurrentvcs

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestDefaultRepositoryConfig(t *testing.T) {
	config := DefaultRepositoryConfig()
	
	if config.Name != "repository" {
		t.Errorf("Expected default name 'repository', got '%s'", config.Name)
	}
	
	if config.DefaultBranch != "main" {
		t.Errorf("Expected default branch 'main', got '%s'", config.DefaultBranch)
	}
	
	if config.AllowForcePush {
		t.Error("Expected force push to be disabled by default")
	}
	
	if config.AutoMergeStrategy != ThreeWay {
		t.Errorf("Expected default merge strategy ThreeWay, got %v", config.AutoMergeStrategy)
	}
	
	if config.ConflictResolution != Manual {
		t.Errorf("Expected default conflict resolution Manual, got %v", config.ConflictResolution)
	}
	
	if config.MaxFileSize != 100*1024*1024 {
		t.Errorf("Expected default max file size 100MB, got %d", config.MaxFileSize)
	}
	
	if len(config.IgnorePatterns) == 0 {
		t.Error("Expected default ignore patterns to be set")
	}
}

func TestNewRepository(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	config := DefaultRepositoryConfig()
	config.Name = "test-repository"
	
	repo, err := NewRepository(tempDir, config)
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	if repo == nil {
		t.Fatal("Repository should not be nil")
	}
	
	if repo.workingDir != tempDir {
		t.Errorf("Expected working dir %s, got %s", tempDir, repo.workingDir)
	}
	
	if repo.currentBranch != config.DefaultBranch {
		t.Errorf("Expected current branch %s, got %s", config.DefaultBranch, repo.currentBranch)
	}
	
	// Check if .git directory was created
	gitDir := filepath.Join(tempDir, ".git")
	if _, err := os.Stat(gitDir); os.IsNotExist(err) {
		t.Error("Git directory should be created")
	}
	
	// Check if main branch was created
	if _, exists := repo.branches[config.DefaultBranch]; !exists {
		t.Error("Default branch should be created")
	}
	
	if repo.statistics == nil {
		t.Error("Statistics should be initialized")
	}
	
	if repo.lockManager == nil {
		t.Error("Lock manager should be initialized")
	}
}

func TestInvalidRepositoryCreation(t *testing.T) {
	testCases := []struct {
		name       string
		workingDir string
		config     RepositoryConfig
	}{
		{
			name:       "Empty working directory",
			workingDir: "",
			config:     DefaultRepositoryConfig(),
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRepository(tc.workingDir, tc.config)
			if err == nil {
				t.Error("Expected error for invalid configuration")
			}
		})
	}
}

func TestLoadRepository(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	// Create repository
	config := DefaultRepositoryConfig()
	originalRepo, err := NewRepository(tempDir, config)
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	// Load repository
	loadedRepo, err := LoadRepository(tempDir)
	if err != nil {
		t.Fatalf("Failed to load repository: %v", err)
	}
	
	if loadedRepo.workingDir != originalRepo.workingDir {
		t.Errorf("Loaded repo working dir mismatch")
	}
	
	if loadedRepo.config.Name != originalRepo.config.Name {
		t.Errorf("Loaded repo config mismatch")
	}
}

func TestLoadNonExistentRepository(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-non-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	_, err = LoadRepository(tempDir)
	if err == nil {
		t.Error("Expected error when loading non-existent repository")
	}
}

func TestSessionManagement(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	// Create session
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	if session == nil {
		t.Fatal("Session should not be nil")
	}
	
	if session.UserID != "user1" {
		t.Errorf("Expected user ID 'user1', got '%s'", session.UserID)
	}
	
	if session.UserName != "John Doe" {
		t.Errorf("Expected user name 'John Doe', got '%s'", session.UserName)
	}
	
	if session.UserEmail != "john@example.com" {
		t.Errorf("Expected user email 'john@example.com', got '%s'", session.UserEmail)
	}
	
	if session.ID == "" {
		t.Error("Session ID should not be empty")
	}
	
	// Validate session
	validatedSession, err := repo.validateSession(session.ID)
	if err != nil {
		t.Fatalf("Failed to validate session: %v", err)
	}
	
	if validatedSession.ID != session.ID {
		t.Error("Validated session mismatch")
	}
	
	// Close session
	err = repo.CloseSession(session.ID)
	if err != nil {
		t.Fatalf("Failed to close session: %v", err)
	}
	
	// Validate closed session should fail
	_, err = repo.validateSession(session.ID)
	if err == nil {
		t.Error("Expected error when validating closed session")
	}
}

func TestSessionTimeout(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	// Create session
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Manually set last active time to simulate timeout
	session.LastActive = time.Now().Add(-2 * time.Hour)
	
	// Validate expired session should fail
	_, err = repo.validateSession(session.ID)
	if err == nil {
		t.Error("Expected error when validating expired session")
	}
}

func TestConcurrentSessions(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	// Create multiple sessions concurrently
	numSessions := 10
	sessions := make([]*UserSession, numSessions)
	var wg sync.WaitGroup
	
	for i := 0; i < numSessions; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			
			userID := fmt.Sprintf("user%d", idx)
			userName := fmt.Sprintf("User %d", idx)
			userEmail := fmt.Sprintf("user%d@example.com", idx)
			
			session, err := repo.CreateSession(userID, userName, userEmail)
			if err != nil {
				t.Errorf("Failed to create session %d: %v", idx, err)
				return
			}
			
			sessions[idx] = session
		}(i)
	}
	
	wg.Wait()
	
	// Verify all sessions were created
	for i, session := range sessions {
		if session == nil {
			t.Errorf("Session %d was not created", i)
			continue
		}
		
		// Validate session
		_, err := repo.validateSession(session.ID)
		if err != nil {
			t.Errorf("Failed to validate session %d: %v", i, err)
		}
	}
	
	// Close all sessions concurrently
	for i := 0; i < numSessions; i++ {
		if sessions[i] != nil {
			wg.Add(1)
			go func(session *UserSession) {
				defer wg.Done()
				repo.CloseSession(session.ID)
			}(sessions[i])
		}
	}
	
	wg.Wait()
}

func TestAddFiles(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Create test files
	testFiles := []struct {
		path    string
		content string
	}{
		{"file1.txt", "Hello, World!"},
		{"dir/file2.txt", "Another file"},
		{"README.md", "# Test Repository"},
	}
	
	for _, file := range testFiles {
		fullPath := filepath.Join(tempDir, file.path)
		dir := filepath.Dir(fullPath)
		
		// Create directory if needed
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatalf("Failed to create directory: %v", err)
		}
		
		// Write file
		if err := os.WriteFile(fullPath, []byte(file.content), 0644); err != nil {
			t.Fatalf("Failed to write file: %v", err)
		}
	}
	
	// Add files to repository
	paths := make([]string, len(testFiles))
	for i, file := range testFiles {
		paths[i] = file.path
	}
	
	err = repo.Add(session.ID, paths)
	if err != nil {
		t.Fatalf("Failed to add files: %v", err)
	}
	
	// Verify files are in index
	for _, file := range testFiles {
		entry, exists := repo.index[file.path]
		if !exists {
			t.Errorf("File %s not found in index", file.path)
			continue
		}
		
		if entry.Path != file.path {
			t.Errorf("Index entry path mismatch: expected %s, got %s", file.path, entry.Path)
		}
		
		if string(entry.Content) != file.content {
			t.Errorf("Index entry content mismatch for %s", file.path)
		}
	}
}

func TestAddNonExistentFile(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Try to add non-existent file
	err = repo.Add(session.ID, []string{"nonexistent.txt"})
	if err == nil {
		t.Error("Expected error when adding non-existent file")
	}
}

func TestCommit(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Create and add test file
	testFile := "test.txt"
	testContent := "Test content"
	fullPath := filepath.Join(tempDir, testFile)
	
	if err := os.WriteFile(fullPath, []byte(testContent), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}
	
	err = repo.Add(session.ID, []string{testFile})
	if err != nil {
		t.Fatalf("Failed to add file: %v", err)
	}
	
	// Create commit
	commitMessage := "Initial commit"
	commit, err := repo.Commit(session.ID, commitMessage, nil)
	if err != nil {
		t.Fatalf("Failed to create commit: %v", err)
	}
	
	if commit == nil {
		t.Fatal("Commit should not be nil")
	}
	
	if commit.Message != commitMessage {
		t.Errorf("Expected commit message '%s', got '%s'", commitMessage, commit.Message)
	}
	
	if commit.Author.Name != session.UserName {
		t.Errorf("Expected author name '%s', got '%s'", session.UserName, commit.Author.Name)
	}
	
	if commit.Author.Email != session.UserEmail {
		t.Errorf("Expected author email '%s', got '%s'", session.UserEmail, commit.Author.Email)
	}
	
	if commit.Hash == "" {
		t.Error("Commit hash should not be empty")
	}
	
	// Verify commit is stored
	storedCommit, exists := repo.commits[commit.Hash]
	if !exists {
		t.Error("Commit should be stored in repository")
	}
	
	if storedCommit.Hash != commit.Hash {
		t.Error("Stored commit hash mismatch")
	}
	
	// Verify branch head is updated
	branch := repo.branches[repo.currentBranch]
	if branch.Head != commit.Hash {
		t.Error("Branch head should be updated to new commit")
	}
	
	// Verify index is cleared
	if len(repo.index) != 0 {
		t.Error("Index should be cleared after commit")
	}
	
	// Verify statistics are updated
	stats := repo.GetStatistics()
	if stats.TotalCommits != 1 {
		t.Errorf("Expected 1 commit in statistics, got %d", stats.TotalCommits)
	}
	
	if stats.CommitsByAuthor[session.UserName] != 1 {
		t.Errorf("Expected 1 commit by author, got %d", stats.CommitsByAuthor[session.UserName])
	}
}

func TestCommitWithoutStagedChanges(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Try to commit without staged changes
	_, err = repo.Commit(session.ID, "Empty commit", nil)
	if err == nil {
		t.Error("Expected error when committing without staged changes")
	}
}

func TestCommitWithEmptyMessage(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Create and add test file
	testFile := "test.txt"
	fullPath := filepath.Join(tempDir, testFile)
	
	if err := os.WriteFile(fullPath, []byte("content"), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}
	
	err = repo.Add(session.ID, []string{testFile})
	if err != nil {
		t.Fatalf("Failed to add file: %v", err)
	}
	
	// Try to commit with empty message
	_, err = repo.Commit(session.ID, "", nil)
	if err == nil {
		t.Error("Expected error when committing with empty message")
	}
}

func TestBranchManagement(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Create initial commit
	testFile := "test.txt"
	fullPath := filepath.Join(tempDir, testFile)
	
	if err := os.WriteFile(fullPath, []byte("content"), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}
	
	err = repo.Add(session.ID, []string{testFile})
	if err != nil {
		t.Fatalf("Failed to add file: %v", err)
	}
	
	commit, err := repo.Commit(session.ID, "Initial commit", nil)
	if err != nil {
		t.Fatalf("Failed to create commit: %v", err)
	}
	
	// Create new branch
	newBranchName := "feature-branch"
	err = repo.CreateBranch(session.ID, newBranchName, "")
	if err != nil {
		t.Fatalf("Failed to create branch: %v", err)
	}
	
	// Verify branch was created
	branch, exists := repo.branches[newBranchName]
	if !exists {
		t.Error("Branch should be created")
	}
	
	if branch.Name != newBranchName {
		t.Errorf("Expected branch name '%s', got '%s'", newBranchName, branch.Name)
	}
	
	if branch.Head != commit.Hash {
		t.Error("New branch should point to current commit")
	}
	
	if branch.CreatedBy != session.UserID {
		t.Errorf("Expected branch creator '%s', got '%s'", session.UserID, branch.CreatedBy)
	}
	
	// Switch to new branch
	err = repo.SwitchBranch(session.ID, newBranchName)
	if err != nil {
		t.Fatalf("Failed to switch branch: %v", err)
	}
	
	if repo.currentBranch != newBranchName {
		t.Errorf("Expected current branch '%s', got '%s'", newBranchName, repo.currentBranch)
	}
	
	// Switch back to main branch
	err = repo.SwitchBranch(session.ID, "main")
	if err != nil {
		t.Fatalf("Failed to switch back to main: %v", err)
	}
	
	if repo.currentBranch != "main" {
		t.Error("Should be back on main branch")
	}
}

func TestCreateDuplicateBranch(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Try to create branch with same name as default branch
	err = repo.CreateBranch(session.ID, "main", "")
	if err == nil {
		t.Error("Expected error when creating duplicate branch")
	}
}

func TestSwitchToNonExistentBranch(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Try to switch to non-existent branch
	err = repo.SwitchBranch(session.ID, "nonexistent")
	if err == nil {
		t.Error("Expected error when switching to non-existent branch")
	}
}

func TestStatus(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Create test files
	trackedFile := "tracked.txt"
	untrackedFile := "untracked.txt"
	
	// Create and track one file
	trackedPath := filepath.Join(tempDir, trackedFile)
	if err := os.WriteFile(trackedPath, []byte("tracked content"), 0644); err != nil {
		t.Fatalf("Failed to write tracked file: %v", err)
	}
	
	err = repo.Add(session.ID, []string{trackedFile})
	if err != nil {
		t.Fatalf("Failed to add tracked file: %v", err)
	}
	
	// Create untracked file
	untrackedPath := filepath.Join(tempDir, untrackedFile)
	if err := os.WriteFile(untrackedPath, []byte("untracked content"), 0644); err != nil {
		t.Fatalf("Failed to write untracked file: %v", err)
	}
	
	// Get status
	status, err := repo.Status(session.ID)
	if err != nil {
		t.Fatalf("Failed to get status: %v", err)
	}
	
	if status == nil {
		t.Fatal("Status should not be nil")
	}
	
	if status.Branch != repo.currentBranch {
		t.Errorf("Expected branch '%s', got '%s'", repo.currentBranch, status.Branch)
	}
	
	if status.Clean {
		t.Error("Repository should not be clean with staged changes")
	}
	
	// Check staged files
	if _, exists := status.Staged[trackedFile]; !exists {
		t.Error("Tracked file should be in staged files")
	}
	
	// Check untracked files
	found := false
	for _, file := range status.Untracked {
		if file == untrackedFile {
			found = true
			break
		}
	}
	if !found {
		t.Error("Untracked file should be in untracked files list")
	}
}

func TestLog(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Create multiple commits
	commits := make([]*Commit, 3)
	for i := 0; i < 3; i++ {
		testFile := fmt.Sprintf("file%d.txt", i)
		fullPath := filepath.Join(tempDir, testFile)
		
		if err := os.WriteFile(fullPath, []byte(fmt.Sprintf("content %d", i)), 0644); err != nil {
			t.Fatalf("Failed to write test file: %v", err)
		}
		
		err = repo.Add(session.ID, []string{testFile})
		if err != nil {
			t.Fatalf("Failed to add file: %v", err)
		}
		
		commit, err := repo.Commit(session.ID, fmt.Sprintf("Commit %d", i), nil)
		if err != nil {
			t.Fatalf("Failed to create commit: %v", err)
		}
		
		commits[i] = commit
	}
	
	// Get log
	log, err := repo.Log(session.ID, "", 0)
	if err != nil {
		t.Fatalf("Failed to get log: %v", err)
	}
	
	if len(log) != 3 {
		t.Errorf("Expected 3 commits in log, got %d", len(log))
	}
	
	// Commits should be in reverse chronological order (newest first)
	for i, commit := range log {
		expectedIndex := len(commits) - 1 - i
		if commit.Hash != commits[expectedIndex].Hash {
			t.Errorf("Log commit %d hash mismatch", i)
		}
	}
	
	// Test log with limit
	limitedLog, err := repo.Log(session.ID, "", 2)
	if err != nil {
		t.Fatalf("Failed to get limited log: %v", err)
	}
	
	if len(limitedLog) != 2 {
		t.Errorf("Expected 2 commits in limited log, got %d", len(limitedLog))
	}
}

func TestConcurrentOperations(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	// Create multiple sessions
	numSessions := 5
	sessions := make([]*UserSession, numSessions)
	
	for i := 0; i < numSessions; i++ {
		session, err := repo.CreateSession(
			fmt.Sprintf("user%d", i),
			fmt.Sprintf("User %d", i),
			fmt.Sprintf("user%d@example.com", i),
		)
		if err != nil {
			t.Fatalf("Failed to create session %d: %v", i, err)
		}
		sessions[i] = session
	}
	
	// Perform concurrent operations
	var wg sync.WaitGroup
	errors := make(chan error, numSessions)
	
	for i := 0; i < numSessions; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			
			session := sessions[idx]
			
			// Create unique file for this session
			fileName := fmt.Sprintf("file_%d.txt", idx)
			filePath := filepath.Join(tempDir, fileName)
			content := fmt.Sprintf("Content from session %d", idx)
			
			// Write file
			if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
				errors <- fmt.Errorf("session %d: failed to write file: %v", idx, err)
				return
			}
			
			// Add file
			if err := repo.Add(session.ID, []string{fileName}); err != nil {
				errors <- fmt.Errorf("session %d: failed to add file: %v", idx, err)
				return
			}
			
			// Commit
			commitMsg := fmt.Sprintf("Commit from session %d", idx)
			if _, err := repo.Commit(session.ID, commitMsg, nil); err != nil {
				errors <- fmt.Errorf("session %d: failed to commit: %v", idx, err)
				return
			}
			
			// Get status
			if _, err := repo.Status(session.ID); err != nil {
				errors <- fmt.Errorf("session %d: failed to get status: %v", idx, err)
				return
			}
		}(i)
	}
	
	wg.Wait()
	close(errors)
	
	// Check for errors
	for err := range errors {
		t.Error(err)
	}
	
	// Verify all commits were created
	stats := repo.GetStatistics()
	if stats.TotalCommits != int64(numSessions) {
		t.Errorf("Expected %d commits, got %d", numSessions, stats.TotalCommits)
	}
}

func TestLockManager(t *testing.T) {
	lm := NewLockManager()
	
	// Test acquiring read lock
	err := lm.AcquireLock("resource1", "read", "session1")
	if err != nil {
		t.Fatalf("Failed to acquire read lock: %v", err)
	}
	
	// Test acquiring another read lock on same resource
	err = lm.AcquireLock("resource1", "read", "session2")
	if err != nil {
		t.Fatalf("Failed to acquire second read lock: %v", err)
	}
	
	// Test releasing lock
	err = lm.ReleaseLock("resource1", "session1")
	if err != nil {
		t.Fatalf("Failed to release lock: %v", err)
	}
	
	// Test releasing non-existent lock
	err = lm.ReleaseLock("nonexistent", "session1")
	if err == nil {
		t.Error("Expected error when releasing non-existent lock")
	}
	
	// Test releasing lock not owned by session
	err = lm.ReleaseLock("resource1", "session3")
	if err == nil {
		t.Error("Expected error when releasing lock not owned by session")
	}
}

func TestTransactionManagement(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Begin transaction
	tx, err := repo.beginTransaction(session.ID, "test")
	if err != nil {
		t.Fatalf("Failed to begin transaction: %v", err)
	}
	
	if tx == nil {
		t.Fatal("Transaction should not be nil")
	}
	
	if tx.SessionID != session.ID {
		t.Error("Transaction session ID mismatch")
	}
	
	if tx.Type != "test" {
		t.Error("Transaction type mismatch")
	}
	
	if tx.State != "active" {
		t.Error("Transaction should be active")
	}
	
	// Commit transaction
	err = repo.commitTransaction(tx)
	if err != nil {
		t.Fatalf("Failed to commit transaction: %v", err)
	}
	
	if tx.State != "committed" {
		t.Error("Transaction should be committed")
	}
}

func TestEventLogging(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Perform some operations that should generate events
	testFile := "test.txt"
	fullPath := filepath.Join(tempDir, testFile)
	
	if err := os.WriteFile(fullPath, []byte("content"), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}
	
	err = repo.Add(session.ID, []string{testFile})
	if err != nil {
		t.Fatalf("Failed to add file: %v", err)
	}
	
	_, err = repo.Commit(session.ID, "Test commit", nil)
	if err != nil {
		t.Fatalf("Failed to commit: %v", err)
	}
	
	// Get event log
	events, err := repo.GetEventLog(session.ID, 10)
	if err != nil {
		t.Fatalf("Failed to get event log: %v", err)
	}
	
	if len(events) == 0 {
		t.Error("Expected events to be logged")
	}
	
	// Check for expected event types
	eventTypes := make(map[string]bool)
	for _, event := range events {
		eventTypes[event.Type] = true
	}
	
	expectedEvents := []string{"session_created", "files_added", "commit_created"}
	for _, expected := range expectedEvents {
		if !eventTypes[expected] {
			t.Errorf("Expected event type '%s' not found", expected)
		}
	}
}

func TestRepositoryStatistics(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	
	// Initial statistics
	stats := repo.GetStatistics()
	if stats.TotalCommits != 0 {
		t.Errorf("Expected 0 initial commits, got %d", stats.TotalCommits)
	}
	
	if stats.TotalBranches != 1 { // main branch
		t.Errorf("Expected 1 initial branch, got %d", stats.TotalBranches)
	}
	
	// Create commit
	testFile := "test.txt"
	fullPath := filepath.Join(tempDir, testFile)
	
	if err := os.WriteFile(fullPath, []byte("content"), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}
	
	err = repo.Add(session.ID, []string{testFile})
	if err != nil {
		t.Fatalf("Failed to add file: %v", err)
	}
	
	commit, err := repo.Commit(session.ID, "Test commit", nil)
	if err != nil {
		t.Fatalf("Failed to commit: %v", err)
	}
	
	// Check updated statistics
	stats = repo.GetStatistics()
	if stats.TotalCommits != 1 {
		t.Errorf("Expected 1 commit after commit, got %d", stats.TotalCommits)
	}
	
	if stats.CommitsByAuthor[commit.Author.Name] != 1 {
		t.Errorf("Expected 1 commit by author, got %d", stats.CommitsByAuthor[commit.Author.Name])
	}
	
	// Create branch
	err = repo.CreateBranch(session.ID, "feature", "")
	if err != nil {
		t.Fatalf("Failed to create branch: %v", err)
	}
	
	stats = repo.GetStatistics()
	if stats.TotalBranches != 2 {
		t.Errorf("Expected 2 branches after creating branch, got %d", stats.TotalBranches)
	}
}

func TestInvalidSessionOperations(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test-repo")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		t.Fatalf("Failed to create repository: %v", err)
	}
	
	invalidSessionID := "invalid-session"
	
	// Test operations with invalid session
	testCases := []struct {
		name string
		op   func() error
	}{
		{
			name: "Add with invalid session",
			op: func() error {
				return repo.Add(invalidSessionID, []string{"test.txt"})
			},
		},
		{
			name: "Commit with invalid session",
			op: func() error {
				_, err := repo.Commit(invalidSessionID, "test", nil)
				return err
			},
		},
		{
			name: "CreateBranch with invalid session",
			op: func() error {
				return repo.CreateBranch(invalidSessionID, "test", "")
			},
		},
		{
			name: "SwitchBranch with invalid session",
			op: func() error {
				return repo.SwitchBranch(invalidSessionID, "main")
			},
		},
		{
			name: "Status with invalid session",
			op: func() error {
				_, err := repo.Status(invalidSessionID)
				return err
			},
		},
		{
			name: "Log with invalid session",
			op: func() error {
				_, err := repo.Log(invalidSessionID, "", 0)
				return err
			},
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.op()
			if err == nil {
				t.Error("Expected error for operation with invalid session")
			}
		})
	}
}

// Benchmark tests

func BenchmarkCreateSession(b *testing.B) {
	tempDir, err := ioutil.TempDir("", "bench-repo")
	if err != nil {
		b.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		b.Fatalf("Failed to create repository: %v", err)
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		userID := fmt.Sprintf("user%d", i)
		userName := fmt.Sprintf("User %d", i)
		userEmail := fmt.Sprintf("user%d@example.com", i)
		
		session, err := repo.CreateSession(userID, userName, userEmail)
		if err != nil {
			b.Fatalf("Failed to create session: %v", err)
		}
		
		// Clean up
		repo.CloseSession(session.ID)
	}
}

func BenchmarkAddFiles(b *testing.B) {
	tempDir, err := ioutil.TempDir("", "bench-repo")
	if err != nil {
		b.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		b.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		b.Fatalf("Failed to create session: %v", err)
	}
	
	// Pre-create test files
	for i := 0; i < b.N; i++ {
		fileName := fmt.Sprintf("file%d.txt", i)
		filePath := filepath.Join(tempDir, fileName)
		content := fmt.Sprintf("Content %d", i)
		
		if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
			b.Fatalf("Failed to write test file: %v", err)
		}
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		fileName := fmt.Sprintf("file%d.txt", i)
		err := repo.Add(session.ID, []string{fileName})
		if err != nil {
			b.Fatalf("Failed to add file: %v", err)
		}
	}
}

func BenchmarkCommit(b *testing.B) {
	tempDir, err := ioutil.TempDir("", "bench-repo")
	if err != nil {
		b.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	repo, err := NewRepository(tempDir, DefaultRepositoryConfig())
	if err != nil {
		b.Fatalf("Failed to create repository: %v", err)
	}
	
	session, err := repo.CreateSession("user1", "John Doe", "john@example.com")
	if err != nil {
		b.Fatalf("Failed to create session: %v", err)
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		// Create and stage a file
		fileName := fmt.Sprintf("file%d.txt", i)
		filePath := filepath.Join(tempDir, fileName)
		content := fmt.Sprintf("Content %d", i)
		
		if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
			b.Fatalf("Failed to write test file: %v", err)
		}
		
		err := repo.Add(session.ID, []string{fileName})
		if err != nil {
			b.Fatalf("Failed to add file: %v", err)
		}
		
		// Commit
		commitMsg := fmt.Sprintf("Commit %d", i)
		_, err = repo.Commit(session.ID, commitMsg, nil)
		if err != nil {
			b.Fatalf("Failed to commit: %v", err)
		}
	}
}