package concurrentvcs

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// FileStatus represents the status of a file in the working directory
type FileStatus int

const (
	Untracked FileStatus = iota
	Modified
	Added
	Deleted
	Renamed
	Copied
	Unchanged
	Conflicted
)

// MergeStrategy defines different merge strategies
type MergeStrategy int

const (
	FastForward MergeStrategy = iota
	ThreeWay
	Octopus
	Ours
	Theirs
	Recursive
)

// ConflictResolution defines how to resolve conflicts
type ConflictResolution int

const (
	Manual ConflictResolution = iota
	AutoOurs
	AutoTheirs
	AutoMerge
)

// FileEntry represents a file in the repository
type FileEntry struct {
	Path         string    `json:"path"`
	Hash         string    `json:"hash"`
	Size         int64     `json:"size"`
	Mode         os.FileMode `json:"mode"`
	LastModified time.Time `json:"last_modified"`
	Content      []byte    `json:"content,omitempty"`
}

// TreeEntry represents a directory tree
type TreeEntry struct {
	Name     string       `json:"name"`
	Type     string       `json:"type"` // "file", "tree"
	Hash     string       `json:"hash"`
	Mode     os.FileMode  `json:"mode"`
	Size     int64        `json:"size,omitempty"`
	Children []*TreeEntry `json:"children,omitempty"`
}

// Commit represents a single commit in the repository
type Commit struct {
	Hash        string            `json:"hash"`
	Tree        string            `json:"tree"`
	Parents     []string          `json:"parents"`
	Author      *Author           `json:"author"`
	Committer   *Author           `json:"committer"`
	Message     string            `json:"message"`
	Timestamp   time.Time         `json:"timestamp"`
	Files       map[string]string `json:"files"` // path -> hash
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// Author represents commit author information
type Author struct {
	Name      string    `json:"name"`
	Email     string    `json:"email"`
	Timestamp time.Time `json:"timestamp"`
}

// Branch represents a branch in the repository
type Branch struct {
	Name      string    `json:"name"`
	Head      string    `json:"head"`
	CreatedAt time.Time `json:"created_at"`
	CreatedBy string    `json:"created_by"`
	Protected bool      `json:"protected"`
	Remote    string    `json:"remote,omitempty"`
}

// Tag represents a tag pointing to a specific commit
type Tag struct {
	Name      string    `json:"name"`
	Commit    string    `json:"commit"`
	Message   string    `json:"message"`
	Tagger    *Author   `json:"tagger"`
	Timestamp time.Time `json:"timestamp"`
}

// Conflict represents a merge conflict
type Conflict struct {
	Path        string   `json:"path"`
	OurContent  []byte   `json:"our_content"`
	TheirContent []byte  `json:"their_content"`
	BaseContent []byte   `json:"base_content,omitempty"`
	Resolved    bool     `json:"resolved"`
}

// MergeResult represents the result of a merge operation
type MergeResult struct {
	Success     bool         `json:"success"`
	Commit      *Commit      `json:"commit,omitempty"`
	Conflicts   []*Conflict  `json:"conflicts,omitempty"`
	Strategy    MergeStrategy `json:"strategy"`
	Message     string       `json:"message"`
}

// Status represents the current repository status
type Status struct {
	Branch        string                `json:"branch"`
	Staged        map[string]FileStatus `json:"staged"`
	Modified      map[string]FileStatus `json:"modified"`
	Untracked     []string              `json:"untracked"`
	Conflicts     []string              `json:"conflicts"`
	Clean         bool                  `json:"clean"`
}

// RemoteRepository represents a remote repository
type RemoteRepository struct {
	Name     string `json:"name"`
	URL      string `json:"url"`
	Username string `json:"username,omitempty"`
	Password string `json:"password,omitempty"`
}

// RepositoryConfig holds repository configuration
type RepositoryConfig struct {
	Name               string                      `json:"name"`
	Description        string                      `json:"description"`
	DefaultBranch      string                      `json:"default_branch"`
	AllowForcePush     bool                        `json:"allow_force_push"`
	RequireSignedOff   bool                        `json:"require_signed_off"`
	AutoMergeStrategy  MergeStrategy               `json:"auto_merge_strategy"`
	ConflictResolution ConflictResolution          `json:"conflict_resolution"`
	Remotes            map[string]*RemoteRepository `json:"remotes"`
	Hooks              map[string]string           `json:"hooks,omitempty"`
	MaxFileSize        int64                       `json:"max_file_size"`
	IgnorePatterns     []string                    `json:"ignore_patterns"`
}

// RepositoryStatistics holds repository statistics
type RepositoryStatistics struct {
	TotalCommits     int64                    `json:"total_commits"`
	TotalBranches    int64                    `json:"total_branches"`
	TotalTags        int64                    `json:"total_tags"`
	TotalFiles       int64                    `json:"total_files"`
	TotalSize        int64                    `json:"total_size"`
	ActiveUsers      int64                    `json:"active_users"`
	CommitsByAuthor  map[string]int64         `json:"commits_by_author"`
	CommitsByDate    map[string]int64         `json:"commits_by_date"`
	FilesByExtension map[string]int64         `json:"files_by_extension"`
	LanguageStats    map[string]int64         `json:"language_stats"`
	BranchActivity   map[string]time.Time     `json:"branch_activity"`
	mutex            sync.RWMutex
}

// Repository represents the main version control repository
type Repository struct {
	config       RepositoryConfig
	workingDir   string
	gitDir       string
	currentBranch string
	
	// Storage
	commits      map[string]*Commit      // hash -> commit
	branches     map[string]*Branch      // name -> branch
	tags         map[string]*Tag         // name -> tag
	objects      map[string][]byte       // hash -> content
	index        map[string]*FileEntry   // staged files
	
	// Concurrent access control
	mutex        sync.RWMutex
	lockManager  *LockManager
	
	// User sessions and transactions
	sessions     map[string]*UserSession // session_id -> session
	transactions map[string]*Transaction // tx_id -> transaction
	
	// Statistics and monitoring
	statistics   *RepositoryStatistics
	eventLog     []*RepositoryEvent
	
	// Background operations
	gcRunning    int32
	packRunning  int32
}

// UserSession represents a user session
type UserSession struct {
	ID          string    `json:"id"`
	UserID      string    `json:"user_id"`
	UserName    string    `json:"user_name"`
	UserEmail   string    `json:"user_email"`
	StartTime   time.Time `json:"start_time"`
	LastActive  time.Time `json:"last_active"`
	WorkingDir  string    `json:"working_dir"`
	CurrentTx   string    `json:"current_transaction,omitempty"`
	Permissions []string  `json:"permissions"`
	mutex       sync.RWMutex
}

// Transaction represents a transaction for atomic operations
type Transaction struct {
	ID          string                  `json:"id"`
	SessionID   string                  `json:"session_id"`
	Type        string                  `json:"type"` // "commit", "merge", "rebase"
	StartTime   time.Time               `json:"start_time"`
	Operations  []*TransactionOperation `json:"operations"`
	State       string                  `json:"state"` // "active", "committed", "aborted"
	LockSet     map[string]string       `json:"lock_set"` // resource -> lock_type
	mutex       sync.RWMutex
}

// TransactionOperation represents an operation within a transaction
type TransactionOperation struct {
	Type      string                 `json:"type"`
	Target    string                 `json:"target"`
	OldValue  interface{}            `json:"old_value,omitempty"`
	NewValue  interface{}            `json:"new_value,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
}

// LockManager handles concurrent access control
type LockManager struct {
	locks     map[string]*ResourceLock
	waitQueue map[string][]*LockRequest
	mutex     sync.RWMutex
}

// ResourceLock represents a lock on a repository resource
type ResourceLock struct {
	Resource    string    `json:"resource"`
	Type        string    `json:"type"` // "read", "write", "exclusive"
	Owner       string    `json:"owner"`
	SessionID   string    `json:"session_id"`
	AcquiredAt  time.Time `json:"acquired_at"`
	ExpiresAt   time.Time `json:"expires_at,omitempty"`
}

// LockRequest represents a pending lock request
type LockRequest struct {
	Resource   string    `json:"resource"`
	Type       string    `json:"type"`
	SessionID  string    `json:"session_id"`
	RequestedAt time.Time `json:"requested_at"`
	Result     chan error
}

// RepositoryEvent represents an event in the repository
type RepositoryEvent struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	SessionID string                 `json:"session_id"`
	UserID    string                 `json:"user_id"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
}

// DefaultRepositoryConfig returns a default repository configuration
func DefaultRepositoryConfig() RepositoryConfig {
	return RepositoryConfig{
		Name:               "repository",
		Description:        "A concurrent version control repository",
		DefaultBranch:      "main",
		AllowForcePush:     false,
		RequireSignedOff:   false,
		AutoMergeStrategy:  ThreeWay,
		ConflictResolution: Manual,
		Remotes:            make(map[string]*RemoteRepository),
		MaxFileSize:        100 * 1024 * 1024, // 100MB
		IgnorePatterns:     []string{".git", "*.tmp", "*.log"},
	}
}

// NewRepository creates a new repository
func NewRepository(workingDir string, config RepositoryConfig) (*Repository, error) {
	if workingDir == "" {
		return nil, errors.New("working directory cannot be empty")
	}
	
	// Create directories
	gitDir := filepath.Join(workingDir, ".git")
	if err := os.MkdirAll(gitDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create git directory: %v", err)
	}
	
	repo := &Repository{
		config:       config,
		workingDir:   workingDir,
		gitDir:       gitDir,
		currentBranch: config.DefaultBranch,
		commits:      make(map[string]*Commit),
		branches:     make(map[string]*Branch),
		tags:         make(map[string]*Tag),
		objects:      make(map[string][]byte),
		index:        make(map[string]*FileEntry),
		sessions:     make(map[string]*UserSession),
		transactions: make(map[string]*Transaction),
		lockManager:  NewLockManager(),
		statistics:   &RepositoryStatistics{
			CommitsByAuthor:  make(map[string]int64),
			CommitsByDate:    make(map[string]int64),
			FilesByExtension: make(map[string]int64),
			LanguageStats:    make(map[string]int64),
			BranchActivity:   make(map[string]time.Time),
		},
		eventLog:     make([]*RepositoryEvent, 0),
	}
	
	// Create initial branch
	mainBranch := &Branch{
		Name:      config.DefaultBranch,
		Head:      "", // Will be set on first commit
		CreatedAt: time.Now(),
		CreatedBy: "system",
		Protected: true,
	}
	repo.branches[config.DefaultBranch] = mainBranch
	
	// Save repository metadata
	if err := repo.saveMetadata(); err != nil {
		return nil, fmt.Errorf("failed to save repository metadata: %v", err)
	}
	
	return repo, nil
}

// LoadRepository loads an existing repository
func LoadRepository(workingDir string) (*Repository, error) {
	gitDir := filepath.Join(workingDir, ".git")
	if _, err := os.Stat(gitDir); os.IsNotExist(err) {
		return nil, errors.New("not a git repository")
	}
	
	repo := &Repository{
		workingDir:   workingDir,
		gitDir:       gitDir,
		commits:      make(map[string]*Commit),
		branches:     make(map[string]*Branch),
		tags:         make(map[string]*Tag),
		objects:      make(map[string][]byte),
		index:        make(map[string]*FileEntry),
		sessions:     make(map[string]*UserSession),
		transactions: make(map[string]*Transaction),
		lockManager:  NewLockManager(),
		statistics:   &RepositoryStatistics{
			CommitsByAuthor:  make(map[string]int64),
			CommitsByDate:    make(map[string]int64),
			FilesByExtension: make(map[string]int64),
			LanguageStats:    make(map[string]int64),
			BranchActivity:   make(map[string]time.Time),
		},
		eventLog:     make([]*RepositoryEvent, 0),
	}
	
	// Load repository metadata
	if err := repo.loadMetadata(); err != nil {
		return nil, fmt.Errorf("failed to load repository metadata: %v", err)
	}
	
	return repo, nil
}

// NewLockManager creates a new lock manager
func NewLockManager() *LockManager {
	return &LockManager{
		locks:     make(map[string]*ResourceLock),
		waitQueue: make(map[string][]*LockRequest),
	}
}

// CreateSession creates a new user session
func (r *Repository) CreateSession(userID, userName, userEmail string) (*UserSession, error) {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	sessionID := generateHash(fmt.Sprintf("%s-%s-%d", userID, userName, time.Now().UnixNano()))
	
	session := &UserSession{
		ID:          sessionID,
		UserID:      userID,
		UserName:    userName,
		UserEmail:   userEmail,
		StartTime:   time.Now(),
		LastActive:  time.Now(),
		WorkingDir:  r.workingDir,
		Permissions: []string{"read", "write"}, // Default permissions
	}
	
	r.sessions[sessionID] = session
	
	// Log event
	r.logEvent("session_created", sessionID, userID, map[string]interface{}{
		"user_name":  userName,
		"user_email": userEmail,
	})
	
	return session, nil
}

// CloseSession closes a user session
func (r *Repository) CloseSession(sessionID string) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	session, exists := r.sessions[sessionID]
	if !exists {
		return errors.New("session not found")
	}
	
	// Abort any active transaction
	if session.CurrentTx != "" {
		if tx, exists := r.transactions[session.CurrentTx]; exists {
			r.abortTransaction(tx)
		}
	}
	
	// Release all locks held by this session
	r.lockManager.ReleaseSessionLocks(sessionID)
	
	delete(r.sessions, sessionID)
	
	// Log event
	r.logEvent("session_closed", sessionID, session.UserID, nil)
	
	return nil
}

// Add adds files to the staging area
func (r *Repository) Add(sessionID string, paths []string) error {
	session, err := r.validateSession(sessionID)
	if err != nil {
		return err
	}
	
	// Acquire write lock
	if err := r.lockManager.AcquireLock("index", "write", sessionID); err != nil {
		return err
	}
	defer r.lockManager.ReleaseLock("index", sessionID)
	
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	for _, path := range paths {
		fullPath := filepath.Join(r.workingDir, path)
		
		// Check if file exists
		info, err := os.Stat(fullPath)
		if err != nil {
			if os.IsNotExist(err) {
				// File deleted, stage deletion
				r.index[path] = &FileEntry{
					Path: path,
					Hash: "",
					Size: 0,
					Mode: 0,
					LastModified: time.Now(),
				}
				continue
			}
			return fmt.Errorf("failed to stat file %s: %v", path, err)
		}
		
		// Read file content
		content, err := os.ReadFile(fullPath)
		if err != nil {
			return fmt.Errorf("failed to read file %s: %v", path, err)
		}
		
		// Check file size limit
		if r.config.MaxFileSize > 0 && info.Size() > r.config.MaxFileSize {
			return fmt.Errorf("file %s exceeds maximum size limit", path)
		}
		
		// Calculate hash
		hash := generateHash(string(content))
		
		// Store object
		r.objects[hash] = content
		
		// Add to index
		r.index[path] = &FileEntry{
			Path:         path,
			Hash:         hash,
			Size:         info.Size(),
			Mode:         info.Mode(),
			LastModified: info.ModTime(),
			Content:      content,
		}
	}
	
	// Update session activity
	session.LastActive = time.Now()
	
	// Log event
	r.logEvent("files_added", sessionID, session.UserID, map[string]interface{}{
		"paths": paths,
	})
	
	return nil
}

// Commit creates a new commit
func (r *Repository) Commit(sessionID, message string, author *Author) (*Commit, error) {
	session, err := r.validateSession(sessionID)
	if err != nil {
		return nil, err
	}
	
	if message == "" {
		return nil, errors.New("commit message cannot be empty")
	}
	
	// Start transaction
	tx, err := r.beginTransaction(sessionID, "commit")
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			r.abortTransaction(tx)
		}
	}()
	
	// Acquire locks
	if err := r.lockManager.AcquireLock("commits", "write", sessionID); err != nil {
		return nil, err
	}
	defer r.lockManager.ReleaseLock("commits", sessionID)
	
	if err := r.lockManager.AcquireLock("branches", "write", sessionID); err != nil {
		return nil, err
	}
	defer r.lockManager.ReleaseLock("branches", sessionID)
	
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	// Check if there are staged changes
	if len(r.index) == 0 {
		return nil, errors.New("no changes staged for commit")
	}
	
	// Get current branch
	branch, exists := r.branches[r.currentBranch]
	if !exists {
		return nil, fmt.Errorf("branch %s not found", r.currentBranch)
	}
	
	// Set default author if not provided
	if author == nil {
		author = &Author{
			Name:      session.UserName,
			Email:     session.UserEmail,
			Timestamp: time.Now(),
		}
	}
	
	// Create tree from index
	tree, err := r.createTree()
	if err != nil {
		return nil, err
	}
	
	// Get parent commits
	var parents []string
	if branch.Head != "" {
		parents = append(parents, branch.Head)
	}
	
	// Create commit
	commit := &Commit{
		Tree:      tree,
		Parents:   parents,
		Author:    author,
		Committer: author,
		Message:   message,
		Timestamp: time.Now(),
		Files:     make(map[string]string),
		Metadata:  make(map[string]string),
	}
	
	// Add files from index
	for path, entry := range r.index {
		commit.Files[path] = entry.Hash
	}
	
	// Calculate commit hash
	commitData, _ := json.Marshal(commit)
	commit.Hash = generateHash(string(commitData))
	
	// Store commit
	r.commits[commit.Hash] = commit
	
	// Update branch head
	branch.Head = commit.Hash
	r.branches[r.currentBranch] = branch
	
	// Clear index
	r.index = make(map[string]*FileEntry)
	
	// Update statistics
	r.updateStatistics(commit)
	
	// Commit transaction
	if err := r.commitTransaction(tx); err != nil {
		return nil, err
	}
	
	// Update session activity
	session.LastActive = time.Now()
	
	// Log event
	r.logEvent("commit_created", sessionID, session.UserID, map[string]interface{}{
		"commit_hash": commit.Hash,
		"message":     message,
		"branch":      r.currentBranch,
	})
	
	return commit, nil
}

// CreateBranch creates a new branch
func (r *Repository) CreateBranch(sessionID, branchName string, startPoint string) error {
	session, err := r.validateSession(sessionID)
	if err != nil {
		return err
	}
	
	if branchName == "" {
		return errors.New("branch name cannot be empty")
	}
	
	// Acquire write lock
	if err := r.lockManager.AcquireLock("branches", "write", sessionID); err != nil {
		return err
	}
	defer r.lockManager.ReleaseLock("branches", sessionID)
	
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	// Check if branch already exists
	if _, exists := r.branches[branchName]; exists {
		return fmt.Errorf("branch %s already exists", branchName)
	}
	
	// Determine start point
	var startCommit string
	if startPoint == "" {
		// Use current branch head
		if currentBranch, exists := r.branches[r.currentBranch]; exists {
			startCommit = currentBranch.Head
		}
	} else {
		// Validate start point (could be commit hash or branch name)
		if _, exists := r.commits[startPoint]; exists {
			startCommit = startPoint
		} else if branch, exists := r.branches[startPoint]; exists {
			startCommit = branch.Head
		} else {
			return fmt.Errorf("invalid start point: %s", startPoint)
		}
	}
	
	// Create branch
	branch := &Branch{
		Name:      branchName,
		Head:      startCommit,
		CreatedAt: time.Now(),
		CreatedBy: session.UserID,
		Protected: false,
	}
	
	r.branches[branchName] = branch
	atomic.AddInt64(&r.statistics.TotalBranches, 1)
	
	// Update session activity
	session.LastActive = time.Now()
	
	// Log event
	r.logEvent("branch_created", sessionID, session.UserID, map[string]interface{}{
		"branch_name": branchName,
		"start_point": startPoint,
		"start_commit": startCommit,
	})
	
	return nil
}

// SwitchBranch switches to a different branch
func (r *Repository) SwitchBranch(sessionID, branchName string) error {
	session, err := r.validateSession(sessionID)
	if err != nil {
		return err
	}
	
	// Acquire read lock
	if err := r.lockManager.AcquireLock("branches", "read", sessionID); err != nil {
		return err
	}
	defer r.lockManager.ReleaseLock("branches", sessionID)
	
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	// Check if branch exists
	branch, exists := r.branches[branchName]
	if !exists {
		return fmt.Errorf("branch %s not found", branchName)
	}
	
	// Check for uncommitted changes
	if len(r.index) > 0 {
		return errors.New("you have uncommitted changes, please commit or stash them first")
	}
	
	// Switch branch
	oldBranch := r.currentBranch
	r.currentBranch = branchName
	
	// Update working directory to match branch head
	if err := r.checkoutCommit(branch.Head); err != nil {
		// Revert branch switch on error
		r.currentBranch = oldBranch
		return fmt.Errorf("failed to checkout branch: %v", err)
	}
	
	// Update branch activity
	r.statistics.BranchActivity[branchName] = time.Now()
	
	// Update session activity
	session.LastActive = time.Now()
	
	// Log event
	r.logEvent("branch_switched", sessionID, session.UserID, map[string]interface{}{
		"from_branch": oldBranch,
		"to_branch":   branchName,
	})
	
	return nil
}

// Merge merges another branch into the current branch
func (r *Repository) Merge(sessionID, sourceBranch string, strategy MergeStrategy) (*MergeResult, error) {
	session, err := r.validateSession(sessionID)
	if err != nil {
		return nil, err
	}
	
	// Start transaction
	tx, err := r.beginTransaction(sessionID, "merge")
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			r.abortTransaction(tx)
		}
	}()
	
	// Acquire locks
	if err := r.lockManager.AcquireLock("branches", "write", sessionID); err != nil {
		return nil, err
	}
	defer r.lockManager.ReleaseLock("branches", sessionID)
	
	if err := r.lockManager.AcquireLock("commits", "write", sessionID); err != nil {
		return nil, err
	}
	defer r.lockManager.ReleaseLock("commits", sessionID)
	
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	// Get source and target branches
	sourceBr, exists := r.branches[sourceBranch]
	if !exists {
		return nil, fmt.Errorf("source branch %s not found", sourceBranch)
	}
	
	targetBr, exists := r.branches[r.currentBranch]
	if !exists {
		return nil, fmt.Errorf("current branch %s not found", r.currentBranch)
	}
	
	// Check if merge is needed
	if sourceBr.Head == targetBr.Head {
		return &MergeResult{
			Success:  true,
			Strategy: strategy,
			Message:  "Already up to date",
		}, nil
	}
	
	// Perform merge based on strategy
	result, err := r.performMerge(sourceBr.Head, targetBr.Head, strategy, session)
	if err != nil {
		return nil, err
	}
	
	// If successful and no conflicts, update branch
	if result.Success && len(result.Conflicts) == 0 && result.Commit != nil {
		targetBr.Head = result.Commit.Hash
		r.branches[r.currentBranch] = targetBr
		
		// Commit transaction
		if err := r.commitTransaction(tx); err != nil {
			return nil, err
		}
	}
	
	// Update session activity
	session.LastActive = time.Now()
	
	// Log event
	r.logEvent("merge_attempted", sessionID, session.UserID, map[string]interface{}{
		"source_branch": sourceBranch,
		"target_branch": r.currentBranch,
		"strategy":      strategy,
		"success":       result.Success,
		"conflicts":     len(result.Conflicts),
	})
	
	return result, nil
}

// Status returns the current repository status
func (r *Repository) Status(sessionID string) (*Status, error) {
	session, err := r.validateSession(sessionID)
	if err != nil {
		return nil, err
	}
	
	// Acquire read lock
	if err := r.lockManager.AcquireLock("status", "read", sessionID); err != nil {
		return nil, err
	}
	defer r.lockManager.ReleaseLock("status", sessionID)
	
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	
	status := &Status{
		Branch:    r.currentBranch,
		Staged:    make(map[string]FileStatus),
		Modified:  make(map[string]FileStatus),
		Untracked: make([]string, 0),
		Conflicts: make([]string, 0),
		Clean:     true,
	}
	
	// Get staged files
	for path := range r.index {
		status.Staged[path] = Added // Simplified status
		status.Clean = false
	}
	
	// Scan working directory for changes
	if err := r.scanWorkingDirectory(status); err != nil {
		return nil, err
	}
	
	// Update session activity
	session.LastActive = time.Now()
	
	return status, nil
}

// Log returns the commit history
func (r *Repository) Log(sessionID string, branch string, limit int) ([]*Commit, error) {
	session, err := r.validateSession(sessionID)
	if err != nil {
		return nil, err
	}
	
	// Acquire read lock
	if err := r.lockManager.AcquireLock("commits", "read", sessionID); err != nil {
		return nil, err
	}
	defer r.lockManager.ReleaseLock("commits", sessionID)
	
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	
	if branch == "" {
		branch = r.currentBranch
	}
	
	// Get branch
	br, exists := r.branches[branch]
	if !exists {
		return nil, fmt.Errorf("branch %s not found", branch)
	}
	
	// Walk commit history
	commits := make([]*Commit, 0)
	current := br.Head
	
	for current != "" && (limit <= 0 || len(commits) < limit) {
		commit, exists := r.commits[current]
		if !exists {
			break
		}
		
		commits = append(commits, commit)
		
		// Move to parent (for simplicity, just use first parent)
		if len(commit.Parents) > 0 {
			current = commit.Parents[0]
		} else {
			current = ""
		}
	}
	
	// Update session activity
	session.LastActive = time.Now()
	
	return commits, nil
}

// Helper methods

// validateSession validates a user session
func (r *Repository) validateSession(sessionID string) (*UserSession, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	
	session, exists := r.sessions[sessionID]
	if !exists {
		return nil, errors.New("invalid session")
	}
	
	// Check session timeout (1 hour)
	if time.Since(session.LastActive) > time.Hour {
		return nil, errors.New("session expired")
	}
	
	return session, nil
}

// beginTransaction starts a new transaction
func (r *Repository) beginTransaction(sessionID, txType string) (*Transaction, error) {
	session, exists := r.sessions[sessionID]
	if !exists {
		return nil, errors.New("invalid session")
	}
	
	// Check if session already has an active transaction
	if session.CurrentTx != "" {
		return nil, errors.New("session already has an active transaction")
	}
	
	txID := generateHash(fmt.Sprintf("%s-%s-%d", sessionID, txType, time.Now().UnixNano()))
	
	tx := &Transaction{
		ID:         txID,
		SessionID:  sessionID,
		Type:       txType,
		StartTime:  time.Now(),
		Operations: make([]*TransactionOperation, 0),
		State:      "active",
		LockSet:    make(map[string]string),
	}
	
	r.transactions[txID] = tx
	session.CurrentTx = txID
	
	return tx, nil
}

// commitTransaction commits a transaction
func (r *Repository) commitTransaction(tx *Transaction) error {
	tx.mutex.Lock()
	defer tx.mutex.Unlock()
	
	if tx.State != "active" {
		return errors.New("transaction is not active")
	}
	
	tx.State = "committed"
	
	// Clear session transaction
	if session, exists := r.sessions[tx.SessionID]; exists {
		session.CurrentTx = ""
	}
	
	return nil
}

// abortTransaction aborts a transaction
func (r *Repository) abortTransaction(tx *Transaction) error {
	tx.mutex.Lock()
	defer tx.mutex.Unlock()
	
	if tx.State != "active" {
		return errors.New("transaction is not active")
	}
	
	tx.State = "aborted"
	
	// Release locks
	for resource := range tx.LockSet {
		r.lockManager.ReleaseLock(resource, tx.SessionID)
	}
	
	// Clear session transaction
	if session, exists := r.sessions[tx.SessionID]; exists {
		session.CurrentTx = ""
	}
	
	// TODO: Rollback operations
	
	return nil
}

// createTree creates a tree object from the current index
func (r *Repository) createTree() (string, error) {
	if len(r.index) == 0 {
		return "", errors.New("no files in index")
	}
	
	// Create tree structure
	tree := make(map[string]*TreeEntry)
	
	for path, entry := range r.index {
		tree[path] = &TreeEntry{
			Name: filepath.Base(path),
			Type: "file",
			Hash: entry.Hash,
			Mode: entry.Mode,
			Size: entry.Size,
		}
	}
	
	// Serialize tree
	treeData, err := json.Marshal(tree)
	if err != nil {
		return "", err
	}
	
	// Calculate tree hash
	treeHash := generateHash(string(treeData))
	r.objects[treeHash] = treeData
	
	return treeHash, nil
}

// checkoutCommit updates working directory to match a commit
func (r *Repository) checkoutCommit(commitHash string) error {
	if commitHash == "" {
		return nil // Empty repository
	}
	
	commit, exists := r.commits[commitHash]
	if !exists {
		return fmt.Errorf("commit %s not found", commitHash)
	}
	
	// Restore files from commit
	for path, hash := range commit.Files {
		content, exists := r.objects[hash]
		if !exists {
			continue // Skip missing objects
		}
		
		fullPath := filepath.Join(r.workingDir, path)
		dir := filepath.Dir(fullPath)
		
		// Create directory if needed
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
		
		// Write file
		if err := os.WriteFile(fullPath, content, 0644); err != nil {
			return err
		}
	}
	
	return nil
}

// performMerge performs the actual merge operation
func (r *Repository) performMerge(sourceCommit, targetCommit string, strategy MergeStrategy, session *UserSession) (*MergeResult, error) {
	// Simplified merge implementation
	// In a real implementation, this would handle different merge strategies properly
	
	sourceC, exists := r.commits[sourceCommit]
	if !exists {
		return nil, fmt.Errorf("source commit %s not found", sourceCommit)
	}
	
	targetC, exists := r.commits[targetCommit]
	if !exists {
		return nil, fmt.Errorf("target commit %s not found", targetCommit)
	}
	
	// Check for fast-forward merge
	if r.isAncestor(targetCommit, sourceCommit) {
		// Fast-forward merge
		return &MergeResult{
			Success:  true,
			Strategy: FastForward,
			Message:  "Fast-forward merge",
		}, nil
	}
	
	// Create merge commit
	mergeCommit := &Commit{
		Parents:   []string{targetCommit, sourceCommit},
		Author:    &Author{Name: session.UserName, Email: session.UserEmail, Timestamp: time.Now()},
		Committer: &Author{Name: session.UserName, Email: session.UserEmail, Timestamp: time.Now()},
		Message:   fmt.Sprintf("Merge branch '%s'", sourceCommit),
		Timestamp: time.Now(),
		Files:     make(map[string]string),
		Metadata:  make(map[string]string),
	}
	
	// Merge file lists (simplified)
	for path, hash := range targetC.Files {
		mergeCommit.Files[path] = hash
	}
	for path, hash := range sourceC.Files {
		if existingHash, exists := mergeCommit.Files[path]; exists && existingHash != hash {
			// Conflict detected - in real implementation, would handle properly
			return &MergeResult{
				Success: false,
				Conflicts: []*Conflict{
					{
						Path: path,
						OurContent: r.objects[existingHash],
						TheirContent: r.objects[hash],
						Resolved: false,
					},
				},
				Strategy: strategy,
				Message: "Merge conflicts detected",
			}, nil
		}
		mergeCommit.Files[path] = hash
	}
	
	// Create tree and finalize commit
	tree, err := r.createTreeFromFiles(mergeCommit.Files)
	if err != nil {
		return nil, err
	}
	
	mergeCommit.Tree = tree
	commitData, _ := json.Marshal(mergeCommit)
	mergeCommit.Hash = generateHash(string(commitData))
	
	r.commits[mergeCommit.Hash] = mergeCommit
	
	return &MergeResult{
		Success:  true,
		Commit:   mergeCommit,
		Strategy: strategy,
		Message:  "Merge successful",
	}, nil
}

// isAncestor checks if commit1 is an ancestor of commit2
func (r *Repository) isAncestor(commit1, commit2 string) bool {
	// Simple implementation - walk parents of commit2
	current := commit2
	visited := make(map[string]bool)
	
	for current != "" && !visited[current] {
		if current == commit1 {
			return true
		}
		
		visited[current] = true
		commit, exists := r.commits[current]
		if !exists || len(commit.Parents) == 0 {
			break
		}
		
		current = commit.Parents[0] // Follow first parent
	}
	
	return false
}

// createTreeFromFiles creates a tree object from file map
func (r *Repository) createTreeFromFiles(files map[string]string) (string, error) {
	treeData, err := json.Marshal(files)
	if err != nil {
		return "", err
	}
	
	treeHash := generateHash(string(treeData))
	r.objects[treeHash] = treeData
	
	return treeHash, nil
}

// scanWorkingDirectory scans for changes in working directory
func (r *Repository) scanWorkingDirectory(status *Status) error {
	return filepath.Walk(r.workingDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		// Skip .git directory
		if strings.Contains(path, ".git") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		
		if info.IsDir() {
			return nil
		}
		
		relPath, err := filepath.Rel(r.workingDir, path)
		if err != nil {
			return err
		}
		
		// Check if file is ignored
		if r.isIgnored(relPath) {
			return nil
		}
		
		// Check if file is tracked
		if _, staged := status.Staged[relPath]; !staged {
			// Check if file exists in current commit
			if currentBranch, exists := r.branches[r.currentBranch]; exists && currentBranch.Head != "" {
				if commit, exists := r.commits[currentBranch.Head]; exists {
					if _, tracked := commit.Files[relPath]; !tracked {
						status.Untracked = append(status.Untracked, relPath)
						status.Clean = false
					}
				}
			} else {
				// No commits yet, all files are untracked
				status.Untracked = append(status.Untracked, relPath)
				status.Clean = false
			}
		}
		
		return nil
	})
}

// isIgnored checks if a file should be ignored
func (r *Repository) isIgnored(path string) bool {
	for _, pattern := range r.config.IgnorePatterns {
		if matched, _ := filepath.Match(pattern, path); matched {
			return true
		}
		if matched, _ := filepath.Match(pattern, filepath.Base(path)); matched {
			return true
		}
	}
	return false
}

// updateStatistics updates repository statistics
func (r *Repository) updateStatistics(commit *Commit) {
	r.statistics.mutex.Lock()
	defer r.statistics.mutex.Unlock()
	
	atomic.AddInt64(&r.statistics.TotalCommits, 1)
	
	// Update by author
	r.statistics.CommitsByAuthor[commit.Author.Name]++
	
	// Update by date
	dateKey := commit.Timestamp.Format("2006-01-02")
	r.statistics.CommitsByDate[dateKey]++
	
	// Update file statistics
	for path := range commit.Files {
		ext := filepath.Ext(path)
		if ext != "" {
			r.statistics.FilesByExtension[ext]++
		}
	}
}

// logEvent logs a repository event
func (r *Repository) logEvent(eventType, sessionID, userID string, data map[string]interface{}) {
	event := &RepositoryEvent{
		ID:        generateHash(fmt.Sprintf("%s-%s-%d", eventType, sessionID, time.Now().UnixNano())),
		Type:      eventType,
		SessionID: sessionID,
		UserID:    userID,
		Timestamp: time.Now(),
		Data:      data,
	}
	
	r.eventLog = append(r.eventLog, event)
	
	// Keep only recent events (last 1000)
	if len(r.eventLog) > 1000 {
		r.eventLog = r.eventLog[len(r.eventLog)-1000:]
	}
}

// saveMetadata saves repository metadata to disk
func (r *Repository) saveMetadata() error {
	configPath := filepath.Join(r.gitDir, "config.json")
	configData, err := json.MarshalIndent(r.config, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(configPath, configData, 0644)
}

// loadMetadata loads repository metadata from disk
func (r *Repository) loadMetadata() error {
	configPath := filepath.Join(r.gitDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return err
	}
	
	return json.Unmarshal(configData, &r.config)
}

// Lock manager methods

// AcquireLock acquires a lock on a resource
func (lm *LockManager) AcquireLock(resource, lockType, sessionID string) error {
	lm.mutex.Lock()
	defer lm.mutex.Unlock()
	
	// Check if lock can be acquired
	if existingLock, exists := lm.locks[resource]; exists {
		if existingLock.Type == "exclusive" || lockType == "exclusive" {
			// Cannot acquire - add to wait queue
			request := &LockRequest{
				Resource:    resource,
				Type:        lockType,
				SessionID:   sessionID,
				RequestedAt: time.Now(),
				Result:      make(chan error, 1),
			}
			
			lm.waitQueue[resource] = append(lm.waitQueue[resource], request)
			
			// Wait for lock (simplified - in real implementation would handle timeouts)
			lm.mutex.Unlock()
			err := <-request.Result
			lm.mutex.Lock()
			return err
		}
		
		// Read locks can coexist
		if existingLock.Type == "read" && lockType == "read" {
			// Allow multiple read locks
		}
	}
	
	// Acquire lock
	lock := &ResourceLock{
		Resource:   resource,
		Type:       lockType,
		Owner:      sessionID,
		SessionID:  sessionID,
		AcquiredAt: time.Now(),
	}
	
	lm.locks[resource] = lock
	return nil
}

// ReleaseLock releases a lock on a resource
func (lm *LockManager) ReleaseLock(resource, sessionID string) error {
	lm.mutex.Lock()
	defer lm.mutex.Unlock()
	
	lock, exists := lm.locks[resource]
	if !exists {
		return errors.New("lock not found")
	}
	
	if lock.SessionID != sessionID {
		return errors.New("lock not owned by session")
	}
	
	delete(lm.locks, resource)
	
	// Process wait queue
	if queue, exists := lm.waitQueue[resource]; exists && len(queue) > 0 {
		// Grant lock to first waiter
		request := queue[0]
		lm.waitQueue[resource] = queue[1:]
		
		// Grant the lock
		go func() {
			request.Result <- nil
		}()
	}
	
	return nil
}

// ReleaseSessionLocks releases all locks held by a session
func (lm *LockManager) ReleaseSessionLocks(sessionID string) {
	lm.mutex.Lock()
	defer lm.mutex.Unlock()
	
	// Find and release all locks for session
	for resource, lock := range lm.locks {
		if lock.SessionID == sessionID {
			delete(lm.locks, resource)
			
			// Process wait queue for this resource
			if queue, exists := lm.waitQueue[resource]; exists && len(queue) > 0 {
				request := queue[0]
				lm.waitQueue[resource] = queue[1:]
				
				go func() {
					request.Result <- nil
				}()
			}
		}
	}
}

// GetStatistics returns repository statistics
func (r *Repository) GetStatistics() *RepositoryStatistics {
	r.statistics.mutex.RLock()
	defer r.statistics.mutex.RUnlock()
	
	// Return a copy to avoid race conditions
	stats := *r.statistics
	return &stats
}

// generateHash generates a SHA-256 hash of the input string
func generateHash(input string) string {
	hash := sha256.Sum256([]byte(input))
	return hex.EncodeToString(hash[:])
}

// Export/Import functionality for remote operations (simplified)

// Export exports repository data for remote synchronization
func (r *Repository) Export(sessionID string, refs []string) (map[string]interface{}, error) {
	session, err := r.validateSession(sessionID)
	if err != nil {
		return nil, err
	}
	
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	
	export := map[string]interface{}{
		"branches": make(map[string]*Branch),
		"commits":  make(map[string]*Commit),
		"objects":  make(map[string][]byte),
	}
	
	// Export requested refs
	for _, ref := range refs {
		if branch, exists := r.branches[ref]; exists {
			export["branches"].(map[string]*Branch)[ref] = branch
			
			// Export commits reachable from this branch
			r.exportCommitsFrom(branch.Head, export)
		}
	}
	
	// Update session activity
	session.LastActive = time.Now()
	
	return export, nil
}

// exportCommitsFrom recursively exports commits from a starting point
func (r *Repository) exportCommitsFrom(commitHash string, export map[string]interface{}) {
	if commitHash == "" {
		return
	}
	
	commits := export["commits"].(map[string]*Commit)
	objects := export["objects"].(map[string][]byte)
	
	if _, exists := commits[commitHash]; exists {
		return // Already exported
	}
	
	commit, exists := r.commits[commitHash]
	if !exists {
		return
	}
	
	commits[commitHash] = commit
	
	// Export objects referenced by commit
	if treeData, exists := r.objects[commit.Tree]; exists {
		objects[commit.Tree] = treeData
	}
	
	for _, fileHash := range commit.Files {
		if fileData, exists := r.objects[fileHash]; exists {
			objects[fileHash] = fileData
		}
	}
	
	// Recursively export parent commits
	for _, parent := range commit.Parents {
		r.exportCommitsFrom(parent, export)
	}
}

// Import imports repository data from remote synchronization
func (r *Repository) Import(sessionID string, data map[string]interface{}) error {
	session, err := r.validateSession(sessionID)
	if err != nil {
		return err
	}
	
	// Start transaction
	tx, err := r.beginTransaction(sessionID, "import")
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			r.abortTransaction(tx)
		}
	}()
	
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	// Import objects
	if objects, exists := data["objects"].(map[string][]byte); exists {
		for hash, content := range objects {
			r.objects[hash] = content
		}
	}
	
	// Import commits
	if commits, exists := data["commits"].(map[string]*Commit); exists {
		for hash, commit := range commits {
			r.commits[hash] = commit
		}
	}
	
	// Import branches
	if branches, exists := data["branches"].(map[string]*Branch); exists {
		for name, branch := range branches {
			r.branches[name] = branch
		}
	}
	
	// Commit transaction
	if err := r.commitTransaction(tx); err != nil {
		return err
	}
	
	// Update session activity
	session.LastActive = time.Now()
	
	// Log event
	r.logEvent("data_imported", sessionID, session.UserID, map[string]interface{}{
		"branches_count": len(data["branches"].(map[string]*Branch)),
		"commits_count":  len(data["commits"].(map[string]*Commit)),
		"objects_count":  len(data["objects"].(map[string][]byte)),
	})
	
	return nil
}

// GetEventLog returns the repository event log
func (r *Repository) GetEventLog(sessionID string, limit int) ([]*RepositoryEvent, error) {
	session, err := r.validateSession(sessionID)
	if err != nil {
		return nil, err
	}
	
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	
	events := make([]*RepositoryEvent, len(r.eventLog))
	copy(events, r.eventLog)
	
	// Sort by timestamp (newest first)
	sort.Slice(events, func(i, j int) bool {
		return events[i].Timestamp.After(events[j].Timestamp)
	})
	
	// Apply limit
	if limit > 0 && limit < len(events) {
		events = events[:limit]
	}
	
	// Update session activity
	session.LastActive = time.Now()
	
	return events, nil
}