package concurrentftpserver

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// FTPServer represents a concurrent FTP server
type FTPServer struct {
	config     ServerConfig
	listener   net.Listener
	clients    sync.Map // client ID -> *Client
	sessions   sync.Map // session ID -> *Session
	idCounter  int64
	running    bool
	quit       chan bool
	wg         sync.WaitGroup
	metrics    *ServerMetrics
	middleware []Middleware
	mu         sync.RWMutex
}

// ServerConfig holds FTP server configuration
type ServerConfig struct {
	Address         string
	Port            int
	MaxClients      int
	DataPortRange   PortRange
	RootDirectory   string
	AllowAnonymous  bool
	Timeout         time.Duration
	MaxFileSize     int64
	EnableLogging   bool
	TLSEnabled      bool
	Users           map[string]UserConfig
}

// UserConfig represents user configuration
type UserConfig struct {
	Username    string
	Password    string
	HomeDir     string
	Permissions UserPermissions
	MaxSessions int
}

// UserPermissions defines what a user can do
type UserPermissions struct {
	Read    bool
	Write   bool
	Delete  bool
	Rename  bool
	List    bool
	Upload  bool
	Download bool
}

// PortRange defines data connection port range
type PortRange struct {
	Min int
	Max int
}

// Client represents an FTP client connection
type Client struct {
	ID          int64
	Conn        net.Conn
	Session     *Session
	Server      *FTPServer
	Reader      *bufio.Reader
	Writer      *bufio.Writer
	CurrentDir  string
	DataConn    net.Conn
	DataListener net.Listener
	PassiveMode bool
	Binary      bool
	Connected   time.Time
	LastCommand time.Time
	mu          sync.RWMutex
}

// Session represents an authenticated user session
type Session struct {
	ID          string
	User        UserConfig
	LoginTime   time.Time
	LastAccess  time.Time
	CommandCount int64
	BytesUploaded int64
	BytesDownloaded int64
	mu          sync.RWMutex
}

// ServerMetrics tracks server performance
type ServerMetrics struct {
	TotalConnections int64
	ActiveConnections int64
	TotalCommands    int64
	TotalUploads     int64
	TotalDownloads   int64
	BytesTransferred int64
	StartTime        time.Time
	mu               sync.RWMutex
}

// Command represents an FTP command
type Command struct {
	Name string
	Args []string
	Raw  string
}

// Response represents an FTP response
type Response struct {
	Code    int
	Message string
}

// CommandHandler handles FTP commands
type CommandHandler func(client *Client, cmd Command) Response

// Middleware allows request/response interception
type Middleware func(client *Client, cmd Command, next CommandHandler) Response

// Transfer represents a file transfer operation
type Transfer struct {
	Type      TransferType
	FileName  string
	Size      int64
	StartTime time.Time
	Client    *Client
}

// TransferType defines upload/download
type TransferType int

const (
	TransferUpload TransferType = iota
	TransferDownload
)

// FTP Response codes
const (
	CodeDataConnectionOpen     = 125
	CodeCommandOK             = 200
	CodeSystemStatus          = 211
	CodeDirectoryStatus       = 212
	CodeFileStatus            = 213
	CodeHelpMessage           = 214
	CodeSystemType            = 215
	CodeServiceReady          = 220
	CodeClosingControl        = 221
	CodeDataConnectionClosed  = 226
	CodePassiveMode           = 227
	CodeUserLoggedIn          = 230
	CodeFileActionOK          = 250
	CodePathCreated           = 257
	CodeUserOK                = 331
	CodeNeedAccount           = 332
	CodeFileActionPending     = 350
	CodeInsufficientStorage   = 452
	CodeCantOpenData          = 425
	CodeTransferAborted       = 426
	CodeFileActionAborted     = 450
	CodeSyntaxError           = 500
	CodeParameterError        = 501
	CodeCommandNotImplemented = 502
	CodeBadSequence           = 503
	CodeParameterNotImplemented = 504
	CodeNotLoggedIn           = 530
	CodeFileNotFound          = 550
	CodePageTypeUnknown       = 551
	CodeExceededStorage       = 552
	CodeFileNameNotAllowed    = 553
)

// NewFTPServer creates a new FTP server
func NewFTPServer(config ServerConfig) *FTPServer {
	if config.Port == 0 {
		config.Port = 21
	}
	if config.MaxClients == 0 {
		config.MaxClients = 100
	}
	if config.DataPortRange.Min == 0 {
		config.DataPortRange.Min = 1024
	}
	if config.DataPortRange.Max == 0 {
		config.DataPortRange.Max = 65535
	}
	if config.RootDirectory == "" {
		config.RootDirectory = "."
	}
	if config.Timeout == 0 {
		config.Timeout = 5 * time.Minute
	}
	if config.MaxFileSize == 0 {
		config.MaxFileSize = 100 * 1024 * 1024 // 100MB
	}
	if config.Users == nil {
		config.Users = make(map[string]UserConfig)
	}

	return &FTPServer{
		config:  config,
		quit:    make(chan bool),
		metrics: &ServerMetrics{StartTime: time.Now()},
	}
}

// Start starts the FTP server
func (s *FTPServer) Start() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		return fmt.Errorf("server already running")
	}

	address := fmt.Sprintf("%s:%d", s.config.Address, s.config.Port)
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return fmt.Errorf("failed to start listener: %w", err)
	}

	s.listener = listener
	s.running = true

	// Start accepting connections
	s.wg.Add(1)
	go s.acceptConnections()

	if s.config.EnableLogging {
		fmt.Printf("FTP server started on %s\n", address)
	}

	return nil
}

// Stop stops the FTP server
func (s *FTPServer) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		return
	}

	close(s.quit)
	if s.listener != nil {
		s.listener.Close()
	}

	// Close all client connections
	s.clients.Range(func(key, value interface{}) bool {
		client := value.(*Client)
		client.disconnect()
		return true
	})

	s.wg.Wait()
	s.running = false

	if s.config.EnableLogging {
		fmt.Println("FTP server stopped")
	}
}

func (s *FTPServer) acceptConnections() {
	defer s.wg.Done()

	for {
		conn, err := s.listener.Accept()
		if err != nil {
			select {
			case <-s.quit:
				return
			default:
				if s.config.EnableLogging {
					fmt.Printf("Accept error: %v\n", err)
				}
				continue
			}
		}

		// Check max clients
		if s.getActiveConnections() >= int64(s.config.MaxClients) {
			response := Response{CodeInsufficientStorage, "Too many connections"}
			s.sendResponse(conn, response)
			conn.Close()
			continue
		}

		// Handle new client
		s.wg.Add(1)
		go s.handleClient(conn)
	}
}

func (s *FTPServer) handleClient(conn net.Conn) {
	defer s.wg.Done()
	defer conn.Close()

	clientID := atomic.AddInt64(&s.idCounter, 1)
	client := &Client{
		ID:          clientID,
		Conn:        conn,
		Server:      s,
		Reader:      bufio.NewReader(conn),
		Writer:      bufio.NewWriter(conn),
		CurrentDir:  "/",
		Binary:      true,
		Connected:   time.Now(),
		LastCommand: time.Now(),
	}

	s.clients.Store(clientID, client)
	defer s.clients.Delete(clientID)

	atomic.AddInt64(&s.metrics.TotalConnections, 1)
	atomic.AddInt64(&s.metrics.ActiveConnections, 1)
	defer atomic.AddInt64(&s.metrics.ActiveConnections, -1)

	// Send welcome message
	welcome := Response{CodeServiceReady, "FTP Server ready"}
	client.sendResponse(welcome)

	// Handle commands
	for {
		// Set read timeout
		conn.SetReadDeadline(time.Now().Add(s.config.Timeout))

		line, err := client.Reader.ReadString('\n')
		if err != nil {
			if s.config.EnableLogging {
				fmt.Printf("Client %d read error: %v\n", clientID, err)
			}
			break
		}

		client.LastCommand = time.Now()
		atomic.AddInt64(&s.metrics.TotalCommands, 1)

		// Parse command
		cmd := s.parseCommand(strings.TrimSpace(line))
		if s.config.EnableLogging {
			fmt.Printf("Client %d: %s\n", clientID, cmd.Name)
		}

		// Handle command
		response := s.handleCommand(client, cmd)
		client.sendResponse(response)

		// Check for QUIT command
		if cmd.Name == "QUIT" {
			break
		}
	}
}

func (s *FTPServer) parseCommand(line string) Command {
	parts := strings.Fields(line)
	if len(parts) == 0 {
		return Command{Name: "", Args: []string{}, Raw: line}
	}

	name := strings.ToUpper(parts[0])
	args := parts[1:]

	return Command{
		Name: name,
		Args: args,
		Raw:  line,
	}
}

func (s *FTPServer) handleCommand(client *Client, cmd Command) Response {
	// Apply middleware
	handler := s.getCommandHandler(cmd.Name)
	for i := len(s.middleware) - 1; i >= 0; i-- {
		middleware := s.middleware[i]
		originalHandler := handler
		handler = func(c *Client, command Command) Response {
			return middleware(c, command, originalHandler)
		}
	}

	return handler(client, cmd)
}

func (s *FTPServer) getCommandHandler(command string) CommandHandler {
	handlers := map[string]CommandHandler{
		"USER": s.handleUSER,
		"PASS": s.handlePASS,
		"QUIT": s.handleQUIT,
		"PWD":  s.handlePWD,
		"CWD":  s.handleCWD,
		"LIST": s.handleLIST,
		"NLST": s.handleNLST,
		"RETR": s.handleRETR,
		"STOR": s.handleSTOR,
		"DELE": s.handleDELE,
		"MKD":  s.handleMKD,
		"RMD":  s.handleRMD,
		"RNFR": s.handleRNFR,
		"RNTO": s.handleRNTO,
		"SIZE": s.handleSIZE,
		"TYPE": s.handleTYPE,
		"PASV": s.handlePASV,
		"PORT": s.handlePORT,
		"SYST": s.handleSYST,
		"FEAT": s.handleFEAT,
		"NOOP": s.handleNOOP,
	}

	if handler, exists := handlers[command]; exists {
		return handler
	}

	return s.handleUnknown
}

// Command handlers

func (s *FTPServer) handleUSER(client *Client, cmd Command) Response {
	if len(cmd.Args) == 0 {
		return Response{CodeParameterError, "Username required"}
	}

	username := cmd.Args[0]
	client.mu.Lock()
	defer client.mu.Unlock()

	if username == "anonymous" && s.config.AllowAnonymous {
		// Allow anonymous login
		return Response{CodeUserLoggedIn, "Anonymous user logged in"}
	}

	if _, exists := s.config.Users[username]; exists {
		client.Session = &Session{
			ID:        fmt.Sprintf("session_%d", time.Now().UnixNano()),
			LoginTime: time.Now(),
		}
		return Response{CodeUserOK, "Username OK, password required"}
	}

	return Response{CodeNotLoggedIn, "Invalid username"}
}

func (s *FTPServer) handlePASS(client *Client, cmd Command) Response {
	if len(cmd.Args) == 0 {
		return Response{CodeParameterError, "Password required"}
	}

	client.mu.Lock()
	defer client.mu.Unlock()

	if client.Session == nil {
		return Response{CodeBadSequence, "Send username first"}
	}

	password := cmd.Args[0]
	// In a real implementation, you'd validate the password
	client.Session.User = UserConfig{
		Username:    "user", // Would come from USER command
		HomeDir:     s.config.RootDirectory,
		Permissions: UserPermissions{Read: true, Write: true, List: true, Upload: true, Download: true},
	}

	s.sessions.Store(client.Session.ID, client.Session)
	return Response{CodeUserLoggedIn, "User logged in"}
}

func (s *FTPServer) handleQUIT(client *Client, cmd Command) Response {
	return Response{CodeClosingControl, "Goodbye"}
}

func (s *FTPServer) handlePWD(client *Client, cmd Command) Response {
	if !s.isAuthenticated(client) {
		return Response{CodeNotLoggedIn, "Not logged in"}
	}

	return Response{CodePathCreated, fmt.Sprintf("\"%s\" is current directory", client.CurrentDir)}
}

func (s *FTPServer) handleCWD(client *Client, cmd Command) Response {
	if !s.isAuthenticated(client) {
		return Response{CodeNotLoggedIn, "Not logged in"}
	}

	if len(cmd.Args) == 0 {
		return Response{CodeParameterError, "Directory required"}
	}

	newDir := cmd.Args[0]
	fullPath := s.resolvePath(client, newDir)

	if info, err := os.Stat(fullPath); err == nil && info.IsDir() {
		client.mu.Lock()
		client.CurrentDir = s.relativePath(fullPath)
		client.mu.Unlock()
		return Response{CodeFileActionOK, "Directory changed"}
	}

	return Response{CodeFileNotFound, "Directory not found"}
}

func (s *FTPServer) handleLIST(client *Client, cmd Command) Response {
	if !s.isAuthenticated(client) {
		return Response{CodeNotLoggedIn, "Not logged in"}
	}

	if !client.Session.User.Permissions.List {
		return Response{CodeFileActionAborted, "Permission denied"}
	}

	path := client.CurrentDir
	if len(cmd.Args) > 0 {
		path = cmd.Args[0]
	}

	fullPath := s.resolvePath(client, path)
	return s.sendDirectoryListing(client, fullPath, true)
}

func (s *FTPServer) handleNLST(client *Client, cmd Command) Response {
	if !s.isAuthenticated(client) {
		return Response{CodeNotLoggedIn, "Not logged in"}
	}

	path := client.CurrentDir
	if len(cmd.Args) > 0 {
		path = cmd.Args[0]
	}

	fullPath := s.resolvePath(client, path)
	return s.sendDirectoryListing(client, fullPath, false)
}

func (s *FTPServer) handleRETR(client *Client, cmd Command) Response {
	if !s.isAuthenticated(client) {
		return Response{CodeNotLoggedIn, "Not logged in"}
	}

	if !client.Session.User.Permissions.Download {
		return Response{CodeFileActionAborted, "Permission denied"}
	}

	if len(cmd.Args) == 0 {
		return Response{CodeParameterError, "Filename required"}
	}

	filename := cmd.Args[0]
	fullPath := s.resolvePath(client, filename)

	return s.sendFile(client, fullPath)
}

func (s *FTPServer) handleSTOR(client *Client, cmd Command) Response {
	if !s.isAuthenticated(client) {
		return Response{CodeNotLoggedIn, "Not logged in"}
	}

	if !client.Session.User.Permissions.Upload {
		return Response{CodeFileActionAborted, "Permission denied"}
	}

	if len(cmd.Args) == 0 {
		return Response{CodeParameterError, "Filename required"}
	}

	filename := cmd.Args[0]
	fullPath := s.resolvePath(client, filename)

	return s.receiveFile(client, fullPath)
}

func (s *FTPServer) handlePASV(client *Client, cmd Command) Response {
	if !s.isAuthenticated(client) {
		return Response{CodeNotLoggedIn, "Not logged in"}
	}

	// Create passive data connection
	listener, err := net.Listen("tcp", ":")
	if err != nil {
		return Response{CodeCantOpenData, "Cannot create data connection"}
	}

	client.mu.Lock()
	client.DataListener = listener
	client.PassiveMode = true
	client.mu.Unlock()

	// Get address and port
	addr := listener.Addr().(*net.TCPAddr)
	ip := strings.Replace(addr.IP.String(), ".", ",", -1)
	port := addr.Port
	p1 := port / 256
	p2 := port % 256

	message := fmt.Sprintf("Entering Passive Mode (%s,%d,%d)", ip, p1, p2)
	return Response{CodePassiveMode, message}
}

func (s *FTPServer) handleSYST(client *Client, cmd Command) Response {
	return Response{CodeSystemType, "UNIX Type: L8"}
}

func (s *FTPServer) handleTYPE(client *Client, cmd Command) Response {
	if len(cmd.Args) == 0 {
		return Response{CodeParameterError, "Type required"}
	}

	typeArg := strings.ToUpper(cmd.Args[0])
	client.mu.Lock()
	defer client.mu.Unlock()

	switch typeArg {
	case "A":
		client.Binary = false
		return Response{CodeCommandOK, "Type set to ASCII"}
	case "I":
		client.Binary = true
		return Response{CodeCommandOK, "Type set to binary"}
	default:
		return Response{CodeParameterNotImplemented, "Type not supported"}
	}
}

func (s *FTPServer) handleFEAT(client *Client, cmd Command) Response {
	features := []string{
		"211-Features:",
		" SIZE",
		" REST STREAM",
		" MLST type*;size*;modify*;",
		" MLSD",
		" UTF8",
		"211 End",
	}
	return Response{CodeSystemStatus, strings.Join(features, "\r\n")}
}

func (s *FTPServer) handleNOOP(client *Client, cmd Command) Response {
	return Response{CodeCommandOK, "OK"}
}

func (s *FTPServer) handleUnknown(client *Client, cmd Command) Response {
	return Response{CodeCommandNotImplemented, "Command not implemented"}
}

// Placeholder handlers for commands not fully implemented
func (s *FTPServer) handleDELE(client *Client, cmd Command) Response {
	return Response{CodeCommandNotImplemented, "DELE not implemented"}
}

func (s *FTPServer) handleMKD(client *Client, cmd Command) Response {
	return Response{CodeCommandNotImplemented, "MKD not implemented"}
}

func (s *FTPServer) handleRMD(client *Client, cmd Command) Response {
	return Response{CodeCommandNotImplemented, "RMD not implemented"}
}

func (s *FTPServer) handleRNFR(client *Client, cmd Command) Response {
	return Response{CodeCommandNotImplemented, "RNFR not implemented"}
}

func (s *FTPServer) handleRNTO(client *Client, cmd Command) Response {
	return Response{CodeCommandNotImplemented, "RNTO not implemented"}
}

func (s *FTPServer) handleSIZE(client *Client, cmd Command) Response {
	return Response{CodeCommandNotImplemented, "SIZE not implemented"}
}

func (s *FTPServer) handlePORT(client *Client, cmd Command) Response {
	return Response{CodeCommandNotImplemented, "PORT not implemented"}
}

// Helper methods

func (s *FTPServer) isAuthenticated(client *Client) bool {
	client.mu.RLock()
	defer client.mu.RUnlock()
	return client.Session != nil
}

func (s *FTPServer) resolvePath(client *Client, path string) string {
	if filepath.IsAbs(path) {
		return filepath.Join(s.config.RootDirectory, path)
	}
	return filepath.Join(s.config.RootDirectory, client.CurrentDir, path)
}

func (s *FTPServer) relativePath(path string) string {
	rel, err := filepath.Rel(s.config.RootDirectory, path)
	if err != nil {
		return "/"
	}
	return "/" + filepath.ToSlash(rel)
}

func (s *FTPServer) sendDirectoryListing(client *Client, path string, detailed bool) Response {
	dataConn, err := s.getDataConnection(client)
	if err != nil {
		return Response{CodeCantOpenData, "Cannot open data connection"}
	}
	defer dataConn.Close()

	client.sendResponse(Response{CodeDataConnectionOpen, "Directory listing"})

	entries, err := os.ReadDir(path)
	if err != nil {
		return Response{CodeFileNotFound, "Cannot read directory"}
	}

	for _, entry := range entries {
		var line string
		if detailed {
			info, _ := entry.Info()
			perms := "-rw-r--r--"
			if entry.IsDir() {
				perms = "drwxr-xr-x"
			}
			line = fmt.Sprintf("%s 1 user group %8d %s %s\r\n",
				perms, info.Size(), info.ModTime().Format("Jan 02 15:04"), entry.Name())
		} else {
			line = entry.Name() + "\r\n"
		}
		dataConn.Write([]byte(line))
	}

	return Response{CodeDataConnectionClosed, "Directory listing complete"}
}

func (s *FTPServer) sendFile(client *Client, path string) Response {
	file, err := os.Open(path)
	if err != nil {
		return Response{CodeFileNotFound, "File not found"}
	}
	defer file.Close()

	dataConn, err := s.getDataConnection(client)
	if err != nil {
		return Response{CodeCantOpenData, "Cannot open data connection"}
	}
	defer dataConn.Close()

	client.sendResponse(Response{CodeDataConnectionOpen, "Transferring file"})

	bytes, err := io.Copy(dataConn, file)
	if err != nil {
		return Response{CodeTransferAborted, "Transfer failed"}
	}

	atomic.AddInt64(&s.metrics.TotalDownloads, 1)
	atomic.AddInt64(&s.metrics.BytesTransferred, bytes)

	if client.Session != nil {
		client.Session.mu.Lock()
		client.Session.BytesDownloaded += bytes
		client.Session.mu.Unlock()
	}

	return Response{CodeDataConnectionClosed, "Transfer complete"}
}

func (s *FTPServer) receiveFile(client *Client, path string) Response {
	dataConn, err := s.getDataConnection(client)
	if err != nil {
		return Response{CodeCantOpenData, "Cannot open data connection"}
	}
	defer dataConn.Close()

	file, err := os.Create(path)
	if err != nil {
		return Response{CodeFileActionAborted, "Cannot create file"}
	}
	defer file.Close()

	client.sendResponse(Response{CodeDataConnectionOpen, "Ready for transfer"})

	bytes, err := io.Copy(file, dataConn)
	if err != nil {
		return Response{CodeTransferAborted, "Transfer failed"}
	}

	atomic.AddInt64(&s.metrics.TotalUploads, 1)
	atomic.AddInt64(&s.metrics.BytesTransferred, bytes)

	if client.Session != nil {
		client.Session.mu.Lock()
		client.Session.BytesUploaded += bytes
		client.Session.mu.Unlock()
	}

	return Response{CodeDataConnectionClosed, "Transfer complete"}
}

func (s *FTPServer) getDataConnection(client *Client) (net.Conn, error) {
	client.mu.Lock()
	defer client.mu.Unlock()

	if client.PassiveMode && client.DataListener != nil {
		conn, err := client.DataListener.Accept()
		client.DataListener.Close()
		client.DataListener = nil
		return conn, err
	}

	return nil, fmt.Errorf("no data connection available")
}

func (s *FTPServer) getActiveConnections() int64 {
	return atomic.LoadInt64(&s.metrics.ActiveConnections)
}

func (s *FTPServer) sendResponse(conn net.Conn, response Response) {
	message := fmt.Sprintf("%d %s\r\n", response.Code, response.Message)
	conn.Write([]byte(message))
}

func (client *Client) sendResponse(response Response) {
	message := fmt.Sprintf("%d %s\r\n", response.Code, response.Message)
	client.Writer.WriteString(message)
	client.Writer.Flush()
}

func (client *Client) disconnect() {
	client.mu.Lock()
	defer client.mu.Unlock()

	if client.DataConn != nil {
		client.DataConn.Close()
	}
	if client.DataListener != nil {
		client.DataListener.Close()
	}
	client.Conn.Close()
}

// AddMiddleware adds middleware to the server
func (s *FTPServer) AddMiddleware(middleware Middleware) {
	s.middleware = append(s.middleware, middleware)
}

// GetMetrics returns server metrics
func (s *FTPServer) GetMetrics() ServerMetrics {
	s.metrics.mu.RLock()
	defer s.metrics.mu.RUnlock()
	return *s.metrics
}

// GetActiveSessions returns current active sessions
func (s *FTPServer) GetActiveSessions() []*Session {
	var sessions []*Session
	s.sessions.Range(func(key, value interface{}) bool {
		sessions = append(sessions, value.(*Session))
		return true
	})
	return sessions
}

// AddUser adds a user to the server configuration
func (s *FTPServer) AddUser(username string, config UserConfig) {
	s.config.Users[username] = config
}

// Example demonstrates concurrent FTP server
func Example() {
	fmt.Println("=== Concurrent FTP Server Example ===")

	// Create server configuration
	config := ServerConfig{
		Address:        "localhost",
		Port:          2121, // Use non-standard port for testing
		MaxClients:    10,
		RootDirectory: "./ftp_root",
		AllowAnonymous: true,
		Timeout:       time.Minute,
		EnableLogging: true,
		Users: map[string]UserConfig{
			"testuser": {
				Username: "testuser",
				Password: "testpass",
				HomeDir:  "./ftp_root",
				Permissions: UserPermissions{
					Read: true, Write: true, Delete: true,
					Rename: true, List: true, Upload: true, Download: true,
				},
			},
		},
	}

	// Create FTP root directory
	os.MkdirAll(config.RootDirectory, 0755)
	defer os.RemoveAll(config.RootDirectory)

	// Create test file
	testFile := filepath.Join(config.RootDirectory, "test.txt")
	os.WriteFile(testFile, []byte("Hello, FTP World!"), 0644)

	// Create and start server
	server := NewFTPServer(config)

	// Add logging middleware
	server.AddMiddleware(func(client *Client, cmd Command, next CommandHandler) Response {
		start := time.Now()
		response := next(client, cmd)
		duration := time.Since(start)
		fmt.Printf("Command %s took %v\n", cmd.Name, duration)
		return response
	})

	if err := server.Start(); err != nil {
		fmt.Printf("Failed to start server: %v\n", err)
		return
	}

	fmt.Printf("FTP server started on %s:%d\n", config.Address, config.Port)
	fmt.Println("You can connect using: ftp localhost 2121")

	// Let it run for a short time for demonstration
	time.Sleep(2 * time.Second)

	// Show metrics
	metrics := server.GetMetrics()
	fmt.Printf("Server metrics:\n")
	fmt.Printf("  Total connections: %d\n", metrics.TotalConnections)
	fmt.Printf("  Active connections: %d\n", metrics.ActiveConnections)
	fmt.Printf("  Total commands: %d\n", metrics.TotalCommands)
	fmt.Printf("  Uptime: %v\n", time.Since(metrics.StartTime))

	// Stop server
	server.Stop()
	fmt.Println("FTP server stopped")
}