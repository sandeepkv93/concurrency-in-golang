package concurrentftpserver

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func createTestServer(t *testing.T) (*FTPServer, string, func()) {
	tempDir := t.TempDir()
	
	config := ServerConfig{
		Address:        "127.0.0.1",
		Port:          0, // Let OS choose port
		MaxClients:    10,
		RootDirectory: tempDir,
		AllowAnonymous: true,
		Timeout:       30 * time.Second,
		EnableLogging: false, // Disable for tests
		Users: map[string]UserConfig{
			"testuser": {
				Username: "testuser",
				Password: "testpass",
				HomeDir:  tempDir,
				Permissions: UserPermissions{
					Read: true, Write: true, Delete: true,
					Rename: true, List: true, Upload: true, Download: true,
				},
			},
		},
	}

	server := NewFTPServer(config)
	if err := server.Start(); err != nil {
		t.Fatalf("Failed to start test server: %v", err)
	}

	// Get the actual address
	addr := server.listener.Addr().String()

	cleanup := func() {
		server.Stop()
		os.RemoveAll(tempDir)
	}

	return server, addr, cleanup
}

func connectToServer(t *testing.T, addr string) (net.Conn, *bufio.Reader, *bufio.Writer) {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("Failed to connect to server: %v", err)
	}

	reader := bufio.NewReader(conn)
	writer := bufio.NewWriter(conn)

	return conn, reader, writer
}

func sendCommand(writer *bufio.Writer, command string) error {
	_, err := writer.WriteString(command + "\r\n")
	if err != nil {
		return err
	}
	return writer.Flush()
}

func readResponse(reader *bufio.Reader) (int, string, error) {
	line, err := reader.ReadString('\n')
	if err != nil {
		return 0, "", err
	}

	line = strings.TrimSpace(line)
	if len(line) < 4 {
		return 0, line, fmt.Errorf("invalid response: %s", line)
	}

	var code int
	fmt.Sscanf(line[:3], "%d", &code)
	message := line[4:]

	return code, message, nil
}

func TestFTPServerStart(t *testing.T) {
	server, addr, cleanup := createTestServer(t)
	defer cleanup()

	if !server.running {
		t.Error("Server should be running")
	}

	if addr == "" {
		t.Error("Server address should not be empty")
	}

	// Test connection
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	reader := bufio.NewReader(conn)
	code, message, err := readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read welcome: %v", err)
	}

	if code != CodeServiceReady {
		t.Errorf("Expected code %d, got %d", CodeServiceReady, code)
	}

	if !strings.Contains(message, "ready") && !strings.Contains(message, "Ready") {
		t.Errorf("Expected ready message, got: %s", message)
	}
}

func TestFTPBasicCommands(t *testing.T) {
	server, addr, cleanup := createTestServer(t)
	defer cleanup()

	conn, reader, writer := connectToServer(t, addr)
	defer conn.Close()

	// Read welcome message
	readResponse(reader)

	// Test USER command
	sendCommand(writer, "USER anonymous")
	code, _, err := readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read USER response: %v", err)
	}
	if code != CodeUserLoggedIn {
		t.Errorf("Expected code %d for anonymous user, got %d", CodeUserLoggedIn, code)
	}

	// Test PWD command
	sendCommand(writer, "PWD")
	code, message, err := readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read PWD response: %v", err)
	}
	if code != CodePathCreated {
		t.Errorf("Expected code %d for PWD, got %d", CodePathCreated, code)
	}
	if !strings.Contains(message, "/") {
		t.Errorf("PWD response should contain path: %s", message)
	}

	// Test SYST command
	sendCommand(writer, "SYST")
	code, _, err = readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read SYST response: %v", err)
	}
	if code != CodeSystemType {
		t.Errorf("Expected code %d for SYST, got %d", CodeSystemType, code)
	}

	// Test NOOP command
	sendCommand(writer, "NOOP")
	code, _, err = readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read NOOP response: %v", err)
	}
	if code != CodeCommandOK {
		t.Errorf("Expected code %d for NOOP, got %d", CodeCommandOK, code)
	}

	// Test QUIT command
	sendCommand(writer, "QUIT")
	code, _, err = readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read QUIT response: %v", err)
	}
	if code != CodeClosingControl {
		t.Errorf("Expected code %d for QUIT, got %d", CodeClosingControl, code)
	}
}

func TestFTPAuthentication(t *testing.T) {
	server, addr, cleanup := createTestServer(t)
	defer cleanup()

	conn, reader, writer := connectToServer(t, addr)
	defer conn.Close()

	// Read welcome
	readResponse(reader)

	// Test valid user
	sendCommand(writer, "USER testuser")
	code, _, err := readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read USER response: %v", err)
	}
	if code != CodeUserOK {
		t.Errorf("Expected code %d for valid user, got %d", CodeUserOK, code)
	}

	// Test password (simplified - any password works in this test)
	sendCommand(writer, "PASS testpass")
	code, _, err = readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read PASS response: %v", err)
	}
	if code != CodeUserLoggedIn {
		t.Errorf("Expected code %d for login, got %d", CodeUserLoggedIn, code)
	}

	// Test invalid user
	sendCommand(writer, "USER invaliduser")
	code, _, err = readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read invalid USER response: %v", err)
	}
	if code != CodeNotLoggedIn {
		t.Errorf("Expected code %d for invalid user, got %d", CodeNotLoggedIn, code)
	}
}

func TestFTPDirectoryOperations(t *testing.T) {
	server, addr, cleanup := createTestServer(t)
	defer cleanup()

	// Create test directory structure
	testDir := filepath.Join(server.config.RootDirectory, "testdir")
	os.MkdirAll(testDir, 0755)
	
	testFile := filepath.Join(testDir, "testfile.txt")
	os.WriteFile(testFile, []byte("test content"), 0644)

	conn, reader, writer := connectToServer(t, addr)
	defer conn.Close()

	// Read welcome and login
	readResponse(reader)
	sendCommand(writer, "USER anonymous")
	readResponse(reader)

	// Test CWD to existing directory
	sendCommand(writer, "CWD testdir")
	code, _, err := readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read CWD response: %v", err)
	}
	if code != CodeFileActionOK {
		t.Errorf("Expected code %d for CWD, got %d", CodeFileActionOK, code)
	}

	// Test PWD after CWD
	sendCommand(writer, "PWD")
	code, message, err := readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read PWD response: %v", err)
	}
	if !strings.Contains(message, "testdir") {
		t.Errorf("PWD should show testdir: %s", message)
	}

	// Test CWD to non-existent directory
	sendCommand(writer, "CWD nonexistent")
	code, _, err = readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read CWD error response: %v", err)
	}
	if code != CodeFileNotFound {
		t.Errorf("Expected code %d for non-existent dir, got %d", CodeFileNotFound, code)
	}
}

func TestFTPPassiveMode(t *testing.T) {
	server, addr, cleanup := createTestServer(t)
	defer cleanup()

	conn, reader, writer := connectToServer(t, addr)
	defer conn.Close()

	// Login
	readResponse(reader)
	sendCommand(writer, "USER anonymous")
	readResponse(reader)

	// Test PASV command
	sendCommand(writer, "PASV")
	code, message, err := readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read PASV response: %v", err)
	}

	if code != CodePassiveMode {
		t.Errorf("Expected code %d for PASV, got %d", CodePassiveMode, code)
	}

	if !strings.Contains(message, "Passive Mode") {
		t.Errorf("PASV response should mention passive mode: %s", message)
	}

	// The response should contain IP and port information
	if !strings.Contains(message, ",") {
		t.Errorf("PASV response should contain comma-separated values: %s", message)
	}
}

func TestFTPTypeCommand(t *testing.T) {
	server, addr, cleanup := createTestServer(t)
	defer cleanup()

	conn, reader, writer := connectToServer(t, addr)
	defer conn.Close()

	// Login
	readResponse(reader)
	sendCommand(writer, "USER anonymous")
	readResponse(reader)

	// Test TYPE A (ASCII)
	sendCommand(writer, "TYPE A")
	code, message, err := readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read TYPE A response: %v", err)
	}
	if code != CodeCommandOK {
		t.Errorf("Expected code %d for TYPE A, got %d", CodeCommandOK, code)
	}
	if !strings.Contains(strings.ToLower(message), "ascii") {
		t.Errorf("TYPE A response should mention ASCII: %s", message)
	}

	// Test TYPE I (Binary)
	sendCommand(writer, "TYPE I")
	code, message, err = readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read TYPE I response: %v", err)
	}
	if code != CodeCommandOK {
		t.Errorf("Expected code %d for TYPE I, got %d", CodeCommandOK, code)
	}
	if !strings.Contains(strings.ToLower(message), "binary") {
		t.Errorf("TYPE I response should mention binary: %s", message)
	}

	// Test invalid type
	sendCommand(writer, "TYPE X")
	code, _, err = readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read TYPE X response: %v", err)
	}
	if code != CodeParameterNotImplemented {
		t.Errorf("Expected code %d for invalid type, got %d", CodeParameterNotImplemented, code)
	}
}

func TestFTPUnknownCommand(t *testing.T) {
	server, addr, cleanup := createTestServer(t)
	defer cleanup()

	conn, reader, writer := connectToServer(t, addr)
	defer conn.Close()

	// Read welcome
	readResponse(reader)

	// Test unknown command
	sendCommand(writer, "UNKNOWN")
	code, _, err := readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read UNKNOWN response: %v", err)
	}
	if code != CodeCommandNotImplemented {
		t.Errorf("Expected code %d for unknown command, got %d", CodeCommandNotImplemented, code)
	}
}

func TestFTPMaxClients(t *testing.T) {
	tempDir := t.TempDir()
	
	config := ServerConfig{
		Address:        "127.0.0.1",
		Port:          0,
		MaxClients:    2, // Very low limit for testing
		RootDirectory: tempDir,
		AllowAnonymous: true,
		Timeout:       time.Second,
		EnableLogging: false,
	}

	server := NewFTPServer(config)
	if err := server.Start(); err != nil {
		t.Fatalf("Failed to start server: %v", err)
	}
	defer server.Stop()

	addr := server.listener.Addr().String()

	// Connect first two clients
	conn1, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("Failed to connect client 1: %v", err)
	}
	defer conn1.Close()

	conn2, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("Failed to connect client 2: %v", err)
	}
	defer conn2.Close()

	// Try to connect third client - should be rejected
	conn3, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("Failed to connect client 3: %v", err)
	}
	defer conn3.Close()

	// Read response from third client
	reader := bufio.NewReader(conn3)
	code, _, err := readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read rejection: %v", err)
	}

	if code != CodeInsufficientStorage {
		t.Errorf("Expected rejection code %d, got %d", CodeInsufficientStorage, code)
	}
}

func TestFTPConcurrentClients(t *testing.T) {
	server, addr, cleanup := createTestServer(t)
	defer cleanup()

	numClients := 5
	var wg sync.WaitGroup

	// Test concurrent clients
	for i := 0; i < numClients; i++ {
		wg.Add(1)
		go func(clientID int) {
			defer wg.Done()

			conn, reader, writer := connectToServer(t, addr)
			defer conn.Close()

			// Read welcome
			readResponse(reader)

			// Login
			sendCommand(writer, "USER anonymous")
			readResponse(reader)

			// Send some commands
			commands := []string{"PWD", "SYST", "NOOP"}
			for _, cmd := range commands {
				sendCommand(writer, cmd)
				code, _, err := readResponse(reader)
				if err != nil {
					t.Errorf("Client %d command %s failed: %v", clientID, cmd, err)
					return
				}
				if code < 200 || code >= 300 {
					t.Errorf("Client %d command %s returned error code: %d", clientID, cmd, code)
				}
			}

			// Quit
			sendCommand(writer, "QUIT")
			readResponse(reader)
		}(i)
	}

	wg.Wait()

	// Check metrics
	metrics := server.GetMetrics()
	if metrics.TotalConnections < int64(numClients) {
		t.Errorf("Expected at least %d connections, got %d", numClients, metrics.TotalConnections)
	}
}

func TestFTPMetrics(t *testing.T) {
	server, addr, cleanup := createTestServer(t)
	defer cleanup()

	initialMetrics := server.GetMetrics()

	conn, reader, writer := connectToServer(t, addr)
	defer conn.Close()

	// Read welcome
	readResponse(reader)

	// Send some commands
	commands := []string{"USER anonymous", "PWD", "SYST", "NOOP"}
	for _, cmd := range commands {
		sendCommand(writer, cmd)
		readResponse(reader)
	}

	finalMetrics := server.GetMetrics()

	// Check that metrics increased
	if finalMetrics.TotalConnections <= initialMetrics.TotalConnections {
		t.Error("Total connections should have increased")
	}

	if finalMetrics.TotalCommands <= initialMetrics.TotalCommands {
		t.Error("Total commands should have increased")
	}

	if finalMetrics.ActiveConnections <= 0 {
		t.Error("Should have active connections")
	}
}

func TestFTPMiddleware(t *testing.T) {
	server, addr, cleanup := createTestServer(t)
	defer cleanup()

	// Add test middleware
	middlewareCalled := false
	server.AddMiddleware(func(client *Client, cmd Command, next CommandHandler) Response {
		middlewareCalled = true
		return next(client, cmd)
	})

	conn, reader, writer := connectToServer(t, addr)
	defer conn.Close()

	// Read welcome
	readResponse(reader)

	// Send command
	sendCommand(writer, "SYST")
	readResponse(reader)

	if !middlewareCalled {
		t.Error("Middleware should have been called")
	}
}

func TestFTPCommandParsing(t *testing.T) {
	server := &FTPServer{}

	tests := []struct {
		input    string
		expected Command
	}{
		{
			input: "USER testuser",
			expected: Command{
				Name: "USER",
				Args: []string{"testuser"},
				Raw:  "USER testuser",
			},
		},
		{
			input: "CWD /home/user",
			expected: Command{
				Name: "CWD",
				Args: []string{"/home/user"},
				Raw:  "CWD /home/user",
			},
		},
		{
			input: "NOOP",
			expected: Command{
				Name: "NOOP",
				Args: []string{},
				Raw:  "NOOP",
			},
		},
		{
			input: "",
			expected: Command{
				Name: "",
				Args: []string{},
				Raw:  "",
			},
		},
	}

	for _, test := range tests {
		result := server.parseCommand(test.input)
		
		if result.Name != test.expected.Name {
			t.Errorf("Command name: expected %s, got %s", test.expected.Name, result.Name)
		}
		
		if len(result.Args) != len(test.expected.Args) {
			t.Errorf("Args length: expected %d, got %d", len(test.expected.Args), len(result.Args))
			continue
		}
		
		for i, arg := range result.Args {
			if arg != test.expected.Args[i] {
				t.Errorf("Arg %d: expected %s, got %s", i, test.expected.Args[i], arg)
			}
		}
		
		if result.Raw != test.expected.Raw {
			t.Errorf("Raw: expected %s, got %s", test.expected.Raw, result.Raw)
		}
	}
}

func TestFTPUserPermissions(t *testing.T) {
	server, addr, cleanup := createTestServer(t)
	defer cleanup()

	// Add user with limited permissions
	server.AddUser("limiteduser", UserConfig{
		Username: "limiteduser",
		Password: "pass",
		HomeDir:  server.config.RootDirectory,
		Permissions: UserPermissions{
			Read: true, Write: false, Delete: false,
			Rename: false, List: true, Upload: false, Download: false,
		},
	})

	conn, reader, writer := connectToServer(t, addr)
	defer conn.Close()

	// Read welcome
	readResponse(reader)

	// Login as limited user
	sendCommand(writer, "USER limiteduser")
	readResponse(reader)
	sendCommand(writer, "PASS pass")
	readResponse(reader)

	// Test command that requires write permission
	sendCommand(writer, "STOR testfile.txt")
	code, _, err := readResponse(reader)
	if err != nil {
		t.Fatalf("Failed to read STOR response: %v", err)
	}

	if code != CodeFileActionAborted {
		t.Errorf("Expected permission denied code %d, got %d", CodeFileActionAborted, code)
	}
}

func TestFTPTimeout(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping timeout test in short mode")
	}

	tempDir := t.TempDir()
	
	config := ServerConfig{
		Address:        "127.0.0.1",
		Port:          0,
		MaxClients:    10,
		RootDirectory: tempDir,
		AllowAnonymous: true,
		Timeout:       100 * time.Millisecond, // Very short timeout
		EnableLogging: false,
	}

	server := NewFTPServer(config)
	if err := server.Start(); err != nil {
		t.Fatalf("Failed to start server: %v", err)
	}
	defer server.Stop()

	addr := server.listener.Addr().String()

	conn, reader, _ := connectToServer(t, addr)
	defer conn.Close()

	// Read welcome
	readResponse(reader)

	// Wait for timeout
	time.Sleep(200 * time.Millisecond)

	// Try to read - should fail due to timeout
	_, err := reader.ReadString('\n')
	if err == nil {
		t.Error("Expected timeout error")
	}
}

func BenchmarkFTPConnections(b *testing.B) {
	server, addr, cleanup := createTestServer(&testing.T{})
	defer cleanup()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		conn, reader, writer := connectToServer(&testing.T{}, addr)
		
		// Read welcome
		readResponse(reader)
		
		// Quick login and quit
		sendCommand(writer, "USER anonymous")
		readResponse(reader)
		sendCommand(writer, "QUIT")
		readResponse(reader)
		
		conn.Close()
	}
}

func BenchmarkFTPCommands(b *testing.B) {
	server, addr, cleanup := createTestServer(&testing.T{})
	defer cleanup()

	conn, reader, writer := connectToServer(&testing.T{}, addr)
	defer conn.Close()

	// Setup
	readResponse(reader)
	sendCommand(writer, "USER anonymous")
	readResponse(reader)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		sendCommand(writer, "NOOP")
		readResponse(reader)
	}
}