package concurrentchatserver

import (
	"bufio"
	"fmt"
	"net"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestChatServer(t *testing.T) {
	config := ServerConfig{
		Address:         "localhost:0", // Random port
		MaxClients:      10,
		MaxRooms:        5,
		DefaultRoomName: "test-lobby",
	}
	
	server := NewChatServer(config)
	
	// Start server
	err := server.Start()
	if err != nil {
		t.Fatalf("Failed to start server: %v", err)
	}
	defer server.Stop()
	
	// Get actual address
	addr := server.listener.Addr().String()
	
	// Test basic connection
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()
	
	reader := bufio.NewReader(conn)
	
	// Should receive welcome message
	line, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("Failed to read welcome message: %v", err)
	}
	
	if !strings.Contains(line, "Welcome") {
		t.Errorf("Expected welcome message, got: %s", line)
	}
}

func TestNickCommand(t *testing.T) {
	server, addr := startTestServer(t)
	defer server.Stop()
	
	client := connectTestClient(t, addr)
	defer client.Close()
	
	// Skip welcome message
	client.ReadLine()
	
	// Set nickname
	client.WriteLine("/nick testuser")
	response := client.ReadLine()
	
	if !strings.Contains(response, "Nickname set to testuser") {
		t.Errorf("Expected nickname confirmation, got: %s", response)
	}
	
	// Try to set invalid nickname
	client.WriteLine("/nick ab")
	response = client.ReadLine()
	
	if !strings.Contains(response, "3-20 characters") {
		t.Errorf("Expected error for short nickname, got: %s", response)
	}
}

func TestJoinRoom(t *testing.T) {
	server, addr := startTestServer(t)
	defer server.Stop()
	
	client := connectTestClient(t, addr)
	defer client.Close()
	
	// Skip welcome
	client.ReadLine()
	
	// Set nickname first
	client.WriteLine("/nick testuser")
	client.ReadLine()
	
	// Join room
	client.WriteLine("/join testroom")
	response := client.ReadLine()
	
	if !strings.Contains(response, "Joined room: testroom") {
		t.Errorf("Expected join confirmation, got: %s", response)
	}
}

func TestPublicMessaging(t *testing.T) {
	server, addr := startTestServer(t)
	defer server.Stop()
	
	// Connect two clients
	client1 := connectTestClient(t, addr)
	defer client1.Close()
	
	client2 := connectTestClient(t, addr)
	defer client2.Close()
	
	// Setup clients
	setupClient(client1, "user1", "chatroom")
	setupClient(client2, "user2", "chatroom")
	
	// Client2 should see join message
	msg := client2.ReadLine()
	if !strings.Contains(msg, "user1 joined") {
		t.Errorf("Expected join message, got: %s", msg)
	}
	
	// Send message from client1
	client1.WriteLine("Hello from user1!")
	
	// Client2 should receive it
	msg = client2.ReadLine()
	if !strings.Contains(msg, "user1: Hello from user1!") {
		t.Errorf("Expected message from user1, got: %s", msg)
	}
}

func TestPrivateMessaging(t *testing.T) {
	server, addr := startTestServer(t)
	defer server.Stop()
	
	// Connect two clients
	client1 := connectTestClient(t, addr)
	defer client1.Close()
	
	client2 := connectTestClient(t, addr)
	defer client2.Close()
	
	// Setup clients
	setupClient(client1, "user1", "chatroom")
	setupClient(client2, "user2", "chatroom")
	
	// Clear join messages
	client2.ReadLine()
	
	// Send private message
	client1.WriteLine("/msg user2 Private hello!")
	
	// Client1 should see confirmation
	msg := client1.ReadLine()
	if !strings.Contains(msg, "Message sent to user2") {
		t.Errorf("Expected confirmation, got: %s", msg)
	}
	
	// Client2 should receive private message
	msg = client2.ReadLine()
	if !strings.Contains(msg, "[PM from user1]: Private hello!") {
		t.Errorf("Expected private message, got: %s", msg)
	}
}

func TestRoomList(t *testing.T) {
	server, addr := startTestServer(t)
	defer server.Stop()
	
	// Create some rooms
	server.CreateRoom("room1", "First room", false, "")
	server.CreateRoom("room2", "Second room", true, "password")
	
	client := connectTestClient(t, addr)
	defer client.Close()
	
	// Skip welcome
	client.ReadLine()
	
	// List rooms
	client.WriteLine("/list")
	
	// Should see header
	msg := client.ReadLine()
	if !strings.Contains(msg, "Available rooms") {
		t.Errorf("Expected room list header, got: %s", msg)
	}
	
	// Should see rooms
	foundRoom1 := false
	foundRoom2 := false
	foundPrivate := false
	
	for i := 0; i < 5; i++ {
		msg = client.ReadLine()
		if strings.Contains(msg, "room1") {
			foundRoom1 = true
		}
		if strings.Contains(msg, "room2") {
			foundRoom2 = true
			if strings.Contains(msg, "[Private]") {
				foundPrivate = true
			}
		}
	}
	
	if !foundRoom1 {
		t.Error("room1 not found in list")
	}
	if !foundRoom2 {
		t.Error("room2 not found in list")
	}
	if !foundPrivate {
		t.Error("room2 not marked as private")
	}
}

func TestMaxClients(t *testing.T) {
	config := ServerConfig{
		Address:    "localhost:0",
		MaxClients: 2,
	}
	
	server := NewChatServer(config)
	err := server.Start()
	if err != nil {
		t.Fatal(err)
	}
	defer server.Stop()
	
	addr := server.listener.Addr().String()
	
	// Connect max clients
	client1, _ := net.Dial("tcp", addr)
	defer client1.Close()
	
	client2, _ := net.Dial("tcp", addr)
	defer client2.Close()
	
	// Try to connect one more
	client3, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer client3.Close()
	
	reader := bufio.NewReader(client3)
	msg, _ := reader.ReadString('\n')
	
	if !strings.Contains(msg, "Server full") {
		t.Errorf("Expected server full message, got: %s", msg)
	}
}

func TestConcurrentMessages(t *testing.T) {
	server, addr := startTestServer(t)
	defer server.Stop()
	
	numClients := 5
	clients := make([]*testClient, numClients)
	
	// Connect and setup clients
	for i := 0; i < numClients; i++ {
		clients[i] = connectTestClient(t, addr)
		defer clients[i].Close()
		
		nickname := fmt.Sprintf("user%d", i)
		setupClient(clients[i], nickname, "stress-test")
		
		// Clear join messages
		for j := 0; j < i; j++ {
			clients[j].ReadLine()
		}
	}
	
	// Send messages concurrently
	var wg sync.WaitGroup
	messagesPerClient := 10
	
	for i := 0; i < numClients; i++ {
		wg.Add(1)
		go func(clientIdx int) {
			defer wg.Done()
			
			for j := 0; j < messagesPerClient; j++ {
				msg := fmt.Sprintf("Message %d from user%d", j, clientIdx)
				clients[clientIdx].WriteLine(msg)
				time.Sleep(10 * time.Millisecond)
			}
		}(i)
	}
	
	wg.Wait()
	
	// Give time for messages to propagate
	time.Sleep(500 * time.Millisecond)
	
	// Each client should have received messages from all others
	for i, client := range clients {
		messageCount := 0
		
		// Count received messages
		for {
			msg := client.ReadLineTimeout(100 * time.Millisecond)
			if msg == "" {
				break
			}
			if strings.Contains(msg, "Message") && strings.Contains(msg, "from user") {
				messageCount++
			}
		}
		
		// Should receive messages from all other clients
		expectedMessages := messagesPerClient * (numClients - 1)
		if messageCount < expectedMessages/2 { // Allow some message loss
			t.Errorf("Client %d received too few messages: %d", i, messageCount)
		}
	}
}

func TestRoomHistory(t *testing.T) {
	server, addr := startTestServer(t)
	defer server.Stop()
	
	// First client sends messages
	client1 := connectTestClient(t, addr)
	setupClient(client1, "user1", "history-test")
	
	// Send some messages
	for i := 0; i < 5; i++ {
		client1.WriteLine(fmt.Sprintf("History message %d", i))
		time.Sleep(10 * time.Millisecond)
	}
	
	client1.Close()
	
	// Second client joins and should see history
	client2 := connectTestClient(t, addr)
	defer client2.Close()
	setupClient(client2, "user2", "history-test")
	
	// Count history messages received
	historyCount := 0
	for i := 0; i < 10; i++ {
		msg := client2.ReadLineTimeout(100 * time.Millisecond)
		if strings.Contains(msg, "History message") {
			historyCount++
		}
	}
	
	if historyCount == 0 {
		t.Error("No history messages received")
	}
}

func TestIdleTimeout(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping idle timeout test in short mode")
	}
	
	config := ServerConfig{
		Address:     "localhost:0",
		IdleTimeout: 500 * time.Millisecond,
	}
	
	server := NewChatServer(config)
	err := server.Start()
	if err != nil {
		t.Fatal(err)
	}
	defer server.Stop()
	
	addr := server.listener.Addr().String()
	
	conn, _ := net.Dial("tcp", addr)
	defer conn.Close()
	
	// Don't send anything, wait for timeout
	time.Sleep(1 * time.Second)
	
	// Try to read, should fail
	reader := bufio.NewReader(conn)
	_, err = reader.ReadString('\n')
	if err == nil {
		t.Error("Expected connection to be closed due to idle timeout")
	}
}

// Helper types and functions

type testClient struct {
	conn   net.Conn
	reader *bufio.Reader
	writer *bufio.Writer
	t      *testing.T
}

func connectTestClient(t *testing.T, addr string) *testClient {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}
	
	return &testClient{
		conn:   conn,
		reader: bufio.NewReader(conn),
		writer: bufio.NewWriter(conn),
		t:      t,
	}
}

func (tc *testClient) WriteLine(line string) {
	_, err := tc.writer.WriteString(line + "\n")
	if err != nil {
		tc.t.Fatalf("Failed to write: %v", err)
	}
	tc.writer.Flush()
}

func (tc *testClient) ReadLine() string {
	line, err := tc.reader.ReadString('\n')
	if err != nil {
		tc.t.Fatalf("Failed to read: %v", err)
	}
	return strings.TrimSpace(line)
}

func (tc *testClient) ReadLineTimeout(timeout time.Duration) string {
	tc.conn.SetReadDeadline(time.Now().Add(timeout))
	line, _ := tc.reader.ReadString('\n')
	tc.conn.SetReadDeadline(time.Time{})
	return strings.TrimSpace(line)
}

func (tc *testClient) Close() {
	tc.conn.Close()
}

func startTestServer(t *testing.T) (*ChatServer, string) {
	config := ServerConfig{
		Address:         "localhost:0",
		MaxClients:      100,
		MaxRooms:        10,
		DefaultRoomName: "test-lobby",
	}
	
	server := NewChatServer(config)
	err := server.Start()
	if err != nil {
		t.Fatalf("Failed to start server: %v", err)
	}
	
	return server, server.listener.Addr().String()
}

func setupClient(client *testClient, nickname, room string) {
	// Skip welcome
	client.ReadLine()
	
	// Set nickname
	client.WriteLine("/nick " + nickname)
	client.ReadLine()
	
	// Join room
	client.WriteLine("/join " + room)
	client.ReadLine()
}

func BenchmarkMessageBroadcast(b *testing.B) {
	server, addr := startTestServer(&testing.T{})
	defer server.Stop()
	
	// Create clients
	numClients := 10
	clients := make([]*testClient, numClients)
	
	for i := 0; i < numClients; i++ {
		clients[i] = connectTestClient(&testing.T{}, addr)
		defer clients[i].Close()
		
		setupClient(clients[i], fmt.Sprintf("user%d", i), "bench-room")
		
		// Clear join messages
		for j := 0; j < i; j++ {
			clients[j].ReadLineTimeout(100 * time.Millisecond)
		}
	}
	
	sender := clients[0]
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		sender.WriteLine(fmt.Sprintf("Benchmark message %d", i))
		
		// Wait for all clients to receive
		for j := 1; j < numClients; j++ {
			clients[j].ReadLineTimeout(100 * time.Millisecond)
		}
	}
}

func BenchmarkConcurrentClients(b *testing.B) {
	server, addr := startTestServer(&testing.T{})
	defer server.Stop()
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			conn, err := net.Dial("tcp", addr)
			if err != nil {
				continue
			}
			
			reader := bufio.NewReader(conn)
			writer := bufio.NewWriter(conn)
			
			// Skip welcome
			reader.ReadString('\n')
			
			// Quick operations
			writer.WriteString("/nick bench\n")
			writer.Flush()
			reader.ReadString('\n')
			
			writer.WriteString("/join bench\n")
			writer.Flush()
			reader.ReadString('\n')
			
			writer.WriteString("test message\n")
			writer.Flush()
			
			conn.Close()
		}
	})
}