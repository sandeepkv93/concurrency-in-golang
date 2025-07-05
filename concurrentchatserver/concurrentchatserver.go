package concurrentchatserver

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Message represents a chat message
type Message struct {
	Type      MessageType
	Sender    string
	Recipient string
	Content   string
	Timestamp time.Time
}

// MessageType defines different message types
type MessageType int

const (
	PublicMessage MessageType = iota
	PrivateMessage
	SystemMessage
	JoinMessage
	LeaveMessage
	CommandMessage
)

// Client represents a connected chat client
type Client struct {
	ID       string
	Nickname string
	Conn     net.Conn
	Server   *ChatServer
	Send     chan *Message
	Room     *Room
	reader   *bufio.Reader
	writer   *bufio.Writer
	active   atomic.Bool
}

// Room represents a chat room
type Room struct {
	Name        string
	Description string
	Clients     sync.Map // ID -> *Client
	History     []*Message
	historyLock sync.RWMutex
	MaxHistory  int
	Created     time.Time
	Topic       string
	Private     bool
	Password    string
}

// ChatServer represents the chat server
type ChatServer struct {
	Rooms       sync.Map // name -> *Room
	Clients     sync.Map // ID -> *Client
	Commands    map[string]CommandHandler
	listener    net.Listener
	quit        chan bool
	wg          sync.WaitGroup
	idCounter   atomic.Int64
	config      ServerConfig
}

// ServerConfig holds server configuration
type ServerConfig struct {
	Address         string
	MaxClients      int
	MaxRooms        int
	MaxMessageSize  int
	MaxHistorySize  int
	IdleTimeout     time.Duration
	DefaultRoomName string
}

// CommandHandler handles chat commands
type CommandHandler func(client *Client, args []string) error

// NewChatServer creates a new chat server
func NewChatServer(config ServerConfig) *ChatServer {
	server := &ChatServer{
		Commands: make(map[string]CommandHandler),
		quit:     make(chan bool),
		config:   config,
	}
	
	// Set defaults
	if server.config.MaxClients <= 0 {
		server.config.MaxClients = 100
	}
	if server.config.MaxRooms <= 0 {
		server.config.MaxRooms = 10
	}
	if server.config.MaxMessageSize <= 0 {
		server.config.MaxMessageSize = 1024
	}
	if server.config.MaxHistorySize <= 0 {
		server.config.MaxHistorySize = 100
	}
	if server.config.IdleTimeout <= 0 {
		server.config.IdleTimeout = 30 * time.Minute
	}
	if server.config.DefaultRoomName == "" {
		server.config.DefaultRoomName = "lobby"
	}
	
	// Register default commands
	server.registerDefaultCommands()
	
	// Create default room
	server.CreateRoom(server.config.DefaultRoomName, "Welcome to the chat server!", false, "")
	
	return server
}

// Start starts the chat server
func (s *ChatServer) Start() error {
	listener, err := net.Listen("tcp", s.config.Address)
	if err != nil {
		return fmt.Errorf("failed to start server: %w", err)
	}
	
	s.listener = listener
	log.Printf("Chat server started on %s", s.config.Address)
	
	s.wg.Add(1)
	go s.acceptConnections()
	
	return nil
}

// Stop stops the chat server
func (s *ChatServer) Stop() {
	log.Println("Shutting down chat server...")
	
	close(s.quit)
	
	if s.listener != nil {
		s.listener.Close()
	}
	
	// Disconnect all clients
	s.Clients.Range(func(key, value interface{}) bool {
		client := value.(*Client)
		client.Disconnect("Server shutting down")
		return true
	})
	
	s.wg.Wait()
	log.Println("Chat server stopped")
}

func (s *ChatServer) acceptConnections() {
	defer s.wg.Done()
	
	for {
		conn, err := s.listener.Accept()
		if err != nil {
			select {
			case <-s.quit:
				return
			default:
				log.Printf("Accept error: %v", err)
				continue
			}
		}
		
		// Check max clients
		clientCount := 0
		s.Clients.Range(func(_, _ interface{}) bool {
			clientCount++
			return clientCount < s.config.MaxClients
		})
		
		if clientCount >= s.config.MaxClients {
			fmt.Fprintln(conn, "Server full. Please try again later.")
			conn.Close()
			continue
		}
		
		s.wg.Add(1)
		go s.handleConnection(conn)
	}
}

func (s *ChatServer) handleConnection(conn net.Conn) {
	defer s.wg.Done()
	
	// Create client
	clientID := fmt.Sprintf("client_%d", s.idCounter.Add(1))
	client := &Client{
		ID:     clientID,
		Conn:   conn,
		Server: s,
		Send:   make(chan *Message, 10),
		reader: bufio.NewReader(conn),
		writer: bufio.NewWriter(conn),
	}
	client.active.Store(true)
	
	// Register client
	s.Clients.Store(clientID, client)
	
	// Send welcome message
	client.SendMessage(&Message{
		Type:      SystemMessage,
		Content:   "Welcome to the chat server! Please set your nickname with /nick <name>",
		Timestamp: time.Now(),
	})
	
	// Start client handlers
	s.wg.Add(2)
	go client.readLoop()
	go client.writeLoop()
	
	log.Printf("Client %s connected from %s", clientID, conn.RemoteAddr())
}

// Client methods

func (c *Client) readLoop() {
	defer c.Server.wg.Done()
	defer c.Disconnect("Connection closed")
	
	for c.active.Load() {
		// Set read deadline
		c.Conn.SetReadDeadline(time.Now().Add(c.Server.config.IdleTimeout))
		
		line, err := c.reader.ReadString('\n')
		if err != nil {
			if c.active.Load() {
				log.Printf("Read error for %s: %v", c.ID, err)
			}
			return
		}
		
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		
		// Check message size
		if len(line) > c.Server.config.MaxMessageSize {
			c.SendMessage(&Message{
				Type:    SystemMessage,
				Content: "Message too long",
			})
			continue
		}
		
		// Handle command or message
		if strings.HasPrefix(line, "/") {
			c.handleCommand(line)
		} else {
			c.handleMessage(line)
		}
	}
}

func (c *Client) writeLoop() {
	defer c.Server.wg.Done()
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for c.active.Load() {
		select {
		case message, ok := <-c.Send:
			if !ok {
				return
			}
			
			// Set write deadline
			c.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			
			if err := c.writeMessage(message); err != nil {
				log.Printf("Write error for %s: %v", c.ID, err)
				return
			}
			
		case <-ticker.C:
			// Send ping to keep connection alive
			c.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if _, err := c.writer.WriteString("PING\n"); err != nil {
				return
			}
			c.writer.Flush()
		}
	}
}

func (c *Client) writeMessage(msg *Message) error {
	var formatted string
	
	switch msg.Type {
	case PublicMessage:
		formatted = fmt.Sprintf("[%s] %s: %s\n", 
			msg.Timestamp.Format("15:04:05"), msg.Sender, msg.Content)
	case PrivateMessage:
		formatted = fmt.Sprintf("[%s] [PM from %s]: %s\n", 
			msg.Timestamp.Format("15:04:05"), msg.Sender, msg.Content)
	case SystemMessage:
		formatted = fmt.Sprintf("[SYSTEM] %s\n", msg.Content)
	case JoinMessage:
		formatted = fmt.Sprintf("*** %s joined the room\n", msg.Sender)
	case LeaveMessage:
		formatted = fmt.Sprintf("*** %s left the room\n", msg.Sender)
	default:
		formatted = fmt.Sprintf("%s\n", msg.Content)
	}
	
	_, err := c.writer.WriteString(formatted)
	if err != nil {
		return err
	}
	
	return c.writer.Flush()
}

func (c *Client) handleCommand(line string) {
	parts := strings.Fields(line)
	if len(parts) == 0 {
		return
	}
	
	cmd := strings.ToLower(parts[0])
	args := parts[1:]
	
	handler, exists := c.Server.Commands[cmd]
	if !exists {
		c.SendMessage(&Message{
			Type:    SystemMessage,
			Content: fmt.Sprintf("Unknown command: %s", cmd),
		})
		return
	}
	
	if err := handler(c, args); err != nil {
		c.SendMessage(&Message{
			Type:    SystemMessage,
			Content: fmt.Sprintf("Error: %s", err.Error()),
		})
	}
}

func (c *Client) handleMessage(content string) {
	if c.Nickname == "" {
		c.SendMessage(&Message{
			Type:    SystemMessage,
			Content: "Please set your nickname first with /nick <name>",
		})
		return
	}
	
	if c.Room == nil {
		c.SendMessage(&Message{
			Type:    SystemMessage,
			Content: "You must join a room first. Use /join <room>",
		})
		return
	}
	
	msg := &Message{
		Type:      PublicMessage,
		Sender:    c.Nickname,
		Content:   content,
		Timestamp: time.Now(),
	}
	
	c.Room.Broadcast(msg, c.ID)
	c.Room.AddToHistory(msg)
}

func (c *Client) SendMessage(msg *Message) {
	select {
	case c.Send <- msg:
	default:
		// Channel full, drop message
		log.Printf("Dropping message for %s: channel full", c.ID)
	}
}

func (c *Client) Disconnect(reason string) {
	if !c.active.CompareAndSwap(true, false) {
		return
	}
	
	// Leave room
	if c.Room != nil {
		c.Room.RemoveClient(c)
		if c.Nickname != "" {
			c.Room.Broadcast(&Message{
				Type:      LeaveMessage,
				Sender:    c.Nickname,
				Timestamp: time.Now(),
			}, c.ID)
		}
	}
	
	// Remove from server
	c.Server.Clients.Delete(c.ID)
	
	// Close connection
	close(c.Send)
	c.Conn.Close()
	
	log.Printf("Client %s disconnected: %s", c.ID, reason)
}

// Room methods

func (r *Room) AddClient(client *Client) error {
	// Check if room is full (optional limit)
	count := 0
	r.Clients.Range(func(_, _ interface{}) bool {
		count++
		return count < 50 // Max 50 clients per room
	})
	
	if count >= 50 {
		return fmt.Errorf("room is full")
	}
	
	// Add client
	r.Clients.Store(client.ID, client)
	client.Room = r
	
	// Send join message
	if client.Nickname != "" {
		r.Broadcast(&Message{
			Type:      JoinMessage,
			Sender:    client.Nickname,
			Timestamp: time.Now(),
		}, client.ID)
	}
	
	// Send room topic
	if r.Topic != "" {
		client.SendMessage(&Message{
			Type:    SystemMessage,
			Content: fmt.Sprintf("Room topic: %s", r.Topic),
		})
	}
	
	// Send recent history
	r.SendHistory(client, 10)
	
	return nil
}

func (r *Room) RemoveClient(client *Client) {
	r.Clients.Delete(client.ID)
	client.Room = nil
}

func (r *Room) Broadcast(msg *Message, excludeID string) {
	r.Clients.Range(func(key, value interface{}) bool {
		if key.(string) != excludeID {
			client := value.(*Client)
			client.SendMessage(msg)
		}
		return true
	})
}

func (r *Room) AddToHistory(msg *Message) {
	r.historyLock.Lock()
	defer r.historyLock.Unlock()
	
	r.History = append(r.History, msg)
	
	// Trim history if needed
	if len(r.History) > r.MaxHistory {
		r.History = r.History[len(r.History)-r.MaxHistory:]
	}
}

func (r *Room) SendHistory(client *Client, count int) {
	r.historyLock.RLock()
	defer r.historyLock.RUnlock()
	
	start := len(r.History) - count
	if start < 0 {
		start = 0
	}
	
	for i := start; i < len(r.History); i++ {
		client.SendMessage(r.History[i])
	}
}

// Server methods

func (s *ChatServer) CreateRoom(name, description string, private bool, password string) (*Room, error) {
	// Check if room exists
	if _, exists := s.Rooms.Load(name); exists {
		return nil, fmt.Errorf("room already exists")
	}
	
	// Check room limit
	roomCount := 0
	s.Rooms.Range(func(_, _ interface{}) bool {
		roomCount++
		return roomCount < s.config.MaxRooms
	})
	
	if roomCount >= s.config.MaxRooms {
		return nil, fmt.Errorf("maximum number of rooms reached")
	}
	
	room := &Room{
		Name:        name,
		Description: description,
		MaxHistory:  s.config.MaxHistorySize,
		Created:     time.Now(),
		Private:     private,
		Password:    password,
	}
	
	s.Rooms.Store(name, room)
	return room, nil
}

func (s *ChatServer) registerDefaultCommands() {
	// /nick command
	s.Commands["/nick"] = func(client *Client, args []string) error {
		if len(args) != 1 {
			return fmt.Errorf("usage: /nick <nickname>")
		}
		
		newNick := args[0]
		if len(newNick) < 3 || len(newNick) > 20 {
			return fmt.Errorf("nickname must be 3-20 characters")
		}
		
		// Check if nickname is taken
		taken := false
		s.Clients.Range(func(_, value interface{}) bool {
			c := value.(*Client)
			if c.Nickname == newNick && c.ID != client.ID {
				taken = true
				return false
			}
			return true
		})
		
		if taken {
			return fmt.Errorf("nickname already taken")
		}
		
		oldNick := client.Nickname
		client.Nickname = newNick
		
		if oldNick == "" {
			client.SendMessage(&Message{
				Type:    SystemMessage,
				Content: fmt.Sprintf("Nickname set to %s", newNick),
			})
		} else {
			client.SendMessage(&Message{
				Type:    SystemMessage,
				Content: fmt.Sprintf("Nickname changed from %s to %s", oldNick, newNick),
			})
		}
		
		return nil
	}
	
	// /join command
	s.Commands["/join"] = func(client *Client, args []string) error {
		if len(args) < 1 {
			return fmt.Errorf("usage: /join <room> [password]")
		}
		
		roomName := args[0]
		password := ""
		if len(args) > 1 {
			password = args[1]
		}
		
		// Leave current room
		if client.Room != nil {
			client.Room.RemoveClient(client)
			if client.Nickname != "" {
				client.Room.Broadcast(&Message{
					Type:      LeaveMessage,
					Sender:    client.Nickname,
					Timestamp: time.Now(),
				}, client.ID)
			}
		}
		
		// Find or create room
		var room *Room
		if r, exists := s.Rooms.Load(roomName); exists {
			room = r.(*Room)
			
			// Check password
			if room.Private && room.Password != password {
				return fmt.Errorf("incorrect password")
			}
		} else {
			// Create new room
			r, err := s.CreateRoom(roomName, "", false, "")
			if err != nil {
				return err
			}
			room = r
		}
		
		// Join room
		if err := room.AddClient(client); err != nil {
			return err
		}
		
		client.SendMessage(&Message{
			Type:    SystemMessage,
			Content: fmt.Sprintf("Joined room: %s", roomName),
		})
		
		return nil
	}
	
	// /list command
	s.Commands["/list"] = func(client *Client, args []string) error {
		client.SendMessage(&Message{
			Type:    SystemMessage,
			Content: "Available rooms:",
		})
		
		s.Rooms.Range(func(key, value interface{}) bool {
			room := value.(*Room)
			count := 0
			room.Clients.Range(func(_, _ interface{}) bool {
				count++
				return true
			})
			
			status := ""
			if room.Private {
				status = " [Private]"
			}
			
			client.SendMessage(&Message{
				Type:    SystemMessage,
				Content: fmt.Sprintf("  %s (%d users)%s - %s", 
					room.Name, count, status, room.Description),
			})
			return true
		})
		
		return nil
	}
	
	// /msg command (private message)
	s.Commands["/msg"] = func(client *Client, args []string) error {
		if len(args) < 2 {
			return fmt.Errorf("usage: /msg <nickname> <message>")
		}
		
		targetNick := args[0]
		message := strings.Join(args[1:], " ")
		
		// Find target client
		var target *Client
		s.Clients.Range(func(_, value interface{}) bool {
			c := value.(*Client)
			if c.Nickname == targetNick {
				target = c
				return false
			}
			return true
		})
		
		if target == nil {
			return fmt.Errorf("user not found: %s", targetNick)
		}
		
		// Send private message
		target.SendMessage(&Message{
			Type:      PrivateMessage,
			Sender:    client.Nickname,
			Content:   message,
			Timestamp: time.Now(),
		})
		
		// Confirm to sender
		client.SendMessage(&Message{
			Type:    SystemMessage,
			Content: fmt.Sprintf("Message sent to %s", targetNick),
		})
		
		return nil
	}
	
	// /help command
	s.Commands["/help"] = func(client *Client, args []string) error {
		client.SendMessage(&Message{
			Type: SystemMessage,
			Content: `Available commands:
  /nick <name>         - Set your nickname
  /join <room> [pass]  - Join a room
  /list               - List all rooms
  /msg <user> <text>  - Send private message
  /quit               - Disconnect from server
  /help               - Show this help`,
		})
		return nil
	}
	
	// /quit command
	s.Commands["/quit"] = func(client *Client, args []string) error {
		client.Disconnect("User quit")
		return nil
	}
}

// Example demonstrates the chat server
func Example() {
	fmt.Println("=== Concurrent Chat Server Example ===")
	
	// Create server
	config := ServerConfig{
		Address:         ":8080",
		MaxClients:      100,
		MaxRooms:        10,
		DefaultRoomName: "lobby",
	}
	
	server := NewChatServer(config)
	
	// Start server
	if err := server.Start(); err != nil {
		log.Fatal(err)
	}
	
	fmt.Println("Chat server running on :8080")
	fmt.Println("Connect with: telnet localhost 8080")
	
	// Run for a while (in real usage, run until interrupted)
	time.Sleep(5 * time.Second)
	
	// Stop server
	server.Stop()
}