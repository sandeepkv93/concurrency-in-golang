# Concurrent Chat Server

## Problem Description

Building a real-time chat server requires handling multiple concurrent connections while maintaining message ordering and implementing features like:

1. Multiple clients connecting and disconnecting simultaneously
2. Real-time message broadcasting to all connected clients
3. Private messaging between specific users
4. Chat rooms with user management
5. Command processing and system messages
6. Connection state management and cleanup

## Solution Approach

This implementation provides a full-featured concurrent chat server using Go's concurrency primitives:

### Key Components

1. **Server Architecture**: Central server managing multiple client connections
2. **Client Management**: Individual goroutines for each client connection
3. **Room System**: Multiple chat rooms with user management
4. **Message Routing**: Efficient message distribution system
5. **Command Processing**: Built-in command system for server operations
6. **Connection Management**: Robust connection lifecycle handling

### Concurrency Model

- **One Goroutine per Client**: Each connection handled by dedicated goroutine
- **Message Broadcasting**: Fan-out pattern for message distribution
- **Room Management**: Thread-safe room operations with mutex protection
- **Command Processing**: Separate goroutine for command execution
- **Connection Cleanup**: Graceful handling of client disconnections

### Implementation Details

- **TCP Server**: Raw TCP socket handling for efficient communication
- **Message Types**: Support for public, private, system, and command messages
- **User Authentication**: Basic nickname-based user identification
- **Room Operations**: Join, leave, list users, and room management
- **Command System**: Built-in commands like /join, /leave, /users, /msg
- **Connection Pooling**: Efficient connection management and reuse

## Usage Example

```go
// Start server
server := NewChatServer(ServerConfig{
    Address:    "localhost",
    Port:       8080,
    MaxClients: 100,
})

go server.Start()

// Client connection handling
client := &Client{
    ID:       generateID(),
    Nickname: "user1",
    Conn:     conn,
    Server:   server,
    Send:     make(chan *Message, 100),
}

server.RegisterClient(client)
```

## Technical Features

- **Concurrent Connection Handling**: Multiple simultaneous client connections
- **Real-time Messaging**: Instant message delivery with low latency
- **Room-based Communication**: Multiple chat rooms with user management
- **Private Messaging**: Direct messages between users
- **Command System**: Built-in server commands and extensible command framework
- **Connection Management**: Automatic cleanup of disconnected clients
- **Message History**: Optional message logging and history
- **User Management**: Nickname registration and duplicate handling
- **Graceful Shutdown**: Clean server shutdown with client notification

## Message Types

- **Public Messages**: Broadcast to all users in a room
- **Private Messages**: Direct messages between specific users
- **System Messages**: Server notifications and status updates
- **Join/Leave Messages**: User connection status updates
- **Command Messages**: Server command execution and responses

## Command System

Built-in commands include:
- `/join <room>` - Join a specific chat room
- `/leave` - Leave current room
- `/users` - List users in current room
- `/rooms` - List available rooms
- `/msg <user> <message>` - Send private message
- `/nick <nickname>` - Change nickname
- `/help` - Show available commands

## Advanced Features

- **Message Filtering**: Content filtering and moderation
- **Rate Limiting**: Prevent message flooding
- **Connection Limits**: Maximum clients per room/server
- **Logging**: Comprehensive logging of all activities
- **Metrics**: Real-time server statistics and monitoring
- **Plugin System**: Extensible middleware for custom functionality

## Testing

The implementation includes comprehensive tests covering:
- Multiple concurrent client connections
- Message broadcasting and delivery
- Room management operations
- Command processing and execution
- Connection lifecycle management
- Error handling and recovery
- Performance under load
- Race condition prevention