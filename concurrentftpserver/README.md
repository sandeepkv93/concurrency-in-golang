# Concurrent FTP Server

## Problem Description

Implementing a fully-featured FTP server requires handling multiple concurrent client connections while supporting the complex FTP protocol requirements:

1. Multiple simultaneous client connections
2. FTP protocol command handling (USER, PASS, LIST, RETR, STOR, etc.)
3. Active and passive data connection modes
4. File upload and download operations
5. Directory navigation and management
6. User authentication and access control
7. Connection state management

## Solution Approach

This implementation provides a comprehensive FTP server with full concurrent client support:

### Key Components

1. **Server Core**: Central FTP server managing multiple client sessions
2. **Session Management**: Individual FTP sessions with state tracking
3. **Command Processing**: Complete FTP command set implementation
4. **Data Connections**: Both active and passive data transfer modes
5. **Authentication**: User-based access control with permissions
6. **File Operations**: Concurrent file upload/download handling

### Concurrency Model

- **One Goroutine per Client**: Each FTP session handled by dedicated goroutine
- **Separate Control/Data Channels**: Independent handling of command and data connections
- **Session State Management**: Thread-safe session state tracking
- **Concurrent File Operations**: Multiple simultaneous file transfers
- **Connection Pooling**: Efficient management of data connections

### Implementation Details

- **FTP Protocol Compliance**: Full RFC 959 compliance with common extensions
- **Command Parser**: Robust parsing of FTP commands and parameters
- **Data Connection Management**: Dynamic port allocation for passive mode
- **File System Operations**: Safe file operations with permission checking
- **User Management**: Configurable user accounts with directory restrictions
- **Logging and Monitoring**: Comprehensive logging and metrics collection

## Usage Example

```go
config := ServerConfig{
    Address:        "localhost",
    Port:           21,
    MaxClients:     50,
    DataPortRange:  PortRange{Start: 2000, End: 2100},
    RootDirectory:  "/ftp",
    AllowAnonymous: false,
    Users: map[string]UserConfig{
        "user1": {
            Username: "user1",
            Password: "pass1",
            HomeDir:  "/ftp/user1",
            Permissions: []string{"read", "write"},
        },
    },
}

server := NewFTPServer(config)
go server.Start()
```

## Technical Features

- **Full FTP Protocol Support**: Complete implementation of FTP commands
- **Concurrent Client Handling**: Multiple simultaneous FTP sessions
- **Active/Passive Modes**: Support for both data connection modes
- **User Authentication**: Username/password authentication with permissions
- **Directory Management**: Full directory navigation and management
- **File Transfer**: Concurrent upload/download operations
- **Connection Management**: Automatic cleanup of inactive sessions
- **Security**: Permission-based access control and path validation
- **Logging**: Comprehensive activity logging and audit trails

## Supported FTP Commands

### Authentication Commands
- `USER` - Specify username
- `PASS` - Specify password
- `QUIT` - Logout and close connection

### Data Connection Commands
- `PASV` - Enter passive mode
- `PORT` - Specify active mode data port

### File Operations
- `RETR` - Download file
- `STOR` - Upload file
- `DELE` - Delete file
- `RNFR/RNTO` - Rename file

### Directory Operations
- `LIST` - List directory contents
- `NLST` - List filenames only
- `CWD` - Change working directory
- `PWD` - Print working directory
- `MKD` - Create directory
- `RMD` - Remove directory

### System Commands
- `SYST` - Show system information
- `FEAT` - Show server features
- `NOOP` - No operation (keepalive)

## Advanced Features

- **Middleware System**: Extensible middleware for custom functionality
- **Rate Limiting**: Bandwidth throttling and connection rate limiting
- **SSL/TLS Support**: Secure FTP connections (FTPS)
- **Virtual File System**: Abstracted file system operations
- **Connection Monitoring**: Real-time connection and transfer monitoring
- **Configuration Management**: Dynamic configuration updates
- **Performance Metrics**: Detailed server and client statistics

## Security Features

- **User Permissions**: Fine-grained permission control
- **Path Validation**: Prevent directory traversal attacks
- **Connection Limits**: Maximum connections per user/IP
- **Timeout Management**: Automatic cleanup of inactive sessions
- **Audit Logging**: Complete audit trail of all operations

## Testing

The implementation includes comprehensive tests covering:
- Multiple concurrent client connections
- FTP command processing and responses
- Active and passive data transfers
- Authentication and authorization
- File upload and download operations
- Directory management operations
- Error handling and protocol compliance
- Performance under concurrent load
- Security and access control validation