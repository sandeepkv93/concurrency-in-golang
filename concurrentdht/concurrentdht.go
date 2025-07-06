package concurrentdht

import (
	"context"
	"crypto/md5"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"hash/fnv"
	"log"
	"math/rand"
	"net"
	"sync"
	"time"
)

// HashFunction defines different hash functions available
type HashFunction int

const (
	FNV1 HashFunction = iota
	FNV1a
	CRC32Hash
	MD5Hash
	SHA1Hash
	SHA256Hash
	ConsistentHash
)

// NodeState represents the state of a DHT node
type NodeState int

const (
	Joining NodeState = iota
	Active
	Leaving
	Failed
	Maintenance
)

// MessageType defines different message types in the DHT
type MessageType int

const (
	JoinRequest MessageType = iota
	JoinResponse
	LeaveRequest
	LeaveResponse
	LookupRequest
	LookupResponse
	StoreRequest
	StoreResponse
	ReplicationRequest
	ReplicationResponse
	HeartbeatRequest
	HeartbeatResponse
	TransferRequest
	TransferResponse
	SuccessorUpdate
	PredecessorUpdate
)

// DHTConfig contains configuration for the DHT
type DHTConfig struct {
	NodeID           string
	Address          string
	Port             int
	HashBits         int           // Number of bits in hash space (e.g., 160 for SHA-1)
	ReplicationFactor int          // Number of replicas for each key
	StabilizeInterval time.Duration
	FixFingerInterval time.Duration
	CheckPredecessor  time.Duration
	HeartbeatInterval time.Duration
	RequestTimeout    time.Duration
	MaxRetries       int
	BufferSize       int
	EnableLogging    bool
	EnableMetrics    bool
	NetworkProtocol  string // "tcp" or "udp"
	HashFunction     HashFunction
	SuccessorListSize int
	BackupReplicas   bool
	ConsistentHashing bool
	VirtualNodes     int
}

// DefaultDHTConfig returns default DHT configuration
func DefaultDHTConfig() DHTConfig {
	return DHTConfig{
		NodeID:            generateNodeID(),
		Address:           "localhost",
		Port:              0, // Auto-assign
		HashBits:          160, // SHA-1 size
		ReplicationFactor: 3,
		StabilizeInterval: 5 * time.Second,
		FixFingerInterval: 10 * time.Second,
		CheckPredecessor:  15 * time.Second,
		HeartbeatInterval: 30 * time.Second,
		RequestTimeout:    10 * time.Second,
		MaxRetries:        3,
		BufferSize:        1000,
		EnableLogging:     true,
		EnableMetrics:     true,
		NetworkProtocol:   "tcp",
		HashFunction:      SHA1Hash,
		SuccessorListSize: 8,
		BackupReplicas:    true,
		ConsistentHashing: true,
		VirtualNodes:      150,
	}
}

// Node represents a node in the DHT
type Node struct {
	ID       string
	Address  string
	Port     int
	State    NodeState
	LastSeen time.Time
	Data     map[string]interface{}
	mutex    sync.RWMutex
}

// NewNode creates a new DHT node
func NewNode(id, address string, port int) *Node {
	return &Node{
		ID:       id,
		Address:  address,
		Port:     port,
		State:    Active,
		LastSeen: time.Now(),
		Data:     make(map[string]interface{}),
	}
}

// GetAddress returns the full address of the node
func (n *Node) GetAddress() string {
	return fmt.Sprintf("%s:%d", n.Address, n.Port)
}

// IsAlive checks if the node is considered alive
func (n *Node) IsAlive(timeout time.Duration) bool {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	return time.Since(n.LastSeen) < timeout
}

// UpdateLastSeen updates the last seen timestamp
func (n *Node) UpdateLastSeen() {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	n.LastSeen = time.Now()
}

// FingerTableEntry represents an entry in the finger table
type FingerTableEntry struct {
	Start uint64
	Node  *Node
	mutex sync.RWMutex
}

// DHT represents the distributed hash table
type DHT struct {
	config          DHTConfig
	localNode       *Node
	predecessor     *Node
	successorList   []*Node
	fingerTable     []*FingerTableEntry
	data            map[string]*DataItem
	virtualNodes    map[string]*Node
	listener        net.Listener
	connections     map[string]net.Conn
	messageHandlers map[MessageType]func(*Message) *Message
	running         bool
	ctx             context.Context
	cancel          context.CancelFunc
	hasher          hash.Hash
	statistics      *DHTStatistics
	mutex           sync.RWMutex
	connMutex       sync.RWMutex
	dataMutex       sync.RWMutex
	wg              sync.WaitGroup
}

// DataItem represents a stored data item in the DHT
type DataItem struct {
	Key         string
	Value       interface{}
	Timestamp   time.Time
	TTL         time.Duration
	Replicas    []string
	Version     uint64
	Metadata    map[string]interface{}
	mutex       sync.RWMutex
}

// IsExpired checks if the data item has expired
func (d *DataItem) IsExpired() bool {
	d.mutex.RLock()
	defer d.mutex.RUnlock()
	if d.TTL == 0 {
		return false
	}
	return time.Since(d.Timestamp) > d.TTL
}

// Message represents a DHT message
type Message struct {
	Type        MessageType            `json:"type"`
	From        string                 `json:"from"`
	To          string                 `json:"to"`
	MessageID   string                 `json:"message_id"`
	Key         string                 `json:"key,omitempty"`
	Value       interface{}            `json:"value,omitempty"`
	Data        map[string]interface{} `json:"data,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
	TTL         time.Duration          `json:"ttl,omitempty"`
	NodeInfo    *Node                  `json:"node_info,omitempty"`
	Nodes       []*Node                `json:"nodes,omitempty"`
	SuccessorList []*Node              `json:"successor_list,omitempty"`
	Error       string                 `json:"error,omitempty"`
}

// DHTStatistics tracks DHT performance metrics
type DHTStatistics struct {
	StartTime         time.Time
	MessagesReceived  uint64
	MessagesSent      uint64
	LookupsSuccessful uint64
	LookupsFailed     uint64
	StoresSuccessful  uint64
	StoresFailed      uint64
	NodesJoined       uint64
	NodesLeft         uint64
	NetworkErrors     uint64
	AverageLookupTime time.Duration
	AverageStoreTime  time.Duration
	DataItemsStored   uint64
	ReplicationsSent  uint64
	HeartbeatsSent    uint64
	FingerTableUpdates uint64
	StabilizeOperations uint64
	mutex             sync.RWMutex
}

// NewDHT creates a new DHT instance
func NewDHT(config DHTConfig) (*DHT, error) {
	if config.NodeID == "" {
		config.NodeID = generateNodeID()
	}
	
	if config.HashBits <= 0 || config.HashBits > 256 {
		return nil, errors.New("invalid hash bits: must be between 1 and 256")
	}
	
	if config.ReplicationFactor <= 0 {
		return nil, errors.New("replication factor must be positive")
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	dht := &DHT{
		config:          config,
		data:            make(map[string]*DataItem),
		virtualNodes:    make(map[string]*Node),
		connections:     make(map[string]net.Conn),
		messageHandlers: make(map[MessageType]func(*Message) *Message),
		running:         false,
		ctx:             ctx,
		cancel:          cancel,
		statistics:      &DHTStatistics{StartTime: time.Now()},
	}
	
	// Initialize hasher
	dht.hasher = dht.createHasher()
	
	// Initialize finger table
	dht.fingerTable = make([]*FingerTableEntry, config.HashBits)
	for i := 0; i < config.HashBits; i++ {
		start := dht.calculateFingerStart(i)
		dht.fingerTable[i] = &FingerTableEntry{
			Start: start,
			Node:  nil,
		}
	}
	
	// Initialize successor list
	dht.successorList = make([]*Node, config.SuccessorListSize)
	
	// Setup message handlers
	dht.setupMessageHandlers()
	
	return dht, nil
}

// createHasher creates the appropriate hasher based on configuration
func (dht *DHT) createHasher() hash.Hash {
	switch dht.config.HashFunction {
	case FNV1:
		return fnv.New64()
	case FNV1a:
		return fnv.New64a()
	case CRC32Hash:
		return crc32.NewIEEE()
	case MD5Hash:
		return md5.New()
	case SHA1Hash:
		return sha1.New()
	case SHA256Hash:
		return sha256.New()
	default:
		return sha1.New()
	}
}

// hashKey hashes a key to a position in the hash space
func (dht *DHT) hashKey(key string) uint64 {
	dht.hasher.Reset()
	dht.hasher.Write([]byte(key))
	hashBytes := dht.hasher.Sum(nil)
	
	// Convert hash to uint64
	if len(hashBytes) >= 8 {
		return binary.BigEndian.Uint64(hashBytes[:8])
	}
	
	// For smaller hashes, pad with zeros
	var result uint64
	for i, b := range hashBytes {
		if i >= 8 {
			break
		}
		result |= uint64(b) << (8 * (7 - i))
	}
	
	// Mask to hash space size
	mask := uint64((1 << dht.config.HashBits) - 1)
	return result & mask
}

// calculateFingerStart calculates the start of the i-th finger
func (dht *DHT) calculateFingerStart(i int) uint64 {
	localHash := dht.hashKey(dht.localNode.ID)
	return (localHash + (1 << uint(i))) % (1 << uint(dht.config.HashBits))
}

// Start starts the DHT node
func (dht *DHT) Start() error {
	dht.mutex.Lock()
	defer dht.mutex.Unlock()
	
	if dht.running {
		return errors.New("DHT is already running")
	}
	
	// Create local node
	var err error
	dht.listener, err = net.Listen(dht.config.NetworkProtocol, 
		fmt.Sprintf("%s:%d", dht.config.Address, dht.config.Port))
	if err != nil {
		return fmt.Errorf("failed to start listener: %v", err)
	}
	
	// Get the actual port if auto-assigned
	if dht.config.Port == 0 {
		addr := dht.listener.Addr().(*net.TCPAddr)
		dht.config.Port = addr.Port
	}
	
	dht.localNode = NewNode(dht.config.NodeID, dht.config.Address, dht.config.Port)
	
	// Initialize finger table with self
	for i := 0; i < len(dht.fingerTable); i++ {
		dht.fingerTable[i].Node = dht.localNode
	}
	
	dht.running = true
	
	// Start background goroutines
	dht.wg.Add(4)
	go dht.acceptConnections()
	go dht.stabilize()
	go dht.fixFingers()
	go dht.checkPredecessor()
	
	if dht.config.EnableLogging {
		log.Printf("DHT node %s started on %s", dht.localNode.ID, dht.localNode.GetAddress())
	}
	
	return nil
}

// Join joins an existing DHT network
func (dht *DHT) Join(bootstrapNode string) error {
	if !dht.running {
		return errors.New("DHT node must be started before joining")
	}
	
	// Connect to bootstrap node
	conn, err := dht.connectToNode(bootstrapNode)
	if err != nil {
		return fmt.Errorf("failed to connect to bootstrap node: %v", err)
	}
	
	// Send join request
	joinMsg := &Message{
		Type:      JoinRequest,
		From:      dht.localNode.ID,
		To:        "",
		MessageID: generateMessageID(),
		NodeInfo:  dht.localNode,
		Timestamp: time.Now(),
	}
	
	response, err := dht.sendMessage(conn, joinMsg)
	if err != nil {
		return fmt.Errorf("failed to send join request: %v", err)
	}
	
	if response.Error != "" {
		return fmt.Errorf("join failed: %s", response.Error)
	}
	
	// Update successor and predecessor
	if len(response.Nodes) > 0 {
		dht.updateSuccessor(response.Nodes[0])
	}
	
	if response.NodeInfo != nil {
		dht.updatePredecessor(response.NodeInfo)
	}
	
	// Request data transfer from successor
	if dht.getSuccessor() != nil && dht.getSuccessor().ID != dht.localNode.ID {
		dht.requestDataTransfer()
	}
	
	dht.statistics.mutex.Lock()
	dht.statistics.NodesJoined++
	dht.statistics.mutex.Unlock()
	
	if dht.config.EnableLogging {
		log.Printf("Node %s joined DHT network", dht.localNode.ID)
	}
	
	return nil
}

// Leave gracefully leaves the DHT network
func (dht *DHT) Leave() error {
	dht.mutex.Lock()
	defer dht.mutex.Unlock()
	
	if !dht.running {
		return errors.New("DHT is not running")
	}
	
	// Transfer data to successor
	successor := dht.getSuccessor()
	if successor != nil && successor.ID != dht.localNode.ID {
		dht.transferDataToNode(successor)
	}
	
	// Notify predecessor and successor
	predecessor := dht.getPredecessor()
	if predecessor != nil {
		dht.notifyLeaving(predecessor, successor)
	}
	
	if successor != nil {
		dht.notifyLeaving(successor, predecessor)
	}
	
	dht.statistics.mutex.Lock()
	dht.statistics.NodesLeft++
	dht.statistics.mutex.Unlock()
	
	return dht.Stop()
}

// Stop stops the DHT node
func (dht *DHT) Stop() error {
	dht.mutex.Lock()
	defer dht.mutex.Unlock()
	
	if !dht.running {
		return errors.New("DHT is not running")
	}
	
	dht.running = false
	dht.cancel()
	
	// Close listener
	if dht.listener != nil {
		dht.listener.Close()
	}
	
	// Close all connections
	dht.connMutex.Lock()
	for _, conn := range dht.connections {
		conn.Close()
	}
	dht.connections = make(map[string]net.Conn)
	dht.connMutex.Unlock()
	
	// Wait for goroutines to finish
	dht.wg.Wait()
	
	if dht.config.EnableLogging {
		log.Printf("DHT node %s stopped", dht.localNode.ID)
	}
	
	return nil
}

// Put stores a key-value pair in the DHT
func (dht *DHT) Put(key string, value interface{}) error {
	return dht.PutWithTTL(key, value, 0)
}

// PutWithTTL stores a key-value pair with TTL in the DHT
func (dht *DHT) PutWithTTL(key string, value interface{}, ttl time.Duration) error {
	if !dht.running {
		return errors.New("DHT is not running")
	}
	
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		dht.statistics.mutex.Lock()
		dht.statistics.AverageStoreTime = 
			(dht.statistics.AverageStoreTime + duration) / 2
		dht.statistics.mutex.Unlock()
	}()
	
	// Find responsible node
	responsible := dht.findSuccessor(dht.hashKey(key))
	
	dataItem := &DataItem{
		Key:       key,
		Value:     value,
		Timestamp: time.Now(),
		TTL:       ttl,
		Version:   1,
		Metadata:  make(map[string]interface{}),
	}
	
	// Store locally if we're responsible
	if responsible.ID == dht.localNode.ID {
		dht.dataMutex.Lock()
		dht.data[key] = dataItem
		dht.dataMutex.Unlock()
		
		dht.statistics.mutex.Lock()
		dht.statistics.StoresSuccessful++
		dht.statistics.DataItemsStored++
		dht.statistics.mutex.Unlock()
		
		// Replicate to successors
		dht.replicateToSuccessors(key, dataItem)
		
		return nil
	}
	
	// Forward to responsible node
	conn, err := dht.connectToNode(responsible.GetAddress())
	if err != nil {
		dht.statistics.mutex.Lock()
		dht.statistics.StoresFailed++
		dht.statistics.mutex.Unlock()
		return fmt.Errorf("failed to connect to responsible node: %v", err)
	}
	
	storeMsg := &Message{
		Type:      StoreRequest,
		From:      dht.localNode.ID,
		To:        responsible.ID,
		MessageID: generateMessageID(),
		Key:       key,
		Value:     value,
		TTL:       ttl,
		Timestamp: time.Now(),
	}
	
	response, err := dht.sendMessage(conn, storeMsg)
	if err != nil {
		dht.statistics.mutex.Lock()
		dht.statistics.StoresFailed++
		dht.statistics.mutex.Unlock()
		return fmt.Errorf("failed to store data: %v", err)
	}
	
	if response.Error != "" {
		dht.statistics.mutex.Lock()
		dht.statistics.StoresFailed++
		dht.statistics.mutex.Unlock()
		return fmt.Errorf("store failed: %s", response.Error)
	}
	
	dht.statistics.mutex.Lock()
	dht.statistics.StoresSuccessful++
	dht.statistics.mutex.Unlock()
	
	return nil
}

// Get retrieves a value from the DHT
func (dht *DHT) Get(key string) (interface{}, error) {
	if !dht.running {
		return nil, errors.New("DHT is not running")
	}
	
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		dht.statistics.mutex.Lock()
		dht.statistics.AverageLookupTime = 
			(dht.statistics.AverageLookupTime + duration) / 2
		dht.statistics.mutex.Unlock()
	}()
	
	// Check locally first
	dht.dataMutex.RLock()
	dataItem, exists := dht.data[key]
	dht.dataMutex.RUnlock()
	
	if exists && !dataItem.IsExpired() {
		dht.statistics.mutex.Lock()
		dht.statistics.LookupsSuccessful++
		dht.statistics.mutex.Unlock()
		return dataItem.Value, nil
	}
	
	// Find responsible node
	responsible := dht.findSuccessor(dht.hashKey(key))
	
	if responsible.ID == dht.localNode.ID {
		// We're responsible but don't have it
		dht.statistics.mutex.Lock()
		dht.statistics.LookupsFailed++
		dht.statistics.mutex.Unlock()
		return nil, errors.New("key not found")
	}
	
	// Query responsible node
	conn, err := dht.connectToNode(responsible.GetAddress())
	if err != nil {
		dht.statistics.mutex.Lock()
		dht.statistics.LookupsFailed++
		dht.statistics.mutex.Unlock()
		return nil, fmt.Errorf("failed to connect to responsible node: %v", err)
	}
	
	lookupMsg := &Message{
		Type:      LookupRequest,
		From:      dht.localNode.ID,
		To:        responsible.ID,
		MessageID: generateMessageID(),
		Key:       key,
		Timestamp: time.Now(),
	}
	
	response, err := dht.sendMessage(conn, lookupMsg)
	if err != nil {
		dht.statistics.mutex.Lock()
		dht.statistics.LookupsFailed++
		dht.statistics.mutex.Unlock()
		return nil, fmt.Errorf("failed to lookup data: %v", err)
	}
	
	if response.Error != "" {
		dht.statistics.mutex.Lock()
		dht.statistics.LookupsFailed++
		dht.statistics.mutex.Unlock()
		return nil, fmt.Errorf("lookup failed: %s", response.Error)
	}
	
	dht.statistics.mutex.Lock()
	dht.statistics.LookupsSuccessful++
	dht.statistics.mutex.Unlock()
	
	return response.Value, nil
}

// Delete removes a key from the DHT
func (dht *DHT) Delete(key string) error {
	if !dht.running {
		return errors.New("DHT is not running")
	}
	
	// Find responsible node
	responsible := dht.findSuccessor(dht.hashKey(key))
	
	// Delete locally if we're responsible
	if responsible.ID == dht.localNode.ID {
		dht.dataMutex.Lock()
		delete(dht.data, key)
		dht.dataMutex.Unlock()
		
		// Delete from replicas
		dht.deleteFromReplicas(key)
		
		return nil
	}
	
	// Forward to responsible node
	conn, err := dht.connectToNode(responsible.GetAddress())
	if err != nil {
		return fmt.Errorf("failed to connect to responsible node: %v", err)
	}
	
	deleteMsg := &Message{
		Type:      StoreRequest, // Reuse store with nil value for delete
		From:      dht.localNode.ID,
		To:        responsible.ID,
		MessageID: generateMessageID(),
		Key:       key,
		Value:     nil,
		Timestamp: time.Now(),
	}
	
	response, err := dht.sendMessage(conn, deleteMsg)
	if err != nil {
		return fmt.Errorf("failed to delete data: %v", err)
	}
	
	if response.Error != "" {
		return fmt.Errorf("delete failed: %s", response.Error)
	}
	
	return nil
}

// findSuccessor finds the successor node for a given hash
func (dht *DHT) findSuccessor(hash uint64) *Node {
	predecessor := dht.findPredecessor(hash)
	if predecessor.ID == dht.localNode.ID {
		return dht.getSuccessor()
	}
	
	// Query predecessor for its successor
	conn, err := dht.connectToNode(predecessor.GetAddress())
	if err != nil {
		// Fallback to local successor
		return dht.getSuccessor()
	}
	
	lookupMsg := &Message{
		Type:      SuccessorUpdate,
		From:      dht.localNode.ID,
		To:        predecessor.ID,
		MessageID: generateMessageID(),
		Timestamp: time.Now(),
	}
	
	response, err := dht.sendMessage(conn, lookupMsg)
	if err != nil || len(response.Nodes) == 0 {
		return dht.getSuccessor()
	}
	
	return response.Nodes[0]
}

// findPredecessor finds the predecessor node for a given hash
func (dht *DHT) findPredecessor(hash uint64) *Node {
	current := dht.localNode
	localHash := dht.hashKey(current.ID)
	
	for {
		successor := dht.getSuccessor()
		if successor == nil {
			break
		}
		
		successorHash := dht.hashKey(successor.ID)
		
		// Check if hash is between current and successor
		if dht.betweenInclusive(hash, localHash, successorHash) {
			break
		}
		
		// Find closest preceding finger
		next := dht.closestPrecedingFinger(hash)
		if next.ID == current.ID {
			break
		}
		
		current = next
		localHash = dht.hashKey(current.ID)
	}
	
	return current
}

// closestPrecedingFinger finds the closest preceding finger for a hash
func (dht *DHT) closestPrecedingFinger(hash uint64) *Node {
	localHash := dht.hashKey(dht.localNode.ID)
	
	// Check finger table in reverse order
	for i := len(dht.fingerTable) - 1; i >= 0; i-- {
		dht.fingerTable[i].mutex.RLock()
		finger := dht.fingerTable[i].Node
		dht.fingerTable[i].mutex.RUnlock()
		
		if finger != nil && finger.ID != dht.localNode.ID {
			fingerHash := dht.hashKey(finger.ID)
			if dht.between(fingerHash, localHash, hash) {
				return finger
			}
		}
	}
	
	return dht.localNode
}

// between checks if a value is between start and end in the hash ring
func (dht *DHT) between(value, start, end uint64) bool {
	if start < end {
		return value > start && value < end
	}
	return value > start || value < end
}

// betweenInclusive checks if a value is between start and end (inclusive) in the hash ring
func (dht *DHT) betweenInclusive(value, start, end uint64) bool {
	if start < end {
		return value > start && value <= end
	}
	return value > start || value <= end
}

// setupMessageHandlers sets up message handlers for different message types
func (dht *DHT) setupMessageHandlers() {
	dht.messageHandlers[JoinRequest] = dht.handleJoinRequest
	dht.messageHandlers[LookupRequest] = dht.handleLookupRequest
	dht.messageHandlers[StoreRequest] = dht.handleStoreRequest
	dht.messageHandlers[ReplicationRequest] = dht.handleReplicationRequest
	dht.messageHandlers[HeartbeatRequest] = dht.handleHeartbeatRequest
	dht.messageHandlers[TransferRequest] = dht.handleTransferRequest
	dht.messageHandlers[SuccessorUpdate] = dht.handleSuccessorUpdate
	dht.messageHandlers[PredecessorUpdate] = dht.handlePredecessorUpdate
}

// Background maintenance functions

// acceptConnections accepts incoming connections
func (dht *DHT) acceptConnections() {
	defer dht.wg.Done()
	
	for {
		select {
		case <-dht.ctx.Done():
			return
		default:
			conn, err := dht.listener.Accept()
			if err != nil {
				if dht.running {
					dht.statistics.mutex.Lock()
					dht.statistics.NetworkErrors++
					dht.statistics.mutex.Unlock()
				}
				continue
			}
			
			go dht.handleConnection(conn)
		}
	}
}

// stabilize runs the stabilization protocol
func (dht *DHT) stabilize() {
	defer dht.wg.Done()
	
	ticker := time.NewTicker(dht.config.StabilizeInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-dht.ctx.Done():
			return
		case <-ticker.C:
			dht.performStabilize()
		}
	}
}

// fixFingers runs the finger table maintenance
func (dht *DHT) fixFingers() {
	defer dht.wg.Done()
	
	ticker := time.NewTicker(dht.config.FixFingerInterval)
	defer ticker.Stop()
	
	next := 0
	
	for {
		select {
		case <-dht.ctx.Done():
			return
		case <-ticker.C:
			dht.performFixFingers(&next)
		}
	}
}

// checkPredecessor checks if predecessor is still alive
func (dht *DHT) checkPredecessor() {
	defer dht.wg.Done()
	
	ticker := time.NewTicker(dht.config.CheckPredecessor)
	defer ticker.Stop()
	
	for {
		select {
		case <-dht.ctx.Done():
			return
		case <-ticker.C:
			dht.performCheckPredecessor()
		}
	}
}

// Helper functions for maintenance operations would continue here...
// Due to length constraints, I'll include the key message handlers

// handleJoinRequest handles node join requests
func (dht *DHT) handleJoinRequest(msg *Message) *Message {
	if msg.NodeInfo == nil {
		return &Message{
			Type:      JoinResponse,
			From:      dht.localNode.ID,
			To:        msg.From,
			MessageID: msg.MessageID,
			Error:     "Invalid join request: missing node info",
			Timestamp: time.Now(),
		}
	}
	
	// Find successor for the joining node
	joiningHash := dht.hashKey(msg.NodeInfo.ID)
	successor := dht.findSuccessor(joiningHash)
	
	response := &Message{
		Type:      JoinResponse,
		From:      dht.localNode.ID,
		To:        msg.From,
		MessageID: msg.MessageID,
		Nodes:     []*Node{successor},
		NodeInfo:  dht.getPredecessor(),
		Timestamp: time.Now(),
	}
	
	// Update our predecessor if the joining node should be our predecessor
	localHash := dht.hashKey(dht.localNode.ID)
	predecessor := dht.getPredecessor()
	
	if predecessor == nil || dht.between(joiningHash, dht.hashKey(predecessor.ID), localHash) {
		dht.updatePredecessor(msg.NodeInfo)
	}
	
	return response
}

// handleLookupRequest handles data lookup requests
func (dht *DHT) handleLookupRequest(msg *Message) *Message {
	dht.dataMutex.RLock()
	dataItem, exists := dht.data[msg.Key]
	dht.dataMutex.RUnlock()
	
	response := &Message{
		Type:      LookupResponse,
		From:      dht.localNode.ID,
		To:        msg.From,
		MessageID: msg.MessageID,
		Key:       msg.Key,
		Timestamp: time.Now(),
	}
	
	if exists && !dataItem.IsExpired() {
		response.Value = dataItem.Value
	} else {
		response.Error = "Key not found"
	}
	
	return response
}

// handleStoreRequest handles data store requests
func (dht *DHT) handleStoreRequest(msg *Message) *Message {
	response := &Message{
		Type:      StoreResponse,
		From:      dht.localNode.ID,
		To:        msg.From,
		MessageID: msg.MessageID,
		Key:       msg.Key,
		Timestamp: time.Now(),
	}
	
	if msg.Value == nil {
		// Delete operation
		dht.dataMutex.Lock()
		delete(dht.data, msg.Key)
		dht.dataMutex.Unlock()
	} else {
		// Store operation
		dataItem := &DataItem{
			Key:       msg.Key,
			Value:     msg.Value,
			Timestamp: time.Now(),
			TTL:       msg.TTL,
			Version:   1,
			Metadata:  make(map[string]interface{}),
		}
		
		dht.dataMutex.Lock()
		dht.data[msg.Key] = dataItem
		dht.dataMutex.Unlock()
		
		// Replicate to successors
		dht.replicateToSuccessors(msg.Key, dataItem)
	}
	
	return response
}

// Additional helper functions and message handlers would be implemented here...

// GetStatistics returns current DHT statistics
func (dht *DHT) GetStatistics() DHTStatistics {
	dht.statistics.mutex.RLock()
	defer dht.statistics.mutex.RUnlock()
	return *dht.statistics
}

// GetNodeInfo returns information about the local node
func (dht *DHT) GetNodeInfo() *Node {
	return dht.localNode
}

// GetSuccessorList returns the current successor list
func (dht *DHT) GetSuccessorList() []*Node {
	dht.mutex.RLock()
	defer dht.mutex.RUnlock()
	
	result := make([]*Node, len(dht.successorList))
	copy(result, dht.successorList)
	return result
}

// GetPredecessor returns the current predecessor
func (dht *DHT) GetPredecessor() *Node {
	return dht.getPredecessor()
}

// Helper functions

func generateNodeID() string {
	return fmt.Sprintf("node_%d_%d", time.Now().UnixNano(), rand.Intn(10000))
}

func generateMessageID() string {
	return fmt.Sprintf("msg_%d_%d", time.Now().UnixNano(), rand.Intn(10000))
}

// Getter functions with proper locking
func (dht *DHT) getSuccessor() *Node {
	dht.mutex.RLock()
	defer dht.mutex.RUnlock()
	if len(dht.successorList) > 0 {
		return dht.successorList[0]
	}
	return nil
}

func (dht *DHT) getPredecessor() *Node {
	dht.mutex.RLock()
	defer dht.mutex.RUnlock()
	return dht.predecessor
}

func (dht *DHT) updateSuccessor(node *Node) {
	dht.mutex.Lock()
	defer dht.mutex.Unlock()
	if len(dht.successorList) > 0 {
		dht.successorList[0] = node
	}
}

func (dht *DHT) updatePredecessor(node *Node) {
	dht.mutex.Lock()
	defer dht.mutex.Unlock()
	dht.predecessor = node
}

// Placeholder implementations for remaining functionality
func (dht *DHT) connectToNode(address string) (net.Conn, error) {
	// Implementation would establish connection to node
	return net.Dial(dht.config.NetworkProtocol, address)
}

func (dht *DHT) sendMessage(conn net.Conn, msg *Message) (*Message, error) {
	// Implementation would serialize and send message, wait for response
	return &Message{}, nil
}

func (dht *DHT) handleConnection(conn net.Conn) {
	// Implementation would handle incoming connection
}

func (dht *DHT) replicateToSuccessors(key string, item *DataItem) {
	// Implementation would replicate data to successor nodes
}

func (dht *DHT) deleteFromReplicas(key string) {
	// Implementation would delete key from replica nodes
}

func (dht *DHT) requestDataTransfer() {
	// Implementation would request data transfer from successor
}

func (dht *DHT) transferDataToNode(node *Node) {
	// Implementation would transfer data to specified node
}

func (dht *DHT) notifyLeaving(node *Node, replacement *Node) {
	// Implementation would notify node of our departure
}

func (dht *DHT) performStabilize() {
	// Implementation would perform stabilization protocol
	dht.statistics.mutex.Lock()
	dht.statistics.StabilizeOperations++
	dht.statistics.mutex.Unlock()
}

func (dht *DHT) performFixFingers(next *int) {
	// Implementation would fix finger table entries
	dht.statistics.mutex.Lock()
	dht.statistics.FingerTableUpdates++
	dht.statistics.mutex.Unlock()
}

func (dht *DHT) performCheckPredecessor() {
	// Implementation would check if predecessor is alive
}

func (dht *DHT) handleReplicationRequest(msg *Message) *Message {
	// Implementation would handle replication requests
	return &Message{}
}

func (dht *DHT) handleHeartbeatRequest(msg *Message) *Message {
	// Implementation would handle heartbeat requests
	return &Message{
		Type:      HeartbeatResponse,
		From:      dht.localNode.ID,
		To:        msg.From,
		MessageID: msg.MessageID,
		Timestamp: time.Now(),
	}
}

func (dht *DHT) handleTransferRequest(msg *Message) *Message {
	// Implementation would handle data transfer requests
	return &Message{}
}

func (dht *DHT) handleSuccessorUpdate(msg *Message) *Message {
	// Implementation would handle successor update requests
	successor := dht.getSuccessor()
	return &Message{
		Type:      SuccessorUpdate,
		From:      dht.localNode.ID,
		To:        msg.From,
		MessageID: msg.MessageID,
		Nodes:     []*Node{successor},
		Timestamp: time.Now(),
	}
}

func (dht *DHT) handlePredecessorUpdate(msg *Message) *Message {
	// Implementation would handle predecessor update requests
	return &Message{}
}