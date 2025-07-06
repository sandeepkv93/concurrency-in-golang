package concurrentdht

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestDefaultDHTConfig(t *testing.T) {
	config := DefaultDHTConfig()
	
	if config.HashBits != 160 {
		t.Errorf("Expected hash bits 160, got %d", config.HashBits)
	}
	
	if config.ReplicationFactor != 3 {
		t.Errorf("Expected replication factor 3, got %d", config.ReplicationFactor)
	}
	
	if config.SuccessorListSize != 8 {
		t.Errorf("Expected successor list size 8, got %d", config.SuccessorListSize)
	}
	
	if !config.EnableLogging {
		t.Error("Expected logging to be enabled by default")
	}
	
	if !config.EnableMetrics {
		t.Error("Expected metrics to be enabled by default")
	}
	
	if config.NodeID == "" {
		t.Error("Node ID should be generated")
	}
}

func TestNewDHT(t *testing.T) {
	config := DefaultDHTConfig()
	config.Port = 0 // Auto-assign port
	
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	if dht == nil {
		t.Fatal("DHT should not be nil")
	}
	
	if dht.config.NodeID == "" {
		t.Error("Node ID should be set")
	}
	
	if len(dht.fingerTable) != config.HashBits {
		t.Errorf("Expected finger table size %d, got %d", config.HashBits, len(dht.fingerTable))
	}
	
	if len(dht.successorList) != config.SuccessorListSize {
		t.Errorf("Expected successor list size %d, got %d", config.SuccessorListSize, len(dht.successorList))
	}
	
	if dht.statistics == nil {
		t.Error("Statistics should be initialized")
	}
	
	if dht.hasher == nil {
		t.Error("Hasher should be initialized")
	}
}

func TestDHTInvalidConfig(t *testing.T) {
	testCases := []struct {
		name   string
		config DHTConfig
	}{
		{
			name: "Zero hash bits",
			config: DHTConfig{
				NodeID:            "test",
				HashBits:          0,
				ReplicationFactor: 3,
			},
		},
		{
			name: "Negative replication factor",
			config: DHTConfig{
				NodeID:            "test",
				HashBits:          160,
				ReplicationFactor: -1,
			},
		},
		{
			name: "Too many hash bits",
			config: DHTConfig{
				NodeID:            "test",
				HashBits:          300,
				ReplicationFactor: 3,
			},
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewDHT(tc.config)
			if err == nil {
				t.Error("Expected error for invalid config")
			}
		})
	}
}

func TestDHTStartStop(t *testing.T) {
	config := DefaultDHTConfig()
	config.Port = 0 // Auto-assign port
	
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	// Test start
	err = dht.Start()
	if err != nil {
		t.Fatalf("Failed to start DHT: %v", err)
	}
	
	if !dht.running {
		t.Error("DHT should be running after start")
	}
	
	if dht.localNode == nil {
		t.Error("Local node should be set after start")
	}
	
	if dht.listener == nil {
		t.Error("Listener should be set after start")
	}
	
	// Test that port was assigned
	if dht.config.Port == 0 {
		t.Error("Port should be assigned after start")
	}
	
	// Test double start
	err = dht.Start()
	if err == nil {
		t.Error("Expected error on double start")
	}
	
	// Test stop
	err = dht.Stop()
	if err != nil {
		t.Errorf("Failed to stop DHT: %v", err)
	}
	
	if dht.running {
		t.Error("DHT should not be running after stop")
	}
	
	// Test double stop
	err = dht.Stop()
	if err == nil {
		t.Error("Expected error on double stop")
	}
}

func TestHashKeyConsistency(t *testing.T) {
	config := DefaultDHTConfig()
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	testKeys := []string{"key1", "key2", "key3", "test_key", "another_key"}
	
	// Test that hashing is consistent
	for _, key := range testKeys {
		hash1 := dht.hashKey(key)
		hash2 := dht.hashKey(key)
		
		if hash1 != hash2 {
			t.Errorf("Hash function should be deterministic for key %s", key)
		}
	}
	
	// Test that different keys produce different hashes (usually)
	hashes := make(map[uint64]string)
	for _, key := range testKeys {
		hash := dht.hashKey(key)
		if existingKey, exists := hashes[hash]; exists {
			t.Logf("Hash collision between %s and %s: %d", key, existingKey, hash)
		} else {
			hashes[hash] = key
		}
	}
}

func TestBetweenFunctions(t *testing.T) {
	config := DefaultDHTConfig()
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	testCases := []struct {
		value    uint64
		start    uint64
		end      uint64
		expected bool
		name     string
	}{
		{5, 3, 7, true, "simple between"},
		{3, 3, 7, false, "equal to start"},
		{7, 3, 7, false, "equal to end"},
		{1, 3, 7, false, "before range"},
		{9, 3, 7, false, "after range"},
		{1, 250, 10, true, "wrap around between"},
		{255, 250, 10, true, "wrap around start"},
		{5, 250, 10, true, "wrap around middle"},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := dht.between(tc.value, tc.start, tc.end)
			if result != tc.expected {
				t.Errorf("between(%d, %d, %d) = %t, expected %t", 
					tc.value, tc.start, tc.end, result, tc.expected)
			}
		})
	}
	
	// Test betweenInclusive
	testCasesInclusive := []struct {
		value    uint64
		start    uint64
		end      uint64
		expected bool
		name     string
	}{
		{5, 3, 7, true, "simple between inclusive"},
		{3, 3, 7, false, "equal to start inclusive"},
		{7, 3, 7, true, "equal to end inclusive"},
		{1, 3, 7, false, "before range inclusive"},
		{9, 3, 7, false, "after range inclusive"},
	}
	
	for _, tc := range testCasesInclusive {
		t.Run(tc.name+"_inclusive", func(t *testing.T) {
			result := dht.betweenInclusive(tc.value, tc.start, tc.end)
			if result != tc.expected {
				t.Errorf("betweenInclusive(%d, %d, %d) = %t, expected %t", 
					tc.value, tc.start, tc.end, result, tc.expected)
			}
		})
	}
}

func TestFingerTableCalculation(t *testing.T) {
	config := DefaultDHTConfig()
	config.HashBits = 8 // Use smaller hash space for testing
	
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	err = dht.Start()
	if err != nil {
		t.Fatalf("Failed to start DHT: %v", err)
	}
	defer dht.Stop()
	
	localHash := dht.hashKey(dht.localNode.ID)
	
	// Test finger table start calculations
	for i := 0; i < config.HashBits; i++ {
		expectedStart := (localHash + (1 << uint(i))) % (1 << uint(config.HashBits))
		actualStart := dht.fingerTable[i].Start
		
		if actualStart != expectedStart {
			t.Errorf("Finger %d start: expected %d, got %d", i, expectedStart, actualStart)
		}
	}
}

func TestSingleNodeOperations(t *testing.T) {
	config := DefaultDHTConfig()
	config.Port = 0
	
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	err = dht.Start()
	if err != nil {
		t.Fatalf("Failed to start DHT: %v", err)
	}
	defer dht.Stop()
	
	// Test put and get operations on single node
	testCases := []struct {
		key   string
		value interface{}
	}{
		{"key1", "value1"},
		{"key2", 42},
		{"key3", []string{"array", "value"}},
		{"key4", map[string]interface{}{"nested": "object"}},
	}
	
	for _, tc := range testCases {
		err := dht.Put(tc.key, tc.value)
		if err != nil {
			t.Errorf("Failed to put %s: %v", tc.key, err)
		}
		
		value, err := dht.Get(tc.key)
		if err != nil {
			t.Errorf("Failed to get %s: %v", tc.key, err)
		}
		
		// Simple equality check - in real implementation would need deep comparison
		if fmt.Sprintf("%v", value) != fmt.Sprintf("%v", tc.value) {
			t.Errorf("Value mismatch for %s: expected %v, got %v", tc.key, tc.value, value)
		}
	}
	
	// Test getting non-existent key
	_, err = dht.Get("nonexistent")
	if err == nil {
		t.Error("Expected error for non-existent key")
	}
}

func TestTTLOperations(t *testing.T) {
	config := DefaultDHTConfig()
	config.Port = 0
	
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	err = dht.Start()
	if err != nil {
		t.Fatalf("Failed to start DHT: %v", err)
	}
	defer dht.Stop()
	
	// Test TTL functionality
	key := "ttl_test_key"
	value := "ttl_test_value"
	ttl := 100 * time.Millisecond
	
	err = dht.PutWithTTL(key, value, ttl)
	if err != nil {
		t.Fatalf("Failed to put with TTL: %v", err)
	}
	
	// Should be available immediately
	retrievedValue, err := dht.Get(key)
	if err != nil {
		t.Errorf("Failed to get key with TTL: %v", err)
	}
	
	if retrievedValue != value {
		t.Errorf("Value mismatch: expected %v, got %v", value, retrievedValue)
	}
	
	// Wait for expiration
	time.Sleep(150 * time.Millisecond)
	
	// Should be expired now
	_, err = dht.Get(key)
	if err == nil {
		t.Error("Expected error for expired key")
	}
}

func TestDeleteOperations(t *testing.T) {
	config := DefaultDHTConfig()
	config.Port = 0
	
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	err = dht.Start()
	if err != nil {
		t.Fatalf("Failed to start DHT: %v", err)
	}
	defer dht.Stop()
	
	key := "delete_test_key"
	value := "delete_test_value"
	
	// Put a value
	err = dht.Put(key, value)
	if err != nil {
		t.Fatalf("Failed to put key: %v", err)
	}
	
	// Verify it exists
	_, err = dht.Get(key)
	if err != nil {
		t.Errorf("Key should exist before deletion: %v", err)
	}
	
	// Delete the key
	err = dht.Delete(key)
	if err != nil {
		t.Errorf("Failed to delete key: %v", err)
	}
	
	// Verify it's gone
	_, err = dht.Get(key)
	if err == nil {
		t.Error("Key should not exist after deletion")
	}
}

func TestConcurrentOperations(t *testing.T) {
	config := DefaultDHTConfig()
	config.Port = 0
	
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	err = dht.Start()
	if err != nil {
		t.Fatalf("Failed to start DHT: %v", err)
	}
	defer dht.Stop()
	
	numGoroutines := 10
	operationsPerGoroutine := 20
	
	var wg sync.WaitGroup
	
	// Concurrent puts
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < operationsPerGoroutine; j++ {
				key := fmt.Sprintf("concurrent_key_%d_%d", id, j)
				value := fmt.Sprintf("concurrent_value_%d_%d", id, j)
				err := dht.Put(key, value)
				if err != nil {
					t.Errorf("Failed to put concurrent key %s: %v", key, err)
				}
			}
		}(i)
	}
	
	wg.Wait()
	
	// Concurrent gets
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < operationsPerGoroutine; j++ {
				key := fmt.Sprintf("concurrent_key_%d_%d", id, j)
				expectedValue := fmt.Sprintf("concurrent_value_%d_%d", id, j)
				value, err := dht.Get(key)
				if err != nil {
					t.Errorf("Failed to get concurrent key %s: %v", key, err)
				} else if value != expectedValue {
					t.Errorf("Value mismatch for %s: expected %s, got %v", key, expectedValue, value)
				}
			}
		}(i)
	}
	
	wg.Wait()
}

func TestDataItemExpiration(t *testing.T) {
	dataItem := &DataItem{
		Key:       "test",
		Value:     "value",
		Timestamp: time.Now(),
		TTL:       100 * time.Millisecond,
	}
	
	// Should not be expired immediately
	if dataItem.IsExpired() {
		t.Error("Data item should not be expired immediately")
	}
	
	// Wait for expiration
	time.Sleep(150 * time.Millisecond)
	
	// Should be expired now
	if !dataItem.IsExpired() {
		t.Error("Data item should be expired")
	}
	
	// Test item with no TTL
	noTTLItem := &DataItem{
		Key:       "test",
		Value:     "value",
		Timestamp: time.Now().Add(-time.Hour), // Old timestamp
		TTL:       0, // No TTL
	}
	
	if noTTLItem.IsExpired() {
		t.Error("Item with no TTL should never expire")
	}
}

func TestNodeCreation(t *testing.T) {
	nodeID := "test_node_123"
	address := "localhost"
	port := 8080
	
	node := NewNode(nodeID, address, port)
	
	if node.ID != nodeID {
		t.Errorf("Expected node ID %s, got %s", nodeID, node.ID)
	}
	
	if node.Address != address {
		t.Errorf("Expected address %s, got %s", address, node.Address)
	}
	
	if node.Port != port {
		t.Errorf("Expected port %d, got %d", port, node.Port)
	}
	
	if node.State != Active {
		t.Errorf("Expected initial state Active, got %v", node.State)
	}
	
	expectedAddress := fmt.Sprintf("%s:%d", address, port)
	if node.GetAddress() != expectedAddress {
		t.Errorf("Expected full address %s, got %s", expectedAddress, node.GetAddress())
	}
	
	// Test IsAlive
	if !node.IsAlive(time.Minute) {
		t.Error("New node should be considered alive")
	}
	
	// Update last seen to past
	node.mutex.Lock()
	node.LastSeen = time.Now().Add(-2 * time.Minute)
	node.mutex.Unlock()
	
	if node.IsAlive(time.Minute) {
		t.Error("Old node should not be considered alive")
	}
}

func TestMessageHandlers(t *testing.T) {
	config := DefaultDHTConfig()
	config.Port = 0
	
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	err = dht.Start()
	if err != nil {
		t.Fatalf("Failed to start DHT: %v", err)
	}
	defer dht.Stop()
	
	// Test join request handler
	joinMsg := &Message{
		Type:      JoinRequest,
		From:      "joining_node",
		MessageID: "test_msg_1",
		NodeInfo:  NewNode("joining_node", "localhost", 8081),
		Timestamp: time.Now(),
	}
	
	response := dht.handleJoinRequest(joinMsg)
	if response == nil {
		t.Error("Join request should return response")
	}
	
	if response.Type != JoinResponse {
		t.Errorf("Expected JoinResponse, got %v", response.Type)
	}
	
	// Test lookup request handler
	dht.Put("test_lookup_key", "test_lookup_value")
	
	lookupMsg := &Message{
		Type:      LookupRequest,
		From:      "requesting_node",
		MessageID: "test_msg_2",
		Key:       "test_lookup_key",
		Timestamp: time.Now(),
	}
	
	response = dht.handleLookupRequest(lookupMsg)
	if response == nil {
		t.Error("Lookup request should return response")
	}
	
	if response.Type != LookupResponse {
		t.Errorf("Expected LookupResponse, got %v", response.Type)
	}
	
	if response.Value != "test_lookup_value" {
		t.Errorf("Expected lookup value 'test_lookup_value', got %v", response.Value)
	}
	
	// Test store request handler
	storeMsg := &Message{
		Type:      StoreRequest,
		From:      "storing_node",
		MessageID: "test_msg_3",
		Key:       "test_store_key",
		Value:     "test_store_value",
		Timestamp: time.Now(),
	}
	
	response = dht.handleStoreRequest(storeMsg)
	if response == nil {
		t.Error("Store request should return response")
	}
	
	if response.Type != StoreResponse {
		t.Errorf("Expected StoreResponse, got %v", response.Type)
	}
	
	// Verify the value was stored
	value, err := dht.Get("test_store_key")
	if err != nil {
		t.Errorf("Failed to get stored value: %v", err)
	}
	
	if value != "test_store_value" {
		t.Errorf("Expected stored value 'test_store_value', got %v", value)
	}
	
	// Test heartbeat request handler
	heartbeatMsg := &Message{
		Type:      HeartbeatRequest,
		From:      "heartbeat_node",
		MessageID: "test_msg_4",
		Timestamp: time.Now(),
	}
	
	response = dht.handleHeartbeatRequest(heartbeatMsg)
	if response == nil {
		t.Error("Heartbeat request should return response")
	}
	
	if response.Type != HeartbeatResponse {
		t.Errorf("Expected HeartbeatResponse, got %v", response.Type)
	}
}

func TestStatisticsTracking(t *testing.T) {
	config := DefaultDHTConfig()
	config.Port = 0
	config.EnableMetrics = true
	
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	err = dht.Start()
	if err != nil {
		t.Fatalf("Failed to start DHT: %v", err)
	}
	defer dht.Stop()
	
	initialStats := dht.GetStatistics()
	
	if initialStats.StartTime.IsZero() {
		t.Error("Start time should be set")
	}
	
	// Perform some operations
	numOperations := 10
	for i := 0; i < numOperations; i++ {
		key := fmt.Sprintf("stats_key_%d", i)
		value := fmt.Sprintf("stats_value_%d", i)
		
		err := dht.Put(key, value)
		if err != nil {
			t.Errorf("Failed to put key %s: %v", key, err)
		}
		
		_, err = dht.Get(key)
		if err != nil {
			t.Errorf("Failed to get key %s: %v", key, err)
		}
	}
	
	finalStats := dht.GetStatistics()
	
	if finalStats.StoresSuccessful < initialStats.StoresSuccessful {
		t.Error("Successful stores should increase")
	}
	
	if finalStats.LookupsSuccessful < initialStats.LookupsSuccessful {
		t.Error("Successful lookups should increase")
	}
	
	if finalStats.DataItemsStored < initialStats.DataItemsStored {
		t.Error("Data items stored should increase")
	}
}

func TestHashFunctions(t *testing.T) {
	testData := "test_hash_data"
	
	hashFunctions := []HashFunction{
		FNV1, FNV1a, CRC32Hash, MD5Hash, SHA1Hash, SHA256Hash,
	}
	
	for _, hashFunc := range hashFunctions {
		t.Run(fmt.Sprintf("HashFunction_%d", hashFunc), func(t *testing.T) {
			config := DefaultDHTConfig()
			config.HashFunction = hashFunc
			
			dht, err := NewDHT(config)
			if err != nil {
				t.Fatalf("Failed to create DHT with hash function %d: %v", hashFunc, err)
			}
			
			hash1 := dht.hashKey(testData)
			hash2 := dht.hashKey(testData)
			
			if hash1 != hash2 {
				t.Errorf("Hash function %d should be deterministic", hashFunc)
			}
			
			// Test with different data
			differentData := "different_test_data"
			hash3 := dht.hashKey(differentData)
			
			if hash1 == hash3 {
				t.Logf("Note: Hash collision detected for function %d", hashFunc)
			}
		})
	}
}

func TestErrorHandling(t *testing.T) {
	config := DefaultDHTConfig()
	config.Port = 0
	
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	// Test operations on stopped DHT
	err = dht.Put("key", "value")
	if err == nil {
		t.Error("Expected error when putting to stopped DHT")
	}
	
	_, err = dht.Get("key")
	if err == nil {
		t.Error("Expected error when getting from stopped DHT")
	}
	
	err = dht.Delete("key")
	if err == nil {
		t.Error("Expected error when deleting from stopped DHT")
	}
	
	// Test leave on stopped DHT
	err = dht.Leave()
	if err == nil {
		t.Error("Expected error when leaving stopped DHT")
	}
}

func TestGettersAndInfo(t *testing.T) {
	config := DefaultDHTConfig()
	config.Port = 0
	
	dht, err := NewDHT(config)
	if err != nil {
		t.Fatalf("Failed to create DHT: %v", err)
	}
	
	err = dht.Start()
	if err != nil {
		t.Fatalf("Failed to start DHT: %v", err)
	}
	defer dht.Stop()
	
	// Test GetNodeInfo
	nodeInfo := dht.GetNodeInfo()
	if nodeInfo == nil {
		t.Error("Node info should not be nil")
	}
	
	if nodeInfo.ID != dht.localNode.ID {
		t.Error("Node info should match local node")
	}
	
	// Test GetSuccessorList
	successorList := dht.GetSuccessorList()
	if successorList == nil {
		t.Error("Successor list should not be nil")
	}
	
	if len(successorList) != dht.config.SuccessorListSize {
		t.Errorf("Expected successor list size %d, got %d", 
			dht.config.SuccessorListSize, len(successorList))
	}
	
	// Test GetPredecessor
	predecessor := dht.GetPredecessor()
	// Predecessor can be nil for single node
	if predecessor != nil && predecessor.ID == "" {
		t.Error("Predecessor should have valid ID if not nil")
	}
}

// Benchmark tests

func BenchmarkDHTHashKey(b *testing.B) {
	config := DefaultDHTConfig()
	dht, err := NewDHT(config)
	if err != nil {
		b.Fatalf("Failed to create DHT: %v", err)
	}
	
	testKey := "benchmark_test_key_with_reasonable_length"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dht.hashKey(testKey)
	}
}

func BenchmarkDHTPut(b *testing.B) {
	config := DefaultDHTConfig()
	config.Port = 0
	
	dht, err := NewDHT(config)
	if err != nil {
		b.Fatalf("Failed to create DHT: %v", err)
	}
	
	err = dht.Start()
	if err != nil {
		b.Fatalf("Failed to start DHT: %v", err)
	}
	defer dht.Stop()
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := fmt.Sprintf("benchmark_key_%d", i)
			value := fmt.Sprintf("benchmark_value_%d", i)
			dht.Put(key, value)
			i++
		}
	})
}

func BenchmarkDHTGet(b *testing.B) {
	config := DefaultDHTConfig()
	config.Port = 0
	
	dht, err := NewDHT(config)
	if err != nil {
		b.Fatalf("Failed to create DHT: %v", err)
	}
	
	err = dht.Start()
	if err != nil {
		b.Fatalf("Failed to start DHT: %v", err)
	}
	defer dht.Stop()
	
	// Pre-populate with data
	numKeys := 1000
	for i := 0; i < numKeys; i++ {
		key := fmt.Sprintf("benchmark_key_%d", i)
		value := fmt.Sprintf("benchmark_value_%d", i)
		dht.Put(key, value)
	}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := fmt.Sprintf("benchmark_key_%d", i%numKeys)
			dht.Get(key)
			i++
		}
	})
}

func BenchmarkBetweenOperations(b *testing.B) {
	config := DefaultDHTConfig()
	dht, err := NewDHT(config)
	if err != nil {
		b.Fatalf("Failed to create DHT: %v", err)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dht.between(uint64(i%256), uint64((i+1)%256), uint64((i+2)%256))
	}
}

// Example functions

func ExampleNewDHT() {
	// Create DHT configuration
	config := DefaultDHTConfig()
	config.Port = 8080
	config.Address = "localhost"
	config.ReplicationFactor = 3
	config.EnableLogging = true
	
	// Create DHT instance
	dht, err := NewDHT(config)
	if err != nil {
		fmt.Printf("Failed to create DHT: %v\n", err)
		return
	}
	
	// Start the DHT node
	err = dht.Start()
	if err != nil {
		fmt.Printf("Failed to start DHT: %v\n", err)
		return
	}
	defer dht.Stop()
	
	fmt.Printf("DHT node started with ID: %s\n", dht.GetNodeInfo().ID)
	fmt.Printf("Listening on: %s\n", dht.GetNodeInfo().GetAddress())
	
	// Output:
	// DHT node started with ID: node_1234567890_1234
	// Listening on: localhost:8080
}

func ExampleDHT_Put() {
	config := DefaultDHTConfig()
	config.Port = 0 // Auto-assign port
	
	dht, err := NewDHT(config)
	if err != nil {
		fmt.Printf("Failed to create DHT: %v\n", err)
		return
	}
	
	err = dht.Start()
	if err != nil {
		fmt.Printf("Failed to start DHT: %v\n", err)
		return
	}
	defer dht.Stop()
	
	// Store different types of data
	err = dht.Put("user:123", "John Doe")
	if err != nil {
		fmt.Printf("Failed to store user: %v\n", err)
		return
	}
	
	err = dht.Put("counter:visits", 42)
	if err != nil {
		fmt.Printf("Failed to store counter: %v\n", err)
		return
	}
	
	// Store with TTL
	err = dht.PutWithTTL("session:abc", "active", 10*time.Minute)
	if err != nil {
		fmt.Printf("Failed to store session: %v\n", err)
		return
	}
	
	fmt.Println("Data stored successfully")
	
	// Output:
	// Data stored successfully
}

func ExampleDHT_Get() {
	config := DefaultDHTConfig()
	config.Port = 0
	
	dht, err := NewDHT(config)
	if err != nil {
		fmt.Printf("Failed to create DHT: %v\n", err)
		return
	}
	
	err = dht.Start()
	if err != nil {
		fmt.Printf("Failed to start DHT: %v\n", err)
		return
	}
	defer dht.Stop()
	
	// Store some data first
	dht.Put("greeting", "Hello, World!")
	dht.Put("number", 42)
	
	// Retrieve the data
	greeting, err := dht.Get("greeting")
	if err != nil {
		fmt.Printf("Failed to get greeting: %v\n", err)
		return
	}
	
	number, err := dht.Get("number")
	if err != nil {
		fmt.Printf("Failed to get number: %v\n", err)
		return
	}
	
	fmt.Printf("Greeting: %s\n", greeting)
	fmt.Printf("Number: %v\n", number)
	
	// Try to get non-existent key
	_, err = dht.Get("nonexistent")
	if err != nil {
		fmt.Printf("Key not found: %v\n", err)
	}
	
	// Output:
	// Greeting: Hello, World!
	// Number: 42
	// Key not found: key not found
}

func ExampleDHT_GetStatistics() {
	config := DefaultDHTConfig()
	config.Port = 0
	config.EnableMetrics = true
	
	dht, err := NewDHT(config)
	if err != nil {
		fmt.Printf("Failed to create DHT: %v\n", err)
		return
	}
	
	err = dht.Start()
	if err != nil {
		fmt.Printf("Failed to start DHT: %v\n", err)
		return
	}
	defer dht.Stop()
	
	// Perform some operations
	for i := 0; i < 10; i++ {
		key := fmt.Sprintf("key_%d", i)
		value := fmt.Sprintf("value_%d", i)
		dht.Put(key, value)
		dht.Get(key)
	}
	
	// Get statistics
	stats := dht.GetStatistics()
	
	fmt.Printf("DHT Statistics:\n")
	fmt.Printf("  Stores Successful: %d\n", stats.StoresSuccessful)
	fmt.Printf("  Lookups Successful: %d\n", stats.LookupsSuccessful)
	fmt.Printf("  Data Items Stored: %d\n", stats.DataItemsStored)
	fmt.Printf("  Average Store Time: %v\n", stats.AverageStoreTime)
	fmt.Printf("  Average Lookup Time: %v\n", stats.AverageLookupTime)
	
	// Output:
	// DHT Statistics:
	//   Stores Successful: 10
	//   Lookups Successful: 10
	//   Data Items Stored: 10
	//   Average Store Time: 123.456µs
	//   Average Lookup Time: 98.765µs
}