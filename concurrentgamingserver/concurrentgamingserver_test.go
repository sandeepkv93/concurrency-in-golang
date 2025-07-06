package concurrentgamingserver

import (
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"testing"
	"time"
)

func TestDefaultServerConfig(t *testing.T) {
	config := DefaultServerConfig()
	
	if config.Port != 8080 {
		t.Errorf("Expected default port 8080, got %d", config.Port)
	}
	
	if config.MaxPlayers != 1000 {
		t.Errorf("Expected max players 1000, got %d", config.MaxPlayers)
	}
	
	if config.TickRate != 60 {
		t.Errorf("Expected tick rate 60, got %d", config.TickRate)
	}
	
	if !config.AntiCheatEnabled {
		t.Error("Expected anti-cheat to be enabled by default")
	}
	
	if !config.EnableStatistics {
		t.Error("Expected statistics to be enabled by default")
	}
}

func TestNewGameServer(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8081 // Use different port for testing
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	if server == nil {
		t.Fatal("Failed to create game server")
	}
	
	if !server.running {
		t.Error("Server should be running")
	}
	
	if server.config.Port != 8081 {
		t.Errorf("Expected port 8081, got %d", server.config.Port)
	}
	
	if server.antiCheat == nil {
		t.Error("Anti-cheat should be initialized when enabled")
	}
	
	if server.replayManager == nil {
		t.Error("Replay manager should be initialized when enabled")
	}
	
	if server.chatManager == nil {
		t.Error("Chat manager should be initialized")
	}
	
	if server.statistics == nil {
		t.Error("Statistics should be initialized")
	}
}

func TestPlayerCreation(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8082
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	// Create a mock connection
	mockConn := &mockConnection{}
	conn := &Connection{
		Conn:          mockConn,
		MessageQueue:  make(chan Message, 100),
		LastHeartbeat: time.Now(),
	}
	
	// Create join message
	joinMessage := &Message{
		Type:      PlayerJoin,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"username": "testplayer",
		},
	}
	
	server.handlePlayerJoin(conn, joinMessage)
	
	// Check if player was created
	if conn.Player == nil {
		t.Fatal("Player should be created after join")
	}
	
	if conn.Player.Username != "testplayer" {
		t.Errorf("Expected username 'testplayer', got '%s'", conn.Player.Username)
	}
	
	if conn.Player.Status != Online {
		t.Errorf("Expected player status Online, got %v", conn.Player.Status)
	}
	
	if conn.Player.Health != 100 {
		t.Errorf("Expected player health 100, got %d", conn.Player.Health)
	}
	
	// Check if player is in server's player map
	server.mutex.RLock()
	storedPlayer, exists := server.players["testplayer"]
	server.mutex.RUnlock()
	
	if !exists {
		t.Error("Player should be stored in server's player map")
	}
	
	if storedPlayer.ID != conn.Player.ID {
		t.Error("Stored player should match connection player")
	}
}

func TestGameCreation(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8083
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	// Create player
	player := &Player{
		ID:       "player1",
		Username: "testplayer",
		Status:   Online,
	}
	
	mockConn := &mockConnection{}
	conn := &Connection{
		Conn:         mockConn,
		Player:       player,
		MessageQueue: make(chan Message, 100),
	}
	player.Connection = conn
	
	// Create game message
	gameMessage := &Message{
		Type:      GameAction,
		PlayerID:  player.ID,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"action":      "create_game",
			"game_type":   float64(ActionGame),
			"max_players": float64(4),
		},
	}
	
	server.handleCreateGame(conn, gameMessage)
	
	// Check if game was created
	server.mutex.RLock()
	gameCount := len(server.games)
	server.mutex.RUnlock()
	
	if gameCount != 1 {
		t.Errorf("Expected 1 game, got %d", gameCount)
	}
	
	// Get the created game
	var game *Game
	server.mutex.RLock()
	for _, g := range server.games {
		game = g
		break
	}
	server.mutex.RUnlock()
	
	if game == nil {
		t.Fatal("Game should be created")
	}
	
	if game.Type != ActionGame {
		t.Errorf("Expected game type ActionGame, got %v", game.Type)
	}
	
	if game.MaxPlayers != 4 {
		t.Errorf("Expected max players 4, got %d", game.MaxPlayers)
	}
	
	if game.State != WaitingForPlayers {
		t.Errorf("Expected game state WaitingForPlayers, got %v", game.State)
	}
	
	if game.CreatedBy != player.ID {
		t.Errorf("Expected game created by %s, got %s", player.ID, game.CreatedBy)
	}
	
	// Check if player is in the game
	game.mutex.RLock()
	_, playerInGame := game.Players[player.ID]
	game.mutex.RUnlock()
	
	if !playerInGame {
		t.Error("Player should be added to the created game")
	}
}

func TestPlayerJoinGame(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8084
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	// Create game first
	game := &Game{
		ID:             "game1",
		Name:           "Test Game",
		Type:           ActionGame,
		State:          WaitingForPlayers,
		MaxPlayers:     4,
		CurrentPlayers: 0,
		Players:        make(map[string]*Player),
		Spectators:     make(map[string]*Player),
		GameData:       make(map[string]interface{}),
		Settings:       make(map[string]interface{}),
		Events:         make([]*GameEvent, 0),
		CreatedAt:      time.Now(),
	}
	game.Map = server.createGenericMap()
	
	server.mutex.Lock()
	server.games[game.ID] = game
	server.mutex.Unlock()
	
	// Create player
	player := &Player{
		ID:             "player1",
		Username:       "testplayer",
		Status:         Online,
		CurrentGameIDs: make([]string, 0),
	}
	
	mockConn := &mockConnection{}
	conn := &Connection{
		Conn:         mockConn,
		Player:       player,
		MessageQueue: make(chan Message, 100),
	}
	player.Connection = conn
	
	// Join game message
	joinMessage := &Message{
		Type:      GameAction,
		PlayerID:  player.ID,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"action":  "join_game",
			"game_id": game.ID,
		},
	}
	
	server.handleJoinGame(conn, joinMessage)
	
	// Check if player joined the game
	game.mutex.RLock()
	_, playerInGame := game.Players[player.ID]
	currentPlayers := game.CurrentPlayers
	game.mutex.RUnlock()
	
	if !playerInGame {
		t.Error("Player should be in the game after joining")
	}
	
	if currentPlayers != 1 {
		t.Errorf("Expected 1 current player, got %d", currentPlayers)
	}
	
	player.mutex.RLock()
	playerStatus := player.Status
	gameIDs := player.CurrentGameIDs
	player.mutex.RUnlock()
	
	if playerStatus != InGame {
		t.Errorf("Expected player status InGame, got %v", playerStatus)
	}
	
	if len(gameIDs) != 1 || gameIDs[0] != game.ID {
		t.Error("Player should have the game ID in CurrentGameIDs")
	}
}

func TestPlayerMovement(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8085
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	// Create player
	player := &Player{
		ID:       "player1",
		Username: "testplayer",
		Status:   InGame,
		Position: &Position{X: 0, Y: 0, Z: 0},
	}
	
	mockConn := &mockConnection{}
	conn := &Connection{
		Conn:         mockConn,
		Player:       player,
		MessageQueue: make(chan Message, 100),
	}
	player.Connection = conn
	
	// Movement message
	moveMessage := &Message{
		Type:      GameAction,
		PlayerID:  player.ID,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"action": "player_move",
			"position": map[string]interface{}{
				"x":        10.0,
				"y":        15.0,
				"z":        2.0,
				"rotation": 45.0,
			},
		},
	}
	
	server.handlePlayerMove(conn, moveMessage)
	
	// Check if player position was updated
	player.mutex.RLock()
	position := player.Position
	player.mutex.RUnlock()
	
	if position.X != 10.0 {
		t.Errorf("Expected X position 10.0, got %f", position.X)
	}
	
	if position.Y != 15.0 {
		t.Errorf("Expected Y position 15.0, got %f", position.Y)
	}
	
	if position.Z != 2.0 {
		t.Errorf("Expected Z position 2.0, got %f", position.Z)
	}
	
	if position.Rotation != 45.0 {
		t.Errorf("Expected rotation 45.0, got %f", position.Rotation)
	}
}

func TestGameMaps(t *testing.T) {
	config := DefaultServerConfig()
	server := NewGameServer(config)
	defer server.Shutdown()
	
	// Test different map types
	testCases := []struct {
		gameType     GameType
		expectedName string
	}{
		{BattleRoyale, "Battle Royale Island"},
		{ActionGame, "Action Arena"},
		{StrategyGame, "Generic Map"},
	}
	
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("GameType_%d", tc.gameType), func(t *testing.T) {
			gameMap := server.createGameMap(tc.gameType)
			
			if gameMap == nil {
				t.Fatal("Game map should not be nil")
			}
			
			if gameMap.Name != tc.expectedName {
				t.Errorf("Expected map name '%s', got '%s'", tc.expectedName, gameMap.Name)
			}
			
			if len(gameMap.SpawnPoints) == 0 {
				t.Error("Game map should have spawn points")
			}
			
			if gameMap.Objects == nil {
				t.Error("Game map should have objects map initialized")
			}
			
			if gameMap.Zones == nil {
				t.Error("Game map should have zones map initialized")
			}
		})
	}
}

func TestBattleRoyaleMap(t *testing.T) {
	config := DefaultServerConfig()
	server := NewGameServer(config)
	defer server.Shutdown()
	
	gameMap := server.createBattleRoyaleMap()
	
	if gameMap.Name != "Battle Royale Island" {
		t.Errorf("Expected map name 'Battle Royale Island', got '%s'", gameMap.Name)
	}
	
	if gameMap.Width != 1000 || gameMap.Height != 1000 {
		t.Errorf("Expected map size 1000x1000, got %fx%f", gameMap.Width, gameMap.Height)
	}
	
	if len(gameMap.SpawnPoints) != 100 {
		t.Errorf("Expected 100 spawn points, got %d", len(gameMap.SpawnPoints))
	}
	
	// Check if shrink zone exists
	shrinkZone, exists := gameMap.Zones["shrink"]
	if !exists {
		t.Fatal("Shrink zone should exist in battle royale map")
	}
	
	if shrinkZone.Type != "shrink" {
		t.Errorf("Expected shrink zone type 'shrink', got '%s'", shrinkZone.Type)
	}
	
	if shrinkZone.Radius != 500 {
		t.Errorf("Expected shrink zone radius 500, got %f", shrinkZone.Radius)
	}
	
	if !shrinkZone.Active {
		t.Error("Shrink zone should be active")
	}
}

func TestAntiCheatManager(t *testing.T) {
	antiCheat := NewAntiCheatManager()
	
	if antiCheat == nil {
		t.Fatal("Anti-cheat manager should not be nil")
	}
	
	if !antiCheat.enabled {
		t.Error("Anti-cheat should be enabled")
	}
	
	if antiCheat.violations == nil {
		t.Error("Violations map should be initialized")
	}
	
	if antiCheat.thresholds == nil {
		t.Error("Thresholds map should be initialized")
	}
	
	// Check default thresholds
	if antiCheat.thresholds["max_speed"] != 100.0 {
		t.Errorf("Expected max speed threshold 100.0, got %f", antiCheat.thresholds["max_speed"])
	}
	
	if antiCheat.thresholds["teleport"] != 50.0 {
		t.Errorf("Expected teleport threshold 50.0, got %f", antiCheat.thresholds["teleport"])
	}
}

func TestReplayManager(t *testing.T) {
	replayManager := NewReplayManager()
	
	if replayManager == nil {
		t.Fatal("Replay manager should not be nil")
	}
	
	if !replayManager.enabled {
		t.Error("Replay manager should be enabled")
	}
	
	if replayManager.replays == nil {
		t.Error("Replays map should be initialized")
	}
	
	if replayManager.maxSize != 1000 {
		t.Errorf("Expected max size 1000, got %d", replayManager.maxSize)
	}
	
	if !replayManager.compress {
		t.Error("Compression should be enabled by default")
	}
}

func TestChatManager(t *testing.T) {
	chatManager := NewChatManager()
	
	if chatManager == nil {
		t.Fatal("Chat manager should not be nil")
	}
	
	if !chatManager.enabled {
		t.Error("Chat manager should be enabled")
	}
	
	if chatManager.channels == nil {
		t.Error("Channels map should be initialized")
	}
	
	if chatManager.moderation == nil {
		t.Error("Moderation should be initialized")
	}
	
	if !chatManager.moderation.Enabled {
		t.Error("Chat moderation should be enabled")
	}
	
	if len(chatManager.moderation.WordFilter) == 0 {
		t.Error("Word filter should have default words")
	}
}

func TestMessageHandling(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8086
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	// Create player
	player := &Player{
		ID:       "player1",
		Username: "testplayer",
		Status:   Online,
	}
	
	mockConn := &mockConnection{}
	conn := &Connection{
		Conn:         mockConn,
		Player:       player,
		MessageQueue: make(chan Message, 100),
	}
	player.Connection = conn
	
	// Test heartbeat message
	heartbeatMessage := &Message{
		Type:      Heartbeat,
		PlayerID:  player.ID,
		Timestamp: time.Now(),
		Data:      map[string]interface{}{},
	}
	
	oldHeartbeat := conn.LastHeartbeat
	time.Sleep(10 * time.Millisecond)
	
	server.handleHeartbeat(conn, heartbeatMessage)
	
	conn.mutex.RLock()
	newHeartbeat := conn.LastHeartbeat
	conn.mutex.RUnlock()
	
	if !newHeartbeat.After(oldHeartbeat) {
		t.Error("Heartbeat timestamp should be updated")
	}
	
	// Check if response was queued
	select {
	case response := <-conn.MessageQueue:
		if response.Type != Heartbeat {
			t.Errorf("Expected heartbeat response, got %v", response.Type)
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("Should receive heartbeat response")
	}
}

func TestStatisticsTracking(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8087
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	stats := server.statistics
	
	if stats == nil {
		t.Fatal("Statistics should be initialized")
	}
	
	if stats.StartTime.IsZero() {
		t.Error("Start time should be set")
	}
	
	// Test statistics updates
	originalConnections := stats.TotalConnections
	
	// Simulate connection
	stats.mutex.Lock()
	stats.TotalConnections++
	stats.CurrentConnections++
	stats.mutex.Unlock()
	
	if stats.TotalConnections != originalConnections+1 {
		t.Error("Total connections should be incremented")
	}
	
	if stats.CurrentConnections != 1 {
		t.Error("Current connections should be 1")
	}
}

func TestConcurrentConnections(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8088
	config.MaxPlayers = 100
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	const numConnections = 10
	var wg sync.WaitGroup
	
	// Simulate multiple concurrent connections
	for i := 0; i < numConnections; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			player := &Player{
				ID:       fmt.Sprintf("player%d", id),
				Username: fmt.Sprintf("player%d", id),
				Status:   Online,
			}
			
			mockConn := &mockConnection{}
			conn := &Connection{
				Conn:         mockConn,
				Player:       player,
				MessageQueue: make(chan Message, 100),
			}
			player.Connection = conn
			
			// Join message
			joinMessage := &Message{
				Type:      PlayerJoin,
				PlayerID:  player.ID,
				Timestamp: time.Now(),
				Data: map[string]interface{}{
					"username": player.Username,
				},
			}
			
			server.handlePlayerJoin(conn, joinMessage)
		}(i)
	}
	
	wg.Wait()
	
	// Check if all players were added
	server.mutex.RLock()
	playerCount := len(server.players)
	server.mutex.RUnlock()
	
	if playerCount != numConnections {
		t.Errorf("Expected %d players, got %d", numConnections, playerCount)
	}
}

func TestGameStateTransitions(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8089
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	// Create game
	game := &Game{
		ID:             "game1",
		Name:           "Test Game",
		Type:           ActionGame,
		State:          WaitingForPlayers,
		MaxPlayers:     2,
		CurrentPlayers: 0,
		Players:        make(map[string]*Player),
		Spectators:     make(map[string]*Player),
		GameData:       make(map[string]interface{}),
		Settings:       make(map[string]interface{}),
		Events:         make([]*GameEvent, 0),
		CreatedAt:      time.Now(),
	}
	game.Map = server.createGenericMap()
	
	server.mutex.Lock()
	server.games[game.ID] = game
	server.mutex.Unlock()
	
	// Add first player
	player1 := &Player{
		ID:             "player1",
		Username:       "player1",
		Status:         Online,
		CurrentGameIDs: make([]string, 0),
	}
	
	server.addPlayerToGame(player1, game)
	
	// Check game state (should still be waiting)
	if game.State != WaitingForPlayers {
		t.Errorf("Expected game state WaitingForPlayers, got %v", game.State)
	}
	
	// Add second player
	player2 := &Player{
		ID:             "player2",
		Username:       "player2",
		Status:         Online,
		CurrentGameIDs: make([]string, 0),
	}
	
	server.addPlayerToGame(player2, game)
	
	// Start the game
	server.startGame(game)
	
	// Check game state (should be in progress)
	if game.State != InProgress {
		t.Errorf("Expected game state InProgress, got %v", game.State)
	}
	
	if game.StartTime.IsZero() {
		t.Error("Game start time should be set")
	}
	
	// End the game
	server.endGame(game, "Test ended")
	
	// Check game state (should be finished)
	if game.State != Finished {
		t.Errorf("Expected game state Finished, got %v", game.State)
	}
	
	if game.EndTime.IsZero() {
		t.Error("Game end time should be set")
	}
	
	if game.Duration == 0 {
		t.Error("Game duration should be set")
	}
}

func TestPlayerDisconnection(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8090
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	// Create player
	player := &Player{
		ID:       "player1",
		Username: "testplayer",
		Status:   Online,
	}
	
	server.mutex.Lock()
	server.players[player.Username] = player
	server.mutex.Unlock()
	
	// Simulate disconnection
	server.handlePlayerDisconnect(player)
	
	// Check player status
	player.mutex.RLock()
	status := player.Status
	connection := player.Connection
	player.mutex.RUnlock()
	
	if status != Offline {
		t.Errorf("Expected player status Offline, got %v", status)
	}
	
	if connection != nil {
		t.Error("Player connection should be nil after disconnect")
	}
}

func TestIDGeneration(t *testing.T) {
	// Test that generated IDs are unique
	ids := make(map[string]bool)
	const numIDs = 1000
	
	for i := 0; i < numIDs; i++ {
		id := generateID()
		
		if id == "" {
			t.Error("Generated ID should not be empty")
		}
		
		if ids[id] {
			t.Errorf("Duplicate ID generated: %s", id)
		}
		
		ids[id] = true
	}
	
	if len(ids) != numIDs {
		t.Errorf("Expected %d unique IDs, got %d", numIDs, len(ids))
	}
}

func TestErrorHandling(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8091
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	mockConn := &mockConnection{}
	conn := &Connection{
		Conn:         mockConn,
		MessageQueue: make(chan Message, 100),
	}
	
	// Test error when no player is authenticated
	gameMessage := &Message{
		Type:      GameAction,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"action": "join_game",
		},
	}
	
	server.handleGameAction(conn, gameMessage)
	
	// Should receive error message
	select {
	case response := <-conn.MessageQueue:
		if errorMsg, exists := response.Data["error"]; !exists {
			t.Error("Should receive error message for unauthenticated action")
		} else if errorMsg != "Not authenticated" {
			t.Errorf("Expected 'Not authenticated' error, got %v", errorMsg)
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("Should receive error response")
	}
}

func TestShutdown(t *testing.T) {
	config := DefaultServerConfig()
	config.Port = 8092
	
	server := NewGameServer(config)
	
	if !server.running {
		t.Error("Server should be running initially")
	}
	
	err := server.Shutdown()
	if err != nil {
		t.Fatalf("Failed to shutdown server: %v", err)
	}
	
	if server.running {
		t.Error("Server should not be running after shutdown")
	}
	
	// Test double shutdown
	err = server.Shutdown()
	if err == nil {
		t.Error("Expected error on double shutdown")
	}
}

// Mock connection for testing
type mockConnection struct {
	data []byte
}

func (m *mockConnection) Read(b []byte) (n int, err error) {
	if len(m.data) == 0 {
		return 0, fmt.Errorf("no data")
	}
	n = copy(b, m.data)
	m.data = m.data[n:]
	return n, nil
}

func (m *mockConnection) Write(b []byte) (n int, err error) {
	return len(b), nil
}

func (m *mockConnection) Close() error {
	return nil
}

func (m *mockConnection) LocalAddr() net.Addr {
	return &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 8080}
}

func (m *mockConnection) RemoteAddr() net.Addr {
	return &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 12345}
}

func (m *mockConnection) SetDeadline(t time.Time) error {
	return nil
}

func (m *mockConnection) SetReadDeadline(t time.Time) error {
	return nil
}

func (m *mockConnection) SetWriteDeadline(t time.Time) error {
	return nil
}

// Benchmark tests

func BenchmarkPlayerJoin(b *testing.B) {
	config := DefaultServerConfig()
	config.Port = 9000
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mockConn := &mockConnection{}
		conn := &Connection{
			Conn:         mockConn,
			MessageQueue: make(chan Message, 100),
		}
		
		joinMessage := &Message{
			Type:      PlayerJoin,
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"username": fmt.Sprintf("player%d", i),
			},
		}
		
		server.handlePlayerJoin(conn, joinMessage)
	}
}

func BenchmarkGameCreation(b *testing.B) {
	config := DefaultServerConfig()
	config.Port = 9001
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	// Create a player first
	player := &Player{
		ID:       "player1",
		Username: "testplayer",
		Status:   Online,
	}
	
	mockConn := &mockConnection{}
	conn := &Connection{
		Conn:         mockConn,
		Player:       player,
		MessageQueue: make(chan Message, 100),
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gameMessage := &Message{
			Type:      GameAction,
			PlayerID:  player.ID,
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"action":      "create_game",
				"game_type":   float64(ActionGame),
				"max_players": float64(4),
			},
		}
		
		server.handleCreateGame(conn, gameMessage)
	}
}

func BenchmarkMessageProcessing(b *testing.B) {
	config := DefaultServerConfig()
	config.Port = 9002
	
	server := NewGameServer(config)
	defer server.Shutdown()
	
	player := &Player{
		ID:       "player1",
		Username: "testplayer",
		Status:   InGame,
		Position: &Position{X: 0, Y: 0, Z: 0},
	}
	
	mockConn := &mockConnection{}
	conn := &Connection{
		Conn:         mockConn,
		Player:       player,
		MessageQueue: make(chan Message, 100),
	}
	
	moveMessage := &Message{
		Type:      GameAction,
		PlayerID:  player.ID,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"action": "player_move",
			"position": map[string]interface{}{
				"x":        10.0,
				"y":        15.0,
				"z":        2.0,
				"rotation": 45.0,
			},
		},
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		server.processMessage(conn, moveMessage)
	}
}

func BenchmarkIDGeneration(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		generateID()
	}
}

func BenchmarkMapCreation(b *testing.B) {
	config := DefaultServerConfig()
	server := NewGameServer(config)
	defer server.Shutdown()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		server.createBattleRoyaleMap()
	}
}

// Example functions
func ExampleNewGameServer() {
	// Create server configuration
	config := DefaultServerConfig()
	config.Port = 8080
	config.MaxPlayers = 1000
	config.TickRate = 60
	
	// Create game server
	server := NewGameServer(config)
	defer server.Shutdown()
	
	fmt.Printf("Game server created with:\n")
	fmt.Printf("  Port: %d\n", server.config.Port)
	fmt.Printf("  Max Players: %d\n", server.config.MaxPlayers)
	fmt.Printf("  Tick Rate: %d\n", server.config.TickRate)
	fmt.Printf("  Anti-cheat: %t\n", server.config.AntiCheatEnabled)
	
	// Output:
	// Game server created with:
	//   Port: 8080
	//   Max Players: 1000
	//   Tick Rate: 60
	//   Anti-cheat: true
}

func ExampleGameServer_handlePlayerJoin() {
	config := DefaultServerConfig()
	server := NewGameServer(config)
	defer server.Shutdown()
	
	// Create mock connection
	mockConn := &mockConnection{}
	conn := &Connection{
		Conn:         mockConn,
		MessageQueue: make(chan Message, 100),
	}
	
	// Create join message
	joinMessage := &Message{
		Type:      PlayerJoin,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"username": "PlayerOne",
		},
	}
	
	// Handle player join
	server.handlePlayerJoin(conn, joinMessage)
	
	if conn.Player != nil {
		fmt.Printf("Player joined: %s\n", conn.Player.Username)
		fmt.Printf("Player ID: %s\n", conn.Player.ID)
		fmt.Printf("Player Status: %v\n", conn.Player.Status)
		fmt.Printf("Player Health: %d\n", conn.Player.Health)
	}
	
	// Output:
	// Player joined: PlayerOne
	// Player ID: player_123456789_1234
	// Player Status: 0
	// Player Health: 100
}

func ExampleGameServer_createBattleRoyaleMap() {
	config := DefaultServerConfig()
	server := NewGameServer(config)
	defer server.Shutdown()
	
	// Create battle royale map
	gameMap := server.createBattleRoyaleMap()
	
	fmt.Printf("Battle Royale Map:\n")
	fmt.Printf("  Name: %s\n", gameMap.Name)
	fmt.Printf("  Dimensions: %.0fx%.0fx%.0f\n", gameMap.Width, gameMap.Height, gameMap.Depth)
	fmt.Printf("  Spawn Points: %d\n", len(gameMap.SpawnPoints))
	fmt.Printf("  Zones: %d\n", len(gameMap.Zones))
	
	if shrinkZone, exists := gameMap.Zones["shrink"]; exists {
		fmt.Printf("  Shrink Zone Radius: %.0f\n", shrinkZone.Radius)
		fmt.Printf("  Shrink Zone Active: %t\n", shrinkZone.Active)
	}
	
	// Output:
	// Battle Royale Map:
	//   Name: Battle Royale Island
	//   Dimensions: 1000x1000x100
	//   Spawn Points: 100
	//   Zones: 1
	//   Shrink Zone Radius: 500
	//   Shrink Zone Active: true
}