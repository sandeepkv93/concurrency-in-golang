package concurrentblockchainminer

import (
	"context"
	"fmt"
	"math/big"
	"sync"
	"testing"
	"time"
)

func TestNewBlockchain(t *testing.T) {
	bc := NewBlockchain(16)
	
	if bc.difficulty != 16 {
		t.Errorf("Expected difficulty 16, got %d", bc.difficulty)
	}
	
	if len(bc.chain) != 1 {
		t.Errorf("Expected chain length 1, got %d", len(bc.chain))
	}
	
	genesis := bc.chain[0]
	if genesis.Index != 0 {
		t.Errorf("Expected genesis block index 0, got %d", genesis.Index)
	}
	
	if genesis.PreviousHash != "0" {
		t.Errorf("Expected genesis block previous hash '0', got %s", genesis.PreviousHash)
	}
}

func TestCalculateHash(t *testing.T) {
	bc := NewBlockchain(16)
	
	block := &Block{
		Index:        1,
		Timestamp:    1234567890,
		Data:         []byte("Test Block"),
		PreviousHash: "abc123",
		Nonce:        42,
		Difficulty:   16,
	}
	
	hash1 := bc.calculateHash(block)
	hash2 := bc.calculateHash(block)
	
	if hash1 != hash2 {
		t.Error("Hash calculation is not deterministic")
	}
	
	block.Nonce = 43
	hash3 := bc.calculateHash(block)
	
	if hash1 == hash3 {
		t.Error("Hash should change when nonce changes")
	}
}

func TestAddBlock(t *testing.T) {
	bc := NewBlockchain(8)
	genesis := bc.GetLatestBlock()
	
	newBlock := &Block{
		Index:        1,
		Timestamp:    time.Now().Unix(),
		Data:         []byte("Test Block"),
		PreviousHash: genesis.Hash,
		Difficulty:   8,
	}
	
	for nonce := int64(0); ; nonce++ {
		newBlock.Nonce = nonce
		hash := bc.calculateHash(newBlock)
		hashInt := new(big.Int)
		hashInt.SetString(hash, 16)
		
		target := bc.calculateTarget(newBlock.Difficulty)
		if hashInt.Cmp(target) < 0 {
			newBlock.Hash = hash
			break
		}
	}
	
	success := bc.AddBlock(newBlock)
	if !success {
		t.Error("Failed to add valid block")
	}
	
	if len(bc.chain) != 2 {
		t.Errorf("Expected chain length 2, got %d", len(bc.chain))
	}
}

func TestAddBlockInvalidPreviousHash(t *testing.T) {
	bc := NewBlockchain(8)
	
	newBlock := &Block{
		Index:        1,
		Timestamp:    time.Now().Unix(),
		Data:         []byte("Test Block"),
		PreviousHash: "invalid",
		Difficulty:   8,
	}
	
	success := bc.AddBlock(newBlock)
	if success {
		t.Error("Should not add block with invalid previous hash")
	}
}

func TestAddBlockInvalidIndex(t *testing.T) {
	bc := NewBlockchain(8)
	genesis := bc.GetLatestBlock()
	
	newBlock := &Block{
		Index:        5,
		Timestamp:    time.Now().Unix(),
		Data:         []byte("Test Block"),
		PreviousHash: genesis.Hash,
		Difficulty:   8,
	}
	
	success := bc.AddBlock(newBlock)
	if success {
		t.Error("Should not add block with invalid index")
	}
}

func TestMinerBasic(t *testing.T) {
	bc := NewBlockchain(8)
	miner := NewMiner(1, 2, bc)
	
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	miner.Start(ctx)
	defer miner.Stop()
	
	tx := Transaction{
		From:      "Alice",
		To:        "Bob",
		Amount:    10.0,
		Timestamp: time.Now().Unix(),
	}
	miner.AddTransaction(tx)
	
	select {
	case result := <-miner.GetMiningResult():
		if result.Block == nil {
			t.Error("Mining result has nil block")
		}
		if result.MinedBy != 1 {
			t.Errorf("Expected miner ID 1, got %d", result.MinedBy)
		}
		if result.HashRate <= 0 {
			t.Error("Hash rate should be positive")
		}
	case <-ctx.Done():
		t.Error("Mining timed out")
	}
}

func TestMiningPool(t *testing.T) {
	bc := NewBlockchain(10)
	pool := NewMiningPool(4, 2, bc)
	
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	pool.Start(ctx)
	defer pool.Stop()
	
	for i := 0; i < 5; i++ {
		tx := Transaction{
			From:      fmt.Sprintf("User%d", i),
			To:        fmt.Sprintf("User%d", i+1),
			Amount:    float64(i * 10),
			Timestamp: time.Now().Unix(),
		}
		pool.AddTransaction(tx)
	}
	
	blocksFound := 0
	timeout := time.After(20 * time.Second)
	
	for blocksFound < 3 {
		select {
		case result := <-pool.GetResults():
			if result.Block != nil {
				blocksFound++
				t.Logf("Block %d mined by miner %d with hash rate %.2f H/s",
					result.Block.Index, result.MinedBy, result.HashRate)
			}
		case <-timeout:
			t.Fatalf("Only found %d blocks before timeout", blocksFound)
		}
	}
	
	if len(bc.GetChain()) != blocksFound+1 {
		t.Errorf("Expected blockchain length %d, got %d", blocksFound+1, len(bc.GetChain()))
	}
}

func TestPoolHashRate(t *testing.T) {
	bc := NewBlockchain(20)
	pool := NewMiningPool(3, 2, bc)
	
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	pool.Start(ctx)
	defer pool.Stop()
	
	time.Sleep(2 * time.Second)
	
	hashRate := pool.GetPoolHashRate()
	if hashRate <= 0 {
		t.Error("Pool hash rate should be positive")
	}
	
	t.Logf("Pool hash rate: %.2f H/s", hashRate)
}

func TestBlockchainValidation(t *testing.T) {
	bc := NewBlockchain(8)
	
	if !bc.Validate() {
		t.Error("Blockchain with only genesis block should be valid")
	}
	
	for i := 0; i < 3; i++ {
		latestBlock := bc.GetLatestBlock()
		newBlock := &Block{
			Index:        latestBlock.Index + 1,
			Timestamp:    time.Now().Unix(),
			Data:         []byte(fmt.Sprintf("Block %d", i)),
			PreviousHash: latestBlock.Hash,
			Difficulty:   8,
		}
		
		for nonce := int64(0); ; nonce++ {
			newBlock.Nonce = nonce
			hash := bc.calculateHash(newBlock)
			hashInt := new(big.Int)
			hashInt.SetString(hash, 16)
			
			target := bc.calculateTarget(newBlock.Difficulty)
			if hashInt.Cmp(target) < 0 {
				newBlock.Hash = hash
				break
			}
		}
		
		bc.AddBlock(newBlock)
	}
	
	if !bc.Validate() {
		t.Error("Valid blockchain should pass validation")
	}
	
	bc.chain[2].PreviousHash = "invalid"
	if bc.Validate() {
		t.Error("Invalid blockchain should fail validation")
	}
}

func TestDifficultyAdjustment(t *testing.T) {
	bc := NewBlockchain(10)
	pool := NewMiningPool(2, 2, bc)
	
	initialDifficulty := bc.GetDifficulty()
	
	for i := 0; i < 12; i++ {
		latestBlock := bc.GetLatestBlock()
		newBlock := &Block{
			Index:        latestBlock.Index + 1,
			Timestamp:    latestBlock.Timestamp + 1,
			Data:         []byte(fmt.Sprintf("Block %d", i)),
			PreviousHash: latestBlock.Hash,
			Difficulty:   bc.GetDifficulty(),
		}
		
		for nonce := int64(0); ; nonce++ {
			newBlock.Nonce = nonce
			hash := bc.calculateHash(newBlock)
			hashInt := new(big.Int)
			hashInt.SetString(hash, 16)
			
			target := bc.calculateTarget(newBlock.Difficulty)
			if hashInt.Cmp(target) < 0 {
				newBlock.Hash = hash
				break
			}
		}
		
		bc.AddBlock(newBlock)
	}
	
	pool.AdjustDifficulty()
	newDifficulty := bc.GetDifficulty()
	
	if newDifficulty == initialDifficulty {
		t.Log("Difficulty remained the same (expected based on block times)")
	} else if newDifficulty > initialDifficulty {
		t.Log("Difficulty increased (blocks mined too quickly)")
	} else {
		t.Log("Difficulty decreased (blocks mined too slowly)")
	}
}

func TestConcurrentMining(t *testing.T) {
	bc := NewBlockchain(12)
	numMiners := 5
	miners := make([]*Miner, numMiners)
	
	for i := 0; i < numMiners; i++ {
		miners[i] = NewMiner(i, 2, bc)
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	for _, miner := range miners {
		miner.Start(ctx)
		defer miner.Stop()
	}
	
	var wg sync.WaitGroup
	results := make(chan MiningResult, numMiners)
	
	for _, miner := range miners {
		wg.Add(1)
		go func(m *Miner) {
			defer wg.Done()
			select {
			case result := <-m.GetMiningResult():
				results <- result
			case <-ctx.Done():
			}
		}(miner)
	}
	
	go func() {
		wg.Wait()
		close(results)
	}()
	
	firstResult := <-results
	if firstResult.Block == nil {
		t.Error("First mining result has nil block")
	}
	
	t.Logf("First block mined by miner %d", firstResult.MinedBy)
}

func TestTransaction(t *testing.T) {
	bc := NewBlockchain(8)
	miner := NewMiner(1, 2, bc)
	
	tx1 := Transaction{From: "Alice", To: "Bob", Amount: 50.0, Timestamp: time.Now().Unix()}
	tx2 := Transaction{From: "Bob", To: "Charlie", Amount: 30.0, Timestamp: time.Now().Unix()}
	tx3 := Transaction{From: "Charlie", To: "Alice", Amount: 20.0, Timestamp: time.Now().Unix()}
	
	miner.AddTransaction(tx1)
	miner.AddTransaction(tx2)
	miner.AddTransaction(tx3)
	
	data := miner.serializeTransactions([]Transaction{tx1, tx2, tx3})
	if len(data) == 0 {
		t.Error("Serialized transaction data should not be empty")
	}
	
	miner.ClearTransactions()
	emptyData := miner.serializeTransactions(miner.pendingTxs)
	if string(emptyData) != "No transactions" {
		t.Error("Empty transaction list should serialize to 'No transactions'")
	}
}

func TestStatsCollector(t *testing.T) {
	sc := NewStatsCollector()
	
	sc.RecordBlock(1, 5*time.Second)
	sc.RecordBlock(1, 3*time.Second)
	sc.RecordBlock(2, 4*time.Second)
	
	miner1Stats := sc.GetMinerStats(1)
	if miner1Stats.BlocksMined != 2 {
		t.Errorf("Expected miner 1 to have mined 2 blocks, got %d", miner1Stats.BlocksMined)
	}
	
	miner2Stats := sc.GetMinerStats(2)
	if miner2Stats.BlocksMined != 1 {
		t.Errorf("Expected miner 2 to have mined 1 block, got %d", miner2Stats.BlocksMined)
	}
	
	globalStats := sc.GetGlobalStats()
	if globalStats["total_blocks"].(int64) != 3 {
		t.Errorf("Expected total blocks 3, got %v", globalStats["total_blocks"])
	}
}

func TestGetBlockByIndex(t *testing.T) {
	bc := NewBlockchain(8)
	
	genesis := bc.GetBlockByIndex(0)
	if genesis == nil {
		t.Error("Should find genesis block by index 0")
	}
	
	nonExistent := bc.GetBlockByIndex(999)
	if nonExistent != nil {
		t.Error("Should return nil for non-existent block index")
	}
}

func TestGetBlockByHash(t *testing.T) {
	bc := NewBlockchain(8)
	genesis := bc.GetLatestBlock()
	
	block := bc.GetBlockByHash(genesis.Hash)
	if block == nil {
		t.Error("Should find genesis block by hash")
	}
	
	if block.Index != genesis.Index {
		t.Error("Retrieved block should match genesis block")
	}
	
	nonExistent := bc.GetBlockByHash("nonexistent")
	if nonExistent != nil {
		t.Error("Should return nil for non-existent block hash")
	}
}

func TestMerkleTree(t *testing.T) {
	data := [][]byte{
		[]byte("Transaction 1"),
		[]byte("Transaction 2"),
		[]byte("Transaction 3"),
		[]byte("Transaction 4"),
	}
	
	tree := NewMerkleTree(data)
	if tree.Root == nil {
		t.Error("Merkle tree root should not be nil")
	}
	
	singleData := [][]byte{[]byte("Single Transaction")}
	singleTree := NewMerkleTree(singleData)
	if singleTree.Root == nil {
		t.Error("Single element Merkle tree should have root")
	}
}

func TestContextCancellation(t *testing.T) {
	bc := NewBlockchain(20)
	miner := NewMiner(1, 4, bc)
	
	ctx, cancel := context.WithCancel(context.Background())
	
	miner.Start(ctx)
	
	time.Sleep(100 * time.Millisecond)
	cancel()
	
	time.Sleep(100 * time.Millisecond)
	
	select {
	case <-miner.GetMiningResult():
		t.Error("Should not receive mining results after context cancellation")
	default:
	}
	
	miner.Stop()
}

func BenchmarkMining(b *testing.B) {
	bc := NewBlockchain(16)
	miner := NewMiner(1, 4, bc)
	
	ctx := context.Background()
	miner.Start(ctx)
	defer miner.Stop()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		select {
		case <-miner.GetMiningResult():
		case <-time.After(10 * time.Second):
			b.Fatal("Mining took too long")
		}
	}
}

func BenchmarkHashCalculation(b *testing.B) {
	bc := NewBlockchain(16)
	block := &Block{
		Index:        1,
		Timestamp:    time.Now().Unix(),
		Data:         []byte("Benchmark Block"),
		PreviousHash: "abc123",
		Nonce:        0,
		Difficulty:   16,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bc.calculateHash(block)
	}
}

func BenchmarkValidation(b *testing.B) {
	bc := NewBlockchain(8)
	
	for i := 0; i < 10; i++ {
		latestBlock := bc.GetLatestBlock()
		newBlock := &Block{
			Index:        latestBlock.Index + 1,
			Timestamp:    time.Now().Unix(),
			Data:         []byte(fmt.Sprintf("Block %d", i)),
			PreviousHash: latestBlock.Hash,
			Difficulty:   8,
		}
		
		for nonce := int64(0); ; nonce++ {
			newBlock.Nonce = nonce
			hash := bc.calculateHash(newBlock)
			hashInt := new(big.Int)
			hashInt.SetString(hash, 16)
			
			target := bc.calculateTarget(newBlock.Difficulty)
			if hashInt.Cmp(target) < 0 {
				newBlock.Hash = hash
				break
			}
		}
		
		bc.AddBlock(newBlock)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bc.Validate()
	}
}