package concurrentblockchainminer

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"math/big"
	"sync"
	"sync/atomic"
	"time"
)

type Block struct {
	Index        int64
	Timestamp    int64
	Data         []byte
	PreviousHash string
	Hash         string
	Nonce        int64
	Difficulty   int
	MinedBy      int
}

type Transaction struct {
	From      string
	To        string
	Amount    float64
	Timestamp int64
}

type MiningResult struct {
	Block    *Block
	MinedBy  int
	HashRate float64
	Duration time.Duration
}

type Blockchain struct {
	chain      []*Block
	difficulty int
	mutex      sync.RWMutex
}

type Miner struct {
	ID              int
	NumWorkers      int
	blockchain      *Blockchain
	hashCount       int64
	startTime       time.Time
	miningResults   chan MiningResult
	stopChan        chan struct{}
	wg              sync.WaitGroup
	currentTarget   *big.Int
	pendingTxs      []Transaction
	txMutex         sync.RWMutex
	statsCollector  *StatsCollector
}

type MiningPool struct {
	miners          []*Miner
	blockchain      *Blockchain
	resultsChan     chan MiningResult
	stopChan        chan struct{}
	wg              sync.WaitGroup
	statsCollector  *StatsCollector
	poolHashRate    atomic.Value
}

type StatsCollector struct {
	totalBlocks     int64
	totalHashCount  int64
	startTime       time.Time
	blockTimes      []time.Duration
	minerStats      map[int]*MinerStats
	mutex           sync.RWMutex
}

type MinerStats struct {
	BlocksMined    int64
	TotalHashCount int64
	AverageHashRate float64
}

func NewBlockchain(difficulty int) *Blockchain {
	bc := &Blockchain{
		chain:      make([]*Block, 0),
		difficulty: difficulty,
	}
	
	genesis := &Block{
		Index:        0,
		Timestamp:    time.Now().Unix(),
		Data:         []byte("Genesis Block"),
		PreviousHash: "0",
		Difficulty:   difficulty,
	}
	genesis.Hash = bc.calculateHash(genesis)
	bc.chain = append(bc.chain, genesis)
	
	return bc
}

func (bc *Blockchain) calculateHash(block *Block) string {
	record := fmt.Sprintf("%d%d%s%s%d%d", 
		block.Index, block.Timestamp, block.Data, 
		block.PreviousHash, block.Nonce, block.Difficulty)
	h := sha256.New()
	h.Write([]byte(record))
	hashed := h.Sum(nil)
	return hex.EncodeToString(hashed)
}

func (bc *Blockchain) GetLatestBlock() *Block {
	bc.mutex.RLock()
	defer bc.mutex.RUnlock()
	
	if len(bc.chain) == 0 {
		return nil
	}
	return bc.chain[len(bc.chain)-1]
}

func (bc *Blockchain) AddBlock(block *Block) bool {
	bc.mutex.Lock()
	defer bc.mutex.Unlock()
	
	if len(bc.chain) == 0 {
		return false
	}
	
	latestBlock := bc.chain[len(bc.chain)-1]
	if block.PreviousHash != latestBlock.Hash {
		return false
	}
	
	if block.Index != latestBlock.Index+1 {
		return false
	}
	
	if !bc.isValidProofOfWork(block) {
		return false
	}
	
	bc.chain = append(bc.chain, block)
	return true
}

func (bc *Blockchain) isValidProofOfWork(block *Block) bool {
	hash := bc.calculateHash(block)
	target := bc.calculateTarget(block.Difficulty)
	
	hashInt := new(big.Int)
	hashInt.SetString(hash, 16)
	
	return hashInt.Cmp(target) < 0
}

func (bc *Blockchain) calculateTarget(difficulty int) *big.Int {
	target := big.NewInt(1)
	target.Lsh(target, uint(256-difficulty))
	return target
}

func (bc *Blockchain) GetChain() []*Block {
	bc.mutex.RLock()
	defer bc.mutex.RUnlock()
	
	chainCopy := make([]*Block, len(bc.chain))
	copy(chainCopy, bc.chain)
	return chainCopy
}

func (bc *Blockchain) SetDifficulty(difficulty int) {
	bc.mutex.Lock()
	defer bc.mutex.Unlock()
	bc.difficulty = difficulty
}

func (bc *Blockchain) GetDifficulty() int {
	bc.mutex.RLock()
	defer bc.mutex.RUnlock()
	return bc.difficulty
}

func NewMiner(id int, numWorkers int, blockchain *Blockchain) *Miner {
	return &Miner{
		ID:             id,
		NumWorkers:     numWorkers,
		blockchain:     blockchain,
		miningResults:  make(chan MiningResult, 10),
		stopChan:       make(chan struct{}),
		pendingTxs:     make([]Transaction, 0),
		statsCollector: NewStatsCollector(),
	}
}

func (m *Miner) Start(ctx context.Context) {
	m.startTime = time.Now()
	m.currentTarget = m.blockchain.calculateTarget(m.blockchain.GetDifficulty())
	
	for i := 0; i < m.NumWorkers; i++ {
		m.wg.Add(1)
		go m.mineWorker(ctx, i)
	}
}

func (m *Miner) Stop() {
	close(m.stopChan)
	m.wg.Wait()
}

func (m *Miner) mineWorker(ctx context.Context, workerID int) {
	defer m.wg.Done()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-m.stopChan:
			return
		default:
			m.mine(ctx, workerID)
		}
	}
}

func (m *Miner) mine(ctx context.Context, workerID int) {
	latestBlock := m.blockchain.GetLatestBlock()
	if latestBlock == nil {
		return
	}
	
	m.txMutex.RLock()
	txData := m.serializeTransactions(m.pendingTxs)
	m.txMutex.RUnlock()
	
	newBlock := &Block{
		Index:        latestBlock.Index + 1,
		Timestamp:    time.Now().Unix(),
		Data:         txData,
		PreviousHash: latestBlock.Hash,
		Difficulty:   m.blockchain.GetDifficulty(),
		MinedBy:      m.ID,
	}
	
	startNonce := int64(workerID) * math.MaxInt32
	nonce := startNonce
	startTime := time.Now()
	localHashCount := int64(0)
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-m.stopChan:
			return
		default:
			newBlock.Nonce = nonce
			hash := m.blockchain.calculateHash(newBlock)
			localHashCount++
			
			if localHashCount%10000 == 0 {
				atomic.AddInt64(&m.hashCount, 10000)
			}
			
			hashInt := new(big.Int)
			hashInt.SetString(hash, 16)
			
			if hashInt.Cmp(m.currentTarget) < 0 {
				newBlock.Hash = hash
				duration := time.Since(startTime)
				
				result := MiningResult{
					Block:    newBlock,
					MinedBy:  m.ID,
					HashRate: float64(localHashCount) / duration.Seconds(),
					Duration: duration,
				}
				
				select {
				case m.miningResults <- result:
					m.statsCollector.RecordBlock(m.ID, duration)
				case <-ctx.Done():
					return
				}
				
				return
			}
			
			nonce++
		}
	}
}

func (m *Miner) GetMiningResult() <-chan MiningResult {
	return m.miningResults
}

func (m *Miner) GetHashRate() float64 {
	elapsed := time.Since(m.startTime).Seconds()
	if elapsed == 0 {
		return 0
	}
	return float64(atomic.LoadInt64(&m.hashCount)) / elapsed
}

func (m *Miner) AddTransaction(tx Transaction) {
	m.txMutex.Lock()
	defer m.txMutex.Unlock()
	m.pendingTxs = append(m.pendingTxs, tx)
}

func (m *Miner) ClearTransactions() {
	m.txMutex.Lock()
	defer m.txMutex.Unlock()
	m.pendingTxs = make([]Transaction, 0)
}

func (m *Miner) serializeTransactions(txs []Transaction) []byte {
	if len(txs) == 0 {
		return []byte("No transactions")
	}
	
	var buffer bytes.Buffer
	for _, tx := range txs {
		buffer.WriteString(fmt.Sprintf("%s->%s:%.2f@%d;", 
			tx.From, tx.To, tx.Amount, tx.Timestamp))
	}
	return buffer.Bytes()
}

func (m *Miner) GetStats() *MinerStats {
	return m.statsCollector.GetMinerStats(m.ID)
}

func NewMiningPool(numMiners int, workersPerMiner int, blockchain *Blockchain) *MiningPool {
	pool := &MiningPool{
		miners:         make([]*Miner, numMiners),
		blockchain:     blockchain,
		resultsChan:    make(chan MiningResult, numMiners),
		stopChan:       make(chan struct{}),
		statsCollector: NewStatsCollector(),
	}
	
	for i := 0; i < numMiners; i++ {
		pool.miners[i] = NewMiner(i, workersPerMiner, blockchain)
	}
	
	pool.poolHashRate.Store(float64(0))
	
	return pool
}

func (mp *MiningPool) Start(ctx context.Context) {
	for _, miner := range mp.miners {
		miner.Start(ctx)
	}
	
	mp.wg.Add(2)
	go mp.resultCollector(ctx)
	go mp.hashRateMonitor(ctx)
}

func (mp *MiningPool) Stop() {
	close(mp.stopChan)
	
	for _, miner := range mp.miners {
		miner.Stop()
	}
	
	mp.wg.Wait()
}

func (mp *MiningPool) resultCollector(ctx context.Context) {
	defer mp.wg.Done()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-mp.stopChan:
			return
		default:
			for _, miner := range mp.miners {
				select {
				case result := <-miner.GetMiningResult():
					if mp.blockchain.AddBlock(result.Block) {
						mp.resultsChan <- result
						mp.statsCollector.RecordBlock(result.MinedBy, result.Duration)
						
						for _, m := range mp.miners {
							m.ClearTransactions()
						}
					}
				default:
				}
			}
		}
	}
}

func (mp *MiningPool) hashRateMonitor(ctx context.Context) {
	defer mp.wg.Done()
	
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-mp.stopChan:
			return
		case <-ticker.C:
			totalHashRate := float64(0)
			for _, miner := range mp.miners {
				totalHashRate += miner.GetHashRate()
			}
			mp.poolHashRate.Store(totalHashRate)
		}
	}
}

func (mp *MiningPool) GetResults() <-chan MiningResult {
	return mp.resultsChan
}

func (mp *MiningPool) GetPoolHashRate() float64 {
	return mp.poolHashRate.Load().(float64)
}

func (mp *MiningPool) AddTransaction(tx Transaction) {
	for _, miner := range mp.miners {
		miner.AddTransaction(tx)
	}
}

func (mp *MiningPool) GetPoolStats() map[string]interface{} {
	stats := make(map[string]interface{})
	stats["total_hash_rate"] = mp.GetPoolHashRate()
	stats["num_miners"] = len(mp.miners)
	stats["blockchain_height"] = len(mp.blockchain.GetChain())
	stats["current_difficulty"] = mp.blockchain.GetDifficulty()
	
	minerStats := make(map[int]*MinerStats)
	for _, miner := range mp.miners {
		minerStats[miner.ID] = miner.GetStats()
	}
	stats["miner_stats"] = minerStats
	
	return stats
}

func (mp *MiningPool) AdjustDifficulty() {
	chain := mp.blockchain.GetChain()
	if len(chain) < 10 {
		return
	}
	
	recentBlocks := chain[len(chain)-10:]
	var totalTime int64
	
	for i := 1; i < len(recentBlocks); i++ {
		totalTime += recentBlocks[i].Timestamp - recentBlocks[i-1].Timestamp
	}
	
	avgTime := float64(totalTime) / float64(len(recentBlocks)-1)
	targetTime := 10.0
	
	currentDifficulty := mp.blockchain.GetDifficulty()
	
	if avgTime < targetTime*0.5 {
		mp.blockchain.SetDifficulty(currentDifficulty + 1)
	} else if avgTime > targetTime*2 {
		if currentDifficulty > 1 {
			mp.blockchain.SetDifficulty(currentDifficulty - 1)
		}
	}
}

func NewStatsCollector() *StatsCollector {
	return &StatsCollector{
		startTime:  time.Now(),
		blockTimes: make([]time.Duration, 0),
		minerStats: make(map[int]*MinerStats),
	}
}

func (sc *StatsCollector) RecordBlock(minerID int, duration time.Duration) {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()
	
	atomic.AddInt64(&sc.totalBlocks, 1)
	sc.blockTimes = append(sc.blockTimes, duration)
	
	if _, exists := sc.minerStats[minerID]; !exists {
		sc.minerStats[minerID] = &MinerStats{}
	}
	
	sc.minerStats[minerID].BlocksMined++
}

func (sc *StatsCollector) GetMinerStats(minerID int) *MinerStats {
	sc.mutex.RLock()
	defer sc.mutex.RUnlock()
	
	if stats, exists := sc.minerStats[minerID]; exists {
		return &MinerStats{
			BlocksMined:     stats.BlocksMined,
			TotalHashCount:  stats.TotalHashCount,
			AverageHashRate: stats.AverageHashRate,
		}
	}
	
	return &MinerStats{}
}

func (sc *StatsCollector) GetGlobalStats() map[string]interface{} {
	sc.mutex.RLock()
	defer sc.mutex.RUnlock()
	
	stats := make(map[string]interface{})
	stats["total_blocks"] = atomic.LoadInt64(&sc.totalBlocks)
	stats["total_runtime"] = time.Since(sc.startTime).Seconds()
	
	if len(sc.blockTimes) > 0 {
		var totalTime time.Duration
		for _, t := range sc.blockTimes {
			totalTime += t
		}
		stats["avg_block_time"] = totalTime.Seconds() / float64(len(sc.blockTimes))
	}
	
	return stats
}

func SimulateTransactions(pool *MiningPool, numTx int) {
	for i := 0; i < numTx; i++ {
		tx := Transaction{
			From:      fmt.Sprintf("User%d", i%10),
			To:        fmt.Sprintf("User%d", (i+1)%10),
			Amount:    float64(i%100 + 1),
			Timestamp: time.Now().Unix(),
		}
		pool.AddTransaction(tx)
		time.Sleep(100 * time.Millisecond)
	}
}

func (bc *Blockchain) Validate() bool {
	bc.mutex.RLock()
	defer bc.mutex.RUnlock()
	
	for i := 1; i < len(bc.chain); i++ {
		currentBlock := bc.chain[i]
		previousBlock := bc.chain[i-1]
		
		if currentBlock.PreviousHash != previousBlock.Hash {
			return false
		}
		
		if !bc.isValidProofOfWork(currentBlock) {
			return false
		}
		
		if currentBlock.Index != previousBlock.Index+1 {
			return false
		}
	}
	
	return true
}

func (bc *Blockchain) GetBlockByIndex(index int64) *Block {
	bc.mutex.RLock()
	defer bc.mutex.RUnlock()
	
	for _, block := range bc.chain {
		if block.Index == index {
			return block
		}
	}
	
	return nil
}

func (bc *Blockchain) GetBlockByHash(hash string) *Block {
	bc.mutex.RLock()
	defer bc.mutex.RUnlock()
	
	for _, block := range bc.chain {
		if block.Hash == hash {
			return block
		}
	}
	
	return nil
}

type MerkleTree struct {
	Root *MerkleNode
}

type MerkleNode struct {
	Left  *MerkleNode
	Right *MerkleNode
	Hash  []byte
}

func NewMerkleTree(data [][]byte) *MerkleTree {
	var nodes []*MerkleNode
	
	for _, datum := range data {
		node := &MerkleNode{Hash: datum}
		nodes = append(nodes, node)
	}
	
	for len(nodes) > 1 {
		var newNodes []*MerkleNode
		
		for i := 0; i < len(nodes); i += 2 {
			if i+1 < len(nodes) {
				hash := hashPair(nodes[i].Hash, nodes[i+1].Hash)
				node := &MerkleNode{
					Left:  nodes[i],
					Right: nodes[i+1],
					Hash:  hash,
				}
				newNodes = append(newNodes, node)
			} else {
				newNodes = append(newNodes, nodes[i])
			}
		}
		
		nodes = newNodes
	}
	
	return &MerkleTree{Root: nodes[0]}
}

func hashPair(left, right []byte) []byte {
	h := sha256.New()
	h.Write(left)
	h.Write(right)
	return h.Sum(nil)
}

func NonceToBytes(nonce int64) []byte {
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.BigEndian, nonce)
	return buf.Bytes()
}