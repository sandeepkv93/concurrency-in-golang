# Concurrent Blockchain Miner

A simplified blockchain implementation with concurrent mining capabilities, demonstrating parallel proof-of-work consensus, mining pools, and distributed block validation in Go.

## Features

### Core Blockchain
- **Block Structure**: Complete block implementation with index, timestamp, data, hash, and nonce
- **Genesis Block**: Automatic genesis block creation on blockchain initialization
- **Proof of Work**: SHA-256 based proof-of-work consensus mechanism
- **Difficulty Adjustment**: Dynamic difficulty adjustment based on block mining times
- **Chain Validation**: Complete blockchain validation with hash verification
- **Transaction Support**: Transaction structure and serialization for block data

### Concurrent Mining
- **Parallel Miners**: Multiple miners working concurrently to find valid blocks
- **Worker Pools**: Each miner can utilize multiple worker goroutines
- **Mining Pool**: Coordinated mining pool with multiple miners
- **Hash Rate Monitoring**: Real-time hash rate calculation for miners and pools
- **Result Collection**: Efficient collection and validation of mining results
- **Context Support**: Graceful shutdown and cancellation handling

### Advanced Features
- **Statistics Collection**: Comprehensive mining statistics and performance metrics
- **Block Retrieval**: Find blocks by index or hash
- **Merkle Tree**: Implementation for efficient transaction verification
- **Transaction Pool**: Pending transaction management across miners
- **Difficulty Target**: Big integer arithmetic for precise difficulty calculations
- **Concurrent Safety**: Thread-safe blockchain operations with RWMutex

## Usage Examples

### Basic Blockchain Creation

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"
    
    "github.com/yourusername/concurrency-in-golang/concurrentblockchainminer"
)

func main() {
    // Create blockchain with difficulty 16
    blockchain := concurrentblockchainminer.NewBlockchain(16)
    
    // Create a single miner with 4 workers
    miner := concurrentblockchainminer.NewMiner(1, 4, blockchain)
    
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    miner.Start(ctx)
    defer miner.Stop()
    
    // Add a transaction
    tx := concurrentblockchainminer.Transaction{
        From:      "Alice",
        To:        "Bob",
        Amount:    50.0,
        Timestamp: time.Now().Unix(),
    }
    miner.AddTransaction(tx)
    
    // Wait for mining result
    result := <-miner.GetMiningResult()
    
    fmt.Printf("Block mined!\n")
    fmt.Printf("Index: %d\n", result.Block.Index)
    fmt.Printf("Hash: %s\n", result.Block.Hash)
    fmt.Printf("Nonce: %d\n", result.Block.Nonce)
    fmt.Printf("Hash Rate: %.2f H/s\n", result.HashRate)
}
```

### Mining Pool Setup

```go
// Create blockchain with higher difficulty
blockchain := concurrentblockchainminer.NewBlockchain(20)

// Create mining pool with 5 miners, each with 3 workers
pool := concurrentblockchainminer.NewMiningPool(5, 3, blockchain)

ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()

pool.Start(ctx)
defer pool.Stop()

// Add transactions to the pool
for i := 0; i < 10; i++ {
    tx := concurrentblockchainminer.Transaction{
        From:      fmt.Sprintf("User%d", i),
        To:        fmt.Sprintf("User%d", i+1),
        Amount:    float64(i * 10),
        Timestamp: time.Now().Unix(),
    }
    pool.AddTransaction(tx)
}

// Monitor mining results
go func() {
    for result := range pool.GetResults() {
        fmt.Printf("Block %d mined by miner %d\n", 
            result.Block.Index, result.MinedBy)
        fmt.Printf("Hash: %s\n", result.Block.Hash)
        fmt.Printf("Mining took: %v\n", result.Duration)
    }
}()

// Monitor pool hash rate
ticker := time.NewTicker(5 * time.Second)
defer ticker.Stop()

for {
    select {
    case <-ticker.C:
        hashRate := pool.GetPoolHashRate()
        fmt.Printf("Pool hash rate: %.2f MH/s\n", hashRate/1000000)
    case <-ctx.Done():
        return
    }
}
```

### Dynamic Difficulty Adjustment

```go
blockchain := concurrentblockchainminer.NewBlockchain(15)
pool := concurrentblockchainminer.NewMiningPool(4, 2, blockchain)

ctx := context.Background()
pool.Start(ctx)
defer pool.Stop()

// Mine blocks and adjust difficulty
blocksToMine := 20
for i := 0; i < blocksToMine; i++ {
    result := <-pool.GetResults()
    fmt.Printf("Block %d mined in %v\n", result.Block.Index, result.Duration)
    
    // Adjust difficulty every 10 blocks
    if result.Block.Index%10 == 0 {
        pool.AdjustDifficulty()
        fmt.Printf("Difficulty adjusted to: %d\n", blockchain.GetDifficulty())
    }
}
```

### Blockchain Validation

```go
blockchain := concurrentblockchainminer.NewBlockchain(12)

// Mine several blocks
// ... mining code ...

// Validate the entire blockchain
if blockchain.Validate() {
    fmt.Println("Blockchain is valid!")
} else {
    fmt.Println("Blockchain validation failed!")
}

// Retrieve specific blocks
block := blockchain.GetBlockByIndex(5)
if block != nil {
    fmt.Printf("Block 5: %s\n", block.Hash)
}

// Find block by hash
targetBlock := blockchain.GetBlockByHash("abc123...")
if targetBlock != nil {
    fmt.Printf("Found block with index: %d\n", targetBlock.Index)
}
```

### Transaction Simulation

```go
pool := concurrentblockchainminer.NewMiningPool(3, 2, blockchain)

ctx := context.Background()
pool.Start(ctx)
defer pool.Stop()

// Simulate continuous transactions
go concurrentblockchainminer.SimulateTransactions(pool, 50)

// Collect mining statistics
stats := pool.GetPoolStats()
fmt.Printf("Total hash rate: %.2f MH/s\n", 
    stats["total_hash_rate"].(float64)/1000000)
fmt.Printf("Blockchain height: %d\n", 
    stats["blockchain_height"].(int))
fmt.Printf("Current difficulty: %d\n", 
    stats["current_difficulty"].(int))

// Print individual miner statistics
minerStats := stats["miner_stats"].(map[int]*concurrentblockchainminer.MinerStats)
for minerID, stats := range minerStats {
    fmt.Printf("Miner %d: %d blocks mined\n", 
        minerID, stats.BlocksMined)
}
```

### Mining Statistics

```go
miner := concurrentblockchainminer.NewMiner(1, 4, blockchain)
statsCollector := concurrentblockchainminer.NewStatsCollector()

ctx := context.Background()
miner.Start(ctx)

// Mine some blocks
for i := 0; i < 5; i++ {
    result := <-miner.GetMiningResult()
    statsCollector.RecordBlock(result.MinedBy, result.Duration)
}

// Get global statistics
globalStats := statsCollector.GetGlobalStats()
fmt.Printf("Total blocks mined: %d\n", globalStats["total_blocks"])
fmt.Printf("Average block time: %.2f seconds\n", globalStats["avg_block_time"])

// Get miner-specific statistics
minerStats := miner.GetStats()
fmt.Printf("Miner 1 blocks: %d\n", minerStats.BlocksMined)
fmt.Printf("Average hash rate: %.2f H/s\n", miner.GetHashRate())
```

## Architecture

### Core Components

1. **Block**: Individual block in the blockchain
   - Index, Timestamp, Data, PreviousHash, Hash, Nonce
   - Difficulty level and miner identification

2. **Blockchain**: Thread-safe blockchain structure
   - Chain of blocks with validation
   - Difficulty management
   - Block retrieval methods

3. **Miner**: Individual mining entity
   - Configurable worker count
   - Transaction pool management
   - Hash rate monitoring

4. **MiningPool**: Coordinated mining operation
   - Multiple miners working together
   - Result aggregation
   - Pool-wide statistics

5. **StatsCollector**: Performance metrics collection
   - Per-miner statistics
   - Global mining statistics
   - Block time tracking

### Concurrency Model

- **Worker Goroutines**: Each miner spawns multiple workers
- **Channel Communication**: Result collection via channels
- **Context Propagation**: Cancellation support throughout
- **Atomic Operations**: Thread-safe hash counting
- **Mutex Protection**: Safe blockchain modifications

### Mining Algorithm

1. **Nonce Search**: Workers search different nonce ranges
2. **Hash Calculation**: SHA-256 hash of block data
3. **Difficulty Check**: Compare hash with target value
4. **Block Submission**: First valid block wins
5. **Chain Update**: Atomic blockchain update

## Configuration

### Blockchain Parameters
- **Difficulty**: Number of leading zeros required (1-256)
- **Target Calculation**: 2^(256-difficulty) target value
- **Block Time Target**: 10 seconds (adjustable)

### Miner Configuration
- **Worker Count**: 1-N goroutines per miner
- **Nonce Range**: Distributed across workers
- **Transaction Pool**: Shared or individual pools

### Pool Configuration
- **Miner Count**: Number of concurrent miners
- **Workers per Miner**: Goroutines per miner
- **Result Aggregation**: Centralized collection
- **Statistics Update**: Real-time monitoring

## Testing

Run the comprehensive test suite:

```bash
go test -v ./concurrentblockchainminer/
```

Run benchmarks:

```bash
go test -bench=. ./concurrentblockchainminer/
```

### Test Coverage

- Basic blockchain operations
- Block mining and validation
- Concurrent mining scenarios
- Mining pool coordination
- Transaction handling
- Difficulty adjustment
- Statistics collection
- Context cancellation
- Blockchain validation
- Performance benchmarks

## Performance Characteristics

### Hash Rates (Approximate)
- **Single Worker**: 100K-500K H/s
- **4 Workers**: 400K-2M H/s
- **Mining Pool (4x4)**: 1.6M-8M H/s

### Memory Usage
- **Per Block**: ~1KB (depending on data size)
- **Per Miner**: ~10KB overhead
- **Blockchain**: Linear with block count

### Scalability
- **Miners**: 1-100 concurrent miners
- **Workers**: 1-CPU cores per miner
- **Block Size**: Limited by memory
- **Chain Length**: Unlimited (memory bound)

## Security Considerations

1. **51% Attack**: Pool should not control majority hash power
2. **Double Spending**: Not implemented (simplified model)
3. **Fork Resolution**: Longest chain rule not implemented
4. **Network Layer**: No P2P networking (local only)
5. **Cryptographic Security**: SHA-256 proof-of-work

## Use Cases

1. **Educational**: Understanding blockchain and mining concepts
2. **Testing**: Blockchain application development
3. **Simulation**: Mining pool behavior analysis
4. **Benchmarking**: Hardware performance testing
5. **Research**: Consensus algorithm experimentation

## Limitations

This is a simplified blockchain implementation for educational purposes:

- No network layer or P2P communication
- No persistent storage
- No UTXO model or account balances
- No smart contract support
- No fork resolution mechanism
- Simplified transaction model
- No mempool prioritization

## Advanced Features

### Merkle Tree Implementation
- Efficient transaction verification
- Root hash calculation
- Tree construction from transaction data

### Statistics and Monitoring
- Real-time hash rate calculation
- Per-miner performance metrics
- Block time analysis
- Mining success rates

### Difficulty Adjustment Algorithm
- Target block time enforcement
- Automatic difficulty scaling
- Prevents mining monopoly
- Maintains consistent block times