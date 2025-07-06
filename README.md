# Concurrency in Golang

This repository contains a comprehensive collection of concurrency problems implemented in Go, showcasing various concurrent programming patterns and techniques.

## Completed Problems ✅

| Problem                                                                   | Description                                                                                                                    |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| [Concurrent Merge Sort](./concurrentmergesort/)                           | A parallel implementation of the merge sort algorithm using goroutines and channels                                            |
| [Even-Odd Printer](./evenoddprinter/)                                     | A program that uses two goroutines to print even and odd numbers in sequence                                                   |
| [Multiple Producers, Multiple Consumers](./multipleproducersconsumers/)   | A classic producer-consumer problem with multiple producers and consumers                                                      |
| [Parallel File Uploader](./parallelfileuploader/)                         | A program that uploads multiple files in parallel with concurrent workers                                                      |
| [Single Producer, Multiple Consumers](./singleproducermultipleconsumers/) | A producer-consumer problem with a single producer and multiple consumers                                                      |
| [Bounded Buffer Problem](./boundedbuffer/)                                | Producer-consumer scenario with fixed-size buffer using condition variables and mutex synchronization                          |
| [Work Stealing Scheduler](./workstealingscheduler/)                       | Multi-worker task scheduler where workers can "steal" tasks from other workers' queues when idle                               |
| [Rate Limiter](./ratelimiter/)                                            | Multiple rate limiting strategies: Token Bucket, Leaky Bucket, and Sliding Window algorithms                                   |
| [Parallel QuickSort](./parallelquicksort/)                                | Parallel implementation of QuickSort with worker pool management and three-way partitioning                                    |
| [Parallel Matrix Multiplication](./parallelmatrixmultiplication/)         | Multiple parallel approaches including basic parallel, blocked multiplication, and Strassen's algorithm                        |
| [Parallel Web Crawler](./parallelwebcrawler/)                             | Concurrent web crawler with depth limiting, domain restrictions, and polite crawling features                                  |
| [Dining Philosophers Problem](./diningphilosophers/)                      | Multiple solution strategies: Ordered, Arbitrator, Limited, and Try-Lock approaches for deadlock prevention                    |
| [Readers-Writers Problem](./readerswriters/)                              | Various fairness strategies: Readers preference, Writers preference, and Fair queue-based solutions                            |
| [Sleeping Barber Problem](./sleepingbarber/)                              | Classic synchronization problem with single and multi-barber variants, including waiting room management                       |
| [Concurrent Hash Map](./concurrenthashmap/)                               | Thread-safe hash map with sharding for concurrent access and LRU variant with eviction support                                 |
| [Parallel Prim's Algorithm](./parallelprim/)                              | Parallel MST construction using Prim's algorithm with distributed version and graph partitioning                               |
| [Parallel Dijkstra's Algorithm](./paralleldijkstra/)                      | Multiple approaches: concurrent edge relaxation, delta-stepping, and bidirectional search algorithms                           |
| [Concurrent LRU Cache](./concurrentlrucache/)                             | Thread-safe LRU cache with sharding, TTL support, and size-limited variants                                                    |
| [Parallel Text Search](./paralleltextsearch/)                             | Concurrent file searching with regex support, streaming search, and indexed search variants                                    |
| [Concurrent Stock Ticker](./concurrentstockticker/)                       | Real-time stock price monitoring from multiple sources with price aggregation and portfolio tracking                           |
| [Parallel Image Processing](./parallelimageprocessing/)                   | Tile-based parallel image processing with multiple filters: Grayscale, Blur, Edge Detection, Brightness, Contrast, Rotate      |
| [Concurrent Chat Server](./concurrentchatserver/)                         | TCP-based chat server with rooms, private messaging, command system, and connection management                                 |
| [Parallel Log Processor](./parallellogprocessor/)                         | Parallel log file processing with multiple parsers (JSON, Common Log, Syslog), filters, and aggregators                        |
| [Concurrent Backup Utility](./concurrentbackuputility/)                   | Comprehensive backup solution with compression, checksums, incremental backup, extraction, and verification                    |
| [Parallel Monte Carlo Pi Estimation](./parallelmontecarlopiestimation/)   | Monte Carlo simulation with adaptive estimation, distributed processing, and convergence analysis                              |
| [Concurrent Job Scheduler](./concurrentjobscheduler/)                     | Advanced job scheduler with priority queues, retry mechanisms, cron scheduling, worker pools, and metrics                      |
| [Parallel Word Count MapReduce](./parallelwordcountmapreduce/)            | Complete MapReduce framework for word counting with configurable mappers/reducers and chunking support                         |
| [Concurrent FTP Server](./concurrentftpserver/)                           | Full-featured FTP server with authentication, passive mode, file transfers, middleware support, and concurrent client handling |
| [Parallel Video Encoder](./parallelvideoencoder/)                         | Multi-threaded video encoder with segment-based parallel processing, quality control, and real-time progress tracking          |
| [Concurrent DNS Resolver](./concurrentdnsresolver/)                       | High-performance DNS resolver with concurrent queries, caching, racing, async resolution, and multiple record types             |
| [Parallel Genetic Algorithm](./parallelgeneticalgorithm/)                 | Evolutionary optimization with parallel fitness evaluation, island model, multiple selection/crossover/mutation strategies      |
| [Concurrent Blockchain Miner](./concurrentblockchainminer/)               | Simplified blockchain with parallel proof-of-work mining, mining pools, difficulty adjustment, and statistics tracking         |
| [Parallel Ray Tracer](./parallelraytracer/)                               | Photorealistic 3D rendering with parallel processing, multiple materials, depth of field, and adaptive sampling                |
| [Concurrent Game of Life](./concurrentgameoflife/)                        | Conway's cellular automaton with parallel processing, pattern library, real-time simulation, and statistics tracking            |
| [Parallel N-Body Simulation](./parallelnbody/)                            | Gravitational dynamics simulation with parallel force calculation, multiple integrators, and collision detection               |
| [Concurrent Auction House](./concurrentauctionhouse/)                     | Thread-safe auction system with concurrent bidding, auto-bidding, real-time notifications, and payment processing     |
| [Parallel File Compressor](./parallelfilecompressor/)                     | High-performance parallel compression with multiple algorithms, chunk-based processing, and real-time progress tracking |
| [Concurrent Database Connection Pool](./concurrentdbpool/)                 | Thread-safe database connection pool with health monitoring, lifecycle management, and comprehensive metrics tracking |
| [Parallel Sudoku Solver](./parallelsudokusolver/)                         | Multi-strategy concurrent Sudoku solver with backtracking, constraint propagation, and intelligent heuristics       |
| [Concurrent API Gateway](./concurrentapigateway/)                         | High-performance API gateway with load balancing, rate limiting, circuit breaker, and advanced traffic management   |

## Pending Problems ⏳

| Problem                                                     | Description                                                                                                          |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **36. Parallel K-Means Clustering**                         | Implement the K-Means clustering algorithm in parallel                                                               |
| **37. Concurrent Load Balancer**                            | Implement a load balancer that distributes traffic to multiple servers concurrently                                  |
| **38. Parallel Fast Fourier Transform (FFT)**               | Implement the Fast Fourier Transform (FFT) algorithm in parallel                                                     |
| **39. Concurrent Spell Checker**                            | Implement a spell checker that checks a large document in parallel                                                   |
| **40. Parallel Ant Colony Optimization**                    | Implement the Ant Colony Optimization (ACO) algorithm in parallel to solve a combinatorial optimization problem      |
| **41. Concurrent Social Network Feed**                      | Implement a social network feed that can be updated and read by multiple users concurrently                          |
| **42. Parallel Particle Swarm Optimization**                | Implement the Particle Swarm Optimization (PSO) algorithm in parallel to solve an optimization problem               |
| **43. Concurrent Online Gaming Server**                     | Implement a server for an online game that can handle multiple players concurrently                                  |
| **44. Parallel Bloom Filter**                               | Implement a Bloom filter that can be safely accessed by multiple goroutines concurrently                             |
| **45. Concurrent Distributed Hash Table (DHT)**             | Implement a simplified Distributed Hash Table (DHT) where multiple nodes can join and leave the network concurrently |
| **46. Parallel Simulated Annealing**                        | Implement the Simulated Annealing (SA) algorithm in parallel to solve an optimization problem                        |
| **47. Concurrent Real-time Analytics Dashboard**            | Implement a real-time analytics dashboard that can be updated and viewed by multiple users concurrently              |
| **48. Parallel Traveling Salesperson Problem (TSP) Solver** | Implement a solver for the Traveling Salesperson Problem (TSP) in parallel                                           |
| **49. Concurrent Version Control System**                   | Implement a simplified version control system that can handle multiple users concurrently                            |
| **50. Parallel Reinforcement Learning Agent**               | Implement a reinforcement learning agent that learns in parallel                                                     |

## Implementation Highlights

### **Concurrency Patterns Used**

- **Worker Pools:** Efficient task distribution across multiple goroutines
- **Producer-Consumer:** Channel-based communication patterns
- **Fan-out/Fan-in:** Parallel processing with result aggregation
- **Pipeline:** Multi-stage concurrent processing
- **MapReduce:** Distributed computing pattern for large-scale data processing
- **Actor Model:** Message-passing concurrency (in chat server)
- **Semaphores:** Resource limiting and access control
- **Condition Variables:** Complex synchronization scenarios

### **Synchronization Primitives**

- **Channels:** Primary communication mechanism
- **Mutexes/RWMutexes:** Protecting shared state
- **Atomic Operations:** Lock-free programming
- **WaitGroups:** Goroutine coordination
- **Context:** Cancellation and timeout management
- **sync.Once:** One-time initialization
- **sync.Pool:** Object reuse and memory optimization

## Progress

**Completed:** 40/50 problems (80%)  
**Current Focus:** Problem 36 (Parallel K-Means Clustering)  
**Next Batch:** Problems 36-40 (Advanced Algorithms)
