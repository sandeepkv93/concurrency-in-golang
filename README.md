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
| [Parallel K-Means Clustering](./parallelkmeans/)                          | Multi-strategy parallel K-means with multiple initialization methods, distance metrics, and performance optimization |
| [Concurrent Load Balancer](./concurrentloadbalancer/)                     | Advanced load balancer with multiple algorithms, health monitoring, circuit breaker, and session management         |
| [Parallel Fast Fourier Transform](./parallelfft/)                         | High-performance parallel FFT with multiple algorithms, windowing, and advanced optimization techniques              |
| [Concurrent Spell Checker](./concurrentspellchecker/)                     | Advanced spell checker with phonetic matching, contextual suggestions, and parallel document processing             |
| [Parallel Ant Colony Optimization](./parallelantcolony/)                   | Multi-algorithm ACO system with parallel processing, local search, and advanced optimization strategies              |
| [Concurrent Social Network Feed](./concurrentsocialfeed/)                   | Real-time social network with concurrent feeds, notifications, caching, and advanced user interaction features      |
| [Parallel Particle Swarm Optimization](./parallelparticleswarm/)            | Multi-variant PSO with adaptive parameters, multiple topologies, and parallel evaluation for optimization problems   |
| [Concurrent Online Gaming Server](./concurrentgamingserver/)                 | High-performance multiplayer gaming server with real-time communication, anti-cheat, and comprehensive game support |
| [Parallel Bloom Filter](./parallelbloomfilter/)                               | Multi-variant parallel Bloom filter with thread-safe operations, counting support, and advanced optimization       |
| [Concurrent Distributed Hash Table](./concurrentdht/)                         | Chord-based DHT with consistent hashing, fault tolerance, replication, and scalable network communication          |
| [Parallel Simulated Annealing](./parallelsimulatedannealing/)                 | Multi-strategy parallel SA with advanced cooling schedules, perturbation methods, and optimization techniques      |
| [Concurrent Real-time Analytics Dashboard](./concurrentanalyticsadashboard/)   | Real-time analytics with concurrent event processing, WebSocket communication, and dashboard management         |

## Pending Problems ⏳

| Problem                                                     | Description                                                                                                          |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
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

**Completed:** 47/50 problems (94%)  
**Current Focus:** Problem 48 (Parallel Traveling Salesperson Problem Solver)  
**Remaining:** Problems 48-50 (Advanced System Design & AI)
