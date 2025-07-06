# Concurrency in Golang

This repository contains a comprehensive collection of concurrency problems implemented in Go, showcasing various concurrent programming patterns and techniques.

## Problems

| No. | Problem                                                                      | Description                                                                                                                    |
| --- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1   | [Concurrent Merge Sort](./concurrentmergesort/)                              | A parallel implementation of the merge sort algorithm using goroutines and channels                                            |
| 2   | [Even-Odd Printer](./evenoddprinter/)                                        | A program that uses two goroutines to print even and odd numbers in sequence                                                   |
| 3   | [Multiple Producers, Multiple Consumers](./multipleproducersconsumers/)      | A classic producer-consumer problem with multiple producers and consumers                                                      |
| 4   | [Parallel File Uploader](./parallelfileuploader/)                            | A program that uploads multiple files in parallel with concurrent workers                                                      |
| 5   | [Single Producer, Multiple Consumers](./singleproducermultipleconsumers/)    | A producer-consumer problem with a single producer and multiple consumers                                                      |
| 6   | [Bounded Buffer Problem](./boundedbuffer/)                                   | Producer-consumer scenario with fixed-size buffer using condition variables and mutex synchronization                          |
| 7   | [Work Stealing Scheduler](./workstealingscheduler/)                          | Multi-worker task scheduler where workers can "steal" tasks from other workers' queues when idle                               |
| 8   | [Rate Limiter](./ratelimiter/)                                               | Multiple rate limiting strategies: Token Bucket, Leaky Bucket, and Sliding Window algorithms                                   |
| 9   | [Parallel QuickSort](./parallelquicksort/)                                   | Parallel implementation of QuickSort with worker pool management and three-way partitioning                                    |
| 10  | [Parallel Matrix Multiplication](./parallelmatrixmultiplication/)            | Multiple parallel approaches including basic parallel, blocked multiplication, and Strassen's algorithm                        |
| 11  | [Parallel Web Crawler](./parallelwebcrawler/)                                | Concurrent web crawler with depth limiting, domain restrictions, and polite crawling features                                  |
| 12  | [Dining Philosophers Problem](./diningphilosophers/)                         | Multiple solution strategies: Ordered, Arbitrator, Limited, and Try-Lock approaches for deadlock prevention                    |
| 13  | [Readers-Writers Problem](./readerswriters/)                                 | Various fairness strategies: Readers preference, Writers preference, and Fair queue-based solutions                            |
| 14  | [Sleeping Barber Problem](./sleepingbarber/)                                 | Classic synchronization problem with single and multi-barber variants, including waiting room management                       |
| 15  | [Concurrent Hash Map](./concurrenthashmap/)                                  | Thread-safe hash map with sharding for concurrent access and LRU variant with eviction support                                 |
| 16  | [Parallel Prim's Algorithm](./parallelprim/)                                 | Parallel MST construction using Prim's algorithm with distributed version and graph partitioning                               |
| 17  | [Parallel Dijkstra's Algorithm](./paralleldijkstra/)                         | Multiple approaches: concurrent edge relaxation, delta-stepping, and bidirectional search algorithms                           |
| 18  | [Concurrent LRU Cache](./concurrentlrucache/)                                | Thread-safe LRU cache with sharding, TTL support, and size-limited variants                                                    |
| 19  | [Parallel Text Search](./paralleltextsearch/)                                | Concurrent file searching with regex support, streaming search, and indexed search variants                                    |
| 20  | [Concurrent Stock Ticker](./concurrentstockticker/)                          | Real-time stock price monitoring from multiple sources with price aggregation and portfolio tracking                           |
| 21  | [Parallel Image Processing](./parallelimageprocessing/)                      | Tile-based parallel image processing with multiple filters: Grayscale, Blur, Edge Detection, Brightness, Contrast, Rotate      |
| 22  | [Concurrent Chat Server](./concurrentchatserver/)                            | TCP-based chat server with rooms, private messaging, command system, and connection management                                 |
| 23  | [Parallel Log Processor](./parallellogprocessor/)                            | Parallel log file processing with multiple parsers (JSON, Common Log, Syslog), filters, and aggregators                        |
| 24  | [Concurrent Backup Utility](./concurrentbackuputility/)                      | Comprehensive backup solution with compression, checksums, incremental backup, extraction, and verification                    |
| 25  | [Parallel Monte Carlo Pi Estimation](./parallelmontecarlopiestimation/)      | Monte Carlo simulation with adaptive estimation, distributed processing, and convergence analysis                              |
| 26  | [Concurrent Job Scheduler](./concurrentjobscheduler/)                        | Advanced job scheduler with priority queues, retry mechanisms, cron scheduling, worker pools, and metrics                      |
| 27  | [Parallel Word Count MapReduce](./parallelwordcountmapreduce/)               | Complete MapReduce framework for word counting with configurable mappers/reducers and chunking support                         |
| 28  | [Concurrent FTP Server](./concurrentftpserver/)                              | Full-featured FTP server with authentication, passive mode, file transfers, middleware support, and concurrent client handling |
| 29  | [Parallel Video Encoder](./parallelvideoencoder/)                            | Multi-threaded video encoder with segment-based parallel processing, quality control, and real-time progress tracking          |
| 30  | [Concurrent DNS Resolver](./concurrentdnsresolver/)                          | High-performance DNS resolver with concurrent queries, caching, racing, async resolution, and multiple record types            |
| 31  | [Parallel Genetic Algorithm](./parallelgeneticalgorithm/)                    | Evolutionary optimization with parallel fitness evaluation, island model, multiple selection/crossover/mutation strategies     |
| 32  | [Concurrent Blockchain Miner](./concurrentblockchainminer/)                  | Simplified blockchain with parallel proof-of-work mining, mining pools, difficulty adjustment, and statistics tracking         |
| 33  | [Parallel Ray Tracer](./parallelraytracer/)                                  | Photorealistic 3D rendering with parallel processing, multiple materials, depth of field, and adaptive sampling                |
| 34  | [Concurrent Game of Life](./concurrentgameoflife/)                           | Conway's cellular automaton with parallel processing, pattern library, real-time simulation, and statistics tracking           |
| 35  | [Parallel N-Body Simulation](./parallelnbody/)                               | Gravitational dynamics simulation with parallel force calculation, multiple integrators, and collision detection               |
| 36  | [Concurrent Auction House](./concurrentauctionhouse/)                        | Thread-safe auction system with concurrent bidding, auto-bidding, real-time notifications, and payment processing              |
| 37  | [Parallel File Compressor](./parallelfilecompressor/)                        | High-performance parallel compression with multiple algorithms, chunk-based processing, and real-time progress tracking        |
| 38  | [Concurrent Database Connection Pool](./concurrentdbpool/)                   | Thread-safe database connection pool with health monitoring, lifecycle management, and comprehensive metrics tracking          |
| 39  | [Parallel Sudoku Solver](./parallelsudokusolver/)                            | Multi-strategy concurrent Sudoku solver with backtracking, constraint propagation, and intelligent heuristics                  |
| 40  | [Concurrent API Gateway](./concurrentapigateway/)                            | High-performance API gateway with load balancing, rate limiting, circuit breaker, and advanced traffic management              |
| 41  | [Parallel K-Means Clustering](./parallelkmeans/)                             | Multi-strategy parallel K-means with multiple initialization methods, distance metrics, and performance optimization           |
| 42  | [Concurrent Load Balancer](./concurrentloadbalancer/)                        | Advanced load balancer with multiple algorithms, health monitoring, circuit breaker, and session management                    |
| 43  | [Parallel Fast Fourier Transform](./parallelfft/)                            | High-performance parallel FFT with multiple algorithms, windowing, and advanced optimization techniques                        |
| 44  | [Concurrent Spell Checker](./concurrentspellchecker/)                        | Advanced spell checker with phonetic matching, contextual suggestions, and parallel document processing                        |
| 45  | [Parallel Ant Colony Optimization](./parallelantcolony/)                     | Multi-algorithm ACO system with parallel processing, local search, and advanced optimization strategies                        |
| 46  | [Concurrent Social Network Feed](./concurrentsocialfeed/)                    | Real-time social network with concurrent feeds, notifications, caching, and advanced user interaction features                 |
| 47  | [Parallel Particle Swarm Optimization](./parallelparticleswarm/)             | Multi-variant PSO with adaptive parameters, multiple topologies, and parallel evaluation for optimization problems             |
| 48  | [Concurrent Online Gaming Server](./concurrentgamingserver/)                 | High-performance multiplayer gaming server with real-time communication, anti-cheat, and comprehensive game support            |
| 49  | [Parallel Bloom Filter](./parallelbloomfilter/)                              | Multi-variant parallel Bloom filter with thread-safe operations, counting support, and advanced optimization                   |
| 50  | [Concurrent Distributed Hash Table](./concurrentdht/)                        | Chord-based DHT with consistent hashing, fault tolerance, replication, and scalable network communication                      |
| 51  | [Parallel Simulated Annealing](./parallelsimulatedannealing/)                | Multi-strategy parallel SA with advanced cooling schedules, perturbation methods, and optimization techniques                  |
| 52  | [Concurrent Real-time Analytics Dashboard](./concurrentanalyticsadashboard/) | Real-time analytics with concurrent event processing, WebSocket communication, and dashboard management                        |
| 53  | [Parallel Traveling Salesperson Problem (TSP) Solver](./paralleltsp/)        | Multi-algorithm parallel TSP solver with various optimization strategies and distance metrics                                  |
| 54  | [Concurrent Version Control System](./concurrentvcs/)                        | Git-like VCS with concurrent operations, transaction management, and distributed synchronization                               |
| 55  | [Parallel Reinforcement Learning Agent](./parallelrl/)                       | Multi-algorithm RL system with parallel training strategies, experience replay, and distributed learning                       |

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
