# Parallel N-Body Simulation

A high-performance, parallel implementation of N-body gravitational simulation in Go, featuring multiple integration methods, collision detection, and comprehensive physics modeling.

## Features

### Core Physics Simulation
- **Gravitational Dynamics**: Accurate N-body gravitational force calculations
- **Multiple Integrators**: Verlet, Leapfrog, and Runge-Kutta 4th order integration
- **Collision Detection**: Sphere-sphere collision detection with multiple handling strategies
- **Energy Conservation**: Momentum and energy tracking for physics validation
- **3D Vector Mathematics**: Complete 3D vector operations for position, velocity, and acceleration
- **Configurable Parameters**: Gravitational constant, time step, softening parameter

### Parallel Processing
- **Concurrent Force Calculation**: Parallel computation of pairwise gravitational forces
- **Worker Pool Architecture**: Configurable number of workers for optimal performance
- **Load Balancing**: Efficient distribution of force calculations across workers
- **Thread-Safe Operations**: Synchronized access to shared simulation state
- **Context Support**: Graceful cancellation and timeout handling
- **Scalable Performance**: Linear scaling with CPU core count

### Advanced Features
- **Multiple Collision Handlers**: Elastic, inelastic, and merging collision responses
- **Observer Pattern**: Real-time monitoring and event notification system
- **Trail Tracking**: Particle trajectory visualization with configurable trail length
- **Statistics Collection**: Energy, momentum, center of mass, and distance metrics
- **Preset Scenarios**: Solar system, binary systems, galaxy collisions, and random configurations
- **Performance Benchmarking**: Built-in performance testing and optimization tools

## Usage Examples

### Basic N-Body Simulation

```go
package main

import (
    "context"
    "fmt"
    "time"
    
    "github.com/yourusername/concurrency-in-golang/parallelnbody"
)

func main() {
    // Configure simulation parameters
    config := parallelnbody.SystemConfig{
        NumWorkers:           8,
        TimeStep:             0.01,
        GravitationalConstant: 6.67430e-11,
        SofteningParameter:   1e6,
        BarnesHutTheta:       0.5,
        MaxTrailLength:       100,
    }
    
    // Create N-body system
    system := parallelnbody.NewNBodySystem(config)
    
    // Add observer for monitoring
    observer := parallelnbody.NewConsoleObserver(10, true, false)
    system.AddObserver(observer)
    
    // Create solar system
    bodies := parallelnbody.CreateSolarSystem()
    for _, body := range bodies {
        system.AddBody(body)
    }
    
    // Run simulation
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    fmt.Println("Starting solar system simulation...")
    err := system.Run(ctx, 1000)
    if err != nil {
        fmt.Printf("Simulation error: %v\n", err)
    }
    
    // Get final statistics
    stats := system.CalculateStatistics()
    fmt.Printf("Final total energy: %e\n", stats.TotalEnergy)
    fmt.Printf("Center of mass: (%.3f, %.3f, %.3f)\n", 
        stats.CenterOfMass.X, stats.CenterOfMass.Y, stats.CenterOfMass.Z)
}
```

### Custom Body Creation

```go
// Create custom bodies
sun := parallelnbody.NewBody(0, 1.989e30, 
    parallelnbody.Vector3D{0, 0, 0}, 
    parallelnbody.Vector3D{0, 0, 0})
sun.Fixed = true
sun.Color = [3]float64{1, 1, 0}
sun.Radius = 5

earth := parallelnbody.NewBody(1, 5.972e24,
    parallelnbody.Vector3D{1.496e11, 0, 0},
    parallelnbody.Vector3D{0, 29780, 0})
earth.Color = [3]float64{0, 0, 1}
earth.Radius = 2

// Add to system
system.AddBody(sun)
system.AddBody(earth)
```

### Binary Star System

```go
config := parallelnbody.SystemConfig{
    NumWorkers:           4,
    TimeStep:             0.001,
    GravitationalConstant: 6.67430e-11,
    SofteningParameter:   1e8,
    MaxTrailLength:       500,
}

system := parallelnbody.NewNBodySystem(config)

// Create binary system
bodies := parallelnbody.CreateBinarySystem()
for _, body := range bodies {
    system.AddBody(body)
}

// Set up collision handling
collisionHandler := &parallelnbody.ElasticCollisionHandler{}
system.SetCollisionHandler(collisionHandler)

// Run continuous simulation
ctx := context.Background()
go system.RunContinuous(ctx, 10*time.Millisecond)

// Monitor energy conservation
for i := 0; i < 100; i++ {
    time.Sleep(100 * time.Millisecond)
    stats := system.CalculateStatistics()
    fmt.Printf("Step %d: Total Energy = %e\n", i, stats.TotalEnergy)
}
```

### Galaxy Collision Simulation

```go
config := parallelnbody.SystemConfig{
    NumWorkers:           16,
    TimeStep:             0.1,
    GravitationalConstant: 6.67430e-11,
    SofteningParameter:   1e10,
    MaxTrailLength:       200,
}

system := parallelnbody.NewNBodySystem(config)

// Create galaxy collision scenario
bodies := parallelnbody.CreateGalaxyCollision()
for _, body := range bodies {
    system.AddBody(body)
}

fmt.Printf("Simulating %d bodies in galaxy collision...\n", len(bodies))

// Run simulation with progress monitoring
ctx := context.Background()
for step := 0; step < 1000; step++ {
    err := system.Step()
    if err != nil {
        fmt.Printf("Error at step %d: %v\n", step, err)
        break
    }
    
    if step%100 == 0 {
        stats := system.CalculateStatistics()
        fmt.Printf("Step %d: %d bodies, Center of mass: (%.2e, %.2e, %.2e)\n",
            step, system.GetBodyCount(),
            stats.CenterOfMass.X, stats.CenterOfMass.Y, stats.CenterOfMass.Z)
    }
}
```

### Different Integration Methods

```go
system := parallelnbody.NewNBodySystem(config)

// Try different integrators
integrators := map[string]parallelnbody.Integrator{
    "Verlet":      &parallelnbody.VerletIntegrator{},
    "Leapfrog":    &parallelnbody.LeapfrogIntegrator{},
    "Runge-Kutta": &parallelnbody.RungeKutta4Integrator{},
}

for name, integrator := range integrators {
    system.Reset()
    system.SetIntegrator(integrator)
    
    // Add test bodies
    bodies := parallelnbody.CreateBinarySystem()
    for _, body := range bodies {
        system.AddBody(body)
    }
    
    start := time.Now()
    system.Run(context.Background(), 100)
    duration := time.Since(start)
    
    stats := system.CalculateStatistics()
    fmt.Printf("%s: Duration=%v, Final Energy=%e\n", 
        name, duration, stats.TotalEnergy)
}
```

### Collision Handling Comparison

```go
handlers := map[string]parallelnbody.CollisionHandler{
    "Elastic":   &parallelnbody.ElasticCollisionHandler{},
    "Inelastic": &parallelnbody.InelasticCollisionHandler{RestitutionCoeff: 0.5},
    "Merge":     &parallelnbody.MergeCollisionHandler{},
}

for name, handler := range handlers {
    system := parallelnbody.NewNBodySystem(config)
    system.SetCollisionHandler(handler)
    
    // Create bodies on collision course
    body1 := parallelnbody.NewBody(1, 1e24, 
        parallelnbody.Vector3D{-100, 0, 0}, 
        parallelnbody.Vector3D{50, 0, 0})
    body2 := parallelnbody.NewBody(2, 1e24, 
        parallelnbody.Vector3D{100, 0, 0}, 
        parallelnbody.Vector3D{-50, 0, 0})
    
    body1.Radius = 10
    body2.Radius = 10
    
    system.AddBody(body1)
    system.AddBody(body2)
    
    fmt.Printf("\nTesting %s collision handler:\n", name)
    system.Run(context.Background(), 100)
    
    fmt.Printf("Final body count: %d\n", system.GetBodyCount())
}
```

### Performance Benchmarking

```go
// Compare performance with different worker counts
workerCounts := []int{1, 2, 4, 8, 16}
results := parallelnbody.ComparePerformance(50, 100, workerCounts, 1e12)

fmt.Println("Performance Results:")
for workers, duration := range results {
    fmt.Printf("Workers: %2d, Time: %v\n", workers, duration)
}

// Custom benchmark
benchmark := parallelnbody.NewBenchmarkRunner(100, 50, 8, 1e12)
duration := benchmark.RunBenchmark()
fmt.Printf("Custom benchmark: %v for 100 bodies, 50 steps\n", duration)
```

### Real-time Monitoring

```go
// Custom observer for detailed monitoring
type DetailedObserver struct {
    energyLog []float64
}

func (do *DetailedObserver) OnStepComplete(system *parallelnbody.NBodySystem, step int) {
    if step%10 == 0 {
        stats := system.CalculateStatistics()
        do.energyLog = append(do.energyLog, stats.TotalEnergy)
        
        fmt.Printf("Step %d: Bodies=%d, Energy=%e, Max Velocity=%.2f\n",
            step, system.GetBodyCount(), stats.TotalEnergy, stats.MaxVelocity)
    }
}

func (do *DetailedObserver) OnCollision(body1, body2 *parallelnbody.Body) {
    fmt.Printf("Collision detected between bodies %d and %d\n", body1.ID, body2.ID)
}

func (do *DetailedObserver) OnEnergyUpdate(total, kinetic, potential float64) {
    fmt.Printf("Energy update: Total=%e, Kinetic=%e, Potential=%e\n",
        total, kinetic, potential)
}

observer := &DetailedObserver{energyLog: make([]float64, 0)}
system.AddObserver(observer)
```

## Architecture

### Core Components

1. **Vector3D**: 3D vector mathematics
   - Addition, subtraction, multiplication, division
   - Magnitude, normalization, distance calculations
   - Dot product and cross product operations

2. **Body**: Individual particle representation
   - Mass, position, velocity, acceleration
   - Force accumulation and trail tracking
   - Color and radius for visualization
   - Fixed body support for static objects

3. **NBodySystem**: Main simulation controller
   - Body management and coordination
   - Integration and force calculation orchestration
   - Observer pattern implementation
   - Statistics collection and analysis

4. **Force Calculators**: Gravitational force computation
   - Direct O(N²) pairwise calculation
   - Parallel worker distribution
   - Softening parameter for numerical stability

5. **Integrators**: Numerical integration methods
   - Verlet: Symplectic, energy-conserving
   - Leapfrog: Simple, stable for orbital mechanics
   - Runge-Kutta 4: High accuracy, computationally expensive

### Parallel Processing Model

- **Force Calculation Parallelization**: Distribute pairwise force calculations
- **Worker Pool Management**: Fixed number of workers processing force pairs
- **Synchronization Barriers**: Ensure all forces calculated before integration
- **Load Balancing**: Dynamic work distribution for optimal CPU utilization

### Physics Implementation

#### Gravitational Force
```
F = G * m1 * m2 / (r² + ε²)
```
Where:
- G: Gravitational constant
- m1, m2: Masses of interacting bodies
- r: Distance between bodies
- ε: Softening parameter (prevents singularities)

#### Integration Methods
- **Verlet**: Position and velocity from acceleration
- **Leapfrog**: Velocity-position leapfrog scheme
- **RK4**: Fourth-order Runge-Kutta for high precision

## Configuration Options

### System Configuration
```go
type SystemConfig struct {
    NumWorkers           int     // Parallel workers (1-CPU cores)
    TimeStep             float64 // Integration time step (0.001-0.1)
    GravitationalConstant float64 // Physics constant (6.67430e-11)
    SofteningParameter   float64 // Numerical stability (1e6-1e10)
    BarnesHutTheta       float64 // Future Barnes-Hut parameter
    MaxTrailLength       int     // Particle trail length (10-1000)
}
```

### Performance Tuning
- **Workers**: Optimal = CPU cores
- **Time Step**: Smaller = more accurate, slower
- **Softening**: Larger = more stable, less accurate
- **Trail Length**: Longer = more memory usage

## Testing

Run the comprehensive test suite:

```bash
go test -v ./parallelnbody/
```

Run benchmarks:

```bash
go test -bench=. ./parallelnbody/
```

### Test Coverage

- Vector mathematics operations
- Body physics and kinematics
- Force calculation accuracy
- Integration method validation
- Collision detection and response
- Parallel processing correctness
- Energy and momentum conservation
- System configuration and management
- Observer pattern functionality
- Performance benchmarking

## Performance Characteristics

### Computational Complexity
- **Force Calculation**: O(N²) for N bodies
- **Integration**: O(N) per time step
- **Collision Detection**: O(N²) naive implementation
- **Memory Usage**: O(N) for body storage

### Typical Performance
- **Small Systems (10 bodies)**: 1000+ steps/second
- **Medium Systems (100 bodies)**: 100+ steps/second
- **Large Systems (1000 bodies)**: 1+ steps/second

### Scaling
- **Linear speedup** with worker count up to CPU cores
- **Memory bound** for very large systems
- **Network bound** for distributed implementations

## Preset Scenarios

### Solar System
- Sun (fixed), Earth, Mars, Venus
- Realistic masses and orbital velocities
- Demonstrates stable orbital mechanics

### Binary System
- Two equal-mass stars in circular orbit
- Perfect for energy conservation testing
- Shows center-of-mass dynamics

### Random System
- Configurable number of random bodies
- Uniform distribution in space
- Random velocities and masses

### Galaxy Collision
- Two spiral galaxies with multiple bodies
- Demonstrates large-scale dynamics
- Complex gravitational interactions

## Advanced Features

### Statistics Collection
```go
type Statistics struct {
    TotalEnergy      float64  // Total system energy
    KineticEnergy    float64  // Kinetic energy sum
    PotentialEnergy  float64  // Gravitational potential
    CenterOfMass     Vector3D // System center of mass
    TotalMomentum    Vector3D // Total momentum vector
    MaxVelocity      float64  // Maximum body velocity
    MinDistance      float64  // Closest approach distance
    MaxDistance      float64  // Farthest separation
    AverageDistance  float64  // Mean inter-body distance
}
```

### Energy Conservation Monitoring
- Track total energy over time
- Detect numerical drift
- Validate physics accuracy
- Compare integration methods

### Collision Handling Strategies
- **Elastic**: Perfect energy conservation
- **Inelastic**: Configurable restitution coefficient
- **Merge**: Combine bodies on collision

## Use Cases

1. **Astrophysics Simulation**: Solar systems, galaxy formation, stellar dynamics
2. **Molecular Dynamics**: Particle interactions, material properties
3. **Game Physics**: Realistic celestial mechanics for space games
4. **Educational Tools**: Teaching gravitational physics and numerical methods
5. **Algorithm Research**: Testing integration methods and optimization techniques
6. **Performance Testing**: Benchmarking parallel computing architectures

## Limitations

This implementation focuses on classical gravitational dynamics:

- No relativistic effects (special/general relativity)
- No electromagnetic interactions
- Simple sphere collision detection
- No adaptive time stepping
- No spatial acceleration structures (octree, Barnes-Hut)
- Limited to Newtonian gravity

## Future Enhancements

### Performance Optimizations
- **Barnes-Hut Algorithm**: O(N log N) force calculation
- **Tree Codes**: Hierarchical force approximation
- **Adaptive Time Stepping**: Variable time step for stability
- **GPU Acceleration**: CUDA/OpenCL implementation

### Physics Extensions
- **Relativistic Effects**: Special and general relativity corrections
- **Electromagnetic Forces**: Charged particle interactions
- **Tidal Forces**: Extended body deformation
- **Dark Matter**: Modified gravity models

### Visualization
- **3D Graphics**: OpenGL-based real-time visualization
- **Web Interface**: Browser-based simulation viewer
- **Animation Export**: Video generation capabilities
- **Interactive Controls**: Real-time parameter adjustment