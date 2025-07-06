package parallelnbody

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

type Vector3D struct {
	X, Y, Z float64
}

func (v Vector3D) Add(other Vector3D) Vector3D {
	return Vector3D{v.X + other.X, v.Y + other.Y, v.Z + other.Z}
}

func (v Vector3D) Sub(other Vector3D) Vector3D {
	return Vector3D{v.X - other.X, v.Y - other.Y, v.Z - other.Z}
}

func (v Vector3D) Mul(scalar float64) Vector3D {
	return Vector3D{v.X * scalar, v.Y * scalar, v.Z * scalar}
}

func (v Vector3D) Div(scalar float64) Vector3D {
	return Vector3D{v.X / scalar, v.Y / scalar, v.Z / scalar}
}

func (v Vector3D) Magnitude() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y + v.Z*v.Z)
}

func (v Vector3D) MagnitudeSq() float64 {
	return v.X*v.X + v.Y*v.Y + v.Z*v.Z
}

func (v Vector3D) Normalize() Vector3D {
	mag := v.Magnitude()
	if mag == 0 {
		return Vector3D{0, 0, 0}
	}
	return v.Div(mag)
}

func (v Vector3D) Distance(other Vector3D) float64 {
	return v.Sub(other).Magnitude()
}

func (v Vector3D) DistanceSq(other Vector3D) float64 {
	return v.Sub(other).MagnitudeSq()
}

type Body struct {
	ID           int
	Mass         float64
	Position     Vector3D
	Velocity     Vector3D
	Acceleration Vector3D
	Force        Vector3D
	Trail        []Vector3D
	Color        [3]float64
	Radius       float64
	Fixed        bool
}

func NewBody(id int, mass float64, position, velocity Vector3D) *Body {
	return &Body{
		ID:           id,
		Mass:         mass,
		Position:     position,
		Velocity:     velocity,
		Acceleration: Vector3D{0, 0, 0},
		Force:        Vector3D{0, 0, 0},
		Trail:        make([]Vector3D, 0),
		Color:        [3]float64{1, 1, 1},
		Radius:       math.Max(1.0, math.Log(mass+1)),
		Fixed:        false,
	}
}

func (b *Body) Copy() *Body {
	trail := make([]Vector3D, len(b.Trail))
	copy(trail, b.Trail)
	
	return &Body{
		ID:           b.ID,
		Mass:         b.Mass,
		Position:     b.Position,
		Velocity:     b.Velocity,
		Acceleration: b.Acceleration,
		Force:        b.Force,
		Trail:        trail,
		Color:        b.Color,
		Radius:       b.Radius,
		Fixed:        b.Fixed,
	}
}

func (b *Body) UpdateTrail(maxTrailLength int) {
	b.Trail = append(b.Trail, b.Position)
	if len(b.Trail) > maxTrailLength {
		b.Trail = b.Trail[1:]
	}
}

func (b *Body) KineticEnergy() float64 {
	return 0.5 * b.Mass * b.Velocity.MagnitudeSq()
}

type NBodySystem struct {
	bodies           []*Body
	numWorkers       int
	timeStep         float64
	gravitationalConstant float64
	softeningParameter   float64
	currentTime          float64
	totalSteps           int
	mutex                sync.RWMutex
	observers            []SystemObserver
	energyHistory        []float64
	collisionHandler     CollisionHandler
	integrator           Integrator
	forceCalculator      ForceCalculator
	barnesHutTheta       float64
	maxTrailLength       int
}

type SystemConfig struct {
	NumWorkers           int
	TimeStep             float64
	GravitationalConstant float64
	SofteningParameter   float64
	BarnesHutTheta       float64
	MaxTrailLength       int
}

type SystemObserver interface {
	OnStepComplete(system *NBodySystem, step int)
	OnCollision(body1, body2 *Body)
	OnEnergyUpdate(totalEnergy, kineticEnergy, potentialEnergy float64)
}

type CollisionHandler interface {
	HandleCollision(body1, body2 *Body) bool
}

type Integrator interface {
	Integrate(body *Body, timeStep float64)
}

type ForceCalculator interface {
	CalculateForces(bodies []*Body, numWorkers int) error
}

func NewNBodySystem(config SystemConfig) *NBodySystem {
	return &NBodySystem{
		bodies:               make([]*Body, 0),
		numWorkers:           config.NumWorkers,
		timeStep:             config.TimeStep,
		gravitationalConstant: config.GravitationalConstant,
		softeningParameter:   config.SofteningParameter,
		currentTime:          0,
		totalSteps:           0,
		observers:            make([]SystemObserver, 0),
		energyHistory:        make([]float64, 0),
		barnesHutTheta:       config.BarnesHutTheta,
		maxTrailLength:       config.MaxTrailLength,
		integrator:           &VerletIntegrator{},
		forceCalculator:      &DirectForceCalculator{},
	}
}

func (ns *NBodySystem) AddBody(body *Body) {
	ns.mutex.Lock()
	defer ns.mutex.Unlock()
	ns.bodies = append(ns.bodies, body)
}

func (ns *NBodySystem) RemoveBody(id int) bool {
	ns.mutex.Lock()
	defer ns.mutex.Unlock()
	
	for i, body := range ns.bodies {
		if body.ID == id {
			ns.bodies = append(ns.bodies[:i], ns.bodies[i+1:]...)
			return true
		}
	}
	return false
}

func (ns *NBodySystem) GetBodies() []*Body {
	ns.mutex.RLock()
	defer ns.mutex.RUnlock()
	
	bodies := make([]*Body, len(ns.bodies))
	for i, body := range ns.bodies {
		bodies[i] = body.Copy()
	}
	return bodies
}

func (ns *NBodySystem) GetBodyCount() int {
	ns.mutex.RLock()
	defer ns.mutex.RUnlock()
	return len(ns.bodies)
}

func (ns *NBodySystem) SetForceCalculator(calculator ForceCalculator) {
	ns.forceCalculator = calculator
}

func (ns *NBodySystem) SetIntegrator(integrator Integrator) {
	ns.integrator = integrator
}

func (ns *NBodySystem) SetCollisionHandler(handler CollisionHandler) {
	ns.collisionHandler = handler
}

func (ns *NBodySystem) AddObserver(observer SystemObserver) {
	ns.mutex.Lock()
	defer ns.mutex.Unlock()
	ns.observers = append(ns.observers, observer)
}

func (ns *NBodySystem) RemoveObserver(observer SystemObserver) {
	ns.mutex.Lock()
	defer ns.mutex.Unlock()
	
	for i, obs := range ns.observers {
		if obs == observer {
			ns.observers = append(ns.observers[:i], ns.observers[i+1:]...)
			break
		}
	}
}

func (ns *NBodySystem) Step() error {
	ns.mutex.Lock()
	defer ns.mutex.Unlock()
	
	if err := ns.forceCalculator.CalculateForces(ns.bodies, ns.numWorkers); err != nil {
		return err
	}
	
	if ns.collisionHandler != nil {
		ns.handleCollisions()
	}
	
	for _, body := range ns.bodies {
		if !body.Fixed {
			ns.integrator.Integrate(body, ns.timeStep)
			body.UpdateTrail(ns.maxTrailLength)
		}
	}
	
	ns.currentTime += ns.timeStep
	ns.totalSteps++
	
	totalEnergy := ns.calculateTotalEnergy()
	ns.energyHistory = append(ns.energyHistory, totalEnergy)
	if len(ns.energyHistory) > 1000 {
		ns.energyHistory = ns.energyHistory[1:]
	}
	
	ns.notifyObservers()
	
	return nil
}

func (ns *NBodySystem) handleCollisions() {
	for i := 0; i < len(ns.bodies); i++ {
		for j := i + 1; j < len(ns.bodies); j++ {
			body1, body2 := ns.bodies[i], ns.bodies[j]
			distance := body1.Position.Distance(body2.Position)
			
			if distance <= (body1.Radius + body2.Radius) {
				if ns.collisionHandler.HandleCollision(body1, body2) {
					for _, observer := range ns.observers {
						observer.OnCollision(body1, body2)
					}
				}
			}
		}
	}
}

func (ns *NBodySystem) Run(ctx context.Context, steps int) error {
	for i := 0; i < steps; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if err := ns.Step(); err != nil {
				return err
			}
		}
	}
	return nil
}

func (ns *NBodySystem) RunContinuous(ctx context.Context, stepInterval time.Duration) error {
	ticker := time.NewTicker(stepInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := ns.Step(); err != nil {
				return err
			}
		}
	}
}

func (ns *NBodySystem) calculateTotalEnergy() float64 {
	kineticEnergy := 0.0
	potentialEnergy := 0.0
	
	for _, body := range ns.bodies {
		kineticEnergy += body.KineticEnergy()
	}
	
	for i := 0; i < len(ns.bodies); i++ {
		for j := i + 1; j < len(ns.bodies); j++ {
			body1, body2 := ns.bodies[i], ns.bodies[j]
			distance := body1.Position.Distance(body2.Position)
			if distance > 0 {
				potentialEnergy -= ns.gravitationalConstant * body1.Mass * body2.Mass / distance
			}
		}
	}
	
	totalEnergy := kineticEnergy + potentialEnergy
	
	for _, observer := range ns.observers {
		observer.OnEnergyUpdate(totalEnergy, kineticEnergy, potentialEnergy)
	}
	
	return totalEnergy
}

func (ns *NBodySystem) notifyObservers() {
	for _, observer := range ns.observers {
		observer.OnStepComplete(ns, ns.totalSteps)
	}
}

func (ns *NBodySystem) GetCurrentTime() float64 {
	ns.mutex.RLock()
	defer ns.mutex.RUnlock()
	return ns.currentTime
}

func (ns *NBodySystem) GetTotalSteps() int {
	ns.mutex.RLock()
	defer ns.mutex.RUnlock()
	return ns.totalSteps
}

func (ns *NBodySystem) GetEnergyHistory() []float64 {
	ns.mutex.RLock()
	defer ns.mutex.RUnlock()
	
	history := make([]float64, len(ns.energyHistory))
	copy(history, ns.energyHistory)
	return history
}

func (ns *NBodySystem) Reset() {
	ns.mutex.Lock()
	defer ns.mutex.Unlock()
	
	ns.currentTime = 0
	ns.totalSteps = 0
	ns.energyHistory = ns.energyHistory[:0]
	
	for _, body := range ns.bodies {
		body.Trail = body.Trail[:0]
		body.Force = Vector3D{0, 0, 0}
		body.Acceleration = Vector3D{0, 0, 0}
	}
}

type DirectForceCalculator struct {
	gravitationalConstant float64
	softeningParameter   float64
}

func (dfc *DirectForceCalculator) CalculateForces(bodies []*Body, numWorkers int) error {
	n := len(bodies)
	if n == 0 {
		return nil
	}
	
	for _, body := range bodies {
		body.Force = Vector3D{0, 0, 0}
	}
	
	if numWorkers == 1 || n < numWorkers*2 {
		return dfc.calculateForcesSequential(bodies)
	}
	
	return dfc.calculateForcesParallel(bodies, numWorkers)
}

func (dfc *DirectForceCalculator) calculateForcesSequential(bodies []*Body) error {
	n := len(bodies)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dfc.calculatePairwiseForce(bodies[i], bodies[j])
		}
	}
	return nil
}

func (dfc *DirectForceCalculator) calculateForcesParallel(bodies []*Body, numWorkers int) error {
	n := len(bodies)
	jobs := make(chan [2]int, n*(n-1)/2)
	var wg sync.WaitGroup
	
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				i, j := job[0], job[1]
				dfc.calculatePairwiseForce(bodies[i], bodies[j])
			}
		}()
	}
	
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			jobs <- [2]int{i, j}
		}
	}
	close(jobs)
	
	wg.Wait()
	return nil
}

func (dfc *DirectForceCalculator) calculatePairwiseForce(body1, body2 *Body) {
	r := body2.Position.Sub(body1.Position)
	distanceSq := r.MagnitudeSq() + dfc.softeningParameter*dfc.softeningParameter
	
	forceMagnitude := dfc.gravitationalConstant * body1.Mass * body2.Mass / distanceSq
	force := r.Normalize().Mul(forceMagnitude)
	
	body1.Force = body1.Force.Add(force)
	body2.Force = body2.Force.Sub(force)
}

type VerletIntegrator struct{}

func (vi *VerletIntegrator) Integrate(body *Body, timeStep float64) {
	body.Acceleration = body.Force.Div(body.Mass)
	
	body.Position = body.Position.Add(body.Velocity.Mul(timeStep)).Add(body.Acceleration.Mul(0.5 * timeStep * timeStep))
	body.Velocity = body.Velocity.Add(body.Acceleration.Mul(timeStep))
}

type LeapfrogIntegrator struct{}

func (li *LeapfrogIntegrator) Integrate(body *Body, timeStep float64) {
	body.Acceleration = body.Force.Div(body.Mass)
	
	body.Velocity = body.Velocity.Add(body.Acceleration.Mul(timeStep))
	body.Position = body.Position.Add(body.Velocity.Mul(timeStep))
}

type RungeKutta4Integrator struct{}

func (rk4 *RungeKutta4Integrator) Integrate(body *Body, timeStep float64) {
	acc := body.Force.Div(body.Mass)
	
	k1v := acc.Mul(timeStep)
	k1r := body.Velocity.Mul(timeStep)
	
	k2v := acc.Mul(timeStep)
	k2r := body.Velocity.Add(k1v.Mul(0.5)).Mul(timeStep)
	
	k3v := acc.Mul(timeStep)
	k3r := body.Velocity.Add(k2v.Mul(0.5)).Mul(timeStep)
	
	k4v := acc.Mul(timeStep)
	k4r := body.Velocity.Add(k3v).Mul(timeStep)
	
	body.Velocity = body.Velocity.Add(k1v.Add(k2v.Mul(2)).Add(k3v.Mul(2)).Add(k4v).Div(6))
	body.Position = body.Position.Add(k1r.Add(k2r.Mul(2)).Add(k3r.Mul(2)).Add(k4r).Div(6))
}

type ElasticCollisionHandler struct{}

func (ech *ElasticCollisionHandler) HandleCollision(body1, body2 *Body) bool {
	if body1.Fixed && body2.Fixed {
		return false
	}
	
	v1, v2 := body1.Velocity, body2.Velocity
	m1, m2 := body1.Mass, body2.Mass
	
	if body1.Fixed {
		body2.Velocity = v2.Mul(-1)
	} else if body2.Fixed {
		body1.Velocity = v1.Mul(-1)
	} else {
		newV1 := v1.Mul(m1 - m2).Add(v2.Mul(2 * m2)).Div(m1 + m2)
		newV2 := v2.Mul(m2 - m1).Add(v1.Mul(2 * m1)).Div(m1 + m2)
		
		body1.Velocity = newV1
		body2.Velocity = newV2
	}
	
	return true
}

type InelasticCollisionHandler struct {
	RestitutionCoeff float64
}

func (ich *InelasticCollisionHandler) HandleCollision(body1, body2 *Body) bool {
	if body1.Fixed && body2.Fixed {
		return false
	}
	
	relativeVelocity := body1.Velocity.Sub(body2.Velocity)
	collisionNormal := body2.Position.Sub(body1.Position).Normalize()
	
	velocityAlongNormal := relativeVelocity.X*collisionNormal.X + 
		relativeVelocity.Y*collisionNormal.Y + relativeVelocity.Z*collisionNormal.Z
	
	if velocityAlongNormal > 0 {
		return false
	}
	
	restitution := ich.RestitutionCoeff
	impulse := -(1 + restitution) * velocityAlongNormal
	
	if !body1.Fixed && !body2.Fixed {
		impulse /= (1/body1.Mass + 1/body2.Mass)
	}
	
	impulseVector := collisionNormal.Mul(impulse)
	
	if !body1.Fixed {
		body1.Velocity = body1.Velocity.Sub(impulseVector.Div(body1.Mass))
	}
	if !body2.Fixed {
		body2.Velocity = body2.Velocity.Add(impulseVector.Div(body2.Mass))
	}
	
	return true
}

type MergeCollisionHandler struct{}

func (mch *MergeCollisionHandler) HandleCollision(body1, body2 *Body) bool {
	if body1.Fixed || body2.Fixed {
		return false
	}
	
	totalMass := body1.Mass + body2.Mass
	body1.Velocity = body1.Velocity.Mul(body1.Mass).Add(body2.Velocity.Mul(body2.Mass)).Div(totalMass)
	body1.Position = body1.Position.Mul(body1.Mass).Add(body2.Position.Mul(body2.Mass)).Div(totalMass)
	body1.Mass = totalMass
	body1.Radius = math.Max(body1.Radius, body2.Radius) * 1.1
	
	return true
}

type ConsoleObserver struct {
	PrintInterval int
	ShowEnergy    bool
	ShowPositions bool
}

func NewConsoleObserver(printInterval int, showEnergy, showPositions bool) *ConsoleObserver {
	return &ConsoleObserver{
		PrintInterval: printInterval,
		ShowEnergy:    showEnergy,
		ShowPositions: showPositions,
	}
}

func (co *ConsoleObserver) OnStepComplete(system *NBodySystem, step int) {
	if step%co.PrintInterval == 0 {
		fmt.Printf("Step %d, Time: %.3f, Bodies: %d\n", 
			step, system.GetCurrentTime(), system.GetBodyCount())
		
		if co.ShowPositions {
			bodies := system.GetBodies()
			for _, body := range bodies {
				fmt.Printf("  Body %d: Mass=%.2f, Pos=(%.3f,%.3f,%.3f), Vel=(%.3f,%.3f,%.3f)\n",
					body.ID, body.Mass, 
					body.Position.X, body.Position.Y, body.Position.Z,
					body.Velocity.X, body.Velocity.Y, body.Velocity.Z)
			}
		}
	}
}

func (co *ConsoleObserver) OnCollision(body1, body2 *Body) {
	fmt.Printf("Collision between Body %d and Body %d\n", body1.ID, body2.ID)
}

func (co *ConsoleObserver) OnEnergyUpdate(totalEnergy, kineticEnergy, potentialEnergy float64) {
	if co.ShowEnergy {
		fmt.Printf("Energy - Total: %.6f, Kinetic: %.6f, Potential: %.6f\n",
			totalEnergy, kineticEnergy, potentialEnergy)
	}
}

func CreateSolarSystem() []*Body {
	bodies := make([]*Body, 0)
	
	sun := NewBody(0, 1.989e30, Vector3D{0, 0, 0}, Vector3D{0, 0, 0})
	sun.Color = [3]float64{1, 1, 0}
	sun.Radius = 5
	sun.Fixed = true
	bodies = append(bodies, sun)
	
	earth := NewBody(1, 5.972e24, Vector3D{1.496e11, 0, 0}, Vector3D{0, 29780, 0})
	earth.Color = [3]float64{0, 0, 1}
	earth.Radius = 2
	bodies = append(bodies, earth)
	
	mars := NewBody(2, 6.39e23, Vector3D{2.279e11, 0, 0}, Vector3D{0, 24070, 0})
	mars.Color = [3]float64{1, 0, 0}
	mars.Radius = 1.5
	bodies = append(bodies, mars)
	
	venus := NewBody(3, 4.867e24, Vector3D{1.082e11, 0, 0}, Vector3D{0, 35020, 0})
	venus.Color = [3]float64{1, 0.8, 0}
	venus.Radius = 1.8
	bodies = append(bodies, venus)
	
	return bodies
}

func CreateRandomSystem(numBodies int, spaceSize float64) []*Body {
	rand.Seed(time.Now().UnixNano())
	bodies := make([]*Body, numBodies)
	
	for i := 0; i < numBodies; i++ {
		mass := rand.Float64()*1e24 + 1e20
		
		position := Vector3D{
			X: (rand.Float64() - 0.5) * spaceSize,
			Y: (rand.Float64() - 0.5) * spaceSize,
			Z: (rand.Float64() - 0.5) * spaceSize,
		}
		
		velocity := Vector3D{
			X: (rand.Float64() - 0.5) * 1000,
			Y: (rand.Float64() - 0.5) * 1000,
			Z: (rand.Float64() - 0.5) * 1000,
		}
		
		body := NewBody(i, mass, position, velocity)
		body.Color = [3]float64{rand.Float64(), rand.Float64(), rand.Float64()}
		bodies[i] = body
	}
	
	return bodies
}

func CreateBinarySystem() []*Body {
	bodies := make([]*Body, 2)
	
	mass := 1e30
	separation := 1e11
	orbitalVelocity := 30000.0
	
	body1 := NewBody(0, mass, Vector3D{-separation/2, 0, 0}, Vector3D{0, -orbitalVelocity/2, 0})
	body1.Color = [3]float64{1, 0, 0}
	body1.Radius = 3
	
	body2 := NewBody(1, mass, Vector3D{separation/2, 0, 0}, Vector3D{0, orbitalVelocity/2, 0})
	body2.Color = [3]float64{0, 0, 1}
	body2.Radius = 3
	
	bodies[0] = body1
	bodies[1] = body2
	
	return bodies
}

func CreateGalaxyCollision() []*Body {
	bodies := make([]*Body, 0)
	
	for galaxy := 0; galaxy < 2; galaxy++ {
		centerX := float64(galaxy*2-1) * 2e12
		
		for i := 0; i < 50; i++ {
			angle := rand.Float64() * 2 * math.Pi
			radius := rand.Float64() * 1e12
			
			mass := rand.Float64()*1e29 + 1e28
			
			position := Vector3D{
				X: centerX + radius*math.Cos(angle),
				Y: radius * math.Sin(angle),
				Z: (rand.Float64() - 0.5) * 1e11,
			}
			
			orbitalSpeed := math.Sqrt(6.67e-11 * 1e32 / radius)
			velocity := Vector3D{
				X: -orbitalSpeed * math.Sin(angle),
				Y: orbitalSpeed * math.Cos(angle),
				Z: 0,
			}
			
			if galaxy == 1 {
				velocity.X += 1000
			} else {
				velocity.X -= 1000
			}
			
			body := NewBody(len(bodies), mass, position, velocity)
			body.Color = [3]float64{
				0.5 + 0.5*float64(galaxy),
				0.5 - 0.25*float64(galaxy),
				1 - 0.5*float64(galaxy),
			}
			bodies = append(bodies, body)
		}
	}
	
	return bodies
}

type BenchmarkRunner struct {
	NumBodies  int
	NumSteps   int
	NumWorkers int
	SpaceSize  float64
}

func NewBenchmarkRunner(numBodies, numSteps, numWorkers int, spaceSize float64) *BenchmarkRunner {
	return &BenchmarkRunner{
		NumBodies:  numBodies,
		NumSteps:   numSteps,
		NumWorkers: numWorkers,
		SpaceSize:  spaceSize,
	}
}

func (br *BenchmarkRunner) RunBenchmark() time.Duration {
	config := SystemConfig{
		NumWorkers:           br.NumWorkers,
		TimeStep:             0.01,
		GravitationalConstant: 6.67430e-11,
		SofteningParameter:   1e6,
		BarnesHutTheta:       0.5,
		MaxTrailLength:       100,
	}
	
	system := NewNBodySystem(config)
	system.SetForceCalculator(&DirectForceCalculator{
		gravitationalConstant: config.GravitationalConstant,
		softeningParameter:   config.SofteningParameter,
	})
	
	bodies := CreateRandomSystem(br.NumBodies, br.SpaceSize)
	for _, body := range bodies {
		system.AddBody(body)
	}
	
	start := time.Now()
	ctx := context.Background()
	system.Run(ctx, br.NumSteps)
	
	return time.Since(start)
}

func ComparePerformance(numBodies, numSteps int, workerCounts []int, spaceSize float64) map[int]time.Duration {
	results := make(map[int]time.Duration)
	
	for _, workers := range workerCounts {
		br := NewBenchmarkRunner(numBodies, numSteps, workers, spaceSize)
		duration := br.RunBenchmark()
		results[workers] = duration
		fmt.Printf("Bodies: %d, Workers: %d, Time: %v\n", numBodies, workers, duration)
	}
	
	return results
}

type Statistics struct {
	TotalEnergy      float64
	KineticEnergy    float64
	PotentialEnergy  float64
	CenterOfMass     Vector3D
	TotalMomentum    Vector3D
	MaxVelocity      float64
	MinDistance      float64
	MaxDistance      float64
	AverageDistance  float64
}

func (ns *NBodySystem) CalculateStatistics() Statistics {
	ns.mutex.RLock()
	defer ns.mutex.RUnlock()
	
	if len(ns.bodies) == 0 {
		return Statistics{}
	}
	
	var stats Statistics
	totalMass := 0.0
	
	for _, body := range ns.bodies {
		totalMass += body.Mass
		stats.KineticEnergy += body.KineticEnergy()
		stats.CenterOfMass = stats.CenterOfMass.Add(body.Position.Mul(body.Mass))
		stats.TotalMomentum = stats.TotalMomentum.Add(body.Velocity.Mul(body.Mass))
		
		velocity := body.Velocity.Magnitude()
		if velocity > stats.MaxVelocity {
			stats.MaxVelocity = velocity
		}
	}
	
	if totalMass > 0 {
		stats.CenterOfMass = stats.CenterOfMass.Div(totalMass)
	}
	
	minDist := math.Inf(1)
	maxDist := 0.0
	totalDist := 0.0
	pairs := 0
	
	for i := 0; i < len(ns.bodies); i++ {
		for j := i + 1; j < len(ns.bodies); j++ {
			body1, body2 := ns.bodies[i], ns.bodies[j]
			distance := body1.Position.Distance(body2.Position)
			
			if distance < minDist {
				minDist = distance
			}
			if distance > maxDist {
				maxDist = distance
			}
			
			totalDist += distance
			pairs++
			
			stats.PotentialEnergy -= ns.gravitationalConstant * body1.Mass * body2.Mass / distance
		}
	}
	
	stats.MinDistance = minDist
	stats.MaxDistance = maxDist
	if pairs > 0 {
		stats.AverageDistance = totalDist / float64(pairs)
	}
	
	stats.TotalEnergy = stats.KineticEnergy + stats.PotentialEnergy
	
	return stats
}