package parallelnbody

import (
	"context"
	"math"
	"testing"
	"time"
)

func TestVector3D(t *testing.T) {
	v1 := Vector3D{1, 2, 3}
	v2 := Vector3D{4, 5, 6}
	
	result := v1.Add(v2)
	expected := Vector3D{5, 7, 9}
	if result != expected {
		t.Errorf("Add: expected %v, got %v", expected, result)
	}
	
	result = v2.Sub(v1)
	expected = Vector3D{3, 3, 3}
	if result != expected {
		t.Errorf("Sub: expected %v, got %v", expected, result)
	}
	
	result = v1.Mul(2)
	expected = Vector3D{2, 4, 6}
	if result != expected {
		t.Errorf("Mul: expected %v, got %v", expected, result)
	}
	
	result = v2.Div(2)
	expected = Vector3D{2, 2.5, 3}
	if result != expected {
		t.Errorf("Div: expected %v, got %v", expected, result)
	}
	
	magnitude := Vector3D{3, 4, 0}.Magnitude()
	if math.Abs(magnitude-5.0) > 1e-10 {
		t.Errorf("Magnitude: expected 5.0, got %f", magnitude)
	}
	
	normalized := Vector3D{3, 4, 0}.Normalize()
	expectedMag := 1.0
	if math.Abs(normalized.Magnitude()-expectedMag) > 1e-10 {
		t.Errorf("Normalize: expected magnitude 1.0, got %f", normalized.Magnitude())
	}
	
	distance := v1.Distance(v2)
	expectedDist := math.Sqrt(27)
	if math.Abs(distance-expectedDist) > 1e-10 {
		t.Errorf("Distance: expected %f, got %f", expectedDist, distance)
	}
}

func TestNewBody(t *testing.T) {
	position := Vector3D{1, 2, 3}
	velocity := Vector3D{4, 5, 6}
	mass := 10.0
	
	body := NewBody(1, mass, position, velocity)
	
	if body.ID != 1 {
		t.Errorf("Expected ID 1, got %d", body.ID)
	}
	
	if body.Mass != mass {
		t.Errorf("Expected mass %f, got %f", mass, body.Mass)
	}
	
	if body.Position != position {
		t.Errorf("Expected position %v, got %v", position, body.Position)
	}
	
	if body.Velocity != velocity {
		t.Errorf("Expected velocity %v, got %v", velocity, body.Velocity)
	}
	
	if body.Fixed {
		t.Error("Expected body to not be fixed by default")
	}
}

func TestBodyCopy(t *testing.T) {
	original := NewBody(1, 10.0, Vector3D{1, 2, 3}, Vector3D{4, 5, 6})
	original.Trail = append(original.Trail, Vector3D{0, 0, 0})
	
	copy := original.Copy()
	
	if copy.ID != original.ID {
		t.Error("Copy should have same ID")
	}
	
	if len(copy.Trail) != len(original.Trail) {
		t.Error("Copy should have same trail length")
	}
	
	copy.Position.X = 999
	if original.Position.X == 999 {
		t.Error("Modifying copy should not affect original")
	}
}

func TestBodyKineticEnergy(t *testing.T) {
	body := NewBody(1, 2.0, Vector3D{0, 0, 0}, Vector3D{3, 4, 0})
	
	expectedKE := 0.5 * 2.0 * (3*3 + 4*4)
	actualKE := body.KineticEnergy()
	
	if math.Abs(actualKE-expectedKE) > 1e-10 {
		t.Errorf("Expected kinetic energy %f, got %f", expectedKE, actualKE)
	}
}

func TestBodyUpdateTrail(t *testing.T) {
	body := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{0, 0, 0})
	
	body.UpdateTrail(3)
	body.Position = Vector3D{1, 0, 0}
	body.UpdateTrail(3)
	body.Position = Vector3D{2, 0, 0}
	body.UpdateTrail(3)
	
	if len(body.Trail) != 3 {
		t.Errorf("Expected trail length 3, got %d", len(body.Trail))
	}
	
	body.Position = Vector3D{3, 0, 0}
	body.UpdateTrail(3)
	
	if len(body.Trail) != 3 {
		t.Errorf("Expected trail length to be capped at 3, got %d", len(body.Trail))
	}
	
	if body.Trail[0] != (Vector3D{1, 0, 0}) {
		t.Error("Trail should maintain FIFO order")
	}
}

func TestNewNBodySystem(t *testing.T) {
	config := SystemConfig{
		NumWorkers:           4,
		TimeStep:             0.01,
		GravitationalConstant: 6.67430e-11,
		SofteningParameter:   1e6,
		BarnesHutTheta:       0.5,
		MaxTrailLength:       100,
	}
	
	system := NewNBodySystem(config)
	
	if system.numWorkers != 4 {
		t.Error("Expected 4 workers")
	}
	
	if system.timeStep != 0.01 {
		t.Error("Expected time step 0.01")
	}
	
	if system.GetBodyCount() != 0 {
		t.Error("Expected empty system initially")
	}
}

func TestSystemAddRemoveBody(t *testing.T) {
	config := SystemConfig{
		NumWorkers:           2,
		TimeStep:             0.01,
		GravitationalConstant: 1.0,
		SofteningParameter:   0.1,
		MaxTrailLength:       10,
	}
	
	system := NewNBodySystem(config)
	
	body1 := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{0, 0, 0})
	body2 := NewBody(2, 2.0, Vector3D{1, 0, 0}, Vector3D{0, 1, 0})
	
	system.AddBody(body1)
	system.AddBody(body2)
	
	if system.GetBodyCount() != 2 {
		t.Error("Expected 2 bodies after adding")
	}
	
	bodies := system.GetBodies()
	if len(bodies) != 2 {
		t.Error("GetBodies should return 2 bodies")
	}
	
	if !system.RemoveBody(1) {
		t.Error("Should successfully remove body with ID 1")
	}
	
	if system.GetBodyCount() != 1 {
		t.Error("Expected 1 body after removal")
	}
	
	if system.RemoveBody(999) {
		t.Error("Should not remove non-existent body")
	}
}

func TestDirectForceCalculator(t *testing.T) {
	calculator := &DirectForceCalculator{
		gravitationalConstant: 1.0,
		softeningParameter:   0.1,
	}
	
	body1 := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{0, 0, 0})
	body2 := NewBody(2, 1.0, Vector3D{1, 0, 0}, Vector3D{0, 0, 0})
	bodies := []*Body{body1, body2}
	
	err := calculator.CalculateForces(bodies, 1)
	if err != nil {
		t.Errorf("CalculateForces failed: %v", err)
	}
	
	if body1.Force.X <= 0 {
		t.Error("Body1 should have positive force in X direction")
	}
	
	if body2.Force.X >= 0 {
		t.Error("Body2 should have negative force in X direction")
	}
	
	forceMag1 := body1.Force.Magnitude()
	forceMag2 := body2.Force.Magnitude()
	if math.Abs(forceMag1-forceMag2) > 1e-10 {
		t.Error("Force magnitudes should be equal (Newton's third law)")
	}
}

func TestParallelForceCalculation(t *testing.T) {
	calculator := &DirectForceCalculator{
		gravitationalConstant: 1.0,
		softeningParameter:   0.1,
	}
	
	bodies := make([]*Body, 10)
	for i := 0; i < 10; i++ {
		bodies[i] = NewBody(i, 1.0, Vector3D{float64(i), 0, 0}, Vector3D{0, 0, 0})
	}
	
	err := calculator.CalculateForces(bodies, 4)
	if err != nil {
		t.Errorf("Parallel force calculation failed: %v", err)
	}
	
	totalForce := Vector3D{0, 0, 0}
	for _, body := range bodies {
		totalForce = totalForce.Add(body.Force)
	}
	
	if totalForce.Magnitude() > 1e-10 {
		t.Error("Total force should be approximately zero")
	}
}

func TestVerletIntegrator(t *testing.T) {
	integrator := &VerletIntegrator{}
	body := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{1, 0, 0})
	body.Force = Vector3D{1, 0, 0}
	
	initialPos := body.Position
	initialVel := body.Velocity
	
	timeStep := 0.1
	integrator.Integrate(body, timeStep)
	
	expectedPosX := initialPos.X + initialVel.X*timeStep + 0.5*1.0*timeStep*timeStep
	if math.Abs(body.Position.X-expectedPosX) > 1e-10 {
		t.Errorf("Expected position X %f, got %f", expectedPosX, body.Position.X)
	}
	
	expectedVelX := initialVel.X + 1.0*timeStep
	if math.Abs(body.Velocity.X-expectedVelX) > 1e-10 {
		t.Errorf("Expected velocity X %f, got %f", expectedVelX, body.Velocity.X)
	}
}

func TestLeapfrogIntegrator(t *testing.T) {
	integrator := &LeapfrogIntegrator{}
	body := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{1, 0, 0})
	body.Force = Vector3D{1, 0, 0}
	
	initialPos := body.Position
	initialVel := body.Velocity
	
	timeStep := 0.1
	integrator.Integrate(body, timeStep)
	
	expectedVelX := initialVel.X + 1.0*timeStep
	expectedPosX := initialPos.X + expectedVelX*timeStep
	
	if math.Abs(body.Velocity.X-expectedVelX) > 1e-10 {
		t.Errorf("Expected velocity X %f, got %f", expectedVelX, body.Velocity.X)
	}
	
	if math.Abs(body.Position.X-expectedPosX) > 1e-10 {
		t.Errorf("Expected position X %f, got %f", expectedPosX, body.Position.X)
	}
}

func TestElasticCollisionHandler(t *testing.T) {
	handler := &ElasticCollisionHandler{}
	
	body1 := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{1, 0, 0})
	body2 := NewBody(2, 1.0, Vector3D{0, 0, 0}, Vector3D{-1, 0, 0})
	
	initialMomentum := body1.Velocity.Mul(body1.Mass).Add(body2.Velocity.Mul(body2.Mass))
	
	handled := handler.HandleCollision(body1, body2)
	if !handled {
		t.Error("Collision should be handled")
	}
	
	finalMomentum := body1.Velocity.Mul(body1.Mass).Add(body2.Velocity.Mul(body2.Mass))
	
	if math.Abs(initialMomentum.X-finalMomentum.X) > 1e-10 {
		t.Error("Momentum should be conserved")
	}
	
	if body1.Velocity.X >= 0 {
		t.Error("Body1 should reverse direction")
	}
	
	if body2.Velocity.X <= 0 {
		t.Error("Body2 should reverse direction")
	}
}

func TestInelasticCollisionHandler(t *testing.T) {
	handler := &InelasticCollisionHandler{RestitutionCoeff: 0.5}
	
	body1 := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{2, 0, 0})
	body2 := NewBody(2, 1.0, Vector3D{1, 0, 0}, Vector3D{0, 0, 0})
	
	initialKE := body1.KineticEnergy() + body2.KineticEnergy()
	
	handled := handler.HandleCollision(body1, body2)
	if !handled {
		t.Error("Collision should be handled")
	}
	
	finalKE := body1.KineticEnergy() + body2.KineticEnergy()
	
	if finalKE >= initialKE {
		t.Error("Kinetic energy should decrease in inelastic collision")
	}
}

func TestSystemStep(t *testing.T) {
	config := SystemConfig{
		NumWorkers:           2,
		TimeStep:             0.01,
		GravitationalConstant: 1.0,
		SofteningParameter:   0.1,
		MaxTrailLength:       10,
	}
	
	system := NewNBodySystem(config)
	system.SetForceCalculator(&DirectForceCalculator{
		gravitationalConstant: config.GravitationalConstant,
		softeningParameter:   config.SofteningParameter,
	})
	
	body1 := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{0, 0, 0})
	body2 := NewBody(2, 1.0, Vector3D{1, 0, 0}, Vector3D{0, 0, 0})
	
	system.AddBody(body1)
	system.AddBody(body2)
	
	initialTime := system.GetCurrentTime()
	initialSteps := system.GetTotalSteps()
	
	err := system.Step()
	if err != nil {
		t.Errorf("Step failed: %v", err)
	}
	
	if system.GetCurrentTime() <= initialTime {
		t.Error("Time should advance after step")
	}
	
	if system.GetTotalSteps() != initialSteps+1 {
		t.Error("Step count should increase")
	}
}

func TestSystemRun(t *testing.T) {
	config := SystemConfig{
		NumWorkers:           2,
		TimeStep:             0.01,
		GravitationalConstant: 1.0,
		SofteningParameter:   0.1,
		MaxTrailLength:       10,
	}
	
	system := NewNBodySystem(config)
	system.SetForceCalculator(&DirectForceCalculator{
		gravitationalConstant: config.GravitationalConstant,
		softeningParameter:   config.SofteningParameter,
	})
	
	bodies := CreateBinarySystem()
	for _, body := range bodies {
		system.AddBody(body)
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	
	err := system.Run(ctx, 10)
	if err != nil {
		t.Errorf("Run failed: %v", err)
	}
	
	if system.GetTotalSteps() != 10 {
		t.Errorf("Expected 10 steps, got %d", system.GetTotalSteps())
	}
}

func TestSystemObserver(t *testing.T) {
	config := SystemConfig{
		NumWorkers:           1,
		TimeStep:             0.01,
		GravitationalConstant: 1.0,
		SofteningParameter:   0.1,
		MaxTrailLength:       10,
	}
	
	system := NewNBodySystem(config)
	
	observer := NewConsoleObserver(1, false, false)
	system.AddObserver(observer)
	
	body := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{0, 0, 0})
	system.AddBody(body)
	
	err := system.Step()
	if err != nil {
		t.Errorf("Step with observer failed: %v", err)
	}
	
	system.RemoveObserver(observer)
	
	err = system.Step()
	if err != nil {
		t.Errorf("Step after removing observer failed: %v", err)
	}
}

func TestCreateSolarSystem(t *testing.T) {
	bodies := CreateSolarSystem()
	
	if len(bodies) < 2 {
		t.Error("Solar system should have at least 2 bodies")
	}
	
	sun := bodies[0]
	if !sun.Fixed {
		t.Error("Sun should be fixed")
	}
	
	if sun.Mass <= 0 {
		t.Error("Sun should have positive mass")
	}
	
	for i := 1; i < len(bodies); i++ {
		planet := bodies[i]
		if planet.Fixed {
			t.Errorf("Planet %d should not be fixed", i)
		}
		
		if planet.Mass <= 0 {
			t.Errorf("Planet %d should have positive mass", i)
		}
	}
}

func TestCreateRandomSystem(t *testing.T) {
	numBodies := 5
	spaceSize := 1000.0
	
	bodies := CreateRandomSystem(numBodies, spaceSize)
	
	if len(bodies) != numBodies {
		t.Errorf("Expected %d bodies, got %d", numBodies, len(bodies))
	}
	
	for i, body := range bodies {
		if body.ID != i {
			t.Errorf("Body %d should have ID %d", i, i)
		}
		
		if body.Mass <= 0 {
			t.Errorf("Body %d should have positive mass", i)
		}
		
		if math.Abs(body.Position.X) > spaceSize/2 ||
			math.Abs(body.Position.Y) > spaceSize/2 ||
			math.Abs(body.Position.Z) > spaceSize/2 {
			t.Errorf("Body %d position outside expected space", i)
		}
	}
}

func TestCreateBinarySystem(t *testing.T) {
	bodies := CreateBinarySystem()
	
	if len(bodies) != 2 {
		t.Error("Binary system should have exactly 2 bodies")
	}
	
	if bodies[0].Mass != bodies[1].Mass {
		t.Error("Binary system bodies should have equal mass")
	}
	
	centerOfMass := bodies[0].Position.Mul(bodies[0].Mass).Add(bodies[1].Position.Mul(bodies[1].Mass)).Div(bodies[0].Mass + bodies[1].Mass)
	
	if centerOfMass.Magnitude() > 1e-10 {
		t.Error("Binary system center of mass should be at origin")
	}
}

func TestEnergyConservation(t *testing.T) {
	config := SystemConfig{
		NumWorkers:           2,
		TimeStep:             0.001,
		GravitationalConstant: 1.0,
		SofteningParameter:   0.1,
		MaxTrailLength:       10,
	}
	
	system := NewNBodySystem(config)
	system.SetForceCalculator(&DirectForceCalculator{
		gravitationalConstant: config.GravitationalConstant,
		softeningParameter:   config.SofteningParameter,
	})
	
	bodies := CreateBinarySystem()
	for _, body := range bodies {
		system.AddBody(body)
	}
	
	initialStats := system.CalculateStatistics()
	initialEnergy := initialStats.TotalEnergy
	
	ctx := context.Background()
	system.Run(ctx, 100)
	
	finalStats := system.CalculateStatistics()
	finalEnergy := finalStats.TotalEnergy
	
	energyChange := math.Abs(finalEnergy - initialEnergy)
	tolerance := math.Abs(initialEnergy) * 0.01
	
	if energyChange > tolerance {
		t.Errorf("Energy not conserved: initial=%f, final=%f, change=%f", 
			initialEnergy, finalEnergy, energyChange)
	}
}

func TestCalculateStatistics(t *testing.T) {
	config := SystemConfig{
		NumWorkers:           2,
		TimeStep:             0.01,
		GravitationalConstant: 1.0,
		SofteningParameter:   0.1,
		MaxTrailLength:       10,
	}
	
	system := NewNBodySystem(config)
	
	body1 := NewBody(1, 1.0, Vector3D{-1, 0, 0}, Vector3D{0, 1, 0})
	body2 := NewBody(2, 1.0, Vector3D{1, 0, 0}, Vector3D{0, -1, 0})
	
	system.AddBody(body1)
	system.AddBody(body2)
	
	stats := system.CalculateStatistics()
	
	if stats.CenterOfMass.Magnitude() > 1e-10 {
		t.Error("Center of mass should be at origin for symmetric system")
	}
	
	if stats.TotalMomentum.Magnitude() > 1e-10 {
		t.Error("Total momentum should be zero for symmetric system")
	}
	
	if stats.KineticEnergy <= 0 {
		t.Error("Kinetic energy should be positive")
	}
	
	if stats.PotentialEnergy >= 0 {
		t.Error("Gravitational potential energy should be negative")
	}
}

func TestSystemReset(t *testing.T) {
	config := SystemConfig{
		NumWorkers:           2,
		TimeStep:             0.01,
		GravitationalConstant: 1.0,
		SofteningParameter:   0.1,
		MaxTrailLength:       10,
	}
	
	system := NewNBodySystem(config)
	
	body := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{1, 0, 0})
	system.AddBody(body)
	
	ctx := context.Background()
	system.Run(ctx, 5)
	
	if system.GetCurrentTime() == 0 {
		t.Error("Time should have advanced")
	}
	
	system.Reset()
	
	if system.GetCurrentTime() != 0 {
		t.Error("Time should be reset to 0")
	}
	
	if system.GetTotalSteps() != 0 {
		t.Error("Steps should be reset to 0")
	}
	
	history := system.GetEnergyHistory()
	if len(history) != 0 {
		t.Error("Energy history should be cleared")
	}
}

func TestBenchmarkRunner(t *testing.T) {
	br := NewBenchmarkRunner(5, 10, 2, 100.0)
	
	duration := br.RunBenchmark()
	
	if duration <= 0 {
		t.Error("Benchmark duration should be positive")
	}
}

func TestComparePerformance(t *testing.T) {
	results := ComparePerformance(3, 5, []int{1, 2}, 50.0)
	
	if len(results) != 2 {
		t.Error("Should have results for 2 worker configurations")
	}
	
	for workers, duration := range results {
		if duration <= 0 {
			t.Errorf("Duration for %d workers should be positive", workers)
		}
	}
}

func TestRungeKutta4Integrator(t *testing.T) {
	integrator := &RungeKutta4Integrator{}
	body := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{0, 0, 0})
	body.Force = Vector3D{1, 0, 0}
	
	timeStep := 0.1
	integrator.Integrate(body, timeStep)
	
	if body.Position.X <= 0 {
		t.Error("Position should increase with positive force")
	}
	
	if body.Velocity.X <= 0 {
		t.Error("Velocity should increase with positive force")
	}
}

func TestMergeCollisionHandler(t *testing.T) {
	handler := &MergeCollisionHandler{}
	
	body1 := NewBody(1, 2.0, Vector3D{0, 0, 0}, Vector3D{1, 0, 0})
	body2 := NewBody(2, 3.0, Vector3D{0, 0, 0}, Vector3D{2, 0, 0})
	
	initialMomentum := body1.Velocity.Mul(body1.Mass).Add(body2.Velocity.Mul(body2.Mass))
	
	handled := handler.HandleCollision(body1, body2)
	if !handled {
		t.Error("Merge collision should be handled")
	}
	
	if body1.Mass != 5.0 {
		t.Errorf("Merged mass should be 5.0, got %f", body1.Mass)
	}
	
	finalMomentum := body1.Velocity.Mul(body1.Mass)
	
	if math.Abs(initialMomentum.X-finalMomentum.X) > 1e-10 {
		t.Error("Momentum should be conserved in merge")
	}
}

func TestCreateGalaxyCollision(t *testing.T) {
	bodies := CreateGalaxyCollision()
	
	if len(bodies) < 50 {
		t.Error("Galaxy collision should create many bodies")
	}
	
	galaxy1Count := 0
	galaxy2Count := 0
	
	for _, body := range bodies {
		if body.Position.X < 0 {
			galaxy1Count++
		} else {
			galaxy2Count++
		}
	}
	
	if galaxy1Count == 0 || galaxy2Count == 0 {
		t.Error("Should have bodies in both galaxies")
	}
}

func TestSystemWithCollisions(t *testing.T) {
	config := SystemConfig{
		NumWorkers:           2,
		TimeStep:             0.01,
		GravitationalConstant: 1.0,
		SofteningParameter:   0.1,
		MaxTrailLength:       10,
	}
	
	system := NewNBodySystem(config)
	system.SetCollisionHandler(&ElasticCollisionHandler{})
	
	body1 := NewBody(1, 1.0, Vector3D{0, 0, 0}, Vector3D{1, 0, 0})
	body2 := NewBody(2, 1.0, Vector3D{0.5, 0, 0}, Vector3D{-1, 0, 0})
	body1.Radius = 0.3
	body2.Radius = 0.3
	
	system.AddBody(body1)
	system.AddBody(body2)
	
	ctx := context.Background()
	err := system.Run(ctx, 10)
	if err != nil {
		t.Errorf("System with collisions failed: %v", err)
	}
}

func BenchmarkDirectForceCalculation(b *testing.B) {
	calculator := &DirectForceCalculator{
		gravitationalConstant: 1.0,
		softeningParameter:   0.1,
	}
	
	bodies := CreateRandomSystem(20, 1000.0)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calculator.CalculateForces(bodies, 1)
	}
}

func BenchmarkParallelForceCalculation(b *testing.B) {
	calculator := &DirectForceCalculator{
		gravitationalConstant: 1.0,
		softeningParameter:   0.1,
	}
	
	bodies := CreateRandomSystem(20, 1000.0)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calculator.CalculateForces(bodies, 4)
	}
}

func BenchmarkSystemStep(b *testing.B) {
	config := SystemConfig{
		NumWorkers:           4,
		TimeStep:             0.01,
		GravitationalConstant: 1.0,
		SofteningParameter:   0.1,
		MaxTrailLength:       10,
	}
	
	system := NewNBodySystem(config)
	system.SetForceCalculator(&DirectForceCalculator{
		gravitationalConstant: config.GravitationalConstant,
		softeningParameter:   config.SofteningParameter,
	})
	
	bodies := CreateRandomSystem(10, 1000.0)
	for _, body := range bodies {
		system.AddBody(body)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		system.Step()
	}
}