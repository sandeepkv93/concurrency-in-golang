package parallelraytracer

import (
	"bytes"
	"context"
	"image"
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestVec3Operations(t *testing.T) {
	v1 := Vec3{1, 2, 3}
	v2 := Vec3{4, 5, 6}
	
	// Test Add
	result := v1.Add(v2)
	expected := Vec3{5, 7, 9}
	if result != expected {
		t.Errorf("Add: expected %v, got %v", expected, result)
	}
	
	// Test Sub
	result = v2.Sub(v1)
	expected = Vec3{3, 3, 3}
	if result != expected {
		t.Errorf("Sub: expected %v, got %v", expected, result)
	}
	
	// Test Mul
	result = v1.Mul(2)
	expected = Vec3{2, 4, 6}
	if result != expected {
		t.Errorf("Mul: expected %v, got %v", expected, result)
	}
	
	// Test Div
	result = v2.Div(2)
	expected = Vec3{2, 2.5, 3}
	if result != expected {
		t.Errorf("Div: expected %v, got %v", expected, result)
	}
	
	// Test Dot
	dot := v1.Dot(v2)
	expectedDot := 32.0
	if dot != expectedDot {
		t.Errorf("Dot: expected %f, got %f", expectedDot, dot)
	}
	
	// Test Cross
	v3 := Vec3{1, 0, 0}
	v4 := Vec3{0, 1, 0}
	cross := v3.Cross(v4)
	expectedCross := Vec3{0, 0, 1}
	if cross != expectedCross {
		t.Errorf("Cross: expected %v, got %v", expectedCross, cross)
	}
	
	// Test Length
	v5 := Vec3{3, 4, 0}
	length := v5.Length()
	expectedLength := 5.0
	if math.Abs(length-expectedLength) > 1e-10 {
		t.Errorf("Length: expected %f, got %f", expectedLength, length)
	}
	
	// Test Normalize
	normalized := v5.Normalize()
	normalizedLength := normalized.Length()
	if math.Abs(normalizedLength-1.0) > 1e-10 {
		t.Errorf("Normalize: expected length 1, got %f", normalizedLength)
	}
}

func TestRay(t *testing.T) {
	origin := Vec3{0, 0, 0}
	direction := Vec3{1, 0, 0}
	ray := Ray{origin, direction}
	
	point := ray.At(5)
	expected := Vec3{5, 0, 0}
	if point != expected {
		t.Errorf("Ray.At: expected %v, got %v", expected, point)
	}
}

func TestSphereHit(t *testing.T) {
	sphere := Sphere{
		Center:   Vec3{0, 0, -1},
		Radius:   0.5,
		Material: Lambertian{Vec3{1, 0, 0}},
	}
	
	// Test hit
	ray := Ray{Vec3{0, 0, 0}, Vec3{0, 0, -1}}
	hit, record := sphere.Hit(ray, 0, math.Inf(1))
	if !hit {
		t.Error("Expected ray to hit sphere")
	}
	if math.Abs(record.T-0.5) > 1e-10 {
		t.Errorf("Expected t=0.5, got %f", record.T)
	}
	
	// Test miss
	ray2 := Ray{Vec3{0, 0, 0}, Vec3{1, 0, 0}}
	hit, _ = sphere.Hit(ray2, 0, math.Inf(1))
	if hit {
		t.Error("Expected ray to miss sphere")
	}
}

func TestWorld(t *testing.T) {
	world := World{
		Objects: []Hittable{
			Sphere{Vec3{0, 0, -1}, 0.5, Lambertian{Vec3{1, 0, 0}}},
			Sphere{Vec3{0, -100.5, -1}, 100, Lambertian{Vec3{0, 1, 0}}},
		},
	}
	
	ray := Ray{Vec3{0, 0, 0}, Vec3{0, 0, -1}}
	hit, record := world.Hit(ray, 0, math.Inf(1))
	
	if !hit {
		t.Error("Expected ray to hit world")
	}
	
	if math.Abs(record.T-0.5) > 1e-10 {
		t.Errorf("Expected to hit closer sphere, got t=%f", record.T)
	}
}

func TestCamera(t *testing.T) {
	lookFrom := Vec3{0, 0, 0}
	lookAt := Vec3{0, 0, -1}
	vup := Vec3{0, 1, 0}
	vfov := 90.0
	aspectRatio := 16.0 / 9.0
	aperture := 0.0
	focusDist := 1.0
	
	camera := NewCamera(lookFrom, lookAt, vup, vfov, aspectRatio, aperture, focusDist)
	
	rng := rand.New(rand.NewSource(42))
	ray := camera.GetRay(0.5, 0.5, rng)
	
	if ray.Origin != lookFrom {
		t.Errorf("Expected ray origin at camera position, got %v", ray.Origin)
	}
}

func TestRayTracerSmallImage(t *testing.T) {
	config := RenderConfig{
		Width:           10,
		Height:          10,
		SamplesPerPixel: 1,
		MaxDepth:        5,
		NumWorkers:      2,
		BackgroundColor: Vec3{0.5, 0.7, 1.0},
	}
	
	world := CreateSimpleScene()
	
	lookFrom := Vec3{3, 3, 2}
	lookAt := Vec3{0, 0, -1}
	vup := Vec3{0, 1, 0}
	camera := NewCamera(lookFrom, lookAt, vup, 20, float64(config.Width)/float64(config.Height), 0.0, lookFrom.Sub(lookAt).Length())
	
	rayTracer := NewRayTracer(config, world, camera)
	
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	img := rayTracer.Render(ctx)
	
	if img.Bounds().Dx() != 10 || img.Bounds().Dy() != 10 {
		t.Errorf("Expected 10x10 image, got %dx%d", img.Bounds().Dx(), img.Bounds().Dy())
	}
}

func TestTileRenderer(t *testing.T) {
	config := RenderConfig{
		Width:           40,
		Height:          30,
		SamplesPerPixel: 2,
		MaxDepth:        5,
		NumWorkers:      4,
		BackgroundColor: Vec3{0.5, 0.7, 1.0},
	}
	
	world := CreateSimpleScene()
	
	lookFrom := Vec3{3, 3, 2}
	lookAt := Vec3{0, 0, -1}
	vup := Vec3{0, 1, 0}
	camera := NewCamera(lookFrom, lookAt, vup, 20, float64(config.Width)/float64(config.Height), 0.0, lookFrom.Sub(lookAt).Length())
	
	tileRenderer := NewTileRenderer(config, 10, world, camera)
	
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	img := tileRenderer.Render(ctx)
	
	if img.Bounds().Dx() != 40 || img.Bounds().Dy() != 30 {
		t.Errorf("Expected 40x30 image, got %dx%d", img.Bounds().Dx(), img.Bounds().Dy())
	}
}

func TestLambertianMaterial(t *testing.T) {
	material := Lambertian{Vec3{0.5, 0.5, 0.5}}
	rayIn := Ray{Vec3{0, 0, 0}, Vec3{1, 1, 0}.Normalize()}
	hit := HitRecord{
		P:         Vec3{1, 1, 0},
		Normal:    Vec3{0, 0, 1},
		T:         math.Sqrt(2),
		FrontFace: true,
	}
	
	rng := rand.New(rand.NewSource(42))
	scattered, attenuation, ok := material.Scatter(rayIn, hit, rng)
	
	if !ok {
		t.Error("Expected Lambertian scatter to succeed")
	}
	
	if attenuation != material.Albedo {
		t.Errorf("Expected attenuation %v, got %v", material.Albedo, attenuation)
	}
	
	if scattered.Origin != hit.P {
		t.Errorf("Expected scattered ray origin at hit point, got %v", scattered.Origin)
	}
}

func TestMetalMaterial(t *testing.T) {
	material := Metal{Vec3{0.8, 0.8, 0.8}, 0.0}
	rayIn := Ray{Vec3{0, 0, 0}, Vec3{1, -1, 0}.Normalize()}
	hit := HitRecord{
		P:         Vec3{1, 0, 0},
		Normal:    Vec3{0, 1, 0},
		T:         1.0,
		FrontFace: true,
	}
	
	rng := rand.New(rand.NewSource(42))
	scattered, attenuation, ok := material.Scatter(rayIn, hit, rng)
	
	if !ok {
		t.Error("Expected Metal scatter to succeed")
	}
	
	if attenuation != material.Albedo {
		t.Errorf("Expected attenuation %v, got %v", material.Albedo, attenuation)
	}
	
	expectedDirection := Vec3{1, 1, 0}.Normalize()
	if math.Abs(scattered.Direction.X-expectedDirection.X) > 0.1 ||
		math.Abs(scattered.Direction.Y-expectedDirection.Y) > 0.1 {
		t.Errorf("Expected reflected direction near %v, got %v", expectedDirection, scattered.Direction)
	}
}

func TestDielectricMaterial(t *testing.T) {
	material := Dielectric{1.5}
	rayIn := Ray{Vec3{0, 0, 0}, Vec3{0, 0, -1}}
	hit := HitRecord{
		P:         Vec3{0, 0, -1},
		Normal:    Vec3{0, 0, 1},
		T:         1.0,
		FrontFace: true,
	}
	
	rng := rand.New(rand.NewSource(42))
	scattered, attenuation, ok := material.Scatter(rayIn, hit, rng)
	
	if !ok {
		t.Error("Expected Dielectric scatter to succeed")
	}
	
	expectedAttenuation := Vec3{1, 1, 1}
	if attenuation != expectedAttenuation {
		t.Errorf("Expected attenuation %v, got %v", expectedAttenuation, attenuation)
	}
	
	if scattered.Origin != hit.P {
		t.Errorf("Expected scattered ray origin at hit point, got %v", scattered.Origin)
	}
}

func TestProgress(t *testing.T) {
	config := RenderConfig{
		Width:           20,
		Height:          20,
		SamplesPerPixel: 1,
		MaxDepth:        3,
		NumWorkers:      2,
		BackgroundColor: Vec3{0.5, 0.7, 1.0},
	}
	
	world := CreateSimpleScene()
	camera := NewCamera(Vec3{0, 0, 0}, Vec3{0, 0, -1}, Vec3{0, 1, 0}, 90, 1.0, 0.0, 1.0)
	
	rayTracer := NewRayTracer(config, world, camera)
	
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	go rayTracer.Render(ctx)
	
	time.Sleep(100 * time.Millisecond)
	
	progress := rayTracer.GetProgress()
	if progress < 0 || progress > 100 {
		t.Errorf("Progress should be between 0 and 100, got %f", progress)
	}
}

func TestSaveImage(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			img.Set(x, y, image.Black)
		}
	}
	
	var buf bytes.Buffer
	err := SaveImage(img, &buf)
	
	if err != nil {
		t.Fatalf("Failed to save image: %v", err)
	}
	
	if buf.Len() == 0 {
		t.Error("Expected non-empty image data")
	}
}

func TestContextCancellation(t *testing.T) {
	config := RenderConfig{
		Width:           100,
		Height:          100,
		SamplesPerPixel: 10,
		MaxDepth:        10,
		NumWorkers:      4,
		BackgroundColor: Vec3{0.5, 0.7, 1.0},
	}
	
	world := CreateDefaultScene()
	camera := NewCamera(Vec3{13, 2, 3}, Vec3{0, 0, 0}, Vec3{0, 1, 0}, 20, 1.0, 0.0, 10.0)
	
	rayTracer := NewRayTracer(config, world, camera)
	
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	
	start := time.Now()
	rayTracer.Render(ctx)
	duration := time.Since(start)
	
	if duration > 200*time.Millisecond {
		t.Errorf("Render took too long after cancellation: %v", duration)
	}
}

func TestAdaptiveRenderer(t *testing.T) {
	config := RenderConfig{
		Width:           20,
		Height:          20,
		MaxDepth:        5,
		NumWorkers:      2,
		BackgroundColor: Vec3{0.5, 0.7, 1.0},
	}
	
	world := CreateSimpleScene()
	camera := NewCamera(Vec3{3, 3, 2}, Vec3{0, 0, -1}, Vec3{0, 1, 0}, 20, 1.0, 0.0, 4.0)
	
	adaptiveRenderer := NewAdaptiveRenderer(config, 2, 10, 0.1, world, camera)
	
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	img := adaptiveRenderer.Render(ctx)
	
	if img.Bounds().Dx() != 20 || img.Bounds().Dy() != 20 {
		t.Errorf("Expected 20x20 image, got %dx%d", img.Bounds().Dx(), img.Bounds().Dy())
	}
}

func TestRandomFunctions(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	
	// Test randomInUnitSphere
	for i := 0; i < 100; i++ {
		v := randomInUnitSphere(rng)
		if v.Dot(v) >= 1.0 {
			t.Error("randomInUnitSphere returned vector outside unit sphere")
		}
	}
	
	// Test randomUnitVector
	for i := 0; i < 100; i++ {
		v := randomUnitVector(rng)
		length := v.Length()
		if math.Abs(length-1.0) > 1e-10 {
			t.Errorf("randomUnitVector returned vector with length %f", length)
		}
	}
	
	// Test randomInUnitDisk
	for i := 0; i < 100; i++ {
		v := randomInUnitDisk(rng)
		if v.Z != 0 {
			t.Error("randomInUnitDisk returned non-zero Z component")
		}
		if v.X*v.X+v.Y*v.Y >= 1.0 {
			t.Error("randomInUnitDisk returned vector outside unit disk")
		}
	}
}

func TestNearZero(t *testing.T) {
	v1 := Vec3{1e-9, 1e-9, 1e-9}
	if !nearZero(v1) {
		t.Error("Expected nearZero to return true for very small vector")
	}
	
	v2 := Vec3{0.1, 0, 0}
	if nearZero(v2) {
		t.Error("Expected nearZero to return false for non-zero vector")
	}
}

func TestClamp(t *testing.T) {
	if clamp(0.5, 0, 1) != 0.5 {
		t.Error("clamp failed for value within range")
	}
	
	if clamp(-0.5, 0, 1) != 0 {
		t.Error("clamp failed for value below range")
	}
	
	if clamp(1.5, 0, 1) != 1 {
		t.Error("clamp failed for value above range")
	}
}

func TestReflectance(t *testing.T) {
	r := reflectance(1.0, 1.5)
	if r < 0 || r > 1 {
		t.Errorf("reflectance should be between 0 and 1, got %f", r)
	}
	
	r0 := reflectance(0, 1.5)
	if math.Abs(r0-0.04) > 0.01 {
		t.Errorf("Expected reflectance near 0.04 for perpendicular incidence, got %f", r0)
	}
}

func BenchmarkRayTracing(b *testing.B) {
	config := RenderConfig{
		Width:           100,
		Height:          100,
		SamplesPerPixel: 10,
		MaxDepth:        10,
		NumWorkers:      4,
		BackgroundColor: Vec3{0.5, 0.7, 1.0},
	}
	
	world := CreateSimpleScene()
	camera := NewCamera(Vec3{3, 3, 2}, Vec3{0, 0, -1}, Vec3{0, 1, 0}, 20, 1.0, 0.0, 4.0)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rayTracer := NewRayTracer(config, world, camera)
		ctx := context.Background()
		rayTracer.Render(ctx)
	}
}

func BenchmarkSphereIntersection(b *testing.B) {
	sphere := Sphere{
		Center:   Vec3{0, 0, -1},
		Radius:   0.5,
		Material: Lambertian{Vec3{1, 0, 0}},
	}
	
	ray := Ray{Vec3{0, 0, 0}, Vec3{0, 0, -1}}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sphere.Hit(ray, 0, math.Inf(1))
	}
}

func BenchmarkVectorOperations(b *testing.B) {
	v1 := Vec3{1, 2, 3}
	v2 := Vec3{4, 5, 6}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v1.Add(v2).Sub(v1).Mul(2.0).Div(3.0).Normalize()
	}
}