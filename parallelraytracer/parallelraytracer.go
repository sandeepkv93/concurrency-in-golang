package parallelraytracer

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

type Vec3 struct {
	X, Y, Z float64
}

func (v Vec3) Add(other Vec3) Vec3 {
	return Vec3{v.X + other.X, v.Y + other.Y, v.Z + other.Z}
}

func (v Vec3) Sub(other Vec3) Vec3 {
	return Vec3{v.X - other.X, v.Y - other.Y, v.Z - other.Z}
}

func (v Vec3) Mul(scalar float64) Vec3 {
	return Vec3{v.X * scalar, v.Y * scalar, v.Z * scalar}
}

func (v Vec3) Div(scalar float64) Vec3 {
	return Vec3{v.X / scalar, v.Y / scalar, v.Z / scalar}
}

func (v Vec3) Dot(other Vec3) float64 {
	return v.X*other.X + v.Y*other.Y + v.Z*other.Z
}

func (v Vec3) Cross(other Vec3) Vec3 {
	return Vec3{
		X: v.Y*other.Z - v.Z*other.Y,
		Y: v.Z*other.X - v.X*other.Z,
		Z: v.X*other.Y - v.Y*other.X,
	}
}

func (v Vec3) Length() float64 {
	return math.Sqrt(v.Dot(v))
}

func (v Vec3) Normalize() Vec3 {
	length := v.Length()
	if length == 0 {
		return v
	}
	return v.Div(length)
}

func (v Vec3) Reflect(normal Vec3) Vec3 {
	return v.Sub(normal.Mul(2 * v.Dot(normal)))
}

func (v Vec3) Lerp(other Vec3, t float64) Vec3 {
	return v.Mul(1 - t).Add(other.Mul(t))
}

type Ray struct {
	Origin    Vec3
	Direction Vec3
}

func (r Ray) At(t float64) Vec3 {
	return r.Origin.Add(r.Direction.Mul(t))
}

type Material interface {
	Scatter(rayIn Ray, hit HitRecord, rng *rand.Rand) (scattered Ray, attenuation Vec3, ok bool)
}

type Lambertian struct {
	Albedo Vec3
}

func (l Lambertian) Scatter(rayIn Ray, hit HitRecord, rng *rand.Rand) (Ray, Vec3, bool) {
	scatterDirection := hit.Normal.Add(randomUnitVector(rng))
	if nearZero(scatterDirection) {
		scatterDirection = hit.Normal
	}
	scattered := Ray{hit.P, scatterDirection}
	return scattered, l.Albedo, true
}

type Metal struct {
	Albedo Vec3
	Fuzz   float64
}

func (m Metal) Scatter(rayIn Ray, hit HitRecord, rng *rand.Rand) (Ray, Vec3, bool) {
	reflected := rayIn.Direction.Normalize().Reflect(hit.Normal)
	scattered := Ray{
		hit.P,
		reflected.Add(randomInUnitSphere(rng).Mul(m.Fuzz)),
	}
	return scattered, m.Albedo, scattered.Direction.Dot(hit.Normal) > 0
}

type Dielectric struct {
	RefractiveIndex float64
}

func (d Dielectric) Scatter(rayIn Ray, hit HitRecord, rng *rand.Rand) (Ray, Vec3, bool) {
	attenuation := Vec3{1, 1, 1}
	refractionRatio := d.RefractiveIndex
	if hit.FrontFace {
		refractionRatio = 1.0 / d.RefractiveIndex
	}
	
	unitDirection := rayIn.Direction.Normalize()
	cosTheta := math.Min(unitDirection.Mul(-1).Dot(hit.Normal), 1.0)
	sinTheta := math.Sqrt(1.0 - cosTheta*cosTheta)
	
	cannotRefract := refractionRatio*sinTheta > 1.0
	var direction Vec3
	
	if cannotRefract || reflectance(cosTheta, refractionRatio) > rng.Float64() {
		direction = unitDirection.Reflect(hit.Normal)
	} else {
		direction = refract(unitDirection, hit.Normal, refractionRatio)
	}
	
	scattered := Ray{hit.P, direction}
	return scattered, attenuation, true
}

func reflectance(cosine, refIdx float64) float64 {
	r0 := (1 - refIdx) / (1 + refIdx)
	r0 = r0 * r0
	return r0 + (1-r0)*math.Pow(1-cosine, 5)
}

func refract(uv, n Vec3, etaiOverEtat float64) Vec3 {
	cosTheta := math.Min(uv.Mul(-1).Dot(n), 1.0)
	rOutPerp := uv.Add(n.Mul(cosTheta)).Mul(etaiOverEtat)
	rOutParallel := n.Mul(-math.Sqrt(math.Abs(1.0 - rOutPerp.Dot(rOutPerp))))
	return rOutPerp.Add(rOutParallel)
}

type HitRecord struct {
	P         Vec3
	Normal    Vec3
	Material  Material
	T         float64
	FrontFace bool
}

func (hr *HitRecord) SetFaceNormal(r Ray, outwardNormal Vec3) {
	hr.FrontFace = r.Direction.Dot(outwardNormal) < 0
	if hr.FrontFace {
		hr.Normal = outwardNormal
	} else {
		hr.Normal = outwardNormal.Mul(-1)
	}
}

type Hittable interface {
	Hit(r Ray, tMin, tMax float64) (bool, HitRecord)
}

type Sphere struct {
	Center   Vec3
	Radius   float64
	Material Material
}

func (s Sphere) Hit(r Ray, tMin, tMax float64) (bool, HitRecord) {
	oc := r.Origin.Sub(s.Center)
	a := r.Direction.Dot(r.Direction)
	halfB := oc.Dot(r.Direction)
	c := oc.Dot(oc) - s.Radius*s.Radius
	
	discriminant := halfB*halfB - a*c
	if discriminant < 0 {
		return false, HitRecord{}
	}
	
	sqrtd := math.Sqrt(discriminant)
	root := (-halfB - sqrtd) / a
	if root < tMin || tMax < root {
		root = (-halfB + sqrtd) / a
		if root < tMin || tMax < root {
			return false, HitRecord{}
		}
	}
	
	hit := HitRecord{
		T:        root,
		P:        r.At(root),
		Material: s.Material,
	}
	outwardNormal := hit.P.Sub(s.Center).Div(s.Radius)
	hit.SetFaceNormal(r, outwardNormal)
	
	return true, hit
}

type World struct {
	Objects []Hittable
}

func (w World) Hit(r Ray, tMin, tMax float64) (bool, HitRecord) {
	var hit HitRecord
	hitAnything := false
	closestSoFar := tMax
	
	for _, object := range w.Objects {
		if didHit, tempHit := object.Hit(r, tMin, closestSoFar); didHit {
			hitAnything = true
			hit = tempHit
			closestSoFar = tempHit.T
		}
	}
	
	return hitAnything, hit
}

type Camera struct {
	Origin          Vec3
	LowerLeftCorner Vec3
	Horizontal      Vec3
	Vertical        Vec3
	U, V, W         Vec3
	LensRadius      float64
}

func NewCamera(lookFrom, lookAt, vup Vec3, vfov, aspectRatio, aperture, focusDist float64) Camera {
	theta := vfov * math.Pi / 180
	h := math.Tan(theta / 2)
	viewportHeight := 2.0 * h
	viewportWidth := aspectRatio * viewportHeight
	
	w := lookFrom.Sub(lookAt).Normalize()
	u := vup.Cross(w).Normalize()
	v := w.Cross(u)
	
	origin := lookFrom
	horizontal := u.Mul(viewportWidth * focusDist)
	vertical := v.Mul(viewportHeight * focusDist)
	lowerLeftCorner := origin.Sub(horizontal.Div(2)).Sub(vertical.Div(2)).Sub(w.Mul(focusDist))
	
	return Camera{
		Origin:          origin,
		LowerLeftCorner: lowerLeftCorner,
		Horizontal:      horizontal,
		Vertical:        vertical,
		U:               u,
		V:               v,
		W:               w,
		LensRadius:      aperture / 2,
	}
}

func (c Camera) GetRay(s, t float64, rng *rand.Rand) Ray {
	rd := randomInUnitDisk(rng).Mul(c.LensRadius)
	offset := c.U.Mul(rd.X).Add(c.V.Mul(rd.Y))
	
	return Ray{
		Origin: c.Origin.Add(offset),
		Direction: c.LowerLeftCorner.Add(c.Horizontal.Mul(s)).Add(c.Vertical.Mul(t)).
			Sub(c.Origin).Sub(offset),
	}
}

type RayTracer struct {
	Width           int
	Height          int
	SamplesPerPixel int
	MaxDepth        int
	NumWorkers      int
	World           World
	Camera          Camera
	backgroundColor Vec3
	progressChan    chan float64
	pixelsProcessed int64
}

type RenderConfig struct {
	Width           int
	Height          int
	SamplesPerPixel int
	MaxDepth        int
	NumWorkers      int
	BackgroundColor Vec3
}

func NewRayTracer(config RenderConfig, world World, camera Camera) *RayTracer {
	return &RayTracer{
		Width:           config.Width,
		Height:          config.Height,
		SamplesPerPixel: config.SamplesPerPixel,
		MaxDepth:        config.MaxDepth,
		NumWorkers:      config.NumWorkers,
		World:           world,
		Camera:          camera,
		backgroundColor: config.BackgroundColor,
		progressChan:    make(chan float64, 100),
	}
}

func (rt *RayTracer) Render(ctx context.Context) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, rt.Width, rt.Height))
	
	type job struct {
		x, y int
	}
	
	jobs := make(chan job, rt.Width*rt.Height)
	var wg sync.WaitGroup
	
	for i := 0; i < rt.NumWorkers; i++ {
		wg.Add(1)
		go rt.renderWorker(ctx, &wg, jobs, img)
	}
	
	go rt.progressReporter(ctx)
	
	for y := rt.Height - 1; y >= 0; y-- {
		for x := 0; x < rt.Width; x++ {
			select {
			case jobs <- job{x, y}:
			case <-ctx.Done():
				close(jobs)
				wg.Wait()
				return img
			}
		}
	}
	
	close(jobs)
	wg.Wait()
	close(rt.progressChan)
	
	return img
}

func (rt *RayTracer) renderWorker(ctx context.Context, wg *sync.WaitGroup, jobs <-chan job, img *image.RGBA) {
	defer wg.Done()
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	
	for {
		select {
		case job, ok := <-jobs:
			if !ok {
				return
			}
			
			pixelColor := Vec3{0, 0, 0}
			for s := 0; s < rt.SamplesPerPixel; s++ {
				u := (float64(job.x) + rng.Float64()) / float64(rt.Width-1)
				v := (float64(job.y) + rng.Float64()) / float64(rt.Height-1)
				r := rt.Camera.GetRay(u, v, rng)
				pixelColor = pixelColor.Add(rt.rayColor(r, rt.MaxDepth, rng))
			}
			
			pixelColor = pixelColor.Div(float64(rt.SamplesPerPixel))
			pixelColor = Vec3{math.Sqrt(pixelColor.X), math.Sqrt(pixelColor.Y), math.Sqrt(pixelColor.Z)}
			
			ir := uint8(256 * clamp(pixelColor.X, 0, 0.999))
			ig := uint8(256 * clamp(pixelColor.Y, 0, 0.999))
			ib := uint8(256 * clamp(pixelColor.Z, 0, 0.999))
			
			img.Set(job.x, rt.Height-1-job.y, color.RGBA{ir, ig, ib, 255})
			
			processed := atomic.AddInt64(&rt.pixelsProcessed, 1)
			if processed%1000 == 0 {
				progress := float64(processed) / float64(rt.Width*rt.Height) * 100
				select {
				case rt.progressChan <- progress:
				default:
				}
			}
			
		case <-ctx.Done():
			return
		}
	}
}

func (rt *RayTracer) rayColor(r Ray, depth int, rng *rand.Rand) Vec3 {
	if depth <= 0 {
		return Vec3{0, 0, 0}
	}
	
	if hit, rec := rt.World.Hit(r, 0.001, math.Inf(1)); hit {
		if scattered, attenuation, ok := rec.Material.Scatter(r, rec, rng); ok {
			return attenuation.Mul(rt.rayColor(scattered, depth-1, rng).X).
				Add(attenuation.Mul(rt.rayColor(scattered, depth-1, rng).Y)).
				Add(attenuation.Mul(rt.rayColor(scattered, depth-1, rng).Z))
		}
		return Vec3{0, 0, 0}
	}
	
	unitDirection := r.Direction.Normalize()
	t := 0.5 * (unitDirection.Y + 1.0)
	return Vec3{1, 1, 1}.Mul(1 - t).Add(rt.backgroundColor.Mul(t))
}

func (rt *RayTracer) progressReporter(ctx context.Context) {
	for {
		select {
		case progress, ok := <-rt.progressChan:
			if !ok {
				return
			}
			fmt.Printf("\rRendering progress: %.1f%%", progress)
		case <-ctx.Done():
			return
		}
	}
}

func (rt *RayTracer) GetProgress() float64 {
	processed := atomic.LoadInt64(&rt.pixelsProcessed)
	return float64(processed) / float64(rt.Width*rt.Height) * 100
}

type TileRenderer struct {
	Width           int
	Height          int
	TileSize        int
	SamplesPerPixel int
	MaxDepth        int
	NumWorkers      int
	World           World
	Camera          Camera
	backgroundColor Vec3
}

func NewTileRenderer(config RenderConfig, tileSize int, world World, camera Camera) *TileRenderer {
	return &TileRenderer{
		Width:           config.Width,
		Height:          config.Height,
		TileSize:        tileSize,
		SamplesPerPixel: config.SamplesPerPixel,
		MaxDepth:        config.MaxDepth,
		NumWorkers:      config.NumWorkers,
		World:           world,
		Camera:          camera,
		backgroundColor: config.BackgroundColor,
	}
}

func (tr *TileRenderer) Render(ctx context.Context) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, tr.Width, tr.Height))
	
	type tile struct {
		x0, y0, x1, y1 int
	}
	
	tiles := make(chan tile, 100)
	var wg sync.WaitGroup
	
	for i := 0; i < tr.NumWorkers; i++ {
		wg.Add(1)
		go tr.renderTileWorker(ctx, &wg, tiles, img)
	}
	
	for y := 0; y < tr.Height; y += tr.TileSize {
		for x := 0; x < tr.Width; x += tr.TileSize {
			x1 := min(x+tr.TileSize, tr.Width)
			y1 := min(y+tr.TileSize, tr.Height)
			
			select {
			case tiles <- tile{x, y, x1, y1}:
			case <-ctx.Done():
				close(tiles)
				wg.Wait()
				return img
			}
		}
	}
	
	close(tiles)
	wg.Wait()
	
	return img
}

func (tr *TileRenderer) renderTileWorker(ctx context.Context, wg *sync.WaitGroup, tiles <-chan tile, img *image.RGBA) {
	defer wg.Done()
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	
	rt := &RayTracer{
		Width:           tr.Width,
		Height:          tr.Height,
		SamplesPerPixel: tr.SamplesPerPixel,
		MaxDepth:        tr.MaxDepth,
		World:           tr.World,
		Camera:          tr.Camera,
		backgroundColor: tr.backgroundColor,
	}
	
	for {
		select {
		case tile, ok := <-tiles:
			if !ok {
				return
			}
			
			for y := tile.y0; y < tile.y1; y++ {
				for x := tile.x0; x < tile.x1; x++ {
					pixelColor := Vec3{0, 0, 0}
					for s := 0; s < tr.SamplesPerPixel; s++ {
						u := (float64(x) + rng.Float64()) / float64(tr.Width-1)
						v := (float64(tr.Height-1-y) + rng.Float64()) / float64(tr.Height-1)
						r := tr.Camera.GetRay(u, v, rng)
						pixelColor = pixelColor.Add(rt.rayColor(r, tr.MaxDepth, rng))
					}
					
					pixelColor = pixelColor.Div(float64(tr.SamplesPerPixel))
					pixelColor = Vec3{math.Sqrt(pixelColor.X), math.Sqrt(pixelColor.Y), math.Sqrt(pixelColor.Z)}
					
					ir := uint8(256 * clamp(pixelColor.X, 0, 0.999))
					ig := uint8(256 * clamp(pixelColor.Y, 0, 0.999))
					ib := uint8(256 * clamp(pixelColor.Z, 0, 0.999))
					
					img.Set(x, y, color.RGBA{ir, ig, ib, 255})
				}
			}
			
		case <-ctx.Done():
			return
		}
	}
}

func CreateDefaultScene() World {
	world := World{Objects: []Hittable{}}
	
	groundMaterial := Lambertian{Vec3{0.5, 0.5, 0.5}}
	world.Objects = append(world.Objects, Sphere{Vec3{0, -1000, 0}, 1000, groundMaterial})
	
	for a := -11; a < 11; a++ {
		for b := -11; b < 11; b++ {
			chooseMat := rand.Float64()
			center := Vec3{float64(a) + 0.9*rand.Float64(), 0.2, float64(b) + 0.9*rand.Float64()}
			
			if center.Sub(Vec3{4, 0.2, 0}).Length() > 0.9 {
				var sphereMaterial Material
				
				if chooseMat < 0.8 {
					albedo := Vec3{rand.Float64() * rand.Float64(), rand.Float64() * rand.Float64(), rand.Float64() * rand.Float64()}
					sphereMaterial = Lambertian{albedo}
				} else if chooseMat < 0.95 {
					albedo := Vec3{0.5 + 0.5*rand.Float64(), 0.5 + 0.5*rand.Float64(), 0.5 + 0.5*rand.Float64()}
					fuzz := 0.5 * rand.Float64()
					sphereMaterial = Metal{albedo, fuzz}
				} else {
					sphereMaterial = Dielectric{1.5}
				}
				
				world.Objects = append(world.Objects, Sphere{center, 0.2, sphereMaterial})
			}
		}
	}
	
	material1 := Dielectric{1.5}
	world.Objects = append(world.Objects, Sphere{Vec3{0, 1, 0}, 1.0, material1})
	
	material2 := Lambertian{Vec3{0.4, 0.2, 0.1}}
	world.Objects = append(world.Objects, Sphere{Vec3{-4, 1, 0}, 1.0, material2})
	
	material3 := Metal{Vec3{0.7, 0.6, 0.5}, 0.0}
	world.Objects = append(world.Objects, Sphere{Vec3{4, 1, 0}, 1.0, material3})
	
	return world
}

func CreateSimpleScene() World {
	world := World{Objects: []Hittable{}}
	
	groundMaterial := Lambertian{Vec3{0.8, 0.8, 0.0}}
	centerMaterial := Lambertian{Vec3{0.1, 0.2, 0.5}}
	leftMaterial := Dielectric{1.5}
	rightMaterial := Metal{Vec3{0.8, 0.6, 0.2}, 0.0}
	
	world.Objects = append(world.Objects, Sphere{Vec3{0, -100.5, -1}, 100, groundMaterial})
	world.Objects = append(world.Objects, Sphere{Vec3{0, 0, -1}, 0.5, centerMaterial})
	world.Objects = append(world.Objects, Sphere{Vec3{-1, 0, -1}, 0.5, leftMaterial})
	world.Objects = append(world.Objects, Sphere{Vec3{-1, 0, -1}, -0.45, leftMaterial})
	world.Objects = append(world.Objects, Sphere{Vec3{1, 0, -1}, 0.5, rightMaterial})
	
	return world
}

func SaveImage(img image.Image, w io.Writer) error {
	return png.Encode(w, img)
}

func randomInUnitSphere(rng *rand.Rand) Vec3 {
	for {
		p := Vec3{
			2*rng.Float64() - 1,
			2*rng.Float64() - 1,
			2*rng.Float64() - 1,
		}
		if p.Dot(p) < 1 {
			return p
		}
	}
}

func randomUnitVector(rng *rand.Rand) Vec3 {
	return randomInUnitSphere(rng).Normalize()
}

func randomInUnitDisk(rng *rand.Rand) Vec3 {
	for {
		p := Vec3{2*rng.Float64() - 1, 2*rng.Float64() - 1, 0}
		if p.Dot(p) < 1 {
			return p
		}
	}
}

func nearZero(v Vec3) bool {
	const s = 1e-8
	return math.Abs(v.X) < s && math.Abs(v.Y) < s && math.Abs(v.Z) < s
}

func clamp(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type AdaptiveRenderer struct {
	baseConfig      RenderConfig
	minSamples      int
	maxSamples      int
	varianceThresh  float64
	world           World
	camera          Camera
}

func NewAdaptiveRenderer(config RenderConfig, minSamples, maxSamples int, varianceThresh float64, world World, camera Camera) *AdaptiveRenderer {
	return &AdaptiveRenderer{
		baseConfig:     config,
		minSamples:     minSamples,
		maxSamples:     maxSamples,
		varianceThresh: varianceThresh,
		world:          world,
		camera:         camera,
	}
}

func (ar *AdaptiveRenderer) Render(ctx context.Context) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, ar.baseConfig.Width, ar.baseConfig.Height))
	
	type pixelJob struct {
		x, y int
	}
	
	jobs := make(chan pixelJob, ar.baseConfig.Width*ar.baseConfig.Height)
	var wg sync.WaitGroup
	
	for i := 0; i < ar.baseConfig.NumWorkers; i++ {
		wg.Add(1)
		go ar.adaptiveWorker(ctx, &wg, jobs, img)
	}
	
	for y := 0; y < ar.baseConfig.Height; y++ {
		for x := 0; x < ar.baseConfig.Width; x++ {
			select {
			case jobs <- pixelJob{x, y}:
			case <-ctx.Done():
				close(jobs)
				wg.Wait()
				return img
			}
		}
	}
	
	close(jobs)
	wg.Wait()
	
	return img
}

func (ar *AdaptiveRenderer) adaptiveWorker(ctx context.Context, wg *sync.WaitGroup, jobs <-chan pixelJob, img *image.RGBA) {
	defer wg.Done()
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	
	rt := &RayTracer{
		Width:           ar.baseConfig.Width,
		Height:          ar.baseConfig.Height,
		MaxDepth:        ar.baseConfig.MaxDepth,
		World:           ar.world,
		Camera:          ar.camera,
		backgroundColor: ar.baseConfig.BackgroundColor,
	}
	
	for {
		select {
		case job, ok := <-jobs:
			if !ok {
				return
			}
			
			samples := []Vec3{}
			pixelColor := Vec3{0, 0, 0}
			
			for s := 0; s < ar.minSamples; s++ {
				u := (float64(job.x) + rng.Float64()) / float64(ar.baseConfig.Width-1)
				v := (float64(ar.baseConfig.Height-1-job.y) + rng.Float64()) / float64(ar.baseConfig.Height-1)
				r := ar.camera.GetRay(u, v, rng)
				sample := rt.rayColor(r, ar.baseConfig.MaxDepth, rng)
				samples = append(samples, sample)
				pixelColor = pixelColor.Add(sample)
			}
			
			variance := ar.calculateVariance(samples)
			totalSamples := ar.minSamples
			
			for variance > ar.varianceThresh && totalSamples < ar.maxSamples {
				u := (float64(job.x) + rng.Float64()) / float64(ar.baseConfig.Width-1)
				v := (float64(ar.baseConfig.Height-1-job.y) + rng.Float64()) / float64(ar.baseConfig.Height-1)
				r := ar.camera.GetRay(u, v, rng)
				sample := rt.rayColor(r, ar.baseConfig.MaxDepth, rng)
				samples = append(samples, sample)
				pixelColor = pixelColor.Add(sample)
				totalSamples++
				
				if totalSamples%10 == 0 {
					variance = ar.calculateVariance(samples)
				}
			}
			
			pixelColor = pixelColor.Div(float64(totalSamples))
			pixelColor = Vec3{math.Sqrt(pixelColor.X), math.Sqrt(pixelColor.Y), math.Sqrt(pixelColor.Z)}
			
			ir := uint8(256 * clamp(pixelColor.X, 0, 0.999))
			ig := uint8(256 * clamp(pixelColor.Y, 0, 0.999))
			ib := uint8(256 * clamp(pixelColor.Z, 0, 0.999))
			
			img.Set(job.x, job.y, color.RGBA{ir, ig, ib, 255})
			
		case <-ctx.Done():
			return
		}
	}
}

func (ar *AdaptiveRenderer) calculateVariance(samples []Vec3) float64 {
	if len(samples) < 2 {
		return 0
	}
	
	mean := Vec3{0, 0, 0}
	for _, s := range samples {
		mean = mean.Add(s)
	}
	mean = mean.Div(float64(len(samples)))
	
	variance := 0.0
	for _, s := range samples {
		diff := s.Sub(mean)
		variance += diff.Dot(diff)
	}
	
	return variance / float64(len(samples)-1)
}