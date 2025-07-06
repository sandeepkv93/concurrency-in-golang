# Parallel Ray Tracer

A high-performance, parallel ray tracer implementation in Go that renders photorealistic 3D scenes using concurrent processing and advanced rendering techniques.

## Features

### Core Ray Tracing
- **Physically-Based Rendering**: Accurate light transport simulation with material properties
- **Multiple Materials**: Lambertian (diffuse), Metal (reflective), and Dielectric (refractive) materials
- **Sphere Primitives**: Efficient sphere intersection with surface normal calculation
- **Camera System**: Configurable camera with field of view, depth of field, and positioning
- **Anti-Aliasing**: Supersampling for smooth edges and reduced aliasing artifacts

### Parallel Processing
- **Worker Pool Architecture**: Configurable number of concurrent rendering workers
- **Pixel-Level Parallelism**: Independent pixel rendering for maximum scalability
- **Tile-Based Rendering**: Optional tile-based approach for memory efficiency
- **Adaptive Sampling**: Dynamic sample count based on pixel variance
- **Progress Monitoring**: Real-time rendering progress reporting
- **Context Support**: Graceful cancellation and timeout handling

### Advanced Features
- **Depth of Field**: Realistic camera blur effects with configurable aperture
- **Multiple Sampling**: Configurable samples per pixel for quality vs. speed trade-offs
- **Recursive Ray Bouncing**: Global illumination through multiple ray bounces
- **Background Gradients**: Smooth sky gradients for realistic lighting
- **Random Sampling**: Monte Carlo integration for realistic light distribution
- **PNG Export**: Standard image format output for rendered scenes

## Usage Examples

### Basic Ray Tracing

```go
package main

import (
    "context"
    "fmt"
    "os"
    "time"
    
    "github.com/yourusername/concurrency-in-golang/parallelraytracer"
)

func main() {
    // Configure render settings
    config := parallelraytracer.RenderConfig{
        Width:           800,
        Height:          600,
        SamplesPerPixel: 100,
        MaxDepth:        50,
        NumWorkers:      8,
        BackgroundColor: parallelraytracer.Vec3{0.5, 0.7, 1.0}, // Sky blue
    }
    
    // Create a simple scene
    world := parallelraytracer.CreateSimpleScene()
    
    // Set up camera
    lookFrom := parallelraytracer.Vec3{3, 3, 2}
    lookAt := parallelraytracer.Vec3{0, 0, -1}
    vup := parallelraytracer.Vec3{0, 1, 0}
    aspectRatio := float64(config.Width) / float64(config.Height)
    
    camera := parallelraytracer.NewCamera(
        lookFrom, lookAt, vup, 
        20.0,        // Field of view
        aspectRatio, // Aspect ratio
        0.1,         // Aperture (depth of field)
        lookFrom.Sub(lookAt).Length(), // Focus distance
    )
    
    // Create ray tracer
    rayTracer := parallelraytracer.NewRayTracer(config, world, camera)
    
    // Render with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()
    
    fmt.Println("Starting render...")
    img := rayTracer.Render(ctx)
    
    // Save image
    file, err := os.Create("render.png")
    if err != nil {
        panic(err)
    }
    defer file.Close()
    
    err = parallelraytracer.SaveImage(img, file)
    if err != nil {
        panic(err)
    }
    
    fmt.Println("Render complete! Saved to render.png")
}
```

### Complex Scene Rendering

```go
// Create a complex scene with multiple materials
world := parallelraytracer.CreateDefaultScene()

// Set up camera for artistic composition
lookFrom := parallelraytracer.Vec3{13, 2, 3}
lookAt := parallelraytracer.Vec3{0, 0, 0}
vup := parallelraytracer.Vec3{0, 1, 0}

camera := parallelraytracer.NewCamera(
    lookFrom, lookAt, vup,
    20.0,  // Wide field of view
    16.0/9.0, // Widescreen aspect ratio
    0.1,   // Small aperture for depth of field
    10.0,  // Focus distance
)

config := parallelraytracer.RenderConfig{
    Width:           1920,
    Height:          1080,
    SamplesPerPixel: 500,  // High quality
    MaxDepth:        50,   // Deep recursion for glass
    NumWorkers:      16,   // Use all CPU cores
    BackgroundColor: parallelraytracer.Vec3{0.7, 0.8, 1.0},
}

rayTracer := parallelraytracer.NewRayTracer(config, world, camera)

ctx := context.Background()
img := rayTracer.Render(ctx)
```

### Tile-Based Rendering

```go
config := parallelraytracer.RenderConfig{
    Width:           1200,
    Height:          800,
    SamplesPerPixel: 200,
    MaxDepth:        30,
    NumWorkers:      8,
    BackgroundColor: parallelraytracer.Vec3{0.5, 0.7, 1.0},
}

world := parallelraytracer.CreateDefaultScene()
camera := parallelraytracer.NewCamera(
    parallelraytracer.Vec3{13, 2, 3},
    parallelraytracer.Vec3{0, 0, 0},
    parallelraytracer.Vec3{0, 1, 0},
    20.0, 1.5, 0.1, 10.0,
)

// Use tile-based renderer for memory efficiency
tileSize := 64
tileRenderer := parallelraytracer.NewTileRenderer(config, tileSize, world, camera)

ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
defer cancel()

img := tileRenderer.Render(ctx)
```

### Adaptive Sampling

```go
config := parallelraytracer.RenderConfig{
    Width:           800,
    Height:          600,
    MaxDepth:        20,
    NumWorkers:      6,
    BackgroundColor: parallelraytracer.Vec3{0.5, 0.7, 1.0},
}

// Adaptive renderer adjusts sample count based on pixel complexity
minSamples := 10
maxSamples := 500
varianceThreshold := 0.01  // Lower = higher quality

adaptiveRenderer := parallelraytracer.NewAdaptiveRenderer(
    config, minSamples, maxSamples, varianceThreshold, world, camera,
)

ctx := context.Background()
img := adaptiveRenderer.Render(ctx)
```

### Progress Monitoring

```go
rayTracer := parallelraytracer.NewRayTracer(config, world, camera)

ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
defer cancel()

// Start rendering in background
go func() {
    img := rayTracer.Render(ctx)
    // Handle completed image
}()

// Monitor progress
ticker := time.NewTicker(1 * time.Second)
defer ticker.Stop()

for {
    select {
    case <-ticker.C:
        progress := rayTracer.GetProgress()
        fmt.Printf("\rProgress: %.1f%%", progress)
        if progress >= 100.0 {
            fmt.Println("\nRender complete!")
            return
        }
    case <-ctx.Done():
        fmt.Println("\nRender cancelled")
        return
    }
}
```

### Custom Materials

```go
// Create custom materials
redLambertian := parallelraytracer.Lambertian{
    Albedo: parallelraytracer.Vec3{0.8, 0.3, 0.3},
}

shinyMetal := parallelraytracer.Metal{
    Albedo: parallelraytracer.Vec3{0.8, 0.8, 0.9},
    Fuzz:   0.0, // Perfect mirror
}

glass := parallelraytracer.Dielectric{
    RefractiveIndex: 1.5, // Glass
}

// Create custom scene
world := parallelraytracer.World{
    Objects: []parallelraytracer.Hittable{
        parallelraytracer.Sphere{
            Center:   parallelraytracer.Vec3{0, -100.5, -1},
            Radius:   100,
            Material: redLambertian,
        },
        parallelraytracer.Sphere{
            Center:   parallelraytracer.Vec3{-1, 0, -1},
            Radius:   0.5,
            Material: glass,
        },
        parallelraytracer.Sphere{
            Center:   parallelraytracer.Vec3{1, 0, -1},
            Radius:   0.5,
            Material: shinyMetal,
        },
    },
}
```

## Architecture

### Core Components

1. **Vec3**: 3D vector with mathematical operations
   - Addition, subtraction, multiplication, division
   - Dot product, cross product, normalization
   - Reflection and linear interpolation

2. **Ray**: Ray representation with origin and direction
   - Point calculation along ray path
   - Used for camera rays and scattered rays

3. **Materials**: Different surface material types
   - Lambertian: Diffuse surfaces with matte appearance
   - Metal: Reflective surfaces with optional fuzziness
   - Dielectric: Transparent materials with refraction

4. **Hittable Objects**: Scene geometry that rays can intersect
   - Sphere: Analytical sphere intersection
   - World: Collection of hittable objects

5. **Camera**: Virtual camera system
   - Configurable field of view and aspect ratio
   - Depth of field with adjustable aperture
   - Look-at positioning system

### Parallel Processing Architecture

- **Worker Pool**: Fixed number of worker goroutines
- **Job Queue**: Channel-based work distribution
- **Pixel Independence**: Each pixel rendered independently
- **Context Propagation**: Cancellation support throughout
- **Progress Reporting**: Real-time progress updates

### Rendering Pipeline

1. **Ray Generation**: Camera generates rays for each pixel
2. **Scene Intersection**: Test rays against all objects
3. **Material Interaction**: Calculate surface response
4. **Recursive Tracing**: Trace reflected/refracted rays
5. **Color Accumulation**: Sum contributions from all samples
6. **Gamma Correction**: Apply gamma correction for display

## Material Properties

### Lambertian (Diffuse)
```go
material := parallelraytracer.Lambertian{
    Albedo: parallelraytracer.Vec3{0.7, 0.3, 0.3}, // Surface color
}
```
- Perfect diffuse reflection
- Scattered rays follow cosine distribution
- Suitable for matte surfaces like paper, fabric

### Metal (Reflective)
```go
material := parallelraytracer.Metal{
    Albedo: parallelraytracer.Vec3{0.8, 0.8, 0.9}, // Tint color
    Fuzz:   0.1, // Surface roughness (0.0 = perfect mirror)
}
```
- Specular reflection with optional fuzziness
- Fuzz parameter controls surface roughness
- Perfect for metallic surfaces

### Dielectric (Transparent)
```go
material := parallelraytracer.Dielectric{
    RefractiveIndex: 1.5, // Glass = 1.5, Diamond = 2.4
}
```
- Handles both reflection and refraction
- Uses Schlick's approximation for Fresnel effects
- Perfect for glass, water, crystals

## Camera Configuration

### Field of View
- Vertical field of view in degrees
- Wider angles capture more scene
- Narrower angles provide telephoto effect

### Depth of Field
- Aperture size controls depth of field
- Larger aperture = shallower depth of field
- Focus distance determines sharp plane

### Positioning
- Look-from: Camera position in world space
- Look-at: Point camera is aimed at
- View-up: Camera orientation vector

## Performance Optimization

### Worker Count
- Optimal: Number of CPU cores
- Too few: Underutilized hardware
- Too many: Context switching overhead

### Samples Per Pixel
- More samples = higher quality, longer render time
- Start with 10-50 for preview
- Use 100-1000 for final renders

### Max Depth
- Controls maximum ray bounces
- Higher depth = more realistic lighting
- Diminishing returns after 20-50 bounces

### Tile Size (Tile Renderer)
- 32-128 pixels optimal for most scenes
- Smaller tiles = better load balancing
- Larger tiles = less overhead

## Testing

Run the comprehensive test suite:

```bash
go test -v ./parallelraytracer/
```

Run benchmarks:

```bash
go test -bench=. ./parallelraytracer/
```

### Test Coverage

- Vector mathematics operations
- Ray-sphere intersection
- Material scattering behavior
- Camera ray generation
- Parallel rendering correctness
- Progress monitoring
- Context cancellation
- Image output format
- Adaptive sampling
- Tile-based rendering

## Performance Characteristics

### Render Times (Approximate)
- **800x600, 100 samples**: 1-5 minutes (8 cores)
- **1920x1080, 500 samples**: 15-60 minutes (8 cores)
- **4K, 1000 samples**: 2-8 hours (8 cores)

### Memory Usage
- **Per Pixel**: ~24 bytes during rendering
- **Scene Objects**: ~100 bytes per sphere
- **Image Buffer**: Width × Height × 4 bytes

### Scalability
- **Linear scaling** with CPU core count
- **Memory bound** for very large images
- **I/O bound** for image saving

## Advanced Features

### Monte Carlo Integration
- Random sampling for realistic light distribution
- Importance sampling for efficient convergence
- Variance reduction through stratified sampling

### Global Illumination
- Multiple ray bounces simulate light transport
- Indirect lighting through diffuse surfaces
- Caustics through refractive materials

### Anti-Aliasing
- Supersampling eliminates jagged edges
- Jittered sampling reduces aliasing artifacts
- Adaptive sampling focuses computation

## Use Cases

1. **Computer Graphics Education**: Learning ray tracing fundamentals
2. **Prototyping**: Testing rendering algorithms and techniques
3. **Art and Visualization**: Creating photorealistic images
4. **Performance Testing**: Benchmarking parallel processing
5. **Algorithm Research**: Experimenting with new techniques
6. **Game Development**: Offline rendering for assets

## Limitations

This implementation focuses on educational clarity:

- Only sphere primitives supported
- No triangle mesh support
- No texture mapping
- No volumetric rendering
- No advanced lighting models (BRDF/BSDF)
- No acceleration structures (BVH, octree)
- Limited file format support (PNG only)

## Future Enhancements

### Geometry
- Triangle mesh support
- Constructive solid geometry (CSG)
- Procedural geometry generation

### Materials
- Physically-based materials (PBR)
- Texture mapping and UV coordinates
- Normal and displacement mapping
- Subsurface scattering

### Performance
- Spatial acceleration structures
- GPU acceleration with OpenCL/CUDA
- Denoising algorithms
- Progressive rendering

### Output
- HDR image formats (EXR, HDR)
- Animation sequences
- Real-time preview modes