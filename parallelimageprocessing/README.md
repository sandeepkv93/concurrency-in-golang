# Parallel Image Processing

A comprehensive, high-performance image processing library in Go that demonstrates advanced concurrency patterns for computationally intensive graphics operations including filtering, transformations, and batch processing with optimal CPU utilization.

## Problem Description

Image processing operations present significant computational challenges:

- **CPU-Intensive Operations**: Image filters require processing millions of pixels
- **Memory-Bound Processing**: Large images can consume significant memory
- **Sequential Bottleneck**: Processing images one pixel at a time is extremely slow
- **Cache Efficiency**: Poor memory access patterns degrade performance
- **Batch Processing**: Need to process multiple images efficiently
- **Filter Complexity**: Different filters have varying computational requirements

## Solution Approach

This implementation provides a sophisticated parallel image processing system:

1. **Tile-Based Parallelism**: Divide images into tiles for concurrent processing
2. **Worker Pool Pattern**: Fixed number of workers processing tile queue
3. **Cache-Optimized Processing**: Memory-friendly algorithms for better performance
4. **Filter Pipeline**: Modular filter system with pluggable implementations
5. **Batch Processing**: Concurrent processing of multiple images

## Key Components

### Core Structures

- **ImageProcessor**: Main orchestrator for parallel image processing
- **Filter**: Interface for implementing various image filters
- **BatchProcessor**: Handles concurrent processing of multiple images
- **SubImage**: Represents image tiles for parallel processing

### Filter Implementations

- **GrayscaleFilter**: Converts images to grayscale using luminance formula
- **BlurFilter**: Gaussian blur with configurable radius
- **EdgeDetectionFilter**: Sobel operator for edge detection
- **BrightnessFilter**: Brightness adjustment with factor control
- **ContrastFilter**: Contrast enhancement with factor scaling
- **RotateFilter**: Image rotation with bilinear interpolation

## Technical Features

### Concurrency Patterns

1. **Producer-Consumer**: Tile generation and processing pipeline
2. **Worker Pool**: Fixed workers processing image tiles
3. **Parallel Decomposition**: Spatial decomposition for image processing
4. **Batch Processing**: Concurrent file processing with worker coordination
5. **Channel-based Communication**: Tile distribution using buffered channels

### Advanced Features

- **Adaptive Tiling**: Configurable tile sizes for optimal performance
- **Memory Optimization**: Efficient memory usage for large images
- **Interpolation**: Bilinear interpolation for smooth transformations
- **Format Support**: JPEG and PNG input/output support
- **Error Handling**: Graceful handling of processing failures

## Usage Examples

### Basic Image Processing

```go
// Create image processor with 4 workers
processor := NewImageProcessor(4)

// Load an image (example shows creating a sample image)
img := image.NewRGBA(image.Rect(0, 0, 800, 600))

// Create filters
grayscaleFilter := &GrayscaleFilter{}
blurFilter := &BlurFilter{Radius: 2.0}

// Apply filters
grayImage := processor.ProcessImage(img, grayscaleFilter)
blurredImage := processor.ProcessImage(img, blurFilter)
```

### Advanced Filtering

```go
// Edge detection with threshold
edgeFilter := &EdgeDetectionFilter{Threshold: 30000}
edges := processor.ProcessImage(img, edgeFilter)

// Brightness adjustment
brightnessFilter := &BrightnessFilter{Factor: 0.2} // 20% brighter
brightImage := processor.ProcessImage(img, brightnessFilter)

// Contrast enhancement
contrastFilter := &ContrastFilter{Factor: 1.5} // 50% more contrast
contrastImage := processor.ProcessImage(img, contrastFilter)

// Image rotation
rotateFilter := &RotateFilter{Degrees: 45.0}
rotatedImage := processor.ProcessImage(img, rotateFilter)
```

### Batch Processing

```go
// Create batch processor
batchProcessor := NewBatchProcessor(8) // 8 concurrent workers

// Process all JPEG files in a directory
inputDir := "/path/to/input/images"
outputDir := "/path/to/output/images"
filter := &BlurFilter{Radius: 3.0}

err := batchProcessor.ProcessBatch(inputDir, outputDir, filter, "*.jpg")
if err != nil {
    log.Printf("Batch processing error: %v", err)
}
```

### Custom Filter Implementation

```go
// Implement custom sepia filter
type SepiaFilter struct{}

func (f *SepiaFilter) Apply(img image.Image) image.Image {
    bounds := img.Bounds()
    result := image.NewRGBA(bounds)
    
    for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
        for x := bounds.Min.X; x < bounds.Max.X; x++ {
            c := img.At(x, y)
            r, g, b, a := c.RGBA()
            
            // Sepia transformation
            newR := uint16(0.393*float64(r) + 0.769*float64(g) + 0.189*float64(b))
            newG := uint16(0.349*float64(r) + 0.686*float64(g) + 0.168*float64(b))
            newB := uint16(0.272*float64(r) + 0.534*float64(g) + 0.131*float64(b))
            
            // Clamp values
            if newR > 65535 { newR = 65535 }
            if newG > 65535 { newG = 65535 }
            if newB > 65535 { newB = 65535 }
            
            result.Set(x, y, color.RGBA64{newR, newG, newB, uint16(a)})
        }
    }
    
    return result
}

// Use custom filter
sepiaFilter := &SepiaFilter{}
sepiaImage := processor.ProcessImage(img, sepiaFilter)
```

### Performance Optimization

```go
// Configure processor for large images
processor := NewImageProcessor(runtime.NumCPU())
processor.tileSize = 256 // Larger tiles for big images

// Batch processing with optimal worker count
optimalWorkers := runtime.NumCPU() * 2 // I/O bound operations
batchProcessor := NewBatchProcessor(optimalWorkers)
```

## Implementation Details

### Tile-Based Processing

Images are divided into tiles for parallel processing:

```go
func (ip *ImageProcessor) generateTiles(bounds image.Rectangle) []image.Rectangle {
    tiles := []image.Rectangle{}
    
    for y := bounds.Min.Y; y < bounds.Max.Y; y += ip.tileSize {
        for x := bounds.Min.X; x < bounds.Max.X; x += ip.tileSize {
            tile := image.Rect(
                x, y,
                minInt(x+ip.tileSize, bounds.Max.X),
                minInt(y+ip.tileSize, bounds.Max.Y),
            )
            tiles = append(tiles, tile)
        }
    }
    
    return tiles
}
```

### Parallel Processing Pipeline

```go
func (ip *ImageProcessor) ProcessImage(img image.Image, filter Filter) image.Image {
    bounds := img.Bounds()
    result := image.NewRGBA(bounds)
    
    // Generate tiles for parallel processing
    tiles := ip.generateTiles(bounds)
    
    // Create worker pool
    var wg sync.WaitGroup
    tileChan := make(chan image.Rectangle, len(tiles))
    
    // Start workers
    for i := 0; i < ip.numWorkers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for tile := range tileChan {
                ip.processTile(img, result, tile, filter)
            }
        }()
    }
    
    // Distribute tiles to workers
    for _, tile := range tiles {
        tileChan <- tile
    }
    close(tileChan)
    
    wg.Wait()
    return result
}
```

### Filter Implementations

#### Gaussian Blur with Convolution

```go
func (f *BlurFilter) Apply(img image.Image) image.Image {
    bounds := img.Bounds()
    result := image.NewRGBA(bounds)
    
    // Generate Gaussian kernel
    size := int(f.Radius*2) + 1
    kernel := f.generateGaussianKernel(size, f.Radius)
    
    // Apply convolution
    for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
        for x := bounds.Min.X; x < bounds.Max.X; x++ {
            result.Set(x, y, f.convolve(img, x, y, kernel))
        }
    }
    
    return result
}

func (f *BlurFilter) generateGaussianKernel(size int, sigma float64) [][]float64 {
    kernel := make([][]float64, size)
    sum := 0.0
    center := size / 2
    
    for i := 0; i < size; i++ {
        kernel[i] = make([]float64, size)
        for j := 0; j < size; j++ {
            x := float64(i - center)
            y := float64(j - center)
            kernel[i][j] = math.Exp(-(x*x+y*y)/(2*sigma*sigma)) / (2 * math.Pi * sigma * sigma)
            sum += kernel[i][j]
        }
    }
    
    // Normalize kernel
    for i := 0; i < size; i++ {
        for j := 0; j < size; j++ {
            kernel[i][j] /= sum
        }
    }
    
    return kernel
}
```

#### Edge Detection with Sobel Operator

```go
func (f *EdgeDetectionFilter) Apply(img image.Image) image.Image {
    // Convert to grayscale first
    gray := (&GrayscaleFilter{}).Apply(img)
    
    bounds := gray.Bounds()
    result := image.NewRGBA(bounds)
    
    // Sobel kernels
    sobelX := [][]float64{
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1},
    }
    
    sobelY := [][]float64{
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1},
    }
    
    for y := bounds.Min.Y + 1; y < bounds.Max.Y-1; y++ {
        for x := bounds.Min.X + 1; x < bounds.Max.X-1; x++ {
            gx := f.applyKernel(gray, x, y, sobelX)
            gy := f.applyKernel(gray, x, y, sobelY)
            
            magnitude := math.Sqrt(gx*gx + gy*gy)
            
            if magnitude > f.Threshold {
                result.Set(x, y, color.White)
            } else {
                result.Set(x, y, color.Black)
            }
        }
    }
    
    return result
}
```

### Batch File Processing

```go
func (bp *BatchProcessor) ProcessBatch(inputDir, outputDir string, filter Filter, pattern string) error {
    // Find matching files
    files, err := filepath.Glob(filepath.Join(inputDir, pattern))
    if err != nil {
        return err
    }
    
    // Process files concurrently
    var wg sync.WaitGroup
    semaphore := make(chan struct{}, bp.maxWorkers)
    errors := make(chan error, len(files))
    processed := int32(0)
    
    for _, file := range files {
        wg.Add(1)
        go func(inputPath string) {
            defer wg.Done()
            
            semaphore <- struct{}{}        // Acquire
            defer func() { <-semaphore }() // Release
            
            if err := bp.processFile(inputPath, outputDir, filter); err != nil {
                errors <- fmt.Errorf("error processing %s: %w", inputPath, err)
            } else {
                atomic.AddInt32(&processed, 1)
            }
        }(file)
    }
    
    wg.Wait()
    close(errors)
    
    // Handle errors
    var allErrors []error
    for err := range errors {
        allErrors = append(allErrors, err)
    }
    
    if len(allErrors) > 0 {
        return fmt.Errorf("batch processing completed with %d errors", len(allErrors))
    }
    
    fmt.Printf("Successfully processed %d images\n", atomic.LoadInt32(&processed))
    return nil
}
```

## Testing

The package includes comprehensive tests covering:

- **Filter Correctness**: Verifying filter algorithms produce expected results
- **Parallel Processing**: Ensuring concurrent processing maintains correctness
- **Memory Management**: Testing large image processing without memory leaks
- **Batch Operations**: Validating concurrent file processing
- **Performance**: Benchmarking different tile sizes and worker counts

Run the tests:

```bash
go test -v ./parallelimageprocessing
go test -race ./parallelimageprocessing  # Race condition detection
go test -bench=. ./parallelimageprocessing  # Performance benchmarks
```

## Performance Considerations

1. **Tile Size**: Balance between parallelism and cache efficiency
2. **Worker Count**: Optimal is usually equal to CPU cores for CPU-bound operations
3. **Memory Usage**: Large images require careful memory management
4. **I/O Bottleneck**: Batch processing may be limited by disk I/O
5. **Filter Complexity**: More complex filters benefit more from parallelization

### Performance Tuning

```go
// Optimal tile size for different image sizes
func optimalTileSize(imageSize int) int {
    if imageSize < 512*512 {
        return 64   // Small images
    } else if imageSize < 2048*2048 {
        return 128  // Medium images
    } else {
        return 256  // Large images
    }
}

// Adaptive worker count
func optimalWorkerCount(operation string) int {
    if operation == "cpu_intensive" {
        return runtime.NumCPU()
    } else if operation == "io_bound" {
        return runtime.NumCPU() * 2
    }
    return runtime.NumCPU()
}
```

## Real-World Applications

This parallel image processing library is suitable for:

- **Photo Editing Software**: Real-time filter application
- **Content Management Systems**: Automated image processing pipelines
- **Computer Vision**: Preprocessing for machine learning models
- **Web Applications**: Dynamic image transformation and optimization
- **Medical Imaging**: Processing of diagnostic images
- **Satellite Imagery**: Large-scale geographic image processing

## Advanced Features

### GPU Acceleration Integration

```go
// Interface for GPU-accelerated filters
type GPUFilter interface {
    Filter
    SupportsGPU() bool
    ProcessOnGPU(img image.Image) (image.Image, error)
}
```

### Progressive Processing

```go
// Progressive filter application for real-time preview
type ProgressiveProcessor struct {
    processor *ImageProcessor
    callback  func(progress float64, preview image.Image)
}

func (pp *ProgressiveProcessor) ProcessProgressive(img image.Image, filter Filter) image.Image {
    // Implementation for progressive processing with callbacks
}
```

### Memory Pool Management

```go
// Memory pool for efficient buffer reuse
type ImageBufferPool struct {
    pool sync.Pool
}

func (ibp *ImageBufferPool) Get(size image.Rectangle) *image.RGBA {
    if buf := ibp.pool.Get(); buf != nil {
        return buf.(*image.RGBA)
    }
    return image.NewRGBA(size)
}

func (ibp *ImageBufferPool) Put(img *image.RGBA) {
    ibp.pool.Put(img)
}
```

The implementation demonstrates sophisticated parallel programming patterns for computationally intensive graphics operations, providing a high-performance foundation for image processing applications with excellent scalability and resource utilization.