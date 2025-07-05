package parallelimageprocessing

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
)

// Filter represents an image filter operation
type Filter interface {
	Apply(img image.Image) image.Image
}

// ImageProcessor handles parallel image processing
type ImageProcessor struct {
	numWorkers int
	tileSize   int
}

// NewImageProcessor creates a new image processor
func NewImageProcessor(numWorkers int) *ImageProcessor {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}
	return &ImageProcessor{
		numWorkers: numWorkers,
		tileSize:   128, // Default tile size for parallel processing
	}
}

// ProcessImage applies a filter to an image in parallel
func (ip *ImageProcessor) ProcessImage(img image.Image, filter Filter) image.Image {
	bounds := img.Bounds()
	result := image.NewRGBA(bounds)
	
	// Calculate tiles
	tiles := ip.generateTiles(bounds)
	
	// Process tiles in parallel
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
	
	// Send tiles to workers
	for _, tile := range tiles {
		tileChan <- tile
	}
	close(tileChan)
	
	wg.Wait()
	return result
}

func (ip *ImageProcessor) generateTiles(bounds image.Rectangle) []image.Rectangle {
	tiles := []image.Rectangle{}
	
	for y := bounds.Min.Y; y < bounds.Max.Y; y += ip.tileSize {
		for x := bounds.Min.X; x < bounds.Max.X; x += ip.tileSize {
			tile := image.Rect(
				x,
				y,
				minInt(x+ip.tileSize, bounds.Max.X),
				minInt(y+ip.tileSize, bounds.Max.Y),
			)
			tiles = append(tiles, tile)
		}
	}
	
	return tiles
}

func (ip *ImageProcessor) processTile(src image.Image, dst *image.RGBA, tile image.Rectangle, filter Filter) {
	// Create sub-image for the tile
	subImg := &subImage{
		Image:  src,
		bounds: tile,
	}
	
	// Apply filter to sub-image
	filtered := filter.Apply(subImg)
	
	// Copy result to destination
	draw.Draw(dst, tile, filtered, tile.Min, draw.Src)
}

// subImage represents a portion of an image
type subImage struct {
	image.Image
	bounds image.Rectangle
}

func (s *subImage) Bounds() image.Rectangle {
	return s.bounds
}

func (s *subImage) At(x, y int) color.Color {
	if !image.Pt(x, y).In(s.bounds) {
		return color.RGBA{0, 0, 0, 0}
	}
	return s.Image.At(x, y)
}

// Filters

// GrayscaleFilter converts image to grayscale
type GrayscaleFilter struct{}

func (f *GrayscaleFilter) Apply(img image.Image) image.Image {
	bounds := img.Bounds()
	result := image.NewRGBA(bounds)
	
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.At(x, y)
			r, g, b, a := c.RGBA()
			// Use luminance formula
			gray := uint16(0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b))
			result.Set(x, y, color.RGBA64{gray, gray, gray, uint16(a)})
		}
	}
	
	return result
}

// BlurFilter applies Gaussian blur
type BlurFilter struct {
	Radius float64
}

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
	
	// Normalize
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			kernel[i][j] /= sum
		}
	}
	
	return kernel
}

func (f *BlurFilter) convolve(img image.Image, x, y int, kernel [][]float64) color.Color {
	size := len(kernel)
	offset := size / 2
	
	var r, g, b, a float64
	
	for ky := 0; ky < size; ky++ {
		for kx := 0; kx < size; kx++ {
			px := x + kx - offset
			py := y + ky - offset
			
			// Handle boundaries
			if px < img.Bounds().Min.X {
				px = img.Bounds().Min.X
			}
			if px >= img.Bounds().Max.X {
				px = img.Bounds().Max.X - 1
			}
			if py < img.Bounds().Min.Y {
				py = img.Bounds().Min.Y
			}
			if py >= img.Bounds().Max.Y {
				py = img.Bounds().Max.Y - 1
			}
			
			c := img.At(px, py)
			cr, cg, cb, ca := c.RGBA()
			weight := kernel[ky][kx]
			
			r += float64(cr) * weight
			g += float64(cg) * weight
			b += float64(cb) * weight
			a += float64(ca) * weight
		}
	}
	
	return color.RGBA64{
		uint16(r),
		uint16(g),
		uint16(b),
		uint16(a),
	}
}

// EdgeDetectionFilter detects edges using Sobel operator
type EdgeDetectionFilter struct {
	Threshold float64
}

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

func (f *EdgeDetectionFilter) applyKernel(img image.Image, x, y int, kernel [][]float64) float64 {
	sum := 0.0
	
	for ky := 0; ky < 3; ky++ {
		for kx := 0; kx < 3; kx++ {
			px := x + kx - 1
			py := y + ky - 1
			
			c := img.At(px, py)
			gray, _, _, _ := c.RGBA()
			sum += float64(gray) * kernel[ky][kx]
		}
	}
	
	return sum
}

// BrightnessFilter adjusts image brightness
type BrightnessFilter struct {
	Factor float64 // -1.0 to 1.0
}

func (f *BrightnessFilter) Apply(img image.Image) image.Image {
	bounds := img.Bounds()
	result := image.NewRGBA(bounds)
	
	adjustment := int(f.Factor * 65535)
	
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.At(x, y)
			r, g, b, a := c.RGBA()
			
			r = clampUint32(uint32(int(r) + adjustment))
			g = clampUint32(uint32(int(g) + adjustment))
			b = clampUint32(uint32(int(b) + adjustment))
			
			result.Set(x, y, color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)})
		}
	}
	
	return result
}

// ContrastFilter adjusts image contrast
type ContrastFilter struct {
	Factor float64 // 0.0 to 2.0, where 1.0 is no change
}

func (f *ContrastFilter) Apply(img image.Image) image.Image {
	bounds := img.Bounds()
	result := image.NewRGBA(bounds)
	
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.At(x, y)
			r, g, b, a := c.RGBA()
			
			// Apply contrast formula: (value - 0.5) * factor + 0.5
			r = applyContrast(r, f.Factor)
			g = applyContrast(g, f.Factor)
			b = applyContrast(b, f.Factor)
			
			result.Set(x, y, color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)})
		}
	}
	
	return result
}

func applyContrast(value uint32, factor float64) uint32 {
	normalized := float64(value) / 65535.0
	adjusted := (normalized-0.5)*factor + 0.5
	return uint32(clamp(adjusted*65535.0, 0, 65535))
}

// RotateFilter rotates image by specified degrees
type RotateFilter struct {
	Degrees float64
}

func (f *RotateFilter) Apply(img image.Image) image.Image {
	bounds := img.Bounds()
	radians := f.Degrees * math.Pi / 180
	
	// Calculate new dimensions
	cos := math.Abs(math.Cos(radians))
	sin := math.Abs(math.Sin(radians))
	newWidth := int(float64(bounds.Dx())*cos + float64(bounds.Dy())*sin)
	newHeight := int(float64(bounds.Dx())*sin + float64(bounds.Dy())*cos)
	
	result := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))
	
	// Center points
	cx := float64(bounds.Dx()) / 2
	cy := float64(bounds.Dy()) / 2
	ncx := float64(newWidth) / 2
	ncy := float64(newHeight) / 2
	
	// Perform rotation
	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			// Translate to center
			tx := float64(x) - ncx
			ty := float64(y) - ncy
			
			// Rotate
			rx := tx*math.Cos(-radians) - ty*math.Sin(-radians)
			ry := tx*math.Sin(-radians) + ty*math.Cos(-radians)
			
			// Translate back
			rx += cx
			ry += cy
			
			// Sample from original image
			if rx >= 0 && rx < float64(bounds.Dx()) && ry >= 0 && ry < float64(bounds.Dy()) {
				// Bilinear interpolation
				x0 := int(rx)
				y0 := int(ry)
				x1 := minInt(x0+1, bounds.Dx()-1)
				y1 := minInt(y0+1, bounds.Dy()-1)
				
				dx := rx - float64(x0)
				dy := ry - float64(y0)
				
				c00 := img.At(bounds.Min.X+x0, bounds.Min.Y+y0)
				c01 := img.At(bounds.Min.X+x0, bounds.Min.Y+y1)
				c10 := img.At(bounds.Min.X+x1, bounds.Min.Y+y0)
				c11 := img.At(bounds.Min.X+x1, bounds.Min.Y+y1)
				
				interpolated := interpolateColors(c00, c01, c10, c11, dx, dy)
				result.Set(x, y, interpolated)
			}
		}
	}
	
	return result
}

// BatchProcessor processes multiple images concurrently
type BatchProcessor struct {
	processor  *ImageProcessor
	maxWorkers int
}

// NewBatchProcessor creates a new batch processor
func NewBatchProcessor(maxWorkers int) *BatchProcessor {
	if maxWorkers <= 0 {
		maxWorkers = runtime.NumCPU()
	}
	return &BatchProcessor{
		processor:  NewImageProcessor(runtime.NumCPU()),
		maxWorkers: maxWorkers,
	}
}

// ProcessBatch processes multiple images with the same filter
func (bp *BatchProcessor) ProcessBatch(inputDir, outputDir string, filter Filter, pattern string) error {
	// Find matching files
	files, err := filepath.Glob(filepath.Join(inputDir, pattern))
	if err != nil {
		return err
	}
	
	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
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
			
			semaphore <- struct{}{}
			defer func() { <-semaphore }()
			
			if err := bp.processFile(inputPath, outputDir, filter); err != nil {
				errors <- fmt.Errorf("error processing %s: %w", inputPath, err)
			} else {
				atomic.AddInt32(&processed, 1)
			}
		}(file)
	}
	
	wg.Wait()
	close(errors)
	
	// Collect errors
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

func (bp *BatchProcessor) processFile(inputPath, outputDir string, filter Filter) error {
	// Open input file
	file, err := os.Open(inputPath)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Decode image
	img, format, err := image.Decode(file)
	if err != nil {
		return err
	}
	
	// Process image
	processed := bp.processor.ProcessImage(img, filter)
	
	// Generate output path
	baseName := filepath.Base(inputPath)
	outputPath := filepath.Join(outputDir, baseName)
	
	// Save processed image
	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer outFile.Close()
	
	// Encode based on format
	switch format {
	case "jpeg":
		return jpeg.Encode(outFile, processed, &jpeg.Options{Quality: 90})
	case "png":
		return png.Encode(outFile, processed)
	default:
		return fmt.Errorf("unsupported format: %s", format)
	}
}

// Helper functions

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

func clampUint32(value uint32) uint32 {
	if value > 65535 {
		return 65535
	}
	return value
}

func interpolateColors(c00, c01, c10, c11 color.Color, dx, dy float64) color.Color {
	r00, g00, b00, a00 := c00.RGBA()
	r01, g01, b01, a01 := c01.RGBA()
	r10, g10, b10, a10 := c10.RGBA()
	r11, g11, b11, a11 := c11.RGBA()
	
	// Bilinear interpolation
	r0 := float64(r00)*(1-dx) + float64(r10)*dx
	r1 := float64(r01)*(1-dx) + float64(r11)*dx
	r := uint16(r0*(1-dy) + r1*dy)
	
	g0 := float64(g00)*(1-dx) + float64(g10)*dx
	g1 := float64(g01)*(1-dx) + float64(g11)*dx
	g := uint16(g0*(1-dy) + g1*dy)
	
	b0 := float64(b00)*(1-dx) + float64(b10)*dx
	b1 := float64(b01)*(1-dx) + float64(b11)*dx
	b := uint16(b0*(1-dy) + b1*dy)
	
	a0 := float64(a00)*(1-dx) + float64(a10)*dx
	a1 := float64(a01)*(1-dx) + float64(a11)*dx
	a := uint16(a0*(1-dy) + a1*dy)
	
	return color.RGBA64{r, g, b, a}
}

// Example demonstrates parallel image processing
func Example() {
	fmt.Println("=== Parallel Image Processing Example ===")
	
	// Create a sample image
	img := image.NewRGBA(image.Rect(0, 0, 200, 200))
	
	// Draw something
	for y := 0; y < 200; y++ {
		for x := 0; x < 200; x++ {
			r := uint8(x * 255 / 200)
			g := uint8(y * 255 / 200)
			b := uint8(128)
			img.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	
	// Create processor
	processor := NewImageProcessor(4)
	
	// Apply filters
	filters := []struct {
		name   string
		filter Filter
	}{
		{"Grayscale", &GrayscaleFilter{}},
		{"Blur", &BlurFilter{Radius: 2.0}},
		{"Edge Detection", &EdgeDetectionFilter{Threshold: 30000}},
		{"Brightness", &BrightnessFilter{Factor: 0.2}},
		{"Contrast", &ContrastFilter{Factor: 1.5}},
	}
	
	for _, f := range filters {
		fmt.Printf("Applying %s filter...\n", f.name)
		result := processor.ProcessImage(img, f.filter)
		fmt.Printf("  Processed image size: %v\n", result.Bounds())
	}
}