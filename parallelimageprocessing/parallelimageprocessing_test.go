package parallelimageprocessing

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"
)

func createTestImage(width, height int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	
	// Create a gradient pattern
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r := uint8(x * 255 / width)
			g := uint8(y * 255 / height)
			b := uint8((x + y) * 255 / (width + height))
			img.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	
	return img
}

func TestImageProcessor(t *testing.T) {
	processor := NewImageProcessor(4)
	img := createTestImage(100, 100)
	
	// Test with grayscale filter
	filter := &GrayscaleFilter{}
	result := processor.ProcessImage(img, filter)
	
	// Verify dimensions
	if !result.Bounds().Eq(img.Bounds()) {
		t.Errorf("Result dimensions don't match: got %v, want %v", 
			result.Bounds(), img.Bounds())
	}
	
	// Verify grayscale conversion
	c := result.At(50, 50)
	r, g, b, _ := c.RGBA()
	if r != g || g != b {
		t.Error("Grayscale conversion failed: RGB values should be equal")
	}
}

func TestGrayscaleFilter(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	
	// Set a colored pixel
	img.Set(5, 5, color.RGBA{255, 0, 0, 255}) // Pure red
	
	filter := &GrayscaleFilter{}
	result := filter.Apply(img)
	
	// Check the converted pixel
	c := result.At(5, 5)
	r, g, b, _ := c.RGBA()
	
	// Should be grayscale
	if r != g || g != b {
		t.Error("Grayscale conversion failed")
	}
	
	// Red has luminance ~0.299
	expectedGray := uint32(0.299 * 65535)
	tolerance := uint32(1000)
	
	if r < expectedGray-tolerance || r > expectedGray+tolerance {
		t.Errorf("Incorrect grayscale value: got %d, expected ~%d", r, expectedGray)
	}
}

func TestBlurFilter(t *testing.T) {
	// Create image with sharp edge
	img := image.NewRGBA(image.Rect(0, 0, 20, 20))
	draw.Draw(img, img.Bounds(), &image.Uniform{color.Black}, image.Point{}, draw.Src)
	
	// White square in center
	for y := 8; y < 12; y++ {
		for x := 8; x < 12; x++ {
			img.Set(x, y, color.White)
		}
	}
	
	filter := &BlurFilter{Radius: 1.0}
	result := filter.Apply(img)
	
	// Edge pixels should be blurred (not pure black or white)
	edge := result.At(7, 10)
	r, _, _, _ := edge.RGBA()
	
	if r == 0 || r == 65535 {
		t.Error("Blur filter didn't blur edge pixels")
	}
}

func TestEdgeDetectionFilter(t *testing.T) {
	// Create image with clear edge
	img := image.NewRGBA(image.Rect(0, 0, 20, 20))
	
	// Left half black, right half white
	for y := 0; y < 20; y++ {
		for x := 0; x < 10; x++ {
			img.Set(x, y, color.Black)
		}
		for x := 10; x < 20; x++ {
			img.Set(x, y, color.White)
		}
	}
	
	filter := &EdgeDetectionFilter{Threshold: 10000}
	result := filter.Apply(img)
	
	// Should detect edge at x=10
	edge := result.At(10, 10)
	r, _, _, _ := edge.RGBA()
	
	if r == 0 {
		t.Error("Edge detection failed to detect vertical edge")
	}
}

func TestBrightnessFilter(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	img.Set(5, 5, color.RGBA{128, 128, 128, 255})
	
	// Test brightness increase
	filter := &BrightnessFilter{Factor: 0.5}
	result := filter.Apply(img)
	
	c := result.At(5, 5)
	r, _, _, _ := c.RGBA()
	
	// Should be brighter than original
	original := uint32(128 * 257) // 8-bit to 16-bit conversion
	if r <= original {
		t.Error("Brightness increase failed")
	}
	
	// Test brightness decrease
	filter2 := &BrightnessFilter{Factor: -0.5}
	result2 := filter2.Apply(img)
	
	c2 := result2.At(5, 5)
	r2, _, _, _ := c2.RGBA()
	
	// Should be darker than original
	if r2 >= original {
		t.Error("Brightness decrease failed")
	}
}

func TestContrastFilter(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	
	// Set pixels with different gray levels
	img.Set(0, 0, color.RGBA{64, 64, 64, 255})   // Dark gray
	img.Set(1, 0, color.RGBA{192, 192, 192, 255}) // Light gray
	
	filter := &ContrastFilter{Factor: 2.0}
	result := filter.Apply(img)
	
	// Dark pixel should be darker
	dark := result.At(0, 0)
	dr, _, _, _ := dark.RGBA()
	
	// Light pixel should be lighter
	light := result.At(1, 0)
	lr, _, _, _ := light.RGBA()
	
	// Verify contrast increased
	originalDiff := uint32(192-64) * 257
	newDiff := lr - dr
	
	if newDiff <= originalDiff {
		t.Error("Contrast filter didn't increase contrast")
	}
}

func TestRotateFilter(t *testing.T) {
	// Create small test image with identifiable pattern
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	
	// Set a specific pixel
	img.Set(9, 0, color.RGBA{255, 0, 0, 255}) // Red pixel at top-right
	
	filter := &RotateFilter{Degrees: 90}
	result := filter.Apply(img)
	
	// After 90-degree rotation, top-right should become bottom-right
	// Check that result has correct dimensions (should be same for 90-degree rotation of square)
	if result.Bounds().Dx() != img.Bounds().Dy() || result.Bounds().Dy() != img.Bounds().Dx() {
		t.Error("Rotated image has incorrect dimensions")
	}
}

func TestTileGeneration(t *testing.T) {
	processor := NewImageProcessor(4)
	processor.tileSize = 100
	
	bounds := image.Rect(0, 0, 250, 250)
	tiles := processor.generateTiles(bounds)
	
	// Should have 3x3 = 9 tiles
	if len(tiles) != 9 {
		t.Errorf("Expected 9 tiles, got %d", len(tiles))
	}
	
	// Verify tiles cover entire image
	covered := image.NewRGBA(bounds)
	for _, tile := range tiles {
		draw.Draw(covered, tile, &image.Uniform{color.White}, image.Point{}, draw.Src)
	}
	
	// Check all pixels are covered
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := covered.At(x, y)
			if c != color.White {
				t.Errorf("Pixel at (%d, %d) not covered by tiles", x, y)
			}
		}
	}
}

func TestConcurrentProcessing(t *testing.T) {
	processor := NewImageProcessor(runtime.NumCPU())
	img := createTestImage(1000, 1000)
	
	// Process multiple images concurrently
	filters := []Filter{
		&GrayscaleFilter{},
		&BlurFilter{Radius: 2.0},
		&BrightnessFilter{Factor: 0.1},
	}
	
	var wg sync.WaitGroup
	results := make([]image.Image, len(filters))
	
	for i, filter := range filters {
		wg.Add(1)
		go func(idx int, f Filter) {
			defer wg.Done()
			results[idx] = processor.ProcessImage(img, f)
		}(i, filter)
	}
	
	wg.Wait()
	
	// Verify all results
	for i, result := range results {
		if result == nil {
			t.Errorf("Filter %d returned nil result", i)
		}
		if !result.Bounds().Eq(img.Bounds()) {
			t.Errorf("Filter %d returned incorrect dimensions", i)
		}
	}
}

func TestEdgeCases(t *testing.T) {
	// Test with 1x1 image
	tiny := image.NewRGBA(image.Rect(0, 0, 1, 1))
	tiny.Set(0, 0, color.RGBA{100, 100, 100, 255})
	
	processor := NewImageProcessor(1)
	filters := []Filter{
		&GrayscaleFilter{},
		&BrightnessFilter{Factor: 0.5},
		&ContrastFilter{Factor: 1.5},
	}
	
	for _, filter := range filters {
		result := processor.ProcessImage(tiny, filter)
		if result.Bounds().Dx() != 1 || result.Bounds().Dy() != 1 {
			t.Error("Filter changed dimensions of 1x1 image")
		}
	}
	
	// Test with empty image
	empty := image.NewRGBA(image.Rect(0, 0, 0, 0))
	result := processor.ProcessImage(empty, &GrayscaleFilter{})
	if !result.Bounds().Empty() {
		t.Error("Processing empty image should return empty result")
	}
}

func BenchmarkImageProcessing(b *testing.B) {
	sizes := []int{100, 500, 1000}
	workerCounts := []int{1, 2, 4, 8}
	
	for _, size := range sizes {
		img := createTestImage(size, size)
		
		for _, workers := range workerCounts {
			b.Run(fmt.Sprintf("Size%dx%d_Workers%d", size, size, workers), func(b *testing.B) {
				processor := NewImageProcessor(workers)
				filter := &GrayscaleFilter{}
				
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					processor.ProcessImage(img, filter)
				}
			})
		}
	}
}

func BenchmarkFilters(b *testing.B) {
	img := createTestImage(500, 500)
	processor := NewImageProcessor(4)
	
	filters := []struct {
		name   string
		filter Filter
	}{
		{"Grayscale", &GrayscaleFilter{}},
		{"Blur", &BlurFilter{Radius: 2.0}},
		{"EdgeDetection", &EdgeDetectionFilter{Threshold: 30000}},
		{"Brightness", &BrightnessFilter{Factor: 0.2}},
		{"Contrast", &ContrastFilter{Factor: 1.5}},
		{"Rotate", &RotateFilter{Degrees: 45}},
	}
	
	for _, f := range filters {
		b.Run(f.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				processor.ProcessImage(img, f.filter)
			}
		})
	}
}

func TestProcessorScaling(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping scaling test in short mode")
	}
	
	img := createTestImage(2000, 2000)
	filter := &BlurFilter{Radius: 3.0}
	
	results := make(map[int]time.Duration)
	
	for workers := 1; workers <= runtime.NumCPU(); workers *= 2 {
		processor := NewImageProcessor(workers)
		
		start := time.Now()
		processor.ProcessImage(img, filter)
		elapsed := time.Since(start)
		
		results[workers] = elapsed
		t.Logf("Workers: %d, Time: %v", workers, elapsed)
	}
	
	// Verify that more workers generally means faster processing
	if runtime.NumCPU() > 1 {
		if results[1] <= results[runtime.NumCPU()] {
			t.Error("Parallel processing not faster than sequential")
		}
	}
}

func TestBilinearInterpolation(t *testing.T) {
	// Test color interpolation
	c00 := color.RGBA{0, 0, 0, 255}       // Black
	c01 := color.RGBA{0, 255, 0, 255}     // Green
	c10 := color.RGBA{255, 0, 0, 255}     // Red
	c11 := color.RGBA{255, 255, 0, 255}   // Yellow
	
	// Test center point (0.5, 0.5)
	result := interpolateColors(c00, c01, c10, c11, 0.5, 0.5)
	r, g, b, _ := result.RGBA()
	
	// Should be approximately gray (average of all colors)
	expectedR := uint16(32767) // ~0.5 * 65535
	expectedG := uint16(32767)
	
	tolerance := uint16(1000)
	if r < expectedR-tolerance || r > expectedR+tolerance {
		t.Errorf("Interpolation R incorrect: got %d, expected ~%d", r, expectedR)
	}
	if g < expectedG-tolerance || g > expectedG+tolerance {
		t.Errorf("Interpolation G incorrect: got %d, expected ~%d", g, expectedG)
	}
}

func TestGaussianKernel(t *testing.T) {
	filter := &BlurFilter{Radius: 1.0}
	kernel := filter.generateGaussianKernel(3, 1.0)
	
	// Verify kernel is normalized (sum ~= 1.0)
	sum := 0.0
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			sum += kernel[i][j]
		}
	}
	
	if math.Abs(sum-1.0) > 0.01 {
		t.Errorf("Gaussian kernel not normalized: sum = %f", sum)
	}
	
	// Verify center has highest weight
	center := kernel[1][1]
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if i == 1 && j == 1 {
				continue
			}
			if kernel[i][j] > center {
				t.Error("Gaussian kernel center should have highest weight")
			}
		}
	}
}