package parallelwebcrawler

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestBasicCrawling(t *testing.T) {
	// Create mock server
	mockServer := NewMockWebServer()
	
	// Add pages
	mockServer.AddPage("http://test.com",
		"<html><title>Home</title><body>Welcome</body></html>",
		[]string{"http://test.com/page1", "http://test.com/page2"})
		
	mockServer.AddPage("http://test.com/page1",
		"<html><title>Page 1</title><body>Content 1</body></html>",
		[]string{"http://test.com"})
		
	mockServer.AddPage("http://test.com/page2",
		"<html><title>Page 2</title><body>Content 2</body></html>",
		[]string{"http://test.com", "http://test.com/page1"})
	
	// Create crawler
	crawler := NewMockCrawler(2, 3, mockServer)
	
	// Crawl
	result, err := crawler.Crawl("http://test.com")
	if err != nil {
		t.Fatalf("Crawl failed: %v", err)
	}
	
	// Verify results
	if result.TotalPages != 3 {
		t.Errorf("Expected 3 pages, got %d", result.TotalPages)
	}
	
	if result.Errors != 0 {
		t.Errorf("Expected 0 errors, got %d", result.Errors)
	}
	
	// Check all pages were crawled
	expectedURLs := []string{
		"http://test.com",
		"http://test.com/page1",
		"http://test.com/page2",
	}
	
	for _, url := range expectedURLs {
		if _, exists := result.Pages[url]; !exists {
			t.Errorf("Expected URL %s was not crawled", url)
		}
	}
}

func TestDepthLimit(t *testing.T) {
	// Create a deep link structure
	mockServer := NewMockWebServer()
	
	// Create a chain of pages
	for i := 0; i < 5; i++ {
		url := fmt.Sprintf("http://test.com/level%d", i)
		nextURL := fmt.Sprintf("http://test.com/level%d", i+1)
		content := fmt.Sprintf("<html><title>Level %d</title></html>", i)
		
		if i < 4 {
			mockServer.AddPage(url, content, []string{nextURL})
		} else {
			mockServer.AddPage(url, content, []string{})
		}
	}
	
	// Test with depth limit 2
	crawler := NewMockCrawler(2, 3, mockServer)
	result, err := crawler.Crawl("http://test.com/level0")
	
	if err != nil {
		t.Fatalf("Crawl failed: %v", err)
	}
	
	// Should have crawled levels 0, 1, and 2 (3 pages)
	if result.TotalPages != 3 {
		t.Errorf("Expected 3 pages with depth limit 2, got %d", result.TotalPages)
	}
	
	// Verify correct pages were crawled
	for i := 0; i <= 2; i++ {
		url := fmt.Sprintf("http://test.com/level%d", i)
		if _, exists := result.Pages[url]; !exists {
			t.Errorf("Expected URL %s was not crawled", url)
		}
	}
	
	// Verify deeper pages were not crawled
	for i := 3; i < 5; i++ {
		url := fmt.Sprintf("http://test.com/level%d", i)
		if _, exists := result.Pages[url]; exists {
			t.Errorf("URL %s should not have been crawled (beyond depth limit)", url)
		}
	}
}

func TestConcurrencyLimit(t *testing.T) {
	// Create a test server that tracks concurrent requests
	var concurrentRequests int32
	var maxConcurrent int32
	
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Increment concurrent requests
		current := atomic.AddInt32(&concurrentRequests, 1)
		
		// Update max if needed
		for {
			max := atomic.LoadInt32(&maxConcurrent)
			if current <= max || atomic.CompareAndSwapInt32(&maxConcurrent, max, current) {
				break
			}
		}
		
		// Simulate processing time
		time.Sleep(50 * time.Millisecond)
		
		// Write response
		path := r.URL.Path
		if path == "/" {
			w.Write([]byte(`<html><title>Home</title><body>
				<a href="/page1">Page 1</a>
				<a href="/page2">Page 2</a>
				<a href="/page3">Page 3</a>
				<a href="/page4">Page 4</a>
			</body></html>`))
		} else {
			w.Write([]byte(fmt.Sprintf(`<html><title>%s</title></html>`, path)))
		}
		
		// Decrement concurrent requests
		atomic.AddInt32(&concurrentRequests, -1)
	}))
	defer server.Close()
	
	// Create crawler with concurrency limit of 2
	crawler := NewCrawler(1, 2)
	result, err := crawler.Crawl(server.URL)
	
	if err != nil {
		t.Fatalf("Crawl failed: %v", err)
	}
	
	// Check that max concurrent requests didn't exceed limit
	if atomic.LoadInt32(&maxConcurrent) > 2 {
		t.Errorf("Max concurrent requests %d exceeded limit of 2", maxConcurrent)
	}
	
	// Verify all pages were crawled
	if result.TotalPages < 5 {
		t.Errorf("Expected at least 5 pages, got %d", result.TotalPages)
	}
}

func TestDomainRestriction(t *testing.T) {
	mockServer := NewMockWebServer()
	
	// Add pages from different domains
	mockServer.AddPage("http://allowed.com",
		"<html><title>Allowed</title></html>",
		[]string{"http://allowed.com/page1", "http://forbidden.com/page1"})
		
	mockServer.AddPage("http://allowed.com/page1",
		"<html><title>Allowed Page 1</title></html>",
		[]string{"http://allowed.com"})
		
	mockServer.AddPage("http://forbidden.com/page1",
		"<html><title>Forbidden Page 1</title></html>",
		[]string{"http://forbidden.com"})
	
	// Create crawler with domain restriction
	crawler := NewMockCrawler(2, 3, mockServer)
	crawler.SetAllowedDomains([]string{"allowed.com"})
	
	result, err := crawler.Crawl("http://allowed.com")
	if err != nil {
		t.Fatalf("Crawl failed: %v", err)
	}
	
	// Should only have crawled allowed.com pages
	for url := range result.Pages {
		if !strings.Contains(url, "allowed.com") {
			t.Errorf("Crawled forbidden URL: %s", url)
		}
	}
	
	// Verify allowed pages were crawled
	if _, exists := result.Pages["http://allowed.com"]; !exists {
		t.Error("Main page was not crawled")
	}
	
	if _, exists := result.Pages["http://allowed.com/page1"]; !exists {
		t.Error("Allowed page1 was not crawled")
	}
	
	// Verify forbidden pages were not crawled
	if _, exists := result.Pages["http://forbidden.com/page1"]; exists {
		t.Error("Forbidden page should not have been crawled")
	}
}

func TestErrorHandling(t *testing.T) {
	mockServer := NewMockWebServer()
	
	// Add a valid page with a link to non-existent page
	mockServer.AddPage("http://test.com",
		"<html><title>Home</title></html>",
		[]string{"http://test.com/exists", "http://test.com/notfound"})
		
	mockServer.AddPage("http://test.com/exists",
		"<html><title>Exists</title></html>",
		[]string{})
	
	// Don't add the "notfound" page
	
	crawler := NewMockCrawler(2, 3, mockServer)
	result, err := crawler.Crawl("http://test.com")
	
	if err != nil {
		t.Fatalf("Crawl failed: %v", err)
	}
	
	// Should have attempted to crawl 3 pages
	if result.TotalPages != 3 {
		t.Errorf("Expected 3 pages in result, got %d", result.TotalPages)
	}
	
	// Should have 1 error (the 404 page)
	if result.Errors != 1 {
		t.Errorf("Expected 1 error, got %d", result.Errors)
	}
	
	// Check the error page
	if page, exists := result.Pages["http://test.com/notfound"]; exists {
		if page.Error == nil {
			t.Error("Expected error for notfound page")
		}
		if page.StatusCode != 404 {
			t.Errorf("Expected status 404, got %d", page.StatusCode)
		}
	} else {
		t.Error("Notfound page should be in results with error")
	}
}

func TestLinkExtraction(t *testing.T) {
	tests := []struct {
		html     string
		expected []string
	}{
		{
			html:     `<a href="http://example.com">Link</a>`,
			expected: []string{"http://example.com"},
		},
		{
			html: `<a href="page1.html">Page 1</a><a href='page2.html'>Page 2</a>`,
			expected: []string{"page1.html", "page2.html"},
		},
		{
			html:     `<a href="http://example.com">Link 1</a><a href="http://example.com">Link 2</a>`,
			expected: []string{"http://example.com"}, // Duplicates removed
		},
		{
			html:     `<div>No links here</div>`,
			expected: []string{},
		},
	}
	
	for _, tt := range tests {
		links := extractLinks(tt.html)
		
		if len(links) != len(tt.expected) {
			t.Errorf("Expected %d links, got %d", len(tt.expected), len(links))
			continue
		}
		
		for i, link := range links {
			if link != tt.expected[i] {
				t.Errorf("Expected link %s, got %s", tt.expected[i], link)
			}
		}
	}
}

func TestTitleExtraction(t *testing.T) {
	tests := []struct {
		html     string
		expected string
	}{
		{
			html:     `<html><head><title>Test Title</title></head></html>`,
			expected: "Test Title",
		},
		{
			html:     `<title>  Spaces Around  </title>`,
			expected: "Spaces Around",
		},
		{
			html:     `<html><body>No title</body></html>`,
			expected: "",
		},
	}
	
	for _, tt := range tests {
		title := extractTitle(tt.html)
		if title != tt.expected {
			t.Errorf("Expected title '%s', got '%s'", tt.expected, title)
		}
	}
}

func BenchmarkCrawling(b *testing.B) {
	// Create a mock server with many pages
	mockServer := NewMockWebServer()
	
	// Create a network of pages
	numPages := 100
	for i := 0; i < numPages; i++ {
		url := fmt.Sprintf("http://test.com/page%d", i)
		links := []string{}
		
		// Add links to next 3 pages (creating a connected graph)
		for j := 1; j <= 3; j++ {
			nextPage := (i + j) % numPages
			links = append(links, fmt.Sprintf("http://test.com/page%d", nextPage))
		}
		
		content := fmt.Sprintf("<html><title>Page %d</title></html>", i)
		mockServer.AddPage(url, content, links)
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		crawler := NewMockCrawler(3, 10, mockServer)
		crawler.Crawl("http://test.com/page0")
	}
}

func BenchmarkConcurrentCrawling(b *testing.B) {
	benchmarks := []struct {
		name       string
		concurrent int
	}{
		{"Concurrent1", 1},
		{"Concurrent5", 5},
		{"Concurrent10", 10},
		{"Concurrent20", 20},
	}
	
	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			// Create mock server
			mockServer := NewMockWebServer()
			
			// Create pages
			for i := 0; i < 50; i++ {
				url := fmt.Sprintf("http://test.com/page%d", i)
				links := []string{}
				if i > 0 {
					links = append(links, fmt.Sprintf("http://test.com/page%d", i-1))
				}
				if i < 49 {
					links = append(links, fmt.Sprintf("http://test.com/page%d", i+1))
				}
				
				content := fmt.Sprintf("<html><title>Page %d</title></html>", i)
				mockServer.AddPage(url, content, links)
			}
			
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				crawler := NewMockCrawler(3, bm.concurrent, mockServer)
				crawler.Crawl("http://test.com/page25")
			}
		})
	}
}