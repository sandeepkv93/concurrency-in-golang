package parallelwebcrawler

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"
)

// Page represents a crawled web page
type Page struct {
	URL        string
	Title      string
	Links      []string
	Content    string
	StatusCode int
	Error      error
	CrawledAt  time.Time
}

// CrawlResult represents the result of crawling
type CrawlResult struct {
	Pages      map[string]*Page
	TotalPages int
	TotalLinks int
	Errors     int
	Duration   time.Duration
}

// Crawler represents a web crawler
type Crawler struct {
	maxDepth       int
	maxConcurrent  int
	visitedMutex   sync.RWMutex
	visited        map[string]bool
	pagesMutex     sync.Mutex
	pages          map[string]*Page
	semaphore      chan struct{}
	client         *http.Client
	allowedDomains []string
	userAgent      string
}

// NewCrawler creates a new web crawler
func NewCrawler(maxDepth, maxConcurrent int) *Crawler {
	return &Crawler{
		maxDepth:      maxDepth,
		maxConcurrent: maxConcurrent,
		visited:       make(map[string]bool),
		pages:         make(map[string]*Page),
		semaphore:     make(chan struct{}, maxConcurrent),
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
		userAgent: "Go-WebCrawler/1.0",
	}
}

// SetAllowedDomains sets the domains that the crawler is allowed to visit
func (c *Crawler) SetAllowedDomains(domains []string) {
	c.allowedDomains = domains
}

// Crawl starts crawling from the given URL
func (c *Crawler) Crawl(startURL string) (*CrawlResult, error) {
	startTime := time.Now()
	
	// Normalize the start URL
	normalizedURL, err := normalizeURL(startURL)
	if err != nil {
		return nil, fmt.Errorf("invalid start URL: %w", err)
	}
	
	var wg sync.WaitGroup
	
	// Start crawling
	wg.Add(1)
	go c.crawlURL(normalizedURL, 0, &wg)
	
	// Wait for all goroutines to finish
	wg.Wait()
	
	// Prepare result
	result := &CrawlResult{
		Pages:      c.pages,
		TotalPages: len(c.pages),
		Duration:   time.Since(startTime),
	}
	
	// Count total links and errors
	for _, page := range c.pages {
		result.TotalLinks += len(page.Links)
		if page.Error != nil {
			result.Errors++
		}
	}
	
	return result, nil
}

func (c *Crawler) crawlURL(pageURL string, depth int, wg *sync.WaitGroup) {
	defer wg.Done()
	
	// Check depth limit
	if depth > c.maxDepth {
		return
	}
	
	// Check if already visited
	c.visitedMutex.Lock()
	if c.visited[pageURL] {
		c.visitedMutex.Unlock()
		return
	}
	c.visited[pageURL] = true
	c.visitedMutex.Unlock()
	
	// Check if URL is allowed
	if !c.isAllowed(pageURL) {
		return
	}
	
	// Acquire semaphore
	c.semaphore <- struct{}{}
	defer func() { <-c.semaphore }()
	
	// Fetch the page
	page := c.fetchPage(pageURL)
	
	// Store the page
	c.pagesMutex.Lock()
	c.pages[pageURL] = page
	c.pagesMutex.Unlock()
	
	// If there was an error or we're at max depth, don't follow links
	if page.Error != nil || depth >= c.maxDepth {
		return
	}
	
	// Crawl links in parallel
	for _, link := range page.Links {
		normalizedLink, err := normalizeURL(link)
		if err != nil {
			continue
		}
		
		// Make link absolute if it's relative
		absoluteLink, err := makeAbsolute(normalizedLink, pageURL)
		if err != nil {
			continue
		}
		
		c.visitedMutex.RLock()
		alreadyVisited := c.visited[absoluteLink]
		c.visitedMutex.RUnlock()
		
		if !alreadyVisited {
			wg.Add(1)
			go c.crawlURL(absoluteLink, depth+1, wg)
		}
	}
}

func (c *Crawler) fetchPage(pageURL string) *Page {
	page := &Page{
		URL:       pageURL,
		CrawledAt: time.Now(),
	}
	
	// Create request
	req, err := http.NewRequest("GET", pageURL, nil)
	if err != nil {
		page.Error = err
		return page
	}
	
	req.Header.Set("User-Agent", c.userAgent)
	
	// Make request
	resp, err := c.client.Do(req)
	if err != nil {
		page.Error = err
		return page
	}
	defer resp.Body.Close()
	
	page.StatusCode = resp.StatusCode
	
	// Read body with size limit
	limitedReader := io.LimitReader(resp.Body, 1024*1024) // 1MB limit
	body, err := io.ReadAll(limitedReader)
	if err != nil {
		page.Error = err
		return page
	}
	
	page.Content = string(body)
	
	// Extract title and links if it's HTML
	if strings.Contains(resp.Header.Get("Content-Type"), "text/html") {
		page.Title = extractTitle(page.Content)
		page.Links = extractLinks(page.Content)
	}
	
	return page
}

func (c *Crawler) isAllowed(pageURL string) bool {
	if len(c.allowedDomains) == 0 {
		return true
	}
	
	parsedURL, err := url.Parse(pageURL)
	if err != nil {
		return false
	}
	
	for _, domain := range c.allowedDomains {
		if strings.HasSuffix(parsedURL.Host, domain) {
			return true
		}
	}
	
	return false
}

// Helper functions

func normalizeURL(rawURL string) (string, error) {
	parsedURL, err := url.Parse(rawURL)
	if err != nil {
		return "", err
	}
	
	// Remove fragment
	parsedURL.Fragment = ""
	
	// Ensure scheme
	if parsedURL.Scheme == "" {
		parsedURL.Scheme = "http"
	}
	
	return parsedURL.String(), nil
}

func makeAbsolute(link, base string) (string, error) {
	linkURL, err := url.Parse(link)
	if err != nil {
		return "", err
	}
	
	if linkURL.IsAbs() {
		return link, nil
	}
	
	baseURL, err := url.Parse(base)
	if err != nil {
		return "", err
	}
	
	return baseURL.ResolveReference(linkURL).String(), nil
}

func extractTitle(html string) string {
	titleRegex := regexp.MustCompile(`<title[^>]*>([^<]+)</title>`)
	matches := titleRegex.FindStringSubmatch(html)
	if len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}
	return ""
}

func extractLinks(html string) []string {
	linkRegex := regexp.MustCompile(`<a[^>]+href=["']([^"']+)["']`)
	matches := linkRegex.FindAllStringSubmatch(html, -1)
	
	links := make([]string, 0, len(matches))
	seen := make(map[string]bool)
	
	for _, match := range matches {
		if len(match) > 1 {
			link := match[1]
			if !seen[link] {
				seen[link] = true
				links = append(links, link)
			}
		}
	}
	
	return links
}

// MockWebServer simulates a web server for testing
type MockWebServer struct {
	pages map[string]*MockPage
}

type MockPage struct {
	Content string
	Links   []string
}

// NewMockWebServer creates a new mock web server
func NewMockWebServer() *MockWebServer {
	return &MockWebServer{
		pages: make(map[string]*MockPage),
	}
}

// AddPage adds a page to the mock server
func (m *MockWebServer) AddPage(url, content string, links []string) {
	m.pages[url] = &MockPage{
		Content: content,
		Links:   links,
	}
}

// MockCrawler is a crawler that uses mock data instead of real HTTP requests
type MockCrawler struct {
	*Crawler
	mockServer *MockWebServer
}

// NewMockCrawler creates a new mock crawler
func NewMockCrawler(maxDepth, maxConcurrent int, mockServer *MockWebServer) *MockCrawler {
	crawler := NewCrawler(maxDepth, maxConcurrent)
	return &MockCrawler{
		Crawler:    crawler,
		mockServer: mockServer,
	}
}

func (mc *MockCrawler) fetchPage(pageURL string) *Page {
	page := &Page{
		URL:        pageURL,
		CrawledAt:  time.Now(),
		StatusCode: 200,
	}
	
	mockPage, exists := mc.mockServer.pages[pageURL]
	if !exists {
		page.StatusCode = 404
		page.Error = fmt.Errorf("page not found")
		return page
	}
	
	page.Content = mockPage.Content
	page.Title = extractTitle(mockPage.Content)
	page.Links = mockPage.Links
	
	return page
}

// Example demonstrates web crawling
func Example() {
	fmt.Println("=== Parallel Web Crawler Example ===")
	
	// Create a mock web server
	mockServer := NewMockWebServer()
	
	// Add some pages
	mockServer.AddPage("http://example.com", 
		"<html><title>Example Home</title><body>Welcome!</body></html>",
		[]string{"http://example.com/about", "http://example.com/products"})
		
	mockServer.AddPage("http://example.com/about",
		"<html><title>About Us</title><body>About page</body></html>",
		[]string{"http://example.com", "http://example.com/team"})
		
	mockServer.AddPage("http://example.com/products",
		"<html><title>Products</title><body>Our products</body></html>",
		[]string{"http://example.com", "http://example.com/product1", "http://example.com/product2"})
		
	mockServer.AddPage("http://example.com/team",
		"<html><title>Our Team</title><body>Team info</body></html>",
		[]string{"http://example.com/about"})
		
	mockServer.AddPage("http://example.com/product1",
		"<html><title>Product 1</title><body>First product</body></html>",
		[]string{"http://example.com/products"})
		
	mockServer.AddPage("http://example.com/product2",
		"<html><title>Product 2</title><body>Second product</body></html>",
		[]string{"http://example.com/products"})
	
	// Create a mock crawler
	crawler := NewMockCrawler(2, 3, mockServer)
	crawler.SetAllowedDomains([]string{"example.com"})
	
	// Start crawling
	fmt.Println("Starting crawl from http://example.com with max depth 2...")
	result, err := crawler.Crawl("http://example.com")
	if err != nil {
		fmt.Printf("Crawl error: %v\n", err)
		return
	}
	
	fmt.Printf("\nCrawl completed in %v\n", result.Duration)
	fmt.Printf("Total pages crawled: %d\n", result.TotalPages)
	fmt.Printf("Total links found: %d\n", result.TotalLinks)
	fmt.Printf("Errors encountered: %d\n", result.Errors)
	
	fmt.Println("\nPages crawled:")
	for url, page := range result.Pages {
		if page.Error != nil {
			fmt.Printf("  %s - Error: %v\n", url, page.Error)
		} else {
			fmt.Printf("  %s - %s (%d links)\n", url, page.Title, len(page.Links))
		}
	}
}