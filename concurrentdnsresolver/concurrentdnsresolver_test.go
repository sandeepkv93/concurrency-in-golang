package concurrentdnsresolver

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestNewDNSResolver(t *testing.T) {
	resolver := NewDNSResolver(10, 5*time.Second)
	
	if resolver.maxWorkers != 10 {
		t.Errorf("Expected maxWorkers to be 10, got %d", resolver.maxWorkers)
	}
	
	if resolver.timeout != 5*time.Second {
		t.Errorf("Expected timeout to be 5s, got %v", resolver.timeout)
	}
	
	if resolver.cache == nil {
		t.Error("Expected cache to be initialized")
	}
}

func TestResolveSingle(t *testing.T) {
	resolver := NewDNSResolver(5, 10*time.Second)
	
	record := resolver.ResolveSingle(context.Background(), "google.com")
	
	if record.Domain != "google.com" {
		t.Errorf("Expected domain to be 'google.com', got %s", record.Domain)
	}
	
	if record.Error != nil {
		t.Errorf("Expected no error, got %v", record.Error)
	}
	
	if record.IP == "" {
		t.Error("Expected IP to be resolved")
	}
	
	if record.Type != "A" && record.Type != "AAAA" {
		t.Errorf("Expected record type to be A or AAAA, got %s", record.Type)
	}
}

func TestResolveMultiple(t *testing.T) {
	resolver := NewDNSResolver(3, 10*time.Second)
	domains := []string{"google.com", "github.com", "stackoverflow.com"}
	
	results := resolver.ResolveMultiple(context.Background(), domains)
	
	if len(results) != len(domains) {
		t.Errorf("Expected %d results, got %d", len(domains), len(results))
	}
	
	for i, result := range results {
		if result.Domain != domains[i] {
			t.Errorf("Expected domain %s, got %s", domains[i], result.Domain)
		}
		
		if result.Error != nil {
			t.Errorf("Expected no error for %s, got %v", domains[i], result.Error)
		}
	}
}

func TestResolveWithRace(t *testing.T) {
	resolver := NewDNSResolver(5, 10*time.Second)
	servers := []string{"8.8.8.8", "8.8.4.4", "1.1.1.1"}
	
	record := resolver.ResolveWithRace(context.Background(), "google.com", servers)
	
	if record.Domain != "google.com" {
		t.Errorf("Expected domain to be 'google.com', got %s", record.Domain)
	}
	
	if record.Error != nil {
		t.Errorf("Expected no error, got %v", record.Error)
	}
	
	if record.IP == "" {
		t.Error("Expected IP to be resolved")
	}
}

func TestCaching(t *testing.T) {
	resolver := NewDNSResolver(5, 10*time.Second)
	
	record1 := resolver.ResolveSingle(context.Background(), "google.com")
	record2 := resolver.ResolveSingle(context.Background(), "google.com")
	
	if record1.IP != record2.IP {
		t.Error("Expected cached result to be the same")
	}
	
	if record1.Latency == record2.Latency {
		t.Error("Expected second lookup to be faster (cached)")
	}
}

func TestBulkResolveWithBatching(t *testing.T) {
	resolver := NewDNSResolver(5, 10*time.Second)
	domains := []string{
		"google.com", "github.com", "stackoverflow.com",
		"reddit.com", "twitter.com", "facebook.com",
		"amazon.com", "netflix.com", "youtube.com",
		"linkedin.com",
	}
	
	results := resolver.BulkResolveWithBatching(context.Background(), domains, 3)
	
	if len(results) != len(domains) {
		t.Errorf("Expected %d results, got %d", len(domains), len(results))
	}
	
	for i, result := range results {
		if result.Domain != domains[i] {
			t.Errorf("Expected domain %s, got %s", domains[i], result.Domain)
		}
	}
}

func TestAsyncResolver(t *testing.T) {
	resolver := NewDNSResolver(3, 10*time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	queryQueue, resultChan := resolver.StartAsyncResolver(ctx)
	
	domains := []string{"google.com", "github.com", "stackoverflow.com"}
	
	go func() {
		for _, domain := range domains {
			queryQueue <- DNSQuery{
				Domain:    domain,
				UseRacing: false,
			}
		}
	}()
	
	receivedResults := 0
	timeout := time.After(30 * time.Second)
	
	for receivedResults < len(domains) {
		select {
		case result := <-resultChan:
			if result.Error != nil {
				t.Errorf("Expected no error for domain %s, got %v", result.Domain, result.Error)
			}
			receivedResults++
		case <-timeout:
			t.Fatal("Timeout waiting for async results")
		}
	}
}

func TestResolveWithRetry(t *testing.T) {
	resolver := NewDNSResolver(5, 10*time.Second)
	
	record := resolver.ResolveWithRetry(context.Background(), "google.com", 3)
	
	if record.Domain != "google.com" {
		t.Errorf("Expected domain to be 'google.com', got %s", record.Domain)
	}
	
	if record.Error != nil {
		t.Errorf("Expected no error, got %v", record.Error)
	}
}

func TestResolveWithRetryInvalidDomain(t *testing.T) {
	resolver := NewDNSResolver(5, 1*time.Second)
	
	record := resolver.ResolveWithRetry(context.Background(), "invalid.nonexistent.domain", 2)
	
	if record.Error == nil {
		t.Error("Expected error for invalid domain")
	}
}

func TestContextCancellation(t *testing.T) {
	resolver := NewDNSResolver(5, 10*time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()
	
	record := resolver.ResolveSingle(ctx, "google.com")
	
	if record.Error == nil {
		t.Log("Resolution completed before cancellation")
	} else if record.Error == context.Canceled {
		t.Log("Resolution was cancelled as expected")
	} else {
		t.Errorf("Unexpected error: %v", record.Error)
	}
}

func TestConcurrentAccess(t *testing.T) {
	resolver := NewDNSResolver(10, 10*time.Second)
	domains := []string{"google.com", "github.com", "stackoverflow.com"}
	
	var wg sync.WaitGroup
	numGoroutines := 10
	
	wg.Add(numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			
			for _, domain := range domains {
				record := resolver.ResolveSingle(context.Background(), domain)
				if record.Error != nil {
					t.Errorf("Goroutine %d: Error resolving %s: %v", id, domain, record.Error)
				}
			}
		}(i)
	}
	
	wg.Wait()
}

func TestClearCache(t *testing.T) {
	resolver := NewDNSResolver(5, 10*time.Second)
	
	resolver.ResolveSingle(context.Background(), "google.com")
	
	stats := resolver.GetStats()
	if stats.CacheHits == 0 {
		t.Error("Expected cache to have entries")
	}
	
	resolver.ClearCache()
	
	stats = resolver.GetStats()
	if stats.CacheHits != 0 {
		t.Error("Expected cache to be empty after clearing")
	}
}

func TestResolveMX(t *testing.T) {
	resolver := NewDNSResolver(5, 10*time.Second)
	
	mxRecords, err := resolver.ResolveMX(context.Background(), "google.com")
	
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	
	if len(mxRecords) == 0 {
		t.Error("Expected at least one MX record")
	}
}

func TestResolveTXT(t *testing.T) {
	resolver := NewDNSResolver(5, 10*time.Second)
	
	txtRecords, err := resolver.ResolveTXT(context.Background(), "google.com")
	
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	
	if len(txtRecords) == 0 {
		t.Log("No TXT records found for google.com")
	}
}

func TestResolveCNAME(t *testing.T) {
	resolver := NewDNSResolver(5, 10*time.Second)
	
	cname, err := resolver.ResolveCNAME(context.Background(), "www.github.com")
	
	if err != nil {
		t.Logf("CNAME resolution failed (expected for some domains): %v", err)
	} else {
		t.Logf("CNAME for www.github.com: %s", cname)
	}
}

func TestTimeout(t *testing.T) {
	resolver := NewDNSResolver(5, 1*time.Millisecond)
	
	record := resolver.ResolveSingle(context.Background(), "google.com")
	
	if record.Error == nil {
		t.Log("Resolution completed within timeout")
	} else {
		t.Logf("Resolution timed out as expected: %v", record.Error)
	}
}

func BenchmarkResolveSingle(b *testing.B) {
	resolver := NewDNSResolver(5, 10*time.Second)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resolver.ResolveSingle(context.Background(), "google.com")
	}
}

func BenchmarkResolveMultiple(b *testing.B) {
	resolver := NewDNSResolver(10, 10*time.Second)
	domains := []string{"google.com", "github.com", "stackoverflow.com"}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resolver.ResolveMultiple(context.Background(), domains)
	}
}

func BenchmarkResolveWithRace(b *testing.B) {
	resolver := NewDNSResolver(5, 10*time.Second)
	servers := []string{"8.8.8.8", "8.8.4.4", "1.1.1.1"}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resolver.ResolveWithRace(context.Background(), "google.com", servers)
	}
}