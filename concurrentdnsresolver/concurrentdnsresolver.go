package concurrentdnsresolver

import (
	"context"
	"fmt"
	"net"
	"sync"
	"time"
)

type DNSRecord struct {
	Domain  string
	IP      string
	Type    string
	TTL     time.Duration
	Error   error
	Latency time.Duration
}

type DNSResolver struct {
	cache      map[string]*DNSRecord
	cacheMu    sync.RWMutex
	maxWorkers int
	timeout    time.Duration
}

func NewDNSResolver(maxWorkers int, timeout time.Duration) *DNSResolver {
	return &DNSResolver{
		cache:      make(map[string]*DNSRecord),
		maxWorkers: maxWorkers,
		timeout:    timeout,
	}
}

func (r *DNSResolver) ResolveMultiple(ctx context.Context, domains []string) []*DNSRecord {
	results := make([]*DNSRecord, len(domains))
	jobs := make(chan job, len(domains))
	resultChan := make(chan result, len(domains))
	
	for i := 0; i < r.maxWorkers; i++ {
		go r.worker(ctx, jobs, resultChan)
	}
	
	for i, domain := range domains {
		jobs <- job{domain: domain, index: i}
	}
	close(jobs)
	
	for i := 0; i < len(domains); i++ {
		res := <-resultChan
		results[res.index] = res.record
	}
	
	return results
}

func (r *DNSResolver) ResolveSingle(ctx context.Context, domain string) *DNSRecord {
	if record := r.getCachedRecord(domain); record != nil {
		return record
	}
	
	record := r.performLookup(ctx, domain)
	r.cacheRecord(domain, record)
	return record
}

func (r *DNSResolver) ResolveWithRace(ctx context.Context, domain string, servers []string) *DNSRecord {
	results := make(chan *DNSRecord, len(servers))
	
	for _, server := range servers {
		go func(srv string) {
			results <- r.resolveWithServer(ctx, domain, srv)
		}(server)
	}
	
	select {
	case result := <-results:
		return result
	case <-ctx.Done():
		return &DNSRecord{
			Domain: domain,
			Error:  ctx.Err(),
		}
	}
}

func (r *DNSResolver) StartAsyncResolver(ctx context.Context) (<-chan DNSQuery, chan<- *DNSRecord) {
	queryQueue := make(chan DNSQuery, 100)
	resultChan := make(chan *DNSRecord, 100)
	
	for i := 0; i < r.maxWorkers; i++ {
		go r.asyncWorker(ctx, queryQueue, resultChan)
	}
	
	return queryQueue, resultChan
}

func (r *DNSResolver) BulkResolveWithBatching(ctx context.Context, domains []string, batchSize int) []*DNSRecord {
	results := make([]*DNSRecord, len(domains))
	var wg sync.WaitGroup
	
	for i := 0; i < len(domains); i += batchSize {
		end := i + batchSize
		if end > len(domains) {
			end = len(domains)
		}
		
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			batch := domains[start:end]
			batchResults := r.ResolveMultiple(ctx, batch)
			copy(results[start:end], batchResults)
		}(i, end)
	}
	
	wg.Wait()
	return results
}

func (r *DNSResolver) getCachedRecord(domain string) *DNSRecord {
	r.cacheMu.RLock()
	defer r.cacheMu.RUnlock()
	
	if record, exists := r.cache[domain]; exists {
		if time.Since(time.Now().Add(-record.TTL)) < 0 {
			return record
		}
		delete(r.cache, domain)
	}
	return nil
}

func (r *DNSResolver) cacheRecord(domain string, record *DNSRecord) {
	if record.Error != nil {
		return
	}
	
	r.cacheMu.Lock()
	defer r.cacheMu.Unlock()
	r.cache[domain] = record
}

func (r *DNSResolver) performLookup(ctx context.Context, domain string) *DNSRecord {
	start := time.Now()
	
	timeoutCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	
	resolver := &net.Resolver{}
	ips, err := resolver.LookupIPAddr(timeoutCtx, domain)
	
	latency := time.Since(start)
	
	if err != nil {
		return &DNSRecord{
			Domain:  domain,
			Error:   err,
			Latency: latency,
		}
	}
	
	if len(ips) == 0 {
		return &DNSRecord{
			Domain:  domain,
			Error:   fmt.Errorf("no IP addresses found for domain %s", domain),
			Latency: latency,
		}
	}
	
	return &DNSRecord{
		Domain:  domain,
		IP:      ips[0].IP.String(),
		Type:    getRecordType(ips[0].IP),
		TTL:     time.Hour,
		Latency: latency,
	}
}

func (r *DNSResolver) resolveWithServer(ctx context.Context, domain string, server string) *DNSRecord {
	start := time.Now()
	
	timeoutCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	
	resolver := &net.Resolver{
		PreferGo: true,
		Dial: func(ctx context.Context, network, address string) (net.Conn, error) {
			d := net.Dialer{
				Timeout: r.timeout,
			}
			return d.DialContext(ctx, network, server+":53")
		},
	}
	
	ips, err := resolver.LookupIPAddr(timeoutCtx, domain)
	latency := time.Since(start)
	
	if err != nil {
		return &DNSRecord{
			Domain:  domain,
			Error:   err,
			Latency: latency,
		}
	}
	
	if len(ips) == 0 {
		return &DNSRecord{
			Domain:  domain,
			Error:   fmt.Errorf("no IP addresses found for domain %s", domain),
			Latency: latency,
		}
	}
	
	return &DNSRecord{
		Domain:  domain,
		IP:      ips[0].IP.String(),
		Type:    getRecordType(ips[0].IP),
		TTL:     time.Hour,
		Latency: latency,
	}
}

func (r *DNSResolver) worker(ctx context.Context, jobs <-chan job, results chan<- result) {
	for {
		select {
		case job, ok := <-jobs:
			if !ok {
				return
			}
			record := r.ResolveSingle(ctx, job.domain)
			results <- result{record: record, index: job.index}
		case <-ctx.Done():
			return
		}
	}
}

func (r *DNSResolver) asyncWorker(ctx context.Context, queries <-chan DNSQuery, results chan<- *DNSRecord) {
	for {
		select {
		case query, ok := <-queries:
			if !ok {
				return
			}
			
			var record *DNSRecord
			if query.UseRacing && len(query.Servers) > 0 {
				record = r.ResolveWithRace(ctx, query.Domain, query.Servers)
			} else {
				record = r.ResolveSingle(ctx, query.Domain)
			}
			
			select {
			case results <- record:
			case <-ctx.Done():
				return
			}
		case <-ctx.Done():
			return
		}
	}
}

func getRecordType(ip net.IP) string {
	if ip.To4() != nil {
		return "A"
	}
	return "AAAA"
}

type job struct {
	domain string
	index  int
}

type result struct {
	record *DNSRecord
	index  int
}

type DNSQuery struct {
	Domain    string
	Servers   []string
	UseRacing bool
	Timeout   time.Duration
}

type DNSStats struct {
	TotalQueries    int
	SuccessfulLookups int
	FailedLookups   int
	CacheHits       int
	AverageLatency  time.Duration
}

func (r *DNSResolver) GetStats() DNSStats {
	r.cacheMu.RLock()
	defer r.cacheMu.RUnlock()
	
	return DNSStats{
		CacheHits: len(r.cache),
	}
}

func (r *DNSResolver) ClearCache() {
	r.cacheMu.Lock()
	defer r.cacheMu.Unlock()
	r.cache = make(map[string]*DNSRecord)
}

func (r *DNSResolver) ResolveWithRetry(ctx context.Context, domain string, maxRetries int) *DNSRecord {
	var lastRecord *DNSRecord
	
	for i := 0; i <= maxRetries; i++ {
		record := r.ResolveSingle(ctx, domain)
		if record.Error == nil {
			return record
		}
		
		lastRecord = record
		
		if i < maxRetries {
			select {
			case <-time.After(time.Millisecond * 100 * time.Duration(i+1)):
			case <-ctx.Done():
				return &DNSRecord{
					Domain: domain,
					Error:  ctx.Err(),
				}
			}
		}
	}
	
	return lastRecord
}

func (r *DNSResolver) ResolveMX(ctx context.Context, domain string) ([]*net.MX, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	
	resolver := &net.Resolver{}
	return resolver.LookupMX(timeoutCtx, domain)
}

func (r *DNSResolver) ResolveTXT(ctx context.Context, domain string) ([]string, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	
	resolver := &net.Resolver{}
	return resolver.LookupTXT(timeoutCtx, domain)
}

func (r *DNSResolver) ResolveCNAME(ctx context.Context, domain string) (string, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	
	resolver := &net.Resolver{}
	return resolver.LookupCNAME(timeoutCtx, domain)
}