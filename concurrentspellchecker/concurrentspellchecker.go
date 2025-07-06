package concurrentspellchecker

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
)

// SpellCheckAlgorithm defines different spell checking algorithms
type SpellCheckAlgorithm int

const (
	LevenshteinDistance SpellCheckAlgorithm = iota
	JaroWinklerDistance
	SoundexMatching
	MetaphoneMatching
	CombinedAlgorithm
)

// SuggestionMethod defines suggestion generation methods
type SuggestionMethod int

const (
	EditDistance SuggestionMethod = iota
	PhoneticSimilarity
	FrequencyBased
	ContextualSuggestions
	HybridSuggestions
)

// SpellCheckerConfig contains configuration for the spell checker
type SpellCheckerConfig struct {
	Algorithm           SpellCheckAlgorithm
	SuggestionMethod    SuggestionMethod
	NumWorkers          int
	MaxSuggestions      int
	MinWordLength       int
	MaxEditDistance     int
	CaseSensitive       bool
	EnableCaching       bool
	CacheSize           int
	CustomDictionaries  []string
	IgnoreNumbers       bool
	IgnoreCapitalized   bool
	EnableContextCheck  bool
	ContextWindowSize   int
	MinSuggestionScore  float64
	EnablePhonetics     bool
	EnableStatistics    bool
}

// Dictionary represents a word dictionary with frequency information
type Dictionary struct {
	words      map[string]int
	soundex    map[string][]string
	metaphone  map[string][]string
	totalWords int64
	mutex      sync.RWMutex
}

// WordInfo contains information about a word being checked
type WordInfo struct {
	Word       string
	Position   int
	Line       int
	Column     int
	Context    []string
	IsValid    bool
	Confidence float64
}

// SpellCheckResult contains the result of spell checking a word
type SpellCheckResult struct {
	Word        string
	IsCorrect   bool
	Suggestions []Suggestion
	Confidence  float64
	Algorithm   SpellCheckAlgorithm
	Position    int
	Line        int
	Column      int
	ProcessTime time.Duration
}

// Suggestion represents a spelling suggestion
type Suggestion struct {
	Word       string
	Score      float64
	Algorithm  string
	Frequency  int
	EditOps    []EditOperation
}

// EditOperation represents an edit operation for suggestions
type EditOperation struct {
	Type     string // insert, delete, substitute, transpose
	Position int
	From     string
	To       string
}

// DocumentAnalysis contains comprehensive analysis of a document
type DocumentAnalysis struct {
	TotalWords        int
	UniqueWords       int
	MisspelledWords   int
	CorrectionsMade   int
	ProcessingTime    time.Duration
	ErrorsByCategory  map[string]int
	TopMisspellings   []WordFrequency
	SuggestionStats   SuggestionStatistics
	PerformanceStats  PerformanceStatistics
}

// WordFrequency represents word frequency information
type WordFrequency struct {
	Word      string
	Count     int
	Frequency float64
}

// SuggestionStatistics contains statistics about suggestions
type SuggestionStatistics struct {
	TotalSuggestionsGenerated int
	AverageScore              float64
	AlgorithmUsage            map[string]int
	ResponseTimes             map[string]time.Duration
}

// PerformanceStatistics contains performance metrics
type PerformanceStatistics struct {
	WordsPerSecond    float64
	AverageLatency    time.Duration
	CacheHitRate      float64
	WorkerUtilization []float64
	MemoryUsage       int64
}

// SpellChecker is the main spell checker instance
type SpellChecker struct {
	config         SpellCheckerConfig
	dictionary     *Dictionary
	cache          *SpellCheckCache
	workers        []*SpellCheckWorker
	wordQueue      chan WordCheckTask
	resultQueue    chan SpellCheckResult
	stats          *Statistics
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup
	mutex          sync.RWMutex
	running        bool
}

// SpellCheckWorker represents a worker for parallel spell checking
type SpellCheckWorker struct {
	id            int
	checker       *SpellChecker
	taskQueue     chan WordCheckTask
	resultQueue   chan SpellCheckResult
	stats         *WorkerStats
	phonetics     *PhoneticProcessor
	editDistance  *EditDistanceCalculator
	ctx           context.Context
}

// WordCheckTask represents a task for checking a word
type WordCheckTask struct {
	Word     string
	Context  []string
	Position int
	Line     int
	Column   int
	TaskID   string
}

// SpellCheckCache provides caching for spell check results
type SpellCheckCache struct {
	cache map[string]*CacheEntry
	mutex sync.RWMutex
	size  int
	hits  int64
	misses int64
}

// CacheEntry represents a cached spell check result
type CacheEntry struct {
	Result    SpellCheckResult
	Timestamp time.Time
	AccessCount int64
}

// Statistics contains spell checker statistics
type Statistics struct {
	TotalWords      int64
	CorrectWords    int64
	MisspelledWords int64
	TotalTime       time.Duration
	CacheHits       int64
	CacheMisses     int64
	WorkerStats     []*WorkerStats
	mutex           sync.RWMutex
}

// WorkerStats contains statistics for individual workers
type WorkerStats struct {
	WorkerID      int
	WordsChecked  int64
	ProcessingTime time.Duration
	Utilization   float64
	ErrorCount    int64
}

// PhoneticProcessor handles phonetic matching algorithms
type PhoneticProcessor struct {
	soundexCache   map[string]string
	metaphoneCache map[string]string
	mutex          sync.RWMutex
}

// EditDistanceCalculator calculates edit distances between words
type EditDistanceCalculator struct {
	matrix [][]int
	mutex  sync.Mutex
}

// NewSpellChecker creates a new spell checker instance
func NewSpellChecker(config SpellCheckerConfig) *SpellChecker {
	if config.NumWorkers <= 0 {
		config.NumWorkers = 4
	}
	if config.MaxSuggestions <= 0 {
		config.MaxSuggestions = 5
	}
	if config.MinWordLength <= 0 {
		config.MinWordLength = 2
	}
	if config.MaxEditDistance <= 0 {
		config.MaxEditDistance = 2
	}
	if config.CacheSize <= 0 {
		config.CacheSize = 10000
	}
	if config.ContextWindowSize <= 0 {
		config.ContextWindowSize = 5
	}
	if config.MinSuggestionScore == 0 {
		config.MinSuggestionScore = 0.5
	}

	ctx, cancel := context.WithCancel(context.Background())

	checker := &SpellChecker{
		config:      config,
		dictionary:  NewDictionary(),
		wordQueue:   make(chan WordCheckTask, config.NumWorkers*2),
		resultQueue: make(chan SpellCheckResult, config.NumWorkers*2),
		stats:       NewStatistics(config.NumWorkers),
		ctx:         ctx,
		cancel:      cancel,
	}

	if config.EnableCaching {
		checker.cache = NewSpellCheckCache(config.CacheSize)
	}

	checker.workers = make([]*SpellCheckWorker, config.NumWorkers)
	for i := 0; i < config.NumWorkers; i++ {
		checker.workers[i] = NewSpellCheckWorker(i, checker)
	}

	return checker
}

// NewDictionary creates a new dictionary instance
func NewDictionary() *Dictionary {
	return &Dictionary{
		words:     make(map[string]int),
		soundex:   make(map[string][]string),
		metaphone: make(map[string][]string),
	}
}

// NewSpellCheckCache creates a new spell check cache
func NewSpellCheckCache(size int) *SpellCheckCache {
	return &SpellCheckCache{
		cache: make(map[string]*CacheEntry),
		size:  size,
	}
}

// NewStatistics creates a new statistics instance
func NewStatistics(numWorkers int) *Statistics {
	stats := &Statistics{
		WorkerStats: make([]*WorkerStats, numWorkers),
	}
	for i := 0; i < numWorkers; i++ {
		stats.WorkerStats[i] = &WorkerStats{WorkerID: i}
	}
	return stats
}

// NewSpellCheckWorker creates a new spell check worker
func NewSpellCheckWorker(id int, checker *SpellChecker) *SpellCheckWorker {
	return &SpellCheckWorker{
		id:           id,
		checker:      checker,
		taskQueue:    checker.wordQueue,
		resultQueue:  checker.resultQueue,
		stats:        checker.stats.WorkerStats[id],
		phonetics:    NewPhoneticProcessor(),
		editDistance: NewEditDistanceCalculator(),
		ctx:          checker.ctx,
	}
}

// NewPhoneticProcessor creates a new phonetic processor
func NewPhoneticProcessor() *PhoneticProcessor {
	return &PhoneticProcessor{
		soundexCache:   make(map[string]string),
		metaphoneCache: make(map[string]string),
	}
}

// NewEditDistanceCalculator creates a new edit distance calculator
func NewEditDistanceCalculator() *EditDistanceCalculator {
	return &EditDistanceCalculator{
		matrix: make([][]int, 100),
	}
}

// LoadDictionary loads words from a reader into the dictionary
func (d *Dictionary) LoadDictionary(reader io.Reader) error {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.Fields(line)
		word := strings.ToLower(parts[0])
		frequency := 1

		if len(parts) > 1 {
			if freq, err := parseFrequency(parts[1]); err == nil {
				frequency = freq
			}
		}

		d.words[word] = frequency
		d.totalWords++

		// Generate phonetic representations
		soundex := generateSoundex(word)
		metaphone := generateMetaphone(word)

		d.soundex[soundex] = append(d.soundex[soundex], word)
		d.metaphone[metaphone] = append(d.metaphone[metaphone], word)
	}

	return scanner.Err()
}

// LoadFromWordList loads words from a slice
func (d *Dictionary) LoadFromWordList(words []string) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	for _, word := range words {
		word = strings.ToLower(strings.TrimSpace(word))
		if word == "" {
			continue
		}

		d.words[word] = 1
		d.totalWords++

		soundex := generateSoundex(word)
		metaphone := generateMetaphone(word)

		d.soundex[soundex] = append(d.soundex[soundex], word)
		d.metaphone[metaphone] = append(d.metaphone[metaphone], word)
	}
}

// ContainsWord checks if a word exists in the dictionary
func (d *Dictionary) ContainsWord(word string) bool {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	_, exists := d.words[strings.ToLower(word)]
	return exists
}

// GetWordFrequency returns the frequency of a word
func (d *Dictionary) GetWordFrequency(word string) int {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	return d.words[strings.ToLower(word)]
}

// GetPhoneticMatches returns words with similar phonetic representation
func (d *Dictionary) GetPhoneticMatches(word string) []string {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	var matches []string
	soundex := generateSoundex(word)
	metaphone := generateMetaphone(word)

	if words, exists := d.soundex[soundex]; exists {
		matches = append(matches, words...)
	}

	if words, exists := d.metaphone[metaphone]; exists {
		for _, w := range words {
			if !contains(matches, w) {
				matches = append(matches, w)
			}
		}
	}

	return matches
}

// Start begins the spell checker workers
func (sc *SpellChecker) Start() error {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()

	if sc.running {
		return errors.New("spell checker is already running")
	}

	for _, worker := range sc.workers {
		sc.wg.Add(1)
		go worker.Start()
	}

	sc.running = true
	return nil
}

// Stop stops the spell checker workers
func (sc *SpellChecker) Stop() error {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()

	if !sc.running {
		return errors.New("spell checker is not running")
	}

	sc.cancel()
	close(sc.wordQueue)

	sc.wg.Wait()
	sc.running = false
	return nil
}

// CheckWord checks a single word for spelling
func (sc *SpellChecker) CheckWord(word string) SpellCheckResult {
	if sc.cache != nil {
		if cached := sc.cache.Get(word); cached != nil {
			return *cached
		}
	}

	start := time.Now()
	result := sc.performSpellCheck(word, nil, 0, 0, 0)
	result.ProcessTime = time.Since(start)

	if sc.cache != nil {
		sc.cache.Set(word, result)
	}

	sc.updateStatistics(result)
	return result
}

// CheckText checks all words in a text
func (sc *SpellChecker) CheckText(text string) ([]SpellCheckResult, error) {
	if !sc.running {
		if err := sc.Start(); err != nil {
			return nil, err
		}
		defer sc.Stop()
	}

	words := sc.tokenizeText(text)
	if len(words) == 0 {
		return nil, nil
	}

	results := make([]SpellCheckResult, 0, len(words))
	taskQueue := make(chan WordCheckTask, len(words))
	resultChan := make(chan SpellCheckResult, len(words))

	// Send tasks
	go func() {
		defer close(taskQueue)
		for _, wordInfo := range words {
			task := WordCheckTask{
				Word:     wordInfo.Word,
				Context:  wordInfo.Context,
				Position: wordInfo.Position,
				Line:     wordInfo.Line,
				Column:   wordInfo.Column,
				TaskID:   fmt.Sprintf("%d-%d", wordInfo.Line, wordInfo.Column),
			}
			taskQueue <- task
		}
	}()

	// Process tasks
	go func() {
		defer close(resultChan)
		for task := range taskQueue {
			select {
			case sc.wordQueue <- task:
			case <-sc.ctx.Done():
				return
			}
		}
	}()

	// Collect results
	for i := 0; i < len(words); i++ {
		select {
		case result := <-sc.resultQueue:
			results = append(results, result)
		case <-sc.ctx.Done():
			return nil, sc.ctx.Err()
		}
	}

	// Sort results by position
	sort.Slice(results, func(i, j int) bool {
		if results[i].Line == results[j].Line {
			return results[i].Column < results[j].Column
		}
		return results[i].Line < results[j].Line
	})

	return results, nil
}

// CheckDocument performs comprehensive document analysis
func (sc *SpellChecker) CheckDocument(reader io.Reader) (*DocumentAnalysis, error) {
	start := time.Now()
	
	content, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	results, err := sc.CheckText(string(content))
	if err != nil {
		return nil, err
	}

	analysis := &DocumentAnalysis{
		ProcessingTime:   time.Since(start),
		ErrorsByCategory: make(map[string]int),
		SuggestionStats:  SuggestionStatistics{AlgorithmUsage: make(map[string]int), ResponseTimes: make(map[string]time.Duration)},
	}

	uniqueWords := make(map[string]bool)
	misspelledCount := make(map[string]int)
	totalSuggestions := 0
	var totalScore float64

	for _, result := range results {
		analysis.TotalWords++
		uniqueWords[result.Word] = true

		if !result.IsCorrect {
			analysis.MisspelledWords++
			misspelledCount[result.Word]++
			
			// Categorize errors
			if len(result.Word) <= 3 {
				analysis.ErrorsByCategory["short_words"]++
			} else if hasNumbers(result.Word) {
				analysis.ErrorsByCategory["contains_numbers"]++
			} else if isCapitalized(result.Word) {
				analysis.ErrorsByCategory["capitalized"]++
			} else {
				analysis.ErrorsByCategory["general"]++
			}

			// Suggestion statistics
			totalSuggestions += len(result.Suggestions)
			for _, suggestion := range result.Suggestions {
				totalScore += suggestion.Score
			}

			analysis.SuggestionStats.AlgorithmUsage[result.Algorithm.String()]++
			analysis.SuggestionStats.ResponseTimes[result.Algorithm.String()] += result.ProcessTime
		}
	}

	analysis.UniqueWords = len(uniqueWords)
	
	// Calculate top misspellings
	type wordCount struct {
		word  string
		count int
	}
	var wordCounts []wordCount
	for word, count := range misspelledCount {
		wordCounts = append(wordCounts, wordCount{word, count})
	}
	sort.Slice(wordCounts, func(i, j int) bool {
		return wordCounts[i].count > wordCounts[j].count
	})

	maxTop := 10
	if len(wordCounts) < maxTop {
		maxTop = len(wordCounts)
	}
	
	for i := 0; i < maxTop; i++ {
		analysis.TopMisspellings = append(analysis.TopMisspellings, WordFrequency{
			Word:      wordCounts[i].word,
			Count:     wordCounts[i].count,
			Frequency: float64(wordCounts[i].count) / float64(analysis.TotalWords),
		})
	}

	// Complete suggestion statistics
	analysis.SuggestionStats.TotalSuggestionsGenerated = totalSuggestions
	if totalSuggestions > 0 {
		analysis.SuggestionStats.AverageScore = totalScore / float64(totalSuggestions)
	}

	// Performance statistics
	if sc.stats != nil {
		stats := sc.GetStatistics()
		analysis.PerformanceStats = PerformanceStatistics{
			WordsPerSecond: float64(stats.TotalWords) / analysis.ProcessingTime.Seconds(),
			AverageLatency: time.Duration(stats.TotalTime.Nanoseconds() / stats.TotalWords),
			CacheHitRate:   float64(stats.CacheHits) / float64(stats.CacheHits+stats.CacheMisses),
		}

		for _, workerStat := range stats.WorkerStats {
			analysis.PerformanceStats.WorkerUtilization = append(
				analysis.PerformanceStats.WorkerUtilization, 
				workerStat.Utilization,
			)
		}
	}

	return analysis, nil
}

// Start worker processing
func (w *SpellCheckWorker) Start() {
	defer w.checker.wg.Done()

	for {
		select {
		case task, ok := <-w.taskQueue:
			if !ok {
				return
			}
			
			start := time.Now()
			result := w.processTask(task)
			result.ProcessTime = time.Since(start)

			atomic.AddInt64(&w.stats.WordsChecked, 1)
			w.stats.ProcessingTime += result.ProcessTime

			select {
			case w.resultQueue <- result:
			case <-w.ctx.Done():
				return
			}

		case <-w.ctx.Done():
			return
		}
	}
}

// processTask processes a single word check task
func (w *SpellCheckWorker) processTask(task WordCheckTask) SpellCheckResult {
	return w.checker.performSpellCheck(task.Word, task.Context, task.Position, task.Line, task.Column)
}

// performSpellCheck performs the actual spell checking logic
func (sc *SpellChecker) performSpellCheck(word string, context []string, position, line, column int) SpellCheckResult {
	result := SpellCheckResult{
		Word:      word,
		Position:  position,
		Line:      line,
		Column:    column,
		Algorithm: sc.config.Algorithm,
	}

	// Preprocessing
	normalizedWord := sc.normalizeWord(word)
	if normalizedWord == "" {
		result.IsCorrect = true
		result.Confidence = 1.0
		return result
	}

	// Check if word exists in dictionary
	if sc.dictionary.ContainsWord(normalizedWord) {
		result.IsCorrect = true
		result.Confidence = 1.0
		return result
	}

	// Word is misspelled, generate suggestions
	result.IsCorrect = false
	result.Suggestions = sc.generateSuggestions(normalizedWord, context)
	
	if len(result.Suggestions) > 0 {
		result.Confidence = result.Suggestions[0].Score
	} else {
		result.Confidence = 0.0
	}

	return result
}

// normalizeWord normalizes a word according to configuration
func (sc *SpellChecker) normalizeWord(word string) string {
	// Remove punctuation
	word = strings.Trim(word, ".,!?;:\"'()[]{}/*&^%$#@~`+=<>")
	
	if len(word) < sc.config.MinWordLength {
		return ""
	}

	if sc.config.IgnoreNumbers && hasNumbers(word) {
		return ""
	}

	if sc.config.IgnoreCapitalized && isCapitalized(word) {
		return ""
	}

	if !sc.config.CaseSensitive {
		word = strings.ToLower(word)
	}

	return word
}

// generateSuggestions generates spelling suggestions for a misspelled word
func (sc *SpellChecker) generateSuggestions(word string, context []string) []Suggestion {
	var suggestions []Suggestion

	switch sc.config.SuggestionMethod {
	case EditDistance:
		suggestions = sc.generateEditDistanceSuggestions(word)
	case PhoneticSimilarity:
		suggestions = sc.generatePhoneticSuggestions(word)
	case FrequencyBased:
		suggestions = sc.generateFrequencyBasedSuggestions(word)
	case ContextualSuggestions:
		suggestions = sc.generateContextualSuggestions(word, context)
	case HybridSuggestions:
		suggestions = sc.generateHybridSuggestions(word, context)
	default:
		suggestions = sc.generateEditDistanceSuggestions(word)
	}

	// Filter and sort suggestions
	filtered := make([]Suggestion, 0)
	for _, suggestion := range suggestions {
		if suggestion.Score >= sc.config.MinSuggestionScore {
			filtered = append(filtered, suggestion)
		}
	}

	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].Score > filtered[j].Score
	})

	if len(filtered) > sc.config.MaxSuggestions {
		filtered = filtered[:sc.config.MaxSuggestions]
	}

	return filtered
}

// generateEditDistanceSuggestions generates suggestions based on edit distance
func (sc *SpellChecker) generateEditDistanceSuggestions(word string) []Suggestion {
	var suggestions []Suggestion
	calculator := NewEditDistanceCalculator()

	sc.dictionary.mutex.RLock()
	defer sc.dictionary.mutex.RUnlock()

	for dictWord, frequency := range sc.dictionary.words {
		distance := calculator.CalculateLevenshtein(word, dictWord)
		if distance <= sc.config.MaxEditDistance {
			score := 1.0 - float64(distance)/float64(max(len(word), len(dictWord)))
			
			// Boost score based on frequency
			if frequency > 1 {
				score += math.Log(float64(frequency)) * 0.1
			}

			suggestions = append(suggestions, Suggestion{
				Word:      dictWord,
				Score:     score,
				Algorithm: "EditDistance",
				Frequency: frequency,
				EditOps:   calculator.CalculateEditOperations(word, dictWord),
			})
		}
	}

	return suggestions
}

// generatePhoneticSuggestions generates suggestions based on phonetic similarity
func (sc *SpellChecker) generatePhoneticSuggestions(word string) []Suggestion {
	var suggestions []Suggestion
	
	phoneticMatches := sc.dictionary.GetPhoneticMatches(word)
	calculator := NewEditDistanceCalculator()

	for _, match := range phoneticMatches {
		if match == word {
			continue
		}

		// Calculate combined score based on phonetic similarity and edit distance
		editDistance := calculator.CalculateLevenshtein(word, match)
		phoneticScore := 0.8 // Base phonetic similarity score
		editScore := 1.0 - float64(editDistance)/float64(max(len(word), len(match)))
		
		combinedScore := (phoneticScore + editScore) / 2.0
		frequency := sc.dictionary.GetWordFrequency(match)

		suggestions = append(suggestions, Suggestion{
			Word:      match,
			Score:     combinedScore,
			Algorithm: "Phonetic",
			Frequency: frequency,
		})
	}

	return suggestions
}

// generateFrequencyBasedSuggestions generates suggestions prioritizing frequent words
func (sc *SpellChecker) generateFrequencyBasedSuggestions(word string) []Suggestion {
	suggestions := sc.generateEditDistanceSuggestions(word)
	
	// Boost scores based on word frequency
	for i := range suggestions {
		if suggestions[i].Frequency > 0 {
			frequencyBoost := math.Log(float64(suggestions[i].Frequency)) / 10.0
			suggestions[i].Score = math.Min(1.0, suggestions[i].Score+frequencyBoost)
		}
	}

	return suggestions
}

// generateContextualSuggestions generates suggestions based on context
func (sc *SpellChecker) generateContextualSuggestions(word string, context []string) []Suggestion {
	baseSuggestions := sc.generateEditDistanceSuggestions(word)
	
	if len(context) == 0 {
		return baseSuggestions
	}

	// Boost scores for words that commonly appear in similar contexts
	for i := range baseSuggestions {
		contextScore := sc.calculateContextScore(baseSuggestions[i].Word, context)
		baseSuggestions[i].Score = (baseSuggestions[i].Score + contextScore) / 2.0
	}

	return baseSuggestions
}

// generateHybridSuggestions combines multiple suggestion methods
func (sc *SpellChecker) generateHybridSuggestions(word string, context []string) []Suggestion {
	editSuggestions := sc.generateEditDistanceSuggestions(word)
	phoneticSuggestions := sc.generatePhoneticSuggestions(word)
	frequencySuggestions := sc.generateFrequencyBasedSuggestions(word)
	
	// Combine and deduplicate suggestions
	suggestionMap := make(map[string]Suggestion)
	
	for _, s := range editSuggestions {
		suggestionMap[s.Word] = s
	}
	
	for _, s := range phoneticSuggestions {
		if existing, exists := suggestionMap[s.Word]; exists {
			existing.Score = (existing.Score + s.Score) / 2.0
			suggestionMap[s.Word] = existing
		} else {
			suggestionMap[s.Word] = s
		}
	}
	
	for _, s := range frequencySuggestions {
		if existing, exists := suggestionMap[s.Word]; exists {
			existing.Score = (existing.Score + s.Score) / 2.0
			suggestionMap[s.Word] = existing
		} else {
			suggestionMap[s.Word] = s
		}
	}

	var combined []Suggestion
	for _, suggestion := range suggestionMap {
		combined = append(combined, suggestion)
	}

	return combined
}

// calculateContextScore calculates relevance score based on context
func (sc *SpellChecker) calculateContextScore(word string, context []string) float64 {
	if len(context) == 0 {
		return 0.5
	}

	// Simple context scoring - can be enhanced with n-gram models
	score := 0.5
	for _, contextWord := range context {
		if strings.Contains(contextWord, word[:min(len(word), 3)]) {
			score += 0.1
		}
	}

	return math.Min(1.0, score)
}

// tokenizeText splits text into words with position information
func (sc *SpellChecker) tokenizeText(text string) []WordInfo {
	var words []WordInfo
	re := regexp.MustCompile(`\b\w+\b`)
	
	lines := strings.Split(text, "\n")
	for lineNum, line := range lines {
		matches := re.FindAllStringIndex(line, -1)
		for _, match := range matches {
			word := line[match[0]:match[1]]
			
			// Extract context
			context := sc.extractContext(lines, lineNum, match[0], match[1])
			
			words = append(words, WordInfo{
				Word:     word,
				Position: match[0],
				Line:     lineNum + 1,
				Column:   match[0] + 1,
				Context:  context,
			})
		}
	}
	
	return words
}

// extractContext extracts context words around the current position
func (sc *SpellChecker) extractContext(lines []string, lineNum, start, end int) []string {
	var context []string
	windowSize := sc.config.ContextWindowSize
	
	// Extract words from the same line
	line := lines[lineNum]
	re := regexp.MustCompile(`\b\w+\b`)
	matches := re.FindAllStringIndex(line, -1)
	
	for _, match := range matches {
		if match[1] <= start || match[0] >= end {
			word := line[match[0]:match[1]]
			context = append(context, word)
		}
	}
	
	// Extract from adjacent lines if needed
	if len(context) < windowSize {
		for i := max(0, lineNum-1); i <= min(len(lines)-1, lineNum+1); i++ {
			if i == lineNum {
				continue
			}
			lineMatches := re.FindAllString(lines[i], -1)
			context = append(context, lineMatches...)
		}
	}
	
	// Limit context size
	if len(context) > windowSize {
		context = context[:windowSize]
	}
	
	return context
}

// Cache operations
func (c *SpellCheckCache) Get(key string) *SpellCheckResult {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if entry, exists := c.cache[key]; exists {
		atomic.AddInt64(&entry.AccessCount, 1)
		atomic.AddInt64(&c.hits, 1)
		return &entry.Result
	}

	atomic.AddInt64(&c.misses, 1)
	return nil
}

func (c *SpellCheckCache) Set(key string, result SpellCheckResult) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if len(c.cache) >= c.size {
		// Simple LRU eviction - remove oldest entry
		var oldestKey string
		var oldestTime time.Time
		first := true

		for k, v := range c.cache {
			if first || v.Timestamp.Before(oldestTime) {
				oldestKey = k
				oldestTime = v.Timestamp
				first = false
			}
		}
		delete(c.cache, oldestKey)
	}

	c.cache[key] = &CacheEntry{
		Result:    result,
		Timestamp: time.Now(),
	}
}

// Statistics methods
func (sc *SpellChecker) updateStatistics(result SpellCheckResult) {
	if sc.stats == nil {
		return
	}

	sc.stats.mutex.Lock()
	defer sc.stats.mutex.Unlock()

	atomic.AddInt64(&sc.stats.TotalWords, 1)
	sc.stats.TotalTime += result.ProcessTime

	if result.IsCorrect {
		atomic.AddInt64(&sc.stats.CorrectWords, 1)
	} else {
		atomic.AddInt64(&sc.stats.MisspelledWords, 1)
	}

	if sc.cache != nil {
		sc.stats.CacheHits = atomic.LoadInt64(&sc.cache.hits)
		sc.stats.CacheMisses = atomic.LoadInt64(&sc.cache.misses)
	}
}

func (sc *SpellChecker) GetStatistics() *Statistics {
	if sc.stats == nil {
		return nil
	}

	sc.stats.mutex.RLock()
	defer sc.stats.mutex.RUnlock()

	// Create a copy of statistics
	statsCopy := &Statistics{
		TotalWords:      atomic.LoadInt64(&sc.stats.TotalWords),
		CorrectWords:    atomic.LoadInt64(&sc.stats.CorrectWords),
		MisspelledWords: atomic.LoadInt64(&sc.stats.MisspelledWords),
		TotalTime:       sc.stats.TotalTime,
		CacheHits:       atomic.LoadInt64(&sc.stats.CacheHits),
		CacheMisses:     atomic.LoadInt64(&sc.stats.CacheMisses),
		WorkerStats:     make([]*WorkerStats, len(sc.stats.WorkerStats)),
	}

	for i, ws := range sc.stats.WorkerStats {
		statsCopy.WorkerStats[i] = &WorkerStats{
			WorkerID:       ws.WorkerID,
			WordsChecked:   atomic.LoadInt64(&ws.WordsChecked),
			ProcessingTime: ws.ProcessingTime,
			Utilization:    ws.Utilization,
			ErrorCount:     atomic.LoadInt64(&ws.ErrorCount),
		}
	}

	return statsCopy
}

// Edit distance calculation methods
func (calc *EditDistanceCalculator) CalculateLevenshtein(s1, s2 string) int {
	calc.mutex.Lock()
	defer calc.mutex.Unlock()

	len1, len2 := len(s1), len(s2)
	
	// Ensure matrix is large enough
	if len(calc.matrix) <= len1 {
		calc.matrix = make([][]int, len1+1)
	}
	
	for i := range calc.matrix[:len1+1] {
		if len(calc.matrix[i]) <= len2 {
			calc.matrix[i] = make([]int, len2+1)
		}
	}

	// Initialize matrix
	for i := 0; i <= len1; i++ {
		calc.matrix[i][0] = i
	}
	for j := 0; j <= len2; j++ {
		calc.matrix[0][j] = j
	}

	// Fill matrix
	for i := 1; i <= len1; i++ {
		for j := 1; j <= len2; j++ {
			if s1[i-1] == s2[j-1] {
				calc.matrix[i][j] = calc.matrix[i-1][j-1]
			} else {
				calc.matrix[i][j] = min3(
					calc.matrix[i-1][j]+1,   // deletion
					calc.matrix[i][j-1]+1,   // insertion
					calc.matrix[i-1][j-1]+1, // substitution
				)
			}
		}
	}

	return calc.matrix[len1][len2]
}

func (calc *EditDistanceCalculator) CalculateEditOperations(s1, s2 string) []EditOperation {
	distance := calc.CalculateLevenshtein(s1, s2)
	if distance == 0 {
		return nil
	}

	var operations []EditOperation
	i, j := len(s1), len(s2)

	calc.mutex.Lock()
	defer calc.mutex.Unlock()

	for i > 0 || j > 0 {
		if i > 0 && j > 0 && calc.matrix[i][j] == calc.matrix[i-1][j-1] {
			// No operation needed
			i--
			j--
		} else if i > 0 && j > 0 && calc.matrix[i][j] == calc.matrix[i-1][j-1]+1 {
			// Substitution
			operations = append(operations, EditOperation{
				Type:     "substitute",
				Position: i - 1,
				From:     string(s1[i-1]),
				To:       string(s2[j-1]),
			})
			i--
			j--
		} else if i > 0 && calc.matrix[i][j] == calc.matrix[i-1][j]+1 {
			// Deletion
			operations = append(operations, EditOperation{
				Type:     "delete",
				Position: i - 1,
				From:     string(s1[i-1]),
			})
			i--
		} else if j > 0 && calc.matrix[i][j] == calc.matrix[i][j-1]+1 {
			// Insertion
			operations = append(operations, EditOperation{
				Type:     "insert",
				Position: i,
				To:       string(s2[j-1]),
			})
			j--
		}
	}

	// Reverse operations to get correct order
	for i := 0; i < len(operations)/2; i++ {
		operations[i], operations[len(operations)-1-i] = operations[len(operations)-1-i], operations[i]
	}

	return operations
}

// Phonetic algorithm implementations
func generateSoundex(word string) string {
	if len(word) == 0 {
		return ""
	}

	word = strings.ToUpper(word)
	result := string(word[0])

	// Soundex mapping
	soundexMap := map[rune]string{
		'B': "1", 'F': "1", 'P': "1", 'V': "1",
		'C': "2", 'G': "2", 'J': "2", 'K': "2", 'Q': "2", 'S': "2", 'X': "2", 'Z': "2",
		'D': "3", 'T': "3",
		'L': "4",
		'M': "5", 'N': "5",
		'R': "6",
	}

	prev := ""
	for _, char := range word[1:] {
		if code, exists := soundexMap[char]; exists && code != prev {
			result += code
			prev = code
			if len(result) == 4 {
				break
			}
		} else if !exists {
			prev = ""
		}
	}

	// Pad with zeros
	for len(result) < 4 {
		result += "0"
	}

	return result
}

func generateMetaphone(word string) string {
	if len(word) == 0 {
		return ""
	}

	word = strings.ToUpper(word)
	
	// Simplified Metaphone implementation
	// This is a basic version - a full implementation would be much more complex
	result := ""
	
	for i, char := range word {
		switch char {
		case 'A', 'E', 'I', 'O', 'U':
			if i == 0 {
				result += string(char)
			}
		case 'B':
			if i == len(word)-1 && word[i-1] == 'M' {
				// Silent B after M at end
			} else {
				result += "B"
			}
		case 'C':
			if i > 0 && word[i-1] == 'S' && (i+1 < len(word)) && (word[i+1] == 'H' || word[i+1] == 'I' || word[i+1] == 'E') {
				// SCH, SCI, SCE
				result += "K"
			} else if (i+1 < len(word)) && (word[i+1] == 'H') {
				result += "X"
			} else {
				result += "K"
			}
		case 'D':
			if (i+1 < len(word)) && word[i+1] == 'G' {
				result += "J"
			} else {
				result += "T"
			}
		case 'F':
			result += "F"
		case 'G':
			if (i+1 < len(word)) && word[i+1] == 'H' {
				result += "F"
			} else {
				result += "K"
			}
		case 'H':
			if i == 0 || isVowel(word[i-1]) {
				result += "H"
			}
		case 'J':
			result += "J"
		case 'K':
			if i == 0 || word[i-1] != 'C' {
				result += "K"
			}
		case 'L':
			result += "L"
		case 'M':
			result += "M"
		case 'N':
			result += "N"
		case 'P':
			if (i+1 < len(word)) && word[i+1] == 'H' {
				result += "F"
			} else {
				result += "P"
			}
		case 'Q':
			result += "K"
		case 'R':
			result += "R"
		case 'S':
			if (i+1 < len(word)) && word[i+1] == 'H' {
				result += "X"
			} else {
				result += "S"
			}
		case 'T':
			if (i+1 < len(word)) && word[i+1] == 'H' {
				result += "0"
			} else {
				result += "T"
			}
		case 'V':
			result += "F"
		case 'W', 'Y':
			if i == 0 || isVowel(word[i-1]) {
				result += string(char)
			}
		case 'X':
			result += "KS"
		case 'Z':
			result += "S"
		}
	}

	return result
}

// Utility functions
func (algo SpellCheckAlgorithm) String() string {
	switch algo {
	case LevenshteinDistance:
		return "LevenshteinDistance"
	case JaroWinklerDistance:
		return "JaroWinklerDistance"
	case SoundexMatching:
		return "SoundexMatching"
	case MetaphoneMatching:
		return "MetaphoneMatching"
	case CombinedAlgorithm:
		return "CombinedAlgorithm"
	default:
		return "Unknown"
	}
}

func parseFrequency(s string) (int, error) {
	// Simple frequency parsing - can be enhanced
	if s == "" {
		return 1, nil
	}
	
	// Try to extract numeric frequency
	var freq int
	if _, err := fmt.Sscanf(s, "%d", &freq); err != nil {
		return 1, err
	}
	
	return freq, nil
}

func hasNumbers(s string) bool {
	for _, char := range s {
		if unicode.IsDigit(char) {
			return true
		}
	}
	return false
}

func isCapitalized(s string) bool {
	if len(s) == 0 {
		return false
	}
	return unicode.IsUpper(rune(s[0]))
}

func isVowel(char byte) bool {
	return char == 'A' || char == 'E' || char == 'I' || char == 'O' || char == 'U'
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min3(a, b, c int) int {
	return min(min(a, b), c)
}