package concurrentsocialfeed

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// PostType defines different types of posts in the social network
type PostType int

const (
	TextPost PostType = iota
	ImagePost
	VideoPost
	LinkPost
	PollPost
	EventPost
)

// PostVisibility defines who can see the post
type PostVisibility int

const (
	Public PostVisibility = iota
	Friends
	Private
	Followers
)

// FeedType defines different types of feeds
type FeedType int

const (
	TimelineFeed FeedType = iota
	UserFeed
	HashtagFeed
	TrendingFeed
	RecommendedFeed
)

// ReactionType defines different types of reactions
type ReactionType int

const (
	Like ReactionType = iota
	Love
	Laugh
	Angry
	Sad
	Wow
)

// SocialNetworkConfig contains configuration for the social network
type SocialNetworkConfig struct {
	MaxFeedSize          int
	MaxPostLength        int
	FeedUpdateWorkers    int
	NotificationWorkers  int
	CacheSize           int
	FeedCacheTTL        time.Duration
	EnableRealtimeUpdates bool
	EnableMetrics        bool
	MaxFollowers        int
	MaxHashtags         int
	RateLimitPerUser    int
	RateLimitWindow     time.Duration
}

// DefaultSocialNetworkConfig returns default configuration
func DefaultSocialNetworkConfig() SocialNetworkConfig {
	return SocialNetworkConfig{
		MaxFeedSize:          100,
		MaxPostLength:        280,
		FeedUpdateWorkers:    4,
		NotificationWorkers:  2,
		CacheSize:           1000,
		FeedCacheTTL:        5 * time.Minute,
		EnableRealtimeUpdates: true,
		EnableMetrics:        true,
		MaxFollowers:        10000,
		MaxHashtags:         10,
		RateLimitPerUser:    100,
		RateLimitWindow:     time.Hour,
	}
}

// User represents a user in the social network
type User struct {
	ID          string             `json:"id"`
	Username    string             `json:"username"`
	DisplayName string             `json:"display_name"`
	Bio         string             `json:"bio"`
	AvatarURL   string             `json:"avatar_url"`
	CreatedAt   time.Time          `json:"created_at"`
	UpdatedAt   time.Time          `json:"updated_at"`
	Followers   map[string]bool    `json:"-"`
	Following   map[string]bool    `json:"-"`
	IsVerified  bool               `json:"is_verified"`
	IsActive    bool               `json:"is_active"`
	Settings    map[string]interface{} `json:"settings"`
	mutex       sync.RWMutex       `json:"-"`
}

// Post represents a post in the social network
type Post struct {
	ID          string               `json:"id"`
	UserID      string               `json:"user_id"`
	Username    string               `json:"username"`
	Content     string               `json:"content"`
	Type        PostType             `json:"type"`
	Visibility  PostVisibility       `json:"visibility"`
	MediaURLs   []string             `json:"media_urls"`
	Hashtags    []string             `json:"hashtags"`
	Mentions    []string             `json:"mentions"`
	CreatedAt   time.Time            `json:"created_at"`
	UpdatedAt   time.Time            `json:"updated_at"`
	Reactions   map[ReactionType]int `json:"reactions"`
	Comments    []Comment            `json:"comments"`
	Shares      int                  `json:"shares"`
	Views       int64                `json:"views"`
	Location    string               `json:"location"`
	IsEdited    bool                 `json:"is_edited"`
	ParentID    string               `json:"parent_id,omitempty"` // For replies
	metadata    map[string]interface{} `json:"-"`
	mutex       sync.RWMutex         `json:"-"`
}

// Comment represents a comment on a post
type Comment struct {
	ID        string               `json:"id"`
	PostID    string               `json:"post_id"`
	UserID    string               `json:"user_id"`
	Username  string               `json:"username"`
	Content   string               `json:"content"`
	CreatedAt time.Time            `json:"created_at"`
	Reactions map[ReactionType]int `json:"reactions"`
	ParentID  string               `json:"parent_id,omitempty"` // For nested comments
	mutex     sync.RWMutex         `json:"-"`
}

// Feed represents a user's feed
type Feed struct {
	UserID      string    `json:"user_id"`
	Type        FeedType  `json:"type"`
	Posts       []*Post   `json:"posts"`
	LastUpdated time.Time `json:"last_updated"`
	Version     int64     `json:"version"`
	mutex       sync.RWMutex `json:"-"`
}

// Notification represents a notification
type Notification struct {
	ID        string                 `json:"id"`
	UserID    string                 `json:"user_id"`
	Type      string                 `json:"type"`
	Title     string                 `json:"title"`
	Message   string                 `json:"message"`
	Data      map[string]interface{} `json:"data"`
	CreatedAt time.Time              `json:"created_at"`
	Read      bool                   `json:"read"`
	Priority  int                    `json:"priority"`
}

// FeedCache represents cached feed data
type FeedCache struct {
	feeds       map[string]*Feed
	expiration  map[string]time.Time
	mutex       sync.RWMutex
	maxSize     int
	defaultTTL  time.Duration
}

// NewFeedCache creates a new feed cache
func NewFeedCache(maxSize int, defaultTTL time.Duration) *FeedCache {
	cache := &FeedCache{
		feeds:      make(map[string]*Feed),
		expiration: make(map[string]time.Time),
		maxSize:    maxSize,
		defaultTTL: defaultTTL,
	}
	
	// Start cleanup goroutine
	go cache.cleanup()
	return cache
}

// Get retrieves a feed from cache
func (fc *FeedCache) Get(key string) (*Feed, bool) {
	fc.mutex.RLock()
	defer fc.mutex.RUnlock()
	
	if expiry, exists := fc.expiration[key]; exists {
		if time.Now().Before(expiry) {
			if feed, exists := fc.feeds[key]; exists {
				return feed, true
			}
		}
	}
	return nil, false
}

// Set stores a feed in cache
func (fc *FeedCache) Set(key string, feed *Feed) {
	fc.mutex.Lock()
	defer fc.mutex.Unlock()
	
	// Evict if at capacity
	if len(fc.feeds) >= fc.maxSize {
		fc.evictLRU()
	}
	
	fc.feeds[key] = feed
	fc.expiration[key] = time.Now().Add(fc.defaultTTL)
}

// evictLRU removes the least recently used item
func (fc *FeedCache) evictLRU() {
	var oldestKey string
	var oldestTime time.Time
	
	for key, expiry := range fc.expiration {
		if oldestKey == "" || expiry.Before(oldestTime) {
			oldestKey = key
			oldestTime = expiry
		}
	}
	
	if oldestKey != "" {
		delete(fc.feeds, oldestKey)
		delete(fc.expiration, oldestKey)
	}
}

// cleanup removes expired entries
func (fc *FeedCache) cleanup() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		fc.mutex.Lock()
		now := time.Now()
		for key, expiry := range fc.expiration {
			if now.After(expiry) {
				delete(fc.feeds, key)
				delete(fc.expiration, key)
			}
		}
		fc.mutex.Unlock()
	}
}

// Statistics tracks social network metrics
type Statistics struct {
	TotalUsers          int64
	TotalPosts          int64
	TotalReactions      int64
	TotalComments       int64
	FeedUpdates         int64
	NotificationsSent   int64
	CacheHits           int64
	CacheMisses         int64
	AveragePostLength   float64
	AverageFeedSize     float64
	ActiveUsers         int64
	mutex              sync.RWMutex
}

// SocialNetwork represents the main social network system
type SocialNetwork struct {
	config           SocialNetworkConfig
	users            map[string]*User
	posts            map[string]*Post
	feeds            map[string]*Feed
	notifications    map[string][]*Notification
	cache            *FeedCache
	statistics       *Statistics
	feedWorkerPool   *FeedWorkerPool
	notificationPool *NotificationWorkerPool
	eventBus         *EventBus
	hashtagIndex     map[string][]*Post
	userIndex        map[string]*User
	postCounters     map[string]int64
	rateLimiter      *RateLimiter
	mutex            sync.RWMutex
	ctx              context.Context
	cancel           context.CancelFunc
	wg               sync.WaitGroup
	running          bool
}

// FeedWorkerPool manages workers for feed updates
type FeedWorkerPool struct {
	workers    []*FeedWorker
	taskQueue  chan FeedTask
	resultChan chan FeedResult
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
}

// FeedWorker processes feed update tasks
type FeedWorker struct {
	id         int
	network    *SocialNetwork
	taskQueue  chan FeedTask
	resultChan chan FeedResult
	ctx        context.Context
}

// FeedTask represents a feed update task
type FeedTask struct {
	Type   string
	UserID string
	PostID string
	Data   map[string]interface{}
}

// FeedResult represents the result of a feed task
type FeedResult struct {
	Success bool
	Error   error
	Data    map[string]interface{}
}

// NotificationWorkerPool manages notification workers
type NotificationWorkerPool struct {
	workers    []*NotificationWorker
	taskQueue  chan NotificationTask
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
}

// NotificationWorker processes notification tasks
type NotificationWorker struct {
	id         int
	network    *SocialNetwork
	taskQueue  chan NotificationTask
	ctx        context.Context
}

// NotificationTask represents a notification task
type NotificationTask struct {
	Type         string
	UserID       string
	Notification *Notification
	Data         map[string]interface{}
}

// EventBus handles real-time events
type EventBus struct {
	subscribers map[string][]chan Event
	mutex       sync.RWMutex
}

// Event represents a real-time event
type Event struct {
	Type      string                 `json:"type"`
	UserID    string                 `json:"user_id"`
	Data      map[string]interface{} `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
}

// RateLimiter implements rate limiting per user
type RateLimiter struct {
	limits   map[string]*UserLimit
	config   SocialNetworkConfig
	mutex    sync.RWMutex
}

// UserLimit tracks rate limit for a user
type UserLimit struct {
	count     int64
	resetTime time.Time
	mutex     sync.Mutex
}

// NewSocialNetwork creates a new social network instance
func NewSocialNetwork(config SocialNetworkConfig) *SocialNetwork {
	ctx, cancel := context.WithCancel(context.Background())
	
	sn := &SocialNetwork{
		config:        config,
		users:         make(map[string]*User),
		posts:         make(map[string]*Post),
		feeds:         make(map[string]*Feed),
		notifications: make(map[string][]*Notification),
		cache:         NewFeedCache(config.CacheSize, config.FeedCacheTTL),
		statistics:    &Statistics{},
		hashtagIndex:  make(map[string][]*Post),
		userIndex:     make(map[string]*User),
		postCounters:  make(map[string]int64),
		rateLimiter:   NewRateLimiter(config),
		ctx:           ctx,
		cancel:        cancel,
		running:       true,
	}
	
	sn.feedWorkerPool = NewFeedWorkerPool(config.FeedUpdateWorkers, sn, ctx)
	sn.notificationPool = NewNotificationWorkerPool(config.NotificationWorkers, sn, ctx)
	sn.eventBus = NewEventBus()
	
	return sn
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(config SocialNetworkConfig) *RateLimiter {
	return &RateLimiter{
		limits: make(map[string]*UserLimit),
		config: config,
	}
}

// CheckRateLimit checks if user has exceeded rate limit
func (rl *RateLimiter) CheckRateLimit(userID string) bool {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()
	
	limit, exists := rl.limits[userID]
	if !exists {
		rl.limits[userID] = &UserLimit{
			count:     1,
			resetTime: time.Now().Add(rl.config.RateLimitWindow),
		}
		return true
	}
	
	limit.mutex.Lock()
	defer limit.mutex.Unlock()
	
	if time.Now().After(limit.resetTime) {
		limit.count = 1
		limit.resetTime = time.Now().Add(rl.config.RateLimitWindow)
		return true
	}
	
	if limit.count >= int64(rl.config.RateLimitPerUser) {
		return false
	}
	
	limit.count++
	return true
}

// NewEventBus creates a new event bus
func NewEventBus() *EventBus {
	return &EventBus{
		subscribers: make(map[string][]chan Event),
	}
}

// Subscribe subscribes to events of a specific type
func (eb *EventBus) Subscribe(eventType string, ch chan Event) {
	eb.mutex.Lock()
	defer eb.mutex.Unlock()
	
	eb.subscribers[eventType] = append(eb.subscribers[eventType], ch)
}

// Publish publishes an event to subscribers
func (eb *EventBus) Publish(event Event) {
	eb.mutex.RLock()
	defer eb.mutex.RUnlock()
	
	if subscribers, exists := eb.subscribers[event.Type]; exists {
		for _, ch := range subscribers {
			select {
			case ch <- event:
			default:
				// Channel is full, skip
			}
		}
	}
}

// NewFeedWorkerPool creates a new feed worker pool
func NewFeedWorkerPool(numWorkers int, network *SocialNetwork, ctx context.Context) *FeedWorkerPool {
	pool := &FeedWorkerPool{
		workers:    make([]*FeedWorker, numWorkers),
		taskQueue:  make(chan FeedTask, numWorkers*10),
		resultChan: make(chan FeedResult, numWorkers*10),
		ctx:        ctx,
	}
	
	for i := 0; i < numWorkers; i++ {
		worker := &FeedWorker{
			id:         i,
			network:    network,
			taskQueue:  pool.taskQueue,
			resultChan: pool.resultChan,
			ctx:        ctx,
		}
		pool.workers[i] = worker
		pool.wg.Add(1)
		go worker.start(&pool.wg)
	}
	
	return pool
}

// start starts the feed worker
func (fw *FeedWorker) start(wg *sync.WaitGroup) {
	defer wg.Done()
	
	for {
		select {
		case task := <-fw.taskQueue:
			result := fw.processTask(task)
			select {
			case fw.resultChan <- result:
			case <-fw.ctx.Done():
				return
			}
		case <-fw.ctx.Done():
			return
		}
	}
}

// processTask processes a feed task
func (fw *FeedWorker) processTask(task FeedTask) FeedResult {
	switch task.Type {
	case "update_user_feed":
		return fw.updateUserFeed(task.UserID, task.PostID)
	case "update_follower_feeds":
		return fw.updateFollowerFeeds(task.UserID, task.PostID)
	case "generate_recommended_feed":
		return fw.generateRecommendedFeed(task.UserID)
	default:
		return FeedResult{
			Success: false,
			Error:   fmt.Errorf("unknown task type: %s", task.Type),
		}
	}
}

// updateUserFeed updates a user's personal feed
func (fw *FeedWorker) updateUserFeed(userID, postID string) FeedResult {
	fw.network.mutex.RLock()
	post, exists := fw.network.posts[postID]
	fw.network.mutex.RUnlock()
	
	if !exists {
		return FeedResult{
			Success: false,
			Error:   fmt.Errorf("post not found: %s", postID),
		}
	}
	
	fw.network.mutex.Lock()
	feed, exists := fw.network.feeds[userID]
	if !exists {
		feed = &Feed{
			UserID: userID,
			Type:   TimelineFeed,
			Posts:  make([]*Post, 0),
		}
		fw.network.feeds[userID] = feed
	}
	fw.network.mutex.Unlock()
	
	feed.mutex.Lock()
	defer feed.mutex.Unlock()
	
	// Add post to beginning of feed
	feed.Posts = append([]*Post{post}, feed.Posts...)
	
	// Trim feed if too large
	if len(feed.Posts) > fw.network.config.MaxFeedSize {
		feed.Posts = feed.Posts[:fw.network.config.MaxFeedSize]
	}
	
	feed.LastUpdated = time.Now()
	atomic.AddInt64(&feed.Version, 1)
	
	// Invalidate cache
	fw.network.cache.Set(fmt.Sprintf("feed:%s", userID), feed)
	
	return FeedResult{Success: true}
}

// updateFollowerFeeds updates feeds of all followers
func (fw *FeedWorker) updateFollowerFeeds(userID, postID string) FeedResult {
	fw.network.mutex.RLock()
	user, exists := fw.network.users[userID]
	if !exists {
		fw.network.mutex.RUnlock()
		return FeedResult{
			Success: false,
			Error:   fmt.Errorf("user not found: %s", userID),
		}
	}
	
	_, exists = fw.network.posts[postID]
	if !exists {
		fw.network.mutex.RUnlock()
		return FeedResult{
			Success: false,
			Error:   fmt.Errorf("post not found: %s", postID),
		}
	}
	fw.network.mutex.RUnlock()
	
	user.mutex.RLock()
	followers := make([]string, 0, len(user.Followers))
	for followerID := range user.Followers {
		followers = append(followers, followerID)
	}
	user.mutex.RUnlock()
	
	// Update followers' feeds concurrently
	var wg sync.WaitGroup
	for _, followerID := range followers {
		wg.Add(1)
		go func(fID string) {
			defer wg.Done()
			fw.updateUserFeed(fID, postID)
		}(followerID)
	}
	wg.Wait()
	
	return FeedResult{Success: true}
}

// generateRecommendedFeed generates recommended content for a user
func (fw *FeedWorker) generateRecommendedFeed(userID string) FeedResult {
	// This is a simplified recommendation algorithm
	// In practice, this would use ML models, user behavior analysis, etc.
	
	fw.network.mutex.RLock()
	user, exists := fw.network.users[userID]
	if !exists {
		fw.network.mutex.RUnlock()
		return FeedResult{
			Success: false,
			Error:   fmt.Errorf("user not found: %s", userID),
		}
	}
	
	allPosts := make([]*Post, 0, len(fw.network.posts))
	for _, post := range fw.network.posts {
		if post.UserID != userID && post.Visibility == Public {
			allPosts = append(allPosts, post)
		}
	}
	fw.network.mutex.RUnlock()
	
	// Simple recommendation: posts from users with similar interests
	user.mutex.RLock()
	following := make(map[string]bool)
	for userID := range user.Following {
		following[userID] = true
	}
	user.mutex.RUnlock()
	
	recommendedPosts := make([]*Post, 0)
	for _, post := range allPosts {
		// Score posts based on various factors
		score := fw.calculateRecommendationScore(post, userID, following)
		if score > 0.5 { // Threshold for recommendation
			recommendedPosts = append(recommendedPosts, post)
		}
	}
	
	// Sort by score (simplified - would use actual scoring)
	sort.Slice(recommendedPosts, func(i, j int) bool {
		return recommendedPosts[i].CreatedAt.After(recommendedPosts[j].CreatedAt)
	})
	
	// Limit recommendations
	if len(recommendedPosts) > fw.network.config.MaxFeedSize/2 {
		recommendedPosts = recommendedPosts[:fw.network.config.MaxFeedSize/2]
	}
	
	return FeedResult{
		Success: true,
		Data: map[string]interface{}{
			"posts": recommendedPosts,
		},
	}
}

// calculateRecommendationScore calculates recommendation score for a post
func (fw *FeedWorker) calculateRecommendationScore(post *Post, userID string, following map[string]bool) float64 {
	score := 0.0
	
	// Higher score for recent posts
	age := time.Since(post.CreatedAt).Hours()
	if age < 24 {
		score += 0.3
	} else if age < 72 {
		score += 0.1
	}
	
	// Higher score for posts with more reactions
	totalReactions := 0
	for _, count := range post.Reactions {
		totalReactions += count
	}
	if totalReactions > 10 {
		score += 0.2
	} else if totalReactions > 5 {
		score += 0.1
	}
	
	// Higher score for posts from users you might know
	if following[post.UserID] {
		score += 0.4
	}
	
	return score
}

// NewNotificationWorkerPool creates a new notification worker pool
func NewNotificationWorkerPool(numWorkers int, network *SocialNetwork, ctx context.Context) *NotificationWorkerPool {
	pool := &NotificationWorkerPool{
		workers:   make([]*NotificationWorker, numWorkers),
		taskQueue: make(chan NotificationTask, numWorkers*10),
		ctx:       ctx,
	}
	
	for i := 0; i < numWorkers; i++ {
		worker := &NotificationWorker{
			id:        i,
			network:   network,
			taskQueue: pool.taskQueue,
			ctx:       ctx,
		}
		pool.workers[i] = worker
		pool.wg.Add(1)
		go worker.start(&pool.wg)
	}
	
	return pool
}

// start starts the notification worker
func (nw *NotificationWorker) start(wg *sync.WaitGroup) {
	defer wg.Done()
	
	for {
		select {
		case task := <-nw.taskQueue:
			nw.processTask(task)
		case <-nw.ctx.Done():
			return
		}
	}
}

// processTask processes a notification task
func (nw *NotificationWorker) processTask(task NotificationTask) {
	switch task.Type {
	case "new_follower":
		nw.sendFollowerNotification(task.UserID, task.Data)
	case "post_reaction":
		nw.sendReactionNotification(task.UserID, task.Data)
	case "post_comment":
		nw.sendCommentNotification(task.UserID, task.Data)
	case "mention":
		nw.sendMentionNotification(task.UserID, task.Data)
	}
}

// sendFollowerNotification sends a follower notification
func (nw *NotificationWorker) sendFollowerNotification(userID string, data map[string]interface{}) {
	followerID, ok := data["follower_id"].(string)
	if !ok {
		return
	}
	
	nw.network.mutex.RLock()
	follower, exists := nw.network.users[followerID]
	nw.network.mutex.RUnlock()
	
	if !exists {
		return
	}
	
	notification := &Notification{
		ID:      fmt.Sprintf("notif_%d", time.Now().UnixNano()),
		UserID:  userID,
		Type:    "new_follower",
		Title:   "New Follower",
		Message: fmt.Sprintf("%s started following you", follower.DisplayName),
		Data: map[string]interface{}{
			"follower_id": followerID,
		},
		CreatedAt: time.Now(),
		Priority:  1,
	}
	
	nw.network.AddNotification(userID, notification)
}

// sendReactionNotification sends a reaction notification
func (nw *NotificationWorker) sendReactionNotification(userID string, data map[string]interface{}) {
	postID, ok := data["post_id"].(string)
	if !ok {
		return
	}
	
	reactorID, ok := data["reactor_id"].(string)
	if !ok {
		return
	}
	
	nw.network.mutex.RLock()
	reactor, exists := nw.network.users[reactorID]
	nw.network.mutex.RUnlock()
	
	if !exists {
		return
	}
	
	notification := &Notification{
		ID:      fmt.Sprintf("notif_%d", time.Now().UnixNano()),
		UserID:  userID,
		Type:    "post_reaction",
		Title:   "New Reaction",
		Message: fmt.Sprintf("%s reacted to your post", reactor.DisplayName),
		Data: map[string]interface{}{
			"post_id":    postID,
			"reactor_id": reactorID,
		},
		CreatedAt: time.Now(),
		Priority:  2,
	}
	
	nw.network.AddNotification(userID, notification)
}

// sendCommentNotification sends a comment notification
func (nw *NotificationWorker) sendCommentNotification(userID string, data map[string]interface{}) {
	postID, ok := data["post_id"].(string)
	if !ok {
		return
	}
	
	commenterID, ok := data["commenter_id"].(string)
	if !ok {
		return
	}
	
	nw.network.mutex.RLock()
	commenter, exists := nw.network.users[commenterID]
	nw.network.mutex.RUnlock()
	
	if !exists {
		return
	}
	
	notification := &Notification{
		ID:      fmt.Sprintf("notif_%d", time.Now().UnixNano()),
		UserID:  userID,
		Type:    "post_comment",
		Title:   "New Comment",
		Message: fmt.Sprintf("%s commented on your post", commenter.DisplayName),
		Data: map[string]interface{}{
			"post_id":      postID,
			"commenter_id": commenterID,
		},
		CreatedAt: time.Now(),
		Priority:  2,
	}
	
	nw.network.AddNotification(userID, notification)
}

// sendMentionNotification sends a mention notification
func (nw *NotificationWorker) sendMentionNotification(userID string, data map[string]interface{}) {
	postID, ok := data["post_id"].(string)
	if !ok {
		return
	}
	
	mentionerID, ok := data["mentioner_id"].(string)
	if !ok {
		return
	}
	
	nw.network.mutex.RLock()
	mentioner, exists := nw.network.users[mentionerID]
	nw.network.mutex.RUnlock()
	
	if !exists {
		return
	}
	
	notification := &Notification{
		ID:      fmt.Sprintf("notif_%d", time.Now().UnixNano()),
		UserID:  userID,
		Type:    "mention",
		Title:   "You were mentioned",
		Message: fmt.Sprintf("%s mentioned you in a post", mentioner.DisplayName),
		Data: map[string]interface{}{
			"post_id":      postID,
			"mentioner_id": mentionerID,
		},
		CreatedAt: time.Now(),
		Priority:  3,
	}
	
	nw.network.AddNotification(userID, notification)
}

// CreateUser creates a new user
func (sn *SocialNetwork) CreateUser(username, displayName string) (*User, error) {
	if !sn.running {
		return nil, errors.New("social network is not running")
	}
	
	sn.mutex.Lock()
	defer sn.mutex.Unlock()
	
	// Check if username already exists
	if _, exists := sn.userIndex[username]; exists {
		return nil, fmt.Errorf("username already exists: %s", username)
	}
	
	user := &User{
		ID:          fmt.Sprintf("user_%d", time.Now().UnixNano()),
		Username:    username,
		DisplayName: displayName,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Followers:   make(map[string]bool),
		Following:   make(map[string]bool),
		IsActive:    true,
		Settings:    make(map[string]interface{}),
	}
	
	sn.users[user.ID] = user
	sn.userIndex[username] = user
	atomic.AddInt64(&sn.statistics.TotalUsers, 1)
	
	return user, nil
}

// GetUser gets a user by ID
func (sn *SocialNetwork) GetUser(userID string) (*User, error) {
	sn.mutex.RLock()
	defer sn.mutex.RUnlock()
	
	user, exists := sn.users[userID]
	if !exists {
		return nil, fmt.Errorf("user not found: %s", userID)
	}
	
	return user, nil
}

// GetUserByUsername gets a user by username
func (sn *SocialNetwork) GetUserByUsername(username string) (*User, error) {
	sn.mutex.RLock()
	defer sn.mutex.RUnlock()
	
	user, exists := sn.userIndex[username]
	if !exists {
		return nil, fmt.Errorf("user not found: %s", username)
	}
	
	return user, nil
}

// FollowUser makes one user follow another
func (sn *SocialNetwork) FollowUser(followerID, followeeID string) error {
	if followerID == followeeID {
		return errors.New("user cannot follow themselves")
	}
	
	sn.mutex.RLock()
	follower, followerExists := sn.users[followerID]
	followee, followeeExists := sn.users[followeeID]
	sn.mutex.RUnlock()
	
	if !followerExists {
		return fmt.Errorf("follower not found: %s", followerID)
	}
	
	if !followeeExists {
		return fmt.Errorf("followee not found: %s", followeeID)
	}
	
	// Check if already following
	follower.mutex.RLock()
	alreadyFollowing := follower.Following[followeeID]
	follower.mutex.RUnlock()
	
	if alreadyFollowing {
		return errors.New("already following user")
	}
	
	// Update following relationship
	follower.mutex.Lock()
	follower.Following[followeeID] = true
	follower.mutex.Unlock()
	
	followee.mutex.Lock()
	followee.Followers[followerID] = true
	followee.mutex.Unlock()
	
	// Send notification
	if sn.config.EnableRealtimeUpdates {
		sn.notificationPool.taskQueue <- NotificationTask{
			Type:   "new_follower",
			UserID: followeeID,
			Data: map[string]interface{}{
				"follower_id": followerID,
			},
		}
	}
	
	return nil
}

// UnfollowUser makes one user unfollow another
func (sn *SocialNetwork) UnfollowUser(followerID, followeeID string) error {
	sn.mutex.RLock()
	follower, followerExists := sn.users[followerID]
	followee, followeeExists := sn.users[followeeID]
	sn.mutex.RUnlock()
	
	if !followerExists {
		return fmt.Errorf("follower not found: %s", followerID)
	}
	
	if !followeeExists {
		return fmt.Errorf("followee not found: %s", followeeID)
	}
	
	// Update following relationship
	follower.mutex.Lock()
	delete(follower.Following, followeeID)
	follower.mutex.Unlock()
	
	followee.mutex.Lock()
	delete(followee.Followers, followerID)
	followee.mutex.Unlock()
	
	return nil
}

// CreatePost creates a new post
func (sn *SocialNetwork) CreatePost(userID, content string, postType PostType, visibility PostVisibility) (*Post, error) {
	if !sn.running {
		return nil, errors.New("social network is not running")
	}
	
	// Check rate limit
	if !sn.rateLimiter.CheckRateLimit(userID) {
		return nil, errors.New("rate limit exceeded")
	}
	
	sn.mutex.RLock()
	user, exists := sn.users[userID]
	sn.mutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("user not found: %s", userID)
	}
	
	if len(content) > sn.config.MaxPostLength {
		return nil, fmt.Errorf("post content too long: %d characters (max %d)", 
			len(content), sn.config.MaxPostLength)
	}
	
	post := &Post{
		ID:         fmt.Sprintf("post_%d", time.Now().UnixNano()),
		UserID:     userID,
		Username:   user.Username,
		Content:    content,
		Type:       postType,
		Visibility: visibility,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Reactions:  make(map[ReactionType]int),
		Comments:   make([]Comment, 0),
		metadata:   make(map[string]interface{}),
	}
	
	// Extract hashtags and mentions
	post.Hashtags = extractHashtags(content)
	post.Mentions = extractMentions(content)
	
	sn.mutex.Lock()
	sn.posts[post.ID] = post
	
	// Update hashtag index
	for _, hashtag := range post.Hashtags {
		sn.hashtagIndex[hashtag] = append(sn.hashtagIndex[hashtag], post)
	}
	sn.mutex.Unlock()
	
	atomic.AddInt64(&sn.statistics.TotalPosts, 1)
	
	// Update post counter for user (requires mutex for map access)
	sn.mutex.Lock()
	sn.postCounters[userID]++
	sn.mutex.Unlock()
	
	// Queue feed updates
	sn.feedWorkerPool.taskQueue <- FeedTask{
		Type:   "update_user_feed",
		UserID: userID,
		PostID: post.ID,
	}
	
	sn.feedWorkerPool.taskQueue <- FeedTask{
		Type:   "update_follower_feeds",
		UserID: userID,
		PostID: post.ID,
	}
	
	// Send mention notifications
	if sn.config.EnableRealtimeUpdates {
		for _, mention := range post.Mentions {
			if mentionedUser, err := sn.GetUserByUsername(mention); err == nil {
				sn.notificationPool.taskQueue <- NotificationTask{
					Type:   "mention",
					UserID: mentionedUser.ID,
					Data: map[string]interface{}{
						"post_id":      post.ID,
						"mentioner_id": userID,
					},
				}
			}
		}
	}
	
	// Publish real-time event
	if sn.config.EnableRealtimeUpdates {
		sn.eventBus.Publish(Event{
			Type:   "new_post",
			UserID: userID,
			Data: map[string]interface{}{
				"post_id": post.ID,
				"post":    post,
			},
			Timestamp: time.Now(),
		})
	}
	
	return post, nil
}

// GetPost gets a post by ID
func (sn *SocialNetwork) GetPost(postID string) (*Post, error) {
	sn.mutex.RLock()
	defer sn.mutex.RUnlock()
	
	post, exists := sn.posts[postID]
	if !exists {
		return nil, fmt.Errorf("post not found: %s", postID)
	}
	
	// Increment view count
	atomic.AddInt64(&post.Views, 1)
	
	return post, nil
}

// ReactToPost adds a reaction to a post
func (sn *SocialNetwork) ReactToPost(userID, postID string, reaction ReactionType) error {
	sn.mutex.RLock()
	post, exists := sn.posts[postID]
	sn.mutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("post not found: %s", postID)
	}
	
	post.mutex.Lock()
	post.Reactions[reaction]++
	post.mutex.Unlock()
	
	atomic.AddInt64(&sn.statistics.TotalReactions, 1)
	
	// Send notification to post author
	if sn.config.EnableRealtimeUpdates && post.UserID != userID {
		sn.notificationPool.taskQueue <- NotificationTask{
			Type:   "post_reaction",
			UserID: post.UserID,
			Data: map[string]interface{}{
				"post_id":    postID,
				"reactor_id": userID,
				"reaction":   reaction,
			},
		}
	}
	
	return nil
}

// CommentOnPost adds a comment to a post
func (sn *SocialNetwork) CommentOnPost(userID, postID, content string) (*Comment, error) {
	sn.mutex.RLock()
	post, exists := sn.posts[postID]
	user, userExists := sn.users[userID]
	sn.mutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("post not found: %s", postID)
	}
	
	if !userExists {
		return nil, fmt.Errorf("user not found: %s", userID)
	}
	
	comment := Comment{
		ID:        fmt.Sprintf("comment_%d", time.Now().UnixNano()),
		PostID:    postID,
		UserID:    userID,
		Username:  user.Username,
		Content:   content,
		CreatedAt: time.Now(),
		Reactions: make(map[ReactionType]int),
	}
	
	post.mutex.Lock()
	post.Comments = append(post.Comments, comment)
	post.mutex.Unlock()
	
	atomic.AddInt64(&sn.statistics.TotalComments, 1)
	
	// Send notification to post author
	if sn.config.EnableRealtimeUpdates && post.UserID != userID {
		sn.notificationPool.taskQueue <- NotificationTask{
			Type:   "post_comment",
			UserID: post.UserID,
			Data: map[string]interface{}{
				"post_id":      postID,
				"commenter_id": userID,
				"comment_id":   comment.ID,
			},
		}
	}
	
	return &comment, nil
}

// GetUserFeed gets a user's feed
func (sn *SocialNetwork) GetUserFeed(userID string, feedType FeedType) (*Feed, error) {
	// Check cache first
	cacheKey := fmt.Sprintf("feed:%s:%d", userID, feedType)
	if cachedFeed, found := sn.cache.Get(cacheKey); found {
		atomic.AddInt64(&sn.statistics.CacheHits, 1)
		return cachedFeed, nil
	}
	
	atomic.AddInt64(&sn.statistics.CacheMisses, 1)
	
	switch feedType {
	case TimelineFeed:
		return sn.generateTimelineFeed(userID)
	case UserFeed:
		return sn.generateUserFeed(userID)
	case TrendingFeed:
		return sn.generateTrendingFeed()
	case RecommendedFeed:
		return sn.generateRecommendedFeed(userID)
	default:
		return nil, fmt.Errorf("unsupported feed type: %d", feedType)
	}
}

// generateTimelineFeed generates a timeline feed for a user
func (sn *SocialNetwork) generateTimelineFeed(userID string) (*Feed, error) {
	sn.mutex.RLock()
	user, exists := sn.users[userID]
	if !exists {
		sn.mutex.RUnlock()
		return nil, fmt.Errorf("user not found: %s", userID)
	}
	
	// Get posts from followed users
	following := make([]string, 0, len(user.Following))
	user.mutex.RLock()
	for followedID := range user.Following {
		following = append(following, followedID)
	}
	user.mutex.RUnlock()
	
	// Include own posts
	following = append(following, userID)
	
	posts := make([]*Post, 0)
	for _, post := range sn.posts {
		for _, followedID := range following {
			if post.UserID == followedID {
				posts = append(posts, post)
				break
			}
		}
	}
	sn.mutex.RUnlock()
	
	// Sort by creation time (newest first)
	sort.Slice(posts, func(i, j int) bool {
		return posts[i].CreatedAt.After(posts[j].CreatedAt)
	})
	
	// Limit feed size
	if len(posts) > sn.config.MaxFeedSize {
		posts = posts[:sn.config.MaxFeedSize]
	}
	
	feed := &Feed{
		UserID:      userID,
		Type:        TimelineFeed,
		Posts:       posts,
		LastUpdated: time.Now(),
		Version:     1,
	}
	
	// Cache the feed
	cacheKey := fmt.Sprintf("feed:%s:%d", userID, TimelineFeed)
	sn.cache.Set(cacheKey, feed)
	
	return feed, nil
}

// generateUserFeed generates a feed showing only posts from a specific user
func (sn *SocialNetwork) generateUserFeed(userID string) (*Feed, error) {
	sn.mutex.RLock()
	defer sn.mutex.RUnlock()
	
	posts := make([]*Post, 0)
	for _, post := range sn.posts {
		if post.UserID == userID {
			posts = append(posts, post)
		}
	}
	
	// Sort by creation time (newest first)
	sort.Slice(posts, func(i, j int) bool {
		return posts[i].CreatedAt.After(posts[j].CreatedAt)
	})
	
	// Limit feed size
	if len(posts) > sn.config.MaxFeedSize {
		posts = posts[:sn.config.MaxFeedSize]
	}
	
	feed := &Feed{
		UserID:      userID,
		Type:        UserFeed,
		Posts:       posts,
		LastUpdated: time.Now(),
		Version:     1,
	}
	
	return feed, nil
}

// generateTrendingFeed generates a feed of trending posts
func (sn *SocialNetwork) generateTrendingFeed() (*Feed, error) {
	sn.mutex.RLock()
	defer sn.mutex.RUnlock()
	
	posts := make([]*Post, 0, len(sn.posts))
	for _, post := range sn.posts {
		if post.Visibility == Public {
			posts = append(posts, post)
		}
	}
	
	// Sort by engagement (reactions + comments + shares)
	sort.Slice(posts, func(i, j int) bool {
		engagementI := calculateEngagement(posts[i])
		engagementJ := calculateEngagement(posts[j])
		
		if engagementI == engagementJ {
			return posts[i].CreatedAt.After(posts[j].CreatedAt)
		}
		return engagementI > engagementJ
	})
	
	// Limit feed size
	if len(posts) > sn.config.MaxFeedSize {
		posts = posts[:sn.config.MaxFeedSize]
	}
	
	feed := &Feed{
		UserID:      "trending",
		Type:        TrendingFeed,
		Posts:       posts,
		LastUpdated: time.Now(),
		Version:     1,
	}
	
	return feed, nil
}

// generateRecommendedFeed generates a recommended feed for a user
func (sn *SocialNetwork) generateRecommendedFeed(userID string) (*Feed, error) {
	// Use worker pool for complex recommendation generation
	sn.feedWorkerPool.taskQueue <- FeedTask{
		Type:   "generate_recommended_feed",
		UserID: userID,
	}
	
	// For now, return a simple recommended feed
	// In practice, this would wait for the worker result
	return sn.generateTimelineFeed(userID)
}

// calculateEngagement calculates total engagement for a post
func calculateEngagement(post *Post) int {
	engagement := 0
	
	for _, count := range post.Reactions {
		engagement += count
	}
	
	engagement += len(post.Comments)
	engagement += post.Shares
	
	return engagement
}

// GetPostsByHashtag gets posts by hashtag
func (sn *SocialNetwork) GetPostsByHashtag(hashtag string) ([]*Post, error) {
	sn.mutex.RLock()
	defer sn.mutex.RUnlock()
	
	posts, exists := sn.hashtagIndex[hashtag]
	if !exists {
		return []*Post{}, nil
	}
	
	// Sort by creation time (newest first)
	sort.Slice(posts, func(i, j int) bool {
		return posts[i].CreatedAt.After(posts[j].CreatedAt)
	})
	
	return posts, nil
}

// GetUserNotifications gets notifications for a user
func (sn *SocialNetwork) GetUserNotifications(userID string) ([]*Notification, error) {
	sn.mutex.RLock()
	defer sn.mutex.RUnlock()
	
	notifications, exists := sn.notifications[userID]
	if !exists {
		return []*Notification{}, nil
	}
	
	// Sort by creation time (newest first)
	sort.Slice(notifications, func(i, j int) bool {
		return notifications[i].CreatedAt.After(notifications[j].CreatedAt)
	})
	
	return notifications, nil
}

// AddNotification adds a notification for a user
func (sn *SocialNetwork) AddNotification(userID string, notification *Notification) {
	sn.mutex.Lock()
	defer sn.mutex.Unlock()
	
	if sn.notifications[userID] == nil {
		sn.notifications[userID] = make([]*Notification, 0)
	}
	
	sn.notifications[userID] = append(sn.notifications[userID], notification)
	atomic.AddInt64(&sn.statistics.NotificationsSent, 1)
}

// MarkNotificationAsRead marks a notification as read
func (sn *SocialNetwork) MarkNotificationAsRead(userID, notificationID string) error {
	sn.mutex.RLock()
	notifications, exists := sn.notifications[userID]
	sn.mutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("no notifications for user: %s", userID)
	}
	
	for _, notification := range notifications {
		if notification.ID == notificationID {
			notification.Read = true
			return nil
		}
	}
	
	return fmt.Errorf("notification not found: %s", notificationID)
}

// GetStatistics returns current statistics
func (sn *SocialNetwork) GetStatistics() *Statistics {
	sn.statistics.mutex.RLock()
	defer sn.statistics.mutex.RUnlock()
	
	stats := *sn.statistics
	return &stats
}

// Subscribe subscribes to real-time events
func (sn *SocialNetwork) Subscribe(eventType string) chan Event {
	eventChan := make(chan Event, 100)
	sn.eventBus.Subscribe(eventType, eventChan)
	return eventChan
}

// Shutdown gracefully shuts down the social network
func (sn *SocialNetwork) Shutdown() error {
	if !sn.running {
		return errors.New("social network is not running")
	}
	
	sn.running = false
	sn.cancel()
	
	// Wait for workers to finish
	sn.feedWorkerPool.wg.Wait()
	sn.notificationPool.wg.Wait()
	
	return nil
}

// Helper functions

// extractHashtags extracts hashtags from text
func extractHashtags(text string) []string {
	hashtags := make([]string, 0)
	// Simple hashtag extraction (would use regex in practice)
	// This is a simplified implementation
	return hashtags
}

// extractMentions extracts mentions from text
func extractMentions(text string) []string {
	mentions := make([]string, 0)
	// Simple mention extraction (would use regex in practice)
	// This is a simplified implementation
	return mentions
}