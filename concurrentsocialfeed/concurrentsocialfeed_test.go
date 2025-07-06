package concurrentsocialfeed

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"
)

func TestNewSocialNetwork(t *testing.T) {
	config := DefaultSocialNetworkConfig()
	sn := NewSocialNetwork(config)
	
	if sn == nil {
		t.Fatal("Failed to create social network")
	}
	
	if !sn.running {
		t.Error("Social network should be running")
	}
	
	if sn.config.MaxFeedSize != config.MaxFeedSize {
		t.Errorf("Expected max feed size %d, got %d", config.MaxFeedSize, sn.config.MaxFeedSize)
	}
	
	// Clean up
	sn.Shutdown()
}

func TestCreateUser(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	user, err := sn.CreateUser("testuser", "Test User")
	if err != nil {
		t.Fatalf("Failed to create user: %v", err)
	}
	
	if user.Username != "testuser" {
		t.Errorf("Expected username 'testuser', got '%s'", user.Username)
	}
	
	if user.DisplayName != "Test User" {
		t.Errorf("Expected display name 'Test User', got '%s'", user.DisplayName)
	}
	
	if !user.IsActive {
		t.Error("User should be active by default")
	}
	
	// Test duplicate username
	_, err = sn.CreateUser("testuser", "Another User")
	if err == nil {
		t.Error("Expected error for duplicate username")
	}
}

func TestGetUser(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create user
	user, err := sn.CreateUser("testuser", "Test User")
	if err != nil {
		t.Fatalf("Failed to create user: %v", err)
	}
	
	// Get user by ID
	retrievedUser, err := sn.GetUser(user.ID)
	if err != nil {
		t.Fatalf("Failed to get user: %v", err)
	}
	
	if retrievedUser.ID != user.ID {
		t.Errorf("Expected user ID %s, got %s", user.ID, retrievedUser.ID)
	}
	
	// Get user by username
	retrievedUser, err = sn.GetUserByUsername("testuser")
	if err != nil {
		t.Fatalf("Failed to get user by username: %v", err)
	}
	
	if retrievedUser.Username != "testuser" {
		t.Errorf("Expected username 'testuser', got '%s'", retrievedUser.Username)
	}
	
	// Test non-existent user
	_, err = sn.GetUser("nonexistent")
	if err == nil {
		t.Error("Expected error for non-existent user")
	}
}

func TestFollowUnfollowUser(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users
	user1, _ := sn.CreateUser("user1", "User One")
	user2, _ := sn.CreateUser("user2", "User Two")
	
	// Test following
	err := sn.FollowUser(user1.ID, user2.ID)
	if err != nil {
		t.Fatalf("Failed to follow user: %v", err)
	}
	
	// Check following relationship
	user1, _ = sn.GetUser(user1.ID)
	user2, _ = sn.GetUser(user2.ID)
	
	user1.mutex.RLock()
	following := user1.Following[user2.ID]
	user1.mutex.RUnlock()
	
	user2.mutex.RLock()
	hasFollower := user2.Followers[user1.ID]
	user2.mutex.RUnlock()
	
	if !following {
		t.Error("User1 should be following User2")
	}
	
	if !hasFollower {
		t.Error("User2 should have User1 as follower")
	}
	
	// Test duplicate follow
	err = sn.FollowUser(user1.ID, user2.ID)
	if err == nil {
		t.Error("Expected error for duplicate follow")
	}
	
	// Test self-follow
	err = sn.FollowUser(user1.ID, user1.ID)
	if err == nil {
		t.Error("Expected error for self-follow")
	}
	
	// Test unfollowing
	err = sn.UnfollowUser(user1.ID, user2.ID)
	if err != nil {
		t.Fatalf("Failed to unfollow user: %v", err)
	}
	
	// Check unfollowing relationship
	user1, _ = sn.GetUser(user1.ID)
	user2, _ = sn.GetUser(user2.ID)
	
	user1.mutex.RLock()
	stillFollowing := user1.Following[user2.ID]
	user1.mutex.RUnlock()
	
	user2.mutex.RLock()
	stillHasFollower := user2.Followers[user1.ID]
	user2.mutex.RUnlock()
	
	if stillFollowing {
		t.Error("User1 should not be following User2 after unfollow")
	}
	
	if stillHasFollower {
		t.Error("User2 should not have User1 as follower after unfollow")
	}
}

func TestCreatePost(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create user
	user, _ := sn.CreateUser("testuser", "Test User")
	
	// Create post
	post, err := sn.CreatePost(user.ID, "This is a test post", TextPost, Public)
	if err != nil {
		t.Fatalf("Failed to create post: %v", err)
	}
	
	if post.Content != "This is a test post" {
		t.Errorf("Expected post content 'This is a test post', got '%s'", post.Content)
	}
	
	if post.UserID != user.ID {
		t.Errorf("Expected post user ID %s, got %s", user.ID, post.UserID)
	}
	
	if post.Type != TextPost {
		t.Errorf("Expected post type %d, got %d", TextPost, post.Type)
	}
	
	if post.Visibility != Public {
		t.Errorf("Expected post visibility %d, got %d", Public, post.Visibility)
	}
	
	// Test post length limit
	longContent := string(make([]byte, sn.config.MaxPostLength+1))
	_, err = sn.CreatePost(user.ID, longContent, TextPost, Public)
	if err == nil {
		t.Error("Expected error for post content too long")
	}
	
	// Test non-existent user
	_, err = sn.CreatePost("nonexistent", "Test", TextPost, Public)
	if err == nil {
		t.Error("Expected error for non-existent user")
	}
}

func TestGetPost(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create user and post
	user, _ := sn.CreateUser("testuser", "Test User")
	post, _ := sn.CreatePost(user.ID, "Test post", TextPost, Public)
	
	// Get post
	retrievedPost, err := sn.GetPost(post.ID)
	if err != nil {
		t.Fatalf("Failed to get post: %v", err)
	}
	
	if retrievedPost.ID != post.ID {
		t.Errorf("Expected post ID %s, got %s", post.ID, retrievedPost.ID)
	}
	
	if retrievedPost.Views != 1 {
		t.Errorf("Expected 1 view, got %d", retrievedPost.Views)
	}
	
	// Test non-existent post
	_, err = sn.GetPost("nonexistent")
	if err == nil {
		t.Error("Expected error for non-existent post")
	}
}

func TestReactToPost(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users and post
	user1, _ := sn.CreateUser("user1", "User One")
	user2, _ := sn.CreateUser("user2", "User Two")
	post, _ := sn.CreatePost(user1.ID, "Test post", TextPost, Public)
	
	// React to post
	err := sn.ReactToPost(user2.ID, post.ID, Like)
	if err != nil {
		t.Fatalf("Failed to react to post: %v", err)
	}
	
	// Check reaction count
	updatedPost, _ := sn.GetPost(post.ID)
	if updatedPost.Reactions[Like] != 1 {
		t.Errorf("Expected 1 like, got %d", updatedPost.Reactions[Like])
	}
	
	// React again
	err = sn.ReactToPost(user2.ID, post.ID, Love)
	if err != nil {
		t.Fatalf("Failed to react to post again: %v", err)
	}
	
	updatedPost, _ = sn.GetPost(post.ID)
	if updatedPost.Reactions[Love] != 1 {
		t.Errorf("Expected 1 love reaction, got %d", updatedPost.Reactions[Love])
	}
	
	// Test non-existent post
	err = sn.ReactToPost(user2.ID, "nonexistent", Like)
	if err == nil {
		t.Error("Expected error for non-existent post")
	}
}

func TestCommentOnPost(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users and post
	user1, _ := sn.CreateUser("user1", "User One")
	user2, _ := sn.CreateUser("user2", "User Two")
	post, _ := sn.CreatePost(user1.ID, "Test post", TextPost, Public)
	
	// Comment on post
	comment, err := sn.CommentOnPost(user2.ID, post.ID, "Great post!")
	if err != nil {
		t.Fatalf("Failed to comment on post: %v", err)
	}
	
	if comment.Content != "Great post!" {
		t.Errorf("Expected comment content 'Great post!', got '%s'", comment.Content)
	}
	
	if comment.UserID != user2.ID {
		t.Errorf("Expected comment user ID %s, got %s", user2.ID, comment.UserID)
	}
	
	// Check comment in post
	updatedPost, _ := sn.GetPost(post.ID)
	if len(updatedPost.Comments) != 1 {
		t.Errorf("Expected 1 comment, got %d", len(updatedPost.Comments))
	}
	
	if updatedPost.Comments[0].Content != "Great post!" {
		t.Errorf("Expected first comment content 'Great post!', got '%s'", updatedPost.Comments[0].Content)
	}
	
	// Test non-existent post
	_, err = sn.CommentOnPost(user2.ID, "nonexistent", "Comment")
	if err == nil {
		t.Error("Expected error for non-existent post")
	}
}

func TestGetUserFeed(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users
	user1, _ := sn.CreateUser("user1", "User One")
	user2, _ := sn.CreateUser("user2", "User Two")
	user3, _ := sn.CreateUser("user3", "User Three")
	
	// Create posts
	post1, _ := sn.CreatePost(user1.ID, "Post by user1", TextPost, Public)
	post2, _ := sn.CreatePost(user2.ID, "Post by user2", TextPost, Public)
	post3, _ := sn.CreatePost(user3.ID, "Post by user3", TextPost, Public)
	
	// User1 follows User2
	sn.FollowUser(user1.ID, user2.ID)
	
	// Wait a bit for feed updates
	time.Sleep(100 * time.Millisecond)
	
	// Get timeline feed for user1
	feed, err := sn.GetUserFeed(user1.ID, TimelineFeed)
	if err != nil {
		t.Fatalf("Failed to get user feed: %v", err)
	}
	
	if feed.UserID != user1.ID {
		t.Errorf("Expected feed user ID %s, got %s", user1.ID, feed.UserID)
	}
	
	if feed.Type != TimelineFeed {
		t.Errorf("Expected feed type %d, got %d", TimelineFeed, feed.Type)
	}
	
	// Should contain posts from user1 and user2 (followed), but not user3
	hasPost1 := false
	hasPost2 := false
	hasPost3 := false
	
	for _, post := range feed.Posts {
		if post.ID == post1.ID {
			hasPost1 = true
		}
		if post.ID == post2.ID {
			hasPost2 = true
		}
		if post.ID == post3.ID {
			hasPost3 = true
		}
	}
	
	if !hasPost1 {
		t.Error("Timeline feed should contain user1's own post")
	}
	
	if !hasPost2 {
		t.Error("Timeline feed should contain followed user's post")
	}
	
	if hasPost3 {
		t.Error("Timeline feed should not contain unfollowed user's post")
	}
	
	// Test user feed
	userFeed, err := sn.GetUserFeed(user2.ID, UserFeed)
	if err != nil {
		t.Fatalf("Failed to get user feed: %v", err)
	}
	
	if len(userFeed.Posts) != 1 || userFeed.Posts[0].ID != post2.ID {
		t.Error("User feed should contain only user2's posts")
	}
}

func TestTrendingFeed(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users and posts
	user1, _ := sn.CreateUser("user1", "User One")
	user2, _ := sn.CreateUser("user2", "User Two")
	
	post1, _ := sn.CreatePost(user1.ID, "Popular post", TextPost, Public)
	post2, _ := sn.CreatePost(user2.ID, "Regular post", TextPost, Public)
	
	// Add reactions to make post1 more popular
	sn.ReactToPost(user2.ID, post1.ID, Like)
	sn.ReactToPost(user2.ID, post1.ID, Love)
	sn.CommentOnPost(user2.ID, post1.ID, "Great!")
	
	// Get trending feed
	feed, err := sn.GetUserFeed("trending", TrendingFeed)
	if err != nil {
		t.Fatalf("Failed to get trending feed: %v", err)
	}
	
	if len(feed.Posts) < 2 {
		t.Fatalf("Expected at least 2 posts in trending feed, got %d", len(feed.Posts))
	}
	
	// More popular post should be first
	if feed.Posts[0].ID != post1.ID {
		t.Error("Most popular post should be first in trending feed")
	}
}

func TestNotifications(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users
	user1, _ := sn.CreateUser("user1", "User One")
	user2, _ := sn.CreateUser("user2", "User Two")
	
	// Create post
	post, _ := sn.CreatePost(user1.ID, "Test post", TextPost, Public)
	
	// React to trigger notification
	sn.ReactToPost(user2.ID, post.ID, Like)
	
	// Wait for notification processing
	time.Sleep(100 * time.Millisecond)
	
	// Get notifications
	notifications, err := sn.GetUserNotifications(user1.ID)
	if err != nil {
		t.Fatalf("Failed to get notifications: %v", err)
	}
	
	if len(notifications) == 0 {
		t.Error("Expected at least one notification")
	} else {
		notification := notifications[0]
		if notification.Type != "post_reaction" {
			t.Errorf("Expected notification type 'post_reaction', got '%s'", notification.Type)
		}
		
		if notification.UserID != user1.ID {
			t.Errorf("Expected notification user ID %s, got %s", user1.ID, notification.UserID)
		}
		
		if notification.Read {
			t.Error("Notification should not be read initially")
		}
		
		// Mark as read
		err = sn.MarkNotificationAsRead(user1.ID, notification.ID)
		if err != nil {
			t.Fatalf("Failed to mark notification as read: %v", err)
		}
		
		if !notification.Read {
			t.Error("Notification should be marked as read")
		}
	}
}

func TestFeedCache(t *testing.T) {
	cache := NewFeedCache(2, time.Minute)
	
	// Create test feeds
	feed1 := &Feed{UserID: "user1", Type: TimelineFeed}
	feed2 := &Feed{UserID: "user2", Type: TimelineFeed}
	feed3 := &Feed{UserID: "user3", Type: TimelineFeed}
	
	// Set feeds
	cache.Set("feed1", feed1)
	cache.Set("feed2", feed2)
	
	// Get feeds
	if cachedFeed, found := cache.Get("feed1"); !found || cachedFeed.UserID != "user1" {
		t.Error("Failed to get cached feed1")
	}
	
	if cachedFeed, found := cache.Get("feed2"); !found || cachedFeed.UserID != "user2" {
		t.Error("Failed to get cached feed2")
	}
	
	// Add third feed (should evict oldest)
	cache.Set("feed3", feed3)
	
	// feed1 should be evicted
	if _, found := cache.Get("feed1"); found {
		t.Error("feed1 should have been evicted")
	}
	
	// feed3 should be available
	if cachedFeed, found := cache.Get("feed3"); !found || cachedFeed.UserID != "user3" {
		t.Error("Failed to get cached feed3")
	}
}

func TestRateLimiter(t *testing.T) {
	config := DefaultSocialNetworkConfig()
	config.RateLimitPerUser = 2
	config.RateLimitWindow = time.Second
	
	rateLimiter := NewRateLimiter(config)
	
	userID := "testuser"
	
	// First request should pass
	if !rateLimiter.CheckRateLimit(userID) {
		t.Error("First request should pass rate limit")
	}
	
	// Second request should pass
	if !rateLimiter.CheckRateLimit(userID) {
		t.Error("Second request should pass rate limit")
	}
	
	// Third request should fail
	if rateLimiter.CheckRateLimit(userID) {
		t.Error("Third request should fail rate limit")
	}
	
	// Wait for rate limit window to reset
	time.Sleep(time.Second + 100*time.Millisecond)
	
	// Request should pass again
	if !rateLimiter.CheckRateLimit(userID) {
		t.Error("Request should pass after rate limit reset")
	}
}

func TestConcurrentOperations(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	const numUsers = 10
	const numPosts = 5
	
	var wg sync.WaitGroup
	users := make([]*User, numUsers)
	
	// Create users concurrently
	for i := 0; i < numUsers; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			user, err := sn.CreateUser(fmt.Sprintf("user%d", index), fmt.Sprintf("User %d", index))
			if err != nil {
				t.Errorf("Failed to create user %d: %v", index, err)
				return
			}
			users[index] = user
		}(i)
	}
	wg.Wait()
	
	// Create posts concurrently
	for i := 0; i < numUsers; i++ {
		for j := 0; j < numPosts; j++ {
			wg.Add(1)
			go func(userIndex, postIndex int) {
				defer wg.Done()
				if users[userIndex] != nil {
					content := fmt.Sprintf("Post %d by user %d", postIndex, userIndex)
					_, err := sn.CreatePost(users[userIndex].ID, content, TextPost, Public)
					if err != nil {
						t.Errorf("Failed to create post: %v", err)
					}
				}
			}(i, j)
		}
	}
	wg.Wait()
	
	// Create follow relationships concurrently
	for i := 0; i < numUsers; i++ {
		for j := 0; j < numUsers; j++ {
			if i != j {
				wg.Add(1)
				go func(followerIndex, followeeIndex int) {
					defer wg.Done()
					if users[followerIndex] != nil && users[followeeIndex] != nil {
						err := sn.FollowUser(users[followerIndex].ID, users[followeeIndex].ID)
						if err != nil {
							t.Logf("Follow relationship creation failed (expected in concurrent scenario): %v", err)
						}
					}
				}(i, j)
			}
		}
	}
	wg.Wait()
	
	// Access feeds concurrently
	for i := 0; i < numUsers; i++ {
		wg.Add(1)
		go func(userIndex int) {
			defer wg.Done()
			if users[userIndex] != nil {
				_, err := sn.GetUserFeed(users[userIndex].ID, TimelineFeed)
				if err != nil {
					t.Errorf("Failed to get feed for user %d: %v", userIndex, err)
				}
			}
		}(i)
	}
	wg.Wait()
	
	// Verify final state
	stats := sn.GetStatistics()
	if stats.TotalUsers != int64(numUsers) {
		t.Errorf("Expected %d users, got %d", numUsers, stats.TotalUsers)
	}
	
	if stats.TotalPosts != int64(numUsers*numPosts) {
		t.Errorf("Expected %d posts, got %d", numUsers*numPosts, stats.TotalPosts)
	}
}

func TestEventBus(t *testing.T) {
	eventBus := NewEventBus()
	
	// Create event channels
	ch1 := make(chan Event, 10)
	ch2 := make(chan Event, 10)
	
	// Subscribe to events
	eventBus.Subscribe("test_event", ch1)
	eventBus.Subscribe("test_event", ch2)
	eventBus.Subscribe("other_event", ch1)
	
	// Publish event
	event := Event{
		Type:      "test_event",
		UserID:    "user1",
		Data:      map[string]interface{}{"key": "value"},
		Timestamp: time.Now(),
	}
	
	eventBus.Publish(event)
	
	// Check that both subscribers received the event
	select {
	case receivedEvent := <-ch1:
		if receivedEvent.Type != "test_event" {
			t.Errorf("Expected event type 'test_event', got '%s'", receivedEvent.Type)
		}
	case <-time.After(time.Second):
		t.Error("Expected to receive event on ch1")
	}
	
	select {
	case receivedEvent := <-ch2:
		if receivedEvent.Type != "test_event" {
			t.Errorf("Expected event type 'test_event', got '%s'", receivedEvent.Type)
		}
	case <-time.After(time.Second):
		t.Error("Expected to receive event on ch2")
	}
	
	// Publish different event type
	otherEvent := Event{
		Type:      "other_event",
		UserID:    "user2",
		Data:      map[string]interface{}{},
		Timestamp: time.Now(),
	}
	
	eventBus.Publish(otherEvent)
	
	// Only ch1 should receive this event
	select {
	case receivedEvent := <-ch1:
		if receivedEvent.Type != "other_event" {
			t.Errorf("Expected event type 'other_event', got '%s'", receivedEvent.Type)
		}
	case <-time.After(time.Second):
		t.Error("Expected to receive other_event on ch1")
	}
	
	// ch2 should not receive this event
	select {
	case <-ch2:
		t.Error("ch2 should not receive other_event")
	case <-time.After(100 * time.Millisecond):
		// Expected timeout
	}
}

func TestStatistics(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users and posts
	user1, _ := sn.CreateUser("user1", "User One")
	user2, _ := sn.CreateUser("user2", "User Two")
	
	post, _ := sn.CreatePost(user1.ID, "Test post", TextPost, Public)
	
	sn.ReactToPost(user2.ID, post.ID, Like)
	sn.CommentOnPost(user2.ID, post.ID, "Nice post!")
	
	// Wait for async operations
	time.Sleep(100 * time.Millisecond)
	
	stats := sn.GetStatistics()
	
	if stats.TotalUsers != 2 {
		t.Errorf("Expected 2 users, got %d", stats.TotalUsers)
	}
	
	if stats.TotalPosts != 1 {
		t.Errorf("Expected 1 post, got %d", stats.TotalPosts)
	}
	
	if stats.TotalReactions != 1 {
		t.Errorf("Expected 1 reaction, got %d", stats.TotalReactions)
	}
	
	if stats.TotalComments != 1 {
		t.Errorf("Expected 1 comment, got %d", stats.TotalComments)
	}
}

func TestShutdown(t *testing.T) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	
	if !sn.running {
		t.Error("Social network should be running initially")
	}
	
	err := sn.Shutdown()
	if err != nil {
		t.Fatalf("Failed to shutdown social network: %v", err)
	}
	
	if sn.running {
		t.Error("Social network should not be running after shutdown")
	}
	
	// Test creating user after shutdown
	_, err = sn.CreateUser("testuser", "Test User")
	if err == nil {
		t.Error("Expected error when creating user after shutdown")
	}
	
	// Test double shutdown
	err = sn.Shutdown()
	if err == nil {
		t.Error("Expected error on double shutdown")
	}
}

// Benchmark tests

func BenchmarkCreateUser(b *testing.B) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		username := fmt.Sprintf("user%d", i)
		_, err := sn.CreateUser(username, fmt.Sprintf("User %d", i))
		if err != nil {
			b.Fatalf("Failed to create user: %v", err)
		}
	}
}

func BenchmarkCreatePost(b *testing.B) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create a user first
	user, _ := sn.CreateUser("testuser", "Test User")
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		content := fmt.Sprintf("Post number %d", i)
		_, err := sn.CreatePost(user.ID, content, TextPost, Public)
		if err != nil {
			b.Fatalf("Failed to create post: %v", err)
		}
	}
}

func BenchmarkFollowUser(b *testing.B) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users
	numUsers := 1000
	users := make([]*User, numUsers)
	for i := 0; i < numUsers; i++ {
		user, _ := sn.CreateUser(fmt.Sprintf("user%d", i), fmt.Sprintf("User %d", i))
		users[i] = user
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		followerIndex := i % numUsers
		followeeIndex := (i + 1) % numUsers
		
		if followerIndex != followeeIndex {
			sn.FollowUser(users[followerIndex].ID, users[followeeIndex].ID)
		}
	}
}

func BenchmarkGetUserFeed(b *testing.B) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users and posts
	numUsers := 100
	numPosts := 10
	users := make([]*User, numUsers)
	
	for i := 0; i < numUsers; i++ {
		user, _ := sn.CreateUser(fmt.Sprintf("user%d", i), fmt.Sprintf("User %d", i))
		users[i] = user
		
		for j := 0; j < numPosts; j++ {
			content := fmt.Sprintf("Post %d by user %d", j, i)
			sn.CreatePost(user.ID, content, TextPost, Public)
		}
	}
	
	// Create follow relationships
	for i := 0; i < numUsers; i++ {
		for j := 0; j < 10 && j < numUsers; j++ {
			if i != j {
				sn.FollowUser(users[i].ID, users[j].ID)
			}
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		userIndex := i % numUsers
		_, err := sn.GetUserFeed(users[userIndex].ID, TimelineFeed)
		if err != nil {
			b.Fatalf("Failed to get user feed: %v", err)
		}
	}
}

func BenchmarkReactToPost(b *testing.B) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users and posts
	user1, _ := sn.CreateUser("user1", "User One")
	user2, _ := sn.CreateUser("user2", "User Two")
	
	numPosts := 100
	posts := make([]*Post, numPosts)
	for i := 0; i < numPosts; i++ {
		post, _ := sn.CreatePost(user1.ID, fmt.Sprintf("Post %d", i), TextPost, Public)
		posts[i] = post
	}
	
	reactions := []ReactionType{Like, Love, Laugh, Angry, Sad, Wow}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		postIndex := i % numPosts
		reactionIndex := i % len(reactions)
		
		sn.ReactToPost(user2.ID, posts[postIndex].ID, reactions[reactionIndex])
	}
}

func BenchmarkConcurrentFeedAccess(b *testing.B) {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users and posts
	numUsers := 50
	users := make([]*User, numUsers)
	
	for i := 0; i < numUsers; i++ {
		user, _ := sn.CreateUser(fmt.Sprintf("user%d", i), fmt.Sprintf("User %d", i))
		users[i] = user
		
		// Create some posts
		for j := 0; j < 5; j++ {
			content := fmt.Sprintf("Post %d by user %d", j, i)
			sn.CreatePost(user.ID, content, TextPost, Public)
		}
		
		// Follow some users
		for j := 0; j < 5 && j < numUsers; j++ {
			if i != j {
				sn.FollowUser(user.ID, users[j].ID)
			}
		}
	}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			userIndex := rand.Intn(numUsers)
			sn.GetUserFeed(users[userIndex].ID, TimelineFeed)
		}
	})
}

// Example function
func ExampleSocialNetwork_CreateUser() {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	user, err := sn.CreateUser("johndoe", "John Doe")
	if err != nil {
		panic(err)
	}
	
	fmt.Printf("Created user: %s (%s)\n", user.Username, user.DisplayName)
	fmt.Printf("User ID: %s\n", user.ID)
	fmt.Printf("Created at: %s\n", user.CreatedAt.Format("2006-01-02 15:04:05"))
	
	// Output:
	// Created user: johndoe (John Doe)
	// User ID: user_1234567890
	// Created at: 2023-01-01 12:00:00
}

func ExampleSocialNetwork_CreatePost() {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create a user first
	user, _ := sn.CreateUser("johndoe", "John Doe")
	
	// Create a post
	post, err := sn.CreatePost(user.ID, "Hello, social network!", TextPost, Public)
	if err != nil {
		panic(err)
	}
	
	fmt.Printf("Created post: %s\n", post.Content)
	fmt.Printf("Post ID: %s\n", post.ID)
	fmt.Printf("Author: %s\n", post.Username)
	fmt.Printf("Visibility: %d\n", post.Visibility)
	
	// Output:
	// Created post: Hello, social network!
	// Post ID: post_1234567890
	// Author: johndoe
	// Visibility: 0
}

func ExampleSocialNetwork_GetUserFeed() {
	sn := NewSocialNetwork(DefaultSocialNetworkConfig())
	defer sn.Shutdown()
	
	// Create users
	user1, _ := sn.CreateUser("alice", "Alice Smith")
	user2, _ := sn.CreateUser("bob", "Bob Johnson")
	
	// Create posts
	sn.CreatePost(user1.ID, "Hello from Alice!", TextPost, Public)
	sn.CreatePost(user2.ID, "Hello from Bob!", TextPost, Public)
	
	// Alice follows Bob
	sn.FollowUser(user1.ID, user2.ID)
	
	// Get Alice's timeline feed
	feed, err := sn.GetUserFeed(user1.ID, TimelineFeed)
	if err != nil {
		panic(err)
	}
	
	fmt.Printf("Feed for %s contains %d posts\n", user1.Username, len(feed.Posts))
	for i, post := range feed.Posts {
		fmt.Printf("Post %d: %s by %s\n", i+1, post.Content, post.Username)
	}
	
	// Output:
	// Feed for alice contains 2 posts
	// Post 1: Hello from Bob! by bob
	// Post 2: Hello from Alice! by alice
}