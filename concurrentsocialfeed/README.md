# Concurrent Social Network Feed

A high-performance, scalable social network feed system in Go featuring real-time updates, intelligent feed generation, advanced caching, and comprehensive concurrency patterns for handling thousands of simultaneous users and interactions.

## Features

### Core Social Network Functionality
- **User Management**: Create users, manage profiles, follow/unfollow relationships
- **Post Creation**: Support for multiple post types (text, image, video, link, poll, event)
- **Content Interaction**: Reactions (like, love, laugh, angry, sad, wow), comments, shares
- **Feed Generation**: Timeline, user, trending, and recommended feeds with intelligent algorithms
- **Real-time Notifications**: Instant notifications for follows, reactions, comments, mentions
- **Hashtag Support**: Content discovery through hashtag indexing and search

### Advanced Features
- **Privacy Controls**: Multiple visibility levels (public, friends, private, followers)
- **Content Moderation**: Extensible content filtering and moderation system
- **Analytics**: Comprehensive metrics and performance tracking
- **Rate Limiting**: Per-user rate limiting to prevent abuse
- **Caching System**: LRU cache with TTL for optimized feed performance
- **Event Streaming**: Real-time event bus for live updates

### Concurrency Architecture
- **Worker Pool Design**: Dedicated workers for feed updates and notifications
- **Lock-Free Operations**: Atomic operations and careful synchronization for high throughput
- **Parallel Feed Generation**: Concurrent processing of multiple user feeds
- **Thread-Safe Data Structures**: Safe concurrent access to users, posts, and relationships
- **Context-Based Cancellation**: Proper resource cleanup and graceful shutdown
- **Load Distribution**: Intelligent task distribution across worker threads

## Architecture Overview

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Layer    │    │   Feed Layer    │    │ Notification    │
│                 │    │                 │    │     Layer       │
│ • User Mgmt     │    │ • Timeline      │    │ • Real-time     │
│ • Relationships │    │ • Trending      │    │ • Push/Pull     │
│ • Authentication│    │ • Recommended   │    │ • Event Bus     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────────┐
         │              Concurrency Engine                     │
         │                                                     │
         │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
         │ │Feed Workers │ │Notification │ │Cache Manager│    │
         │ │             │ │   Workers   │ │             │    │
         │ └─────────────┘ └─────────────┘ └─────────────┘    │
         └─────────────────────────────────────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────────┐
         │                Storage Layer                        │
         │                                                     │
         │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
         │ │  Users  │ │  Posts  │ │  Feeds  │ │ Indices │    │
         │ └─────────┘ └─────────┘ └─────────┘ └─────────┘    │
         └─────────────────────────────────────────────────────┘
```

## Usage Examples

### Basic Social Network Setup

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/yourusername/concurrency-in-golang/concurrentsocialfeed"
)

func main() {
    // Create social network with custom configuration
    config := concurrentsocialfeed.SocialNetworkConfig{
        MaxFeedSize:           100,
        MaxPostLength:         280,
        FeedUpdateWorkers:     8,
        NotificationWorkers:   4,
        CacheSize:            10000,
        FeedCacheTTL:         time.Minute * 5,
        EnableRealtimeUpdates: true,
        EnableMetrics:        true,
        MaxFollowers:         50000,
        RateLimitPerUser:     1000,
        RateLimitWindow:      time.Hour,
    }

    sn := concurrentsocialfeed.NewSocialNetwork(config)
    defer sn.Shutdown()

    // Create users
    alice, err := sn.CreateUser("alice", "Alice Smith")
    if err != nil {
        log.Fatalf("Failed to create user: %v", err)
    }

    bob, err := sn.CreateUser("bob", "Bob Johnson")
    if err != nil {
        log.Fatalf("Failed to create user: %v", err)
    }

    charlie, err := sn.CreateUser("charlie", "Charlie Brown")
    if err != nil {
        log.Fatalf("Failed to create user: %v", err)
    }

    fmt.Printf("Created users: %s, %s, %s\n", 
        alice.Username, bob.Username, charlie.Username)
}
```

### Creating and Managing Content

```go
// Create different types of posts
textPost, _ := sn.CreatePost(alice.ID, 
    "Just joined this amazing social network! 🎉", 
    concurrentsocialfeed.TextPost, 
    concurrentsocialfeed.Public)

imagePost, _ := sn.CreatePost(bob.ID,
    "Check out this beautiful sunset!",
    concurrentsocialfeed.ImagePost,
    concurrentsocialfeed.Public)

// Add reactions
err := sn.ReactToPost(bob.ID, textPost.ID, concurrentsocialfeed.Like)
if err != nil {
    log.Printf("Failed to react: %v", err)
}

err = sn.ReactToPost(charlie.ID, textPost.ID, concurrentsocialfeed.Love)
if err != nil {
    log.Printf("Failed to react: %v", err)
}

// Add comments
comment, err := sn.CommentOnPost(bob.ID, textPost.ID, "Welcome to the network!")
if err != nil {
    log.Printf("Failed to comment: %v", err)
} else {
    fmt.Printf("Comment added: %s\n", comment.Content)
}
```

### Building Social Connections

```go
// Create follow relationships
err = sn.FollowUser(alice.ID, bob.ID)
if err != nil {
    log.Printf("Failed to follow: %v", err)
}

err = sn.FollowUser(alice.ID, charlie.ID)
if err != nil {
    log.Printf("Failed to follow: %v", err)
}

err = sn.FollowUser(bob.ID, charlie.ID)
if err != nil {
    log.Printf("Failed to follow: %v", err)
}

// Get user's followers and following
user, _ := sn.GetUser(alice.ID)
user.mutex.RLock()
followerCount := len(user.Followers)
followingCount := len(user.Following)
user.mutex.RUnlock()

fmt.Printf("Alice has %d followers and is following %d users\n", 
    followerCount, followingCount)
```

### Feed Generation and Management

```go
// Get different types of feeds
timelineFeed, err := sn.GetUserFeed(alice.ID, concurrentsocialfeed.TimelineFeed)
if err != nil {
    log.Printf("Failed to get timeline: %v", err)
} else {
    fmt.Printf("Alice's timeline has %d posts\n", len(timelineFeed.Posts))
    
    for i, post := range timelineFeed.Posts {
        fmt.Printf("  %d. %s by %s (%s)\n", 
            i+1, post.Content, post.Username, post.CreatedAt.Format("15:04"))
    }
}

// Get user-specific feed
userFeed, err := sn.GetUserFeed(bob.ID, concurrentsocialfeed.UserFeed)
if err != nil {
    log.Printf("Failed to get user feed: %v", err)
} else {
    fmt.Printf("Bob's posts: %d\n", len(userFeed.Posts))
}

// Get trending feed
trendingFeed, err := sn.GetUserFeed("", concurrentsocialfeed.TrendingFeed)
if err != nil {
    log.Printf("Failed to get trending feed: %v", err)
} else {
    fmt.Printf("Trending posts: %d\n", len(trendingFeed.Posts))
    
    for i, post := range trendingFeed.Posts {
        engagement := 0
        for _, count := range post.Reactions {
            engagement += count
        }
        engagement += len(post.Comments)
        
        fmt.Printf("  %d. \"%s\" by %s (engagement: %d)\n",
            i+1, post.Content, post.Username, engagement)
    }
}
```

### Real-time Notifications and Events

```go
// Subscribe to real-time events
newPostEvents := sn.Subscribe("new_post")
go func() {
    for event := range newPostEvents {
        if postData, ok := event.Data["post"]; ok {
            if post, ok := postData.(*concurrentsocialfeed.Post); ok {
                fmt.Printf("🔔 New post by %s: %s\n", 
                    post.Username, post.Content)
            }
        }
    }
}()

// Get and display notifications
notifications, err := sn.GetUserNotifications(alice.ID)
if err != nil {
    log.Printf("Failed to get notifications: %v", err)
} else {
    fmt.Printf("Alice has %d notifications:\n", len(notifications))
    
    for _, notification := range notifications {
        status := "🔵"
        if notification.Read {
            status = "✅"
        }
        
        fmt.Printf("  %s %s: %s (%s)\n",
            status, notification.Type, notification.Message,
            notification.CreatedAt.Format("15:04"))
    }
}

// Mark notifications as read
for _, notification := range notifications {
    if !notification.Read {
        err := sn.MarkNotificationAsRead(alice.ID, notification.ID)
        if err != nil {
            log.Printf("Failed to mark notification as read: %v", err)
        }
    }
}
```

### Advanced Features and Analytics

```go
// Get comprehensive statistics
stats := sn.GetStatistics()
fmt.Printf("\n📊 Social Network Statistics:\n")
fmt.Printf("  Total Users: %d\n", stats.TotalUsers)
fmt.Printf("  Total Posts: %d\n", stats.TotalPosts)
fmt.Printf("  Total Reactions: %d\n", stats.TotalReactions)
fmt.Printf("  Total Comments: %d\n", stats.TotalComments)
fmt.Printf("  Feed Updates: %d\n", stats.FeedUpdates)
fmt.Printf("  Notifications Sent: %d\n", stats.NotificationsSent)
fmt.Printf("  Cache Hits: %d\n", stats.CacheHits)
fmt.Printf("  Cache Misses: %d\n", stats.CacheMisses)

// Search posts by hashtag
hashtagPosts, err := sn.GetPostsByHashtag("technology")
if err != nil {
    log.Printf("Failed to search hashtag: %v", err)
} else {
    fmt.Printf("Posts with #technology: %d\n", len(hashtagPosts))
}

// Get post details with view tracking
post, err := sn.GetPost(textPost.ID)
if err != nil {
    log.Printf("Failed to get post: %v", err)
} else {
    fmt.Printf("Post views: %d\n", post.Views)
    fmt.Printf("Post reactions: %v\n", post.Reactions)
    fmt.Printf("Post comments: %d\n", len(post.Comments))
}
```

### Concurrent Operations Example

```go
// Demonstrate high concurrency with multiple users
func demonstrateConcurrency() {
    sn := concurrentsocialfeed.NewSocialNetwork(concurrentsocialfeed.DefaultSocialNetworkConfig())
    defer sn.Shutdown()

    const numUsers = 1000
    const numPosts = 5

    var wg sync.WaitGroup
    users := make([]*concurrentsocialfeed.User, numUsers)

    // Create users concurrently
    fmt.Printf("Creating %d users concurrently...\n", numUsers)
    start := time.Now()

    for i := 0; i < numUsers; i++ {
        wg.Add(1)
        go func(index int) {
            defer wg.Done()
            user, err := sn.CreateUser(
                fmt.Sprintf("user%d", index),
                fmt.Sprintf("User %d", index))
            if err != nil {
                log.Printf("Failed to create user %d: %v", index, err)
                return
            }
            users[index] = user
        }(i)
    }
    wg.Wait()

    fmt.Printf("Created %d users in %v\n", numUsers, time.Since(start))

    // Create posts concurrently
    fmt.Printf("Creating posts concurrently...\n")
    start = time.Now()

    for i := 0; i < numUsers; i++ {
        for j := 0; j < numPosts; j++ {
            wg.Add(1)
            go func(userIndex, postIndex int) {
                defer wg.Done()
                if users[userIndex] != nil {
                    content := fmt.Sprintf("Post %d by user %d", postIndex, userIndex)
                    _, err := sn.CreatePost(users[userIndex].ID, content,
                        concurrentsocialfeed.TextPost, concurrentsocialfeed.Public)
                    if err != nil {
                        log.Printf("Failed to create post: %v", err)
                    }
                }
            }(i, j)
        }
    }
    wg.Wait()

    fmt.Printf("Created posts in %v\n", time.Since(start))

    // Create follow relationships concurrently
    fmt.Printf("Creating follow relationships...\n")
    start = time.Now()

    for i := 0; i < numUsers/10; i++ {
        for j := 0; j < 50; j++ {
            wg.Add(1)
            go func(followerIndex, followeeIndex int) {
                defer wg.Done()
                if followerIndex != followeeIndex &&
                   users[followerIndex] != nil && users[followeeIndex] != nil {
                    sn.FollowUser(users[followerIndex].ID, users[followeeIndex].ID)
                }
            }(i, (i+j+1)%numUsers)
        }
    }
    wg.Wait()

    fmt.Printf("Created relationships in %v\n", time.Since(start))

    // Access feeds concurrently
    fmt.Printf("Accessing feeds concurrently...\n")
    start = time.Now()

    for i := 0; i < numUsers/10; i++ {
        wg.Add(1)
        go func(userIndex int) {
            defer wg.Done()
            if users[userIndex] != nil {
                _, err := sn.GetUserFeed(users[userIndex].ID, 
                    concurrentsocialfeed.TimelineFeed)
                if err != nil {
                    log.Printf("Failed to get feed: %v", err)
                }
            }
        }(i)
    }
    wg.Wait()

    fmt.Printf("Accessed feeds in %v\n", time.Since(start))

    // Final statistics
    stats := sn.GetStatistics()
    fmt.Printf("\nFinal Statistics:\n")
    fmt.Printf("  Users: %d\n", stats.TotalUsers)
    fmt.Printf("  Posts: %d\n", stats.TotalPosts)
    fmt.Printf("  Cache Hits: %d\n", stats.CacheHits)
    fmt.Printf("  Cache Misses: %d\n", stats.CacheMisses)
}
```

## Performance Characteristics

| Operation | Time Complexity | Space Complexity | Concurrency |
|-----------|----------------|------------------|-------------|
| Create User | O(1) | O(1) | Thread-safe |
| Create Post | O(1) | O(1) | Lock-free |
| Follow User | O(1) | O(1) | Atomic updates |
| Generate Timeline | O(n log n) | O(n) | Parallel processing |
| Get Trending Feed | O(n log n) | O(n) | Cached results |
| Send Notification | O(1) | O(1) | Async workers |
| Cache Lookup | O(1) | O(1) | Lock-free reads |

## Configuration Options

```go
type SocialNetworkConfig struct {
    MaxFeedSize           int           // Maximum posts per feed (default: 100)
    MaxPostLength         int           // Maximum characters per post (default: 280)
    FeedUpdateWorkers     int           // Workers for feed updates (default: 4)
    NotificationWorkers   int           // Workers for notifications (default: 2)
    CacheSize            int           // Maximum cached feeds (default: 1000)
    FeedCacheTTL         time.Duration // Cache expiration time (default: 5m)
    EnableRealtimeUpdates bool          // Enable real-time events (default: true)
    EnableMetrics        bool          // Enable statistics (default: true)
    MaxFollowers         int           // Maximum followers per user (default: 10000)
    MaxHashtags          int           // Maximum hashtags per post (default: 10)
    RateLimitPerUser     int           // Actions per user per window (default: 100)
    RateLimitWindow      time.Duration // Rate limit window (default: 1h)
}
```

## Testing

The package includes comprehensive tests covering:

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interaction
- **Concurrency Tests**: Thread safety and race conditions
- **Performance Tests**: Benchmarks for critical operations
- **Load Tests**: High-volume concurrent operations

```bash
# Run all tests
go test -v

# Run tests with race detection
go test -race -v

# Run benchmarks
go test -bench=. -benchmem

# Run with coverage
go test -cover -v
```

## Example Benchmarks

```
BenchmarkCreateUser-8                    	 1000000	      1053 ns/op	     480 B/op	      11 allocs/op
BenchmarkCreatePost-8                    	  500000	      2847 ns/op	    1256 B/op	      25 allocs/op
BenchmarkFollowUser-8                    	 2000000	       912 ns/op	      64 B/op	       2 allocs/op
BenchmarkGetUserFeed-8                   	   10000	    145623 ns/op	   45632 B/op	     892 allocs/op
BenchmarkReactToPost-8                   	 1000000	      1205 ns/op	     128 B/op	       4 allocs/op
BenchmarkConcurrentFeedAccess-8          	   50000	     35421 ns/op	   12456 B/op	     234 allocs/op
```

## Production Considerations

### Scalability
- **Horizontal Scaling**: Multiple instances with shared storage
- **Database Integration**: Replace in-memory storage with persistent database
- **Caching Layer**: Redis/Memcached for distributed caching
- **Message Queues**: RabbitMQ/Kafka for async processing

### Monitoring
- **Metrics Collection**: Prometheus integration
- **Logging**: Structured logging with levels
- **Tracing**: Distributed tracing for request flow
- **Alerting**: Real-time alerts for system health

### Security
- **Authentication**: JWT token validation
- **Authorization**: Role-based access control
- **Input Validation**: Sanitize user content
- **Rate Limiting**: DDoS protection
- **Content Moderation**: Automated content filtering

## License

This implementation is part of the Concurrency in Golang project and is provided for educational and demonstration purposes.