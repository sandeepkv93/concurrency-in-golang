package workstealingscheduler

import (
	"sync/atomic"
	"testing"
	"time"
)

func TestWorkStealingSchedulerBasic(t *testing.T) {
	scheduler := NewWorkStealingScheduler(4)
	scheduler.Start()
	
	numTasks := 20
	taskExecuted := make([]int32, numTasks)
	
	// Submit tasks
	for i := 0; i < numTasks; i++ {
		idx := i
		task := &SimpleTask{
			ID: idx,
			Work: func() {
				atomic.StoreInt32(&taskExecuted[idx], 1)
			},
		}
		scheduler.Submit(task)
	}
	
	// Wait for completion
	timeout := time.After(5 * time.Second)
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-timeout:
			t.Fatal("Timeout waiting for tasks to complete")
		case <-ticker.C:
			completed, _, _ := scheduler.GetStats()
			if int(completed) >= numTasks {
				goto done
			}
		}
	}
	
done:
	scheduler.Stop()
	
	// Verify all tasks executed
	for i, executed := range taskExecuted {
		if atomic.LoadInt32(&executed) != 1 {
			t.Errorf("Task %d was not executed", i)
		}
	}
}

func TestWorkStealing(t *testing.T) {
	scheduler := NewWorkStealingScheduler(4)
	scheduler.Start()
	
	numTasks := 100
	
	// Submit all tasks to worker 0 to force stealing
	for i := 0; i < numTasks; i++ {
		task := &SimpleTask{
			ID: i,
			Work: func() {
				time.Sleep(time.Millisecond)
			},
		}
		scheduler.SubmitToWorker(0, task)
	}
	
	// Wait for completion
	timeout := time.After(10 * time.Second)
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-timeout:
			t.Fatal("Timeout waiting for tasks to complete")
		case <-ticker.C:
			completed, _, _ := scheduler.GetStats()
			if int(completed) >= numTasks {
				goto done
			}
		}
	}
	
done:
	scheduler.Stop()
	
	completed, steals, workerStats := scheduler.GetStats()
	
	if int(completed) != numTasks {
		t.Errorf("Expected %d completed tasks, got %d", numTasks, completed)
	}
	
	if steals == 0 {
		t.Error("Expected some work stealing to occur")
	}
	
	// Verify work was distributed among workers
	workersWithTasks := 0
	for _, count := range workerStats {
		if count > 0 {
			workersWithTasks++
		}
	}
	
	if workersWithTasks < 2 {
		t.Error("Expected work to be distributed to multiple workers")
	}
}

func TestSchedulerStop(t *testing.T) {
	scheduler := NewWorkStealingScheduler(2)
	scheduler.Start()
	
	// Submit some long-running tasks
	for i := 0; i < 5; i++ {
		task := &SimpleTask{
			ID: i,
			Work: func() {
				time.Sleep(50 * time.Millisecond)
			},
		}
		scheduler.Submit(task)
	}
	
	// Give tasks time to start
	time.Sleep(20 * time.Millisecond)
	
	// Stop should wait for all tasks to complete
	done := make(chan bool)
	go func() {
		scheduler.Stop()
		done <- true
	}()
	
	select {
	case <-done:
		// Good, Stop() returned
	case <-time.After(5 * time.Second):
		t.Fatal("Stop() did not return in time")
	}
}

func BenchmarkWorkStealingScheduler(b *testing.B) {
	scheduler := NewWorkStealingScheduler(4)
	scheduler.Start()
	defer scheduler.Stop()
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		task := &SimpleTask{
			ID:   i,
			Work: func() {},
		}
		scheduler.Submit(task)
	}
	
	// Wait for all tasks to complete
	for {
		completed, _, _ := scheduler.GetStats()
		if int(completed) >= b.N {
			break
		}
		time.Sleep(time.Millisecond)
	}
}