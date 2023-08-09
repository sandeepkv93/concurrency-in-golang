package main

import (
	"fmt"
	"sync"
)

func main() {
	// Create a Mutex for even and odd
	var evenMutex sync.Mutex
	var oddMutex sync.Mutex

	// Create a WaitGroup to wait for all the go routines to finish
	var wg sync.WaitGroup
	wg.Add(2)
	evenMutex.Lock()

	go func() {
		defer wg.Done()
		for i := 1; i <= 100; i += 2 {
			oddMutex.Lock()
			fmt.Println(i)
			evenMutex.Unlock()
		}
	}()

	go func() {
		defer wg.Done()
		for i := 2; i <= 100; i += 2 {
			evenMutex.Lock()
			fmt.Println(i)
			oddMutex.Unlock()
		}
	}()

	wg.Wait()
}
