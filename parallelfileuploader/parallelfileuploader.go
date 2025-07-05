package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

type File struct {
	Name string
	Size int
}

func Upload(file File, wg *sync.WaitGroup) error {
	defer wg.Done()

	// Sleep for File Size seconds to simulate upload
	time.Sleep(time.Duration(file.Size) * time.Millisecond)
	fmt.Printf("Uploaded the file %s with size %d\n", file.Name, file.Size)
	return nil
}

func main() {
	// Create a loop which creates 100 files of random size
	var files []File = make([]File, 1000)

	for i := 0; i < len(files); i++ {
		files[i] = File{
			Name: fmt.Sprintf("File-%d", i),
			Size: rand.Int() % 10000,
		}
	}

	// Create a WaitGroup
	var wg sync.WaitGroup = sync.WaitGroup{}
	wg.Add(len(files))

	// Upload the files in parallel
	for _, file := range files {
		go Upload(file, &wg)
	}

	// Wait for all files to be uploaded
	wg.Wait()
}
