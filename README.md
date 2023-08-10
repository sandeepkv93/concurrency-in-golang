# Concurrency in Golang

## What is Concurrency?

Concurrency is the ability to run multiple programs or parts of a program in parallel. Concurrency is not parallelism. Concurrency is about dealing with lots of things at once. Parallelism is about doing lots of things at once.

## Why Concurrency?

Concurrency is important because it allows us to write programs that can do more than one thing at a time. For example, we can write a web server that can handle multiple requests at the same time. We can also write a program that can perform multiple tasks at the same time, such as downloading multiple files simultaneously.

## How to do Concurrency in Golang?

Golang provides concurrency primitives such as goroutines and channels to help us write concurrent programs. Goroutines are lightweight threads that are managed by the Go runtime. Channels are used to communicate between goroutines. We can use these primitives to write concurrent programs in Go.

## Example

The following example demonstrates how to write a concurrent program in Go. Simple Example: Hello World

```go
package main

import (
	"fmt"
	"time"
    "sync"
)

func printNumbers() {
    defer wg.Done()
	for i := 1; i <= 5; i++ {
		fmt.Printf("%d ", i)
		time.Sleep(time.Millisecond * 100)
	}
}

func printLetters() {
    defer wg.Done()
	for i := 'a'; i <= 'e'; i++ {
		fmt.Printf("%c ", i)
		time.Sleep(time.Millisecond * 100)
	}
}

func main() {
    wg := sync.WaitGroup{}
    wg.Add(2)

    go printNumbers()
	go printLetters()

    wg.Wait()
}

```

## Table of Contents

| Sl No | Program                                                                               |
| ----- | ------------------------------------------------------------------------------------- |
| 1     | [Consecutive Number Printing using Goroutines](evenoddprinter/README.md)              |
| 2     | [Single Producer Multiple Consumers](singleproducermultipleconsumers/README.md)       |
| 3     | [Multiple Producers Multiple Consumers](multipleproducersmultipleconsumers/README.md) |
