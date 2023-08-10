# Simulate Parallel File Uploader

## Problem Statement
We have a list of files that we want to upload to a server. Each file has a name and a size. We want to upload all the files to the server in parallel. We want to use goroutines to upload the files in parallel. We want to use a WaitGroup to wait for all the files to be uploaded.