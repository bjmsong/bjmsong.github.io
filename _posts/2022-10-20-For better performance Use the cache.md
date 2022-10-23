---
layout:     post
title:      For better performance Use the cache
subtitle:   
date:       2022-10-20
author:     Mike Spertus
header-img: 
catalog: true
tags:
    - C++
---
- from CPP-Summit 2022

## What is a cache? Why do I care?
- CPU is 100x fast than memory
    - Processors typically run one instruction every clock cycle: Billions of computations per second
    - A memory access typically takes over one hundred cycles
- Cache
    - Processors have some very fast memory on chip
    - That can be accessed in as little as one clock cycle
    - 三级缓存
        - A fast but small (kilobytes) L1 cache on each core for instructions
        - A fast but small (kilobytes) L1 cache on each core for data
        - A L2 cache on each core that is larger (hundreds of kilobytes) but not quite as fast
        - A L3 cache on each CPU chip that is even larger (megabytes) but a little less fast than L2
            - Still much faster than main memory
- How does a computer decide what memory to cache?
    - When the memory that a processor reads is already in the cache, that is called a cache hit
    - If it needs to go to slow main memory to read the contents, that is called a cache miss
    - The caching algorithm's goal is to maximize this "hit rate"
    - Because reading main memory is so slow, even a small improvement to the hit rate can mean a big difference to your program
        - A program with a 95% cache hit rate will run almost twice as fast as a program with a 90% cache hit rate because the cache misses will be cut in half
    - In addition to hit rate, there are some optimizations to help the processor quickly access the cached data
- How do computers maximize the hit rate?
    - The challenge is to predict what data we think will be accessed in the future
    - This is a hard problem, but some simple algorithms do surprisingly well
    - First, whenever we read memory we store it in the cache because that shows the data at that location is of interest to the process and is likely to be read again
    - The second techniques is that we cache all of the data near the memory we read because programs often read nearby memory locations
        - Array elements
        - Characters in a string
        - Fields in a struct
        - Consecutive machine instructions
- Cache lines
    - To bring in the nearby data
    - Whenever a computer reads memory，it doesn't just cache the memory it read, But a whole region, called a cache line, that contains the memory
    - The actual size of a cache line varies by processor, but is typically 32-128 bytes
    - C++ gives us tools for working with cache lines
- Perhaps the biggest secret in computer progress is that computer cores have not gotten any faster in 15 years
    - 2005’s Pentium 4 HT 571 ran at 3.8GHz, which is better than many high-end CPUs today
    - https://semiwiki.com/ip/312695-white-paper-scaling-is-falling/
    - The problem with increasing clock speeds is heat
        - A high end CPU dissipates over 100 watts in about 1 cubic centimeter
        - An incandescent light bulb dissipates 100 watts in about 75 cubic centimeters
- To be fast, you have to use threads: multithreaded programming
    - But just because we have 100 cores doesn't mean that our program will be 100 times faster
        - We have to make sure that each core can efficiently do a share of the work
    ```C
    #include <iostream>
    #include <thread>

    void hello_threads(){
        std::cout<<"Hello threads"<<endl;
    }

    int main(){
        // Print in a different thread
        std::thread t(hello_threads);
        // Wait for that thread to complete
        t.join();

    }
    ```
- Threads and Caches
    - Unfortunately, while both threads and caches make your programs much faster, they don't work well together
    - Race condition
        - Race conditions occur when multiple threads use the same memory where at least one thread is modifying the memory
        - Example: If thread 1 is running on core A and is working on an object that it has cached, thread 2 running on core B will not see the cached version of the object but instead the stale version in main memory
        - Sometimes the hardware will force core A to flush its contents back to main memory so that core B will read the correct data values. This is called a cache coherency protocol, but of course writing to main memory and back is very slow, so the performance cost is very high even if a data race is avoided
        - If a program has a race condition, it's behavior is undefined
        - Do not allow race conditions in your programs
    - How to prevent race conditions
        - C++ provides a variety of tools for writing code without race conditions


## Learn by doing
- Our example: A counter
    - Our counter can be incremented or read by any thread
    - In effect, it is the worst case for sharing data among threads since all threads are rapidly modifying the counter
        - We have to prevent a data race in the increment
        - And we have to make effective use of the cache even though the count has to be shared by different processors that can't see each other's caches
- First attempt: Use a lock
    - The most basic way to prevent race conditions is to use a lock or a mutex
    - By putting a lock around the counter, only one thread will be able to access it at a time, and the new value will be written back to main memory after each increment
    - This is a safe and correct implementation
    ```C
    class DistributedCounter {
        public:
            typedef long long value_type;
            static std::string name() { return "locked counter"; };
        private:
            value_type count;
            std::shared_mutex mutable mtx;
        public:
            DistributedCounter() : count(0){}
            void operator++() {
                std::unique_lock lock(mtx);
                ++count;
            }
            value_type get() const {
                std:shared_lock lock(mtx);
                return count;
            }
    };
    ```
    - Performance
        - Running the counter in a program with one hundred threads each counting to ten million takes 113 seconds
        - Even though the computer has 128 cores
        - We probably aren't getting much parallelism because they are all taking turns to get the lock each time they increment the counter
        - We also are probably getting bad memory performance because the different processors need to update main memory after each increment
- Second attempt: Locked array
    - Instead of just storing a single value in our counter, we will store an array of values in the counter
    - Each thread will use a hash function to decide which value to increment and only lock that value
    - This should allow one thread to increment its value at the same time as another thread is incrementing its value
    ```C
    class DistributedCounter {
        typedef size_t value_type;
        struct bucket {
            std::shared_mutex sm;
            value_type count;
        };
        static size_t const bucket{ 128 };
        std:vector<bucket> counts { buckets };
        public:
            static std::string name() { return "locked array"; };
            void operator++() {
                size_t index = std::hash<std::thread::id>() (std::this_thread::get_id()) % buckets;
                std::unique_lock ul(counts[index].sm);
                counts[index].count++;
            }
            value_type get() {
                return std::accmulate(counts.begin(), counts.end(), value_type)0,
                [](auto acc, auto &x){
                    std::shared_lock sl(x.sm);
                    return acc + x.count;
                }
            }
    }
    ```
    - Performance
        - Running the same test now takes 13.86 seconds: Ten times faster!
- Third attempt: Locked padded array
    ```C
    
    ```
    - Performance
        - 
- Fourth attempt: Atomic padded array
    ```C
    
    ```
    - Performance
        - 


## Best practices 
- 




