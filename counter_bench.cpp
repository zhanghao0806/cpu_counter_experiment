// 多核计数实验：真实共享 / 伪共享 / 消除伪共享 / 批量聚合
// C++17, Linux (GCC/Clang); macOS only supports the unbound compatibility mode.
// Build: g++ -O3 -std=c++17 -pthread counter_bench.cpp -o counter_bench
// Usage: ./counter_bench <global|compact|padded|batch> <threads> <events> <batch_size> [cpus]

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

// 109-32cpu 的 sysfs 实测值；启动时再次核对，不默认为跨机器通用。
constexpr std::size_t CACHE_LINE = 64;
using AtomicCounter = std::atomic<std::uint64_t>;
static_assert(sizeof(AtomicCounter) == 8, "This layout requires 8-byte atomics");
static_assert(AtomicCounter::is_always_lock_free, "Lock-free uint64 atomics required");
constexpr std::size_t PER_LINE = CACHE_LINE / sizeof(AtomicCounter);
struct alignas(CACHE_LINE) CompactLine {
    AtomicCounter values[PER_LINE]{};
};
struct alignas(CACHE_LINE) PaddedCounter {
    std::atomic<std::uint64_t> value{0};
};
static_assert(sizeof(CompactLine) == CACHE_LINE && alignof(CompactLine) == CACHE_LINE);
static_assert(sizeof(PaddedCounter) == CACHE_LINE && alignof(PaddedCounter) == CACHE_LINE);
static_assert(offsetof(PaddedCounter, value) == 0);

std::vector<int> parse_cpus(const std::string& text, int threads) {
    if (text == "none") return {};
    std::vector<int> cpus;
    std::set<int> seen;
    std::istringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        std::size_t used = 0;
        int cpu = std::stoi(item, &used);
        if (used != item.size() || cpu < 0 || !seen.insert(cpu).second)
            throw std::invalid_argument("CPU IDs must be unique nonnegative integers");
#ifdef __linux__
        if (cpu >= CPU_SETSIZE) throw std::invalid_argument("CPU ID exceeds CPU_SETSIZE");
#endif
        cpus.push_back(cpu);
    }
    if (static_cast<int>(cpus.size()) != threads)
        throw std::invalid_argument("Provide exactly one CPU per worker");
#ifdef __linux__
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        throw std::runtime_error("Cannot read process affinity");
    for (int cpu : cpus) {
        if (!CPU_ISSET(cpu, &allowed))
            throw std::invalid_argument("Requested CPU is outside process affinity");
        std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu)
                           + "/cache/index0/coherency_line_size");
        int line = 0;
        if (!(file >> line) || line != CACHE_LINE)
            throw std::runtime_error("Cannot confirm 64-byte cache line for selected CPU");
    }
#else
    throw std::runtime_error("Explicit CPU binding requires Linux");
#endif
    return cpus;
}

bool bind_worker(int cpu) {
#ifdef __linux__
    cpu_set_t wanted, actual;
    CPU_ZERO(&wanted);
    CPU_ZERO(&actual);
    CPU_SET(cpu, &wanted);
    return pthread_setaffinity_np(pthread_self(), sizeof(wanted), &wanted) == 0
        && pthread_getaffinity_np(pthread_self(), sizeof(actual), &actual) == 0
        && CPU_COUNT(&actual) == 1 && CPU_ISSET(cpu, &actual)
        && sched_getcpu() == cpu;
#else
    (void)cpu;
    return false;
#endif
}

// 各方案共有的一点轻量、具有数据依赖的“请求处理”工作。
// 防止批量模式的循环被编译器等价地简化为单次常量计数。
inline std::uint64_t step(std::uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

int main(int argc, char** argv) {
    try {
        const bool inspect = argc >= 2 && std::string(argv[1]) == "--inspect";
        if ((inspect && argc != 4) || (!inspect && argc != 5 && argc != 6)) {
            std::cerr << "Usage: " << argv[0]
                      << " <global|compact|padded|batch> <threads> <events> <batch_size> [cpu0,cpu1,...|none]\n"
                      << "       " << argv[0] << " --inspect <threads> <cpu0,cpu1,...|none>\n";
            return 2;
        }
        const std::string mode = inspect ? "compact" : argv[1];
        const int threads = std::stoi(argv[2]);
        const std::uint64_t events = inspect ? threads : std::stoull(argv[3]);
        const std::uint64_t batch_size = inspect ? 1 : std::stoull(argv[4]);
        if ((mode != "global" && mode != "compact" && mode != "padded" && mode != "batch") ||
            threads < 1 || events < static_cast<std::uint64_t>(threads) || batch_size < 1) {
            throw std::invalid_argument("Invalid mode, threads, events, or batch_size");
        }

        const auto cpus = parse_cpus(inspect ? argv[3] : (argc == 6 ? argv[5] : "none"), threads);
        PaddedCounter global_storage;
        auto& global_count = global_storage.value;
        std::vector<CompactLine> compact((threads + PER_LINE - 1) / PER_LINE);
        std::vector<PaddedCounter> padded(threads);
        auto compact_at = [&](int tid) -> AtomicCounter& {
            return compact[tid / PER_LINE].values[tid % PER_LINE];
        };
        const auto compact_base = reinterpret_cast<std::uintptr_t>(&compact_at(0));
        const auto padded_base = reinterpret_cast<std::uintptr_t>(&padded[0].value);
        if (compact_base % CACHE_LINE || padded_base % CACHE_LINE
            || reinterpret_cast<std::uintptr_t>(&global_count) % CACHE_LINE)
            throw std::runtime_error("Counter base alignment check failed");
        for (int i = 0; i < threads; ++i) {
            const auto c = reinterpret_cast<std::uintptr_t>(&compact_at(i));
            const auto p = reinterpret_cast<std::uintptr_t>(&padded[i].value);
            if (c - compact_base != i * sizeof(AtomicCounter)
                || p - padded_base != i * CACHE_LINE
                || c / CACHE_LINE != (c + sizeof(AtomicCounter) - 1) / CACHE_LINE)
                throw std::runtime_error("Counter stride or cache-line check failed");
            compact_at(i).store(0, std::memory_order_relaxed);
            padded[i].value.store(0, std::memory_order_relaxed);
        }
        // 完成校验值也隔行存放，避免线程结束写回引入额外伪共享。
        std::vector<PaddedCounter> final_states(threads);

        // 所有工作线程先创建好并等待，避免把线程创建成本计入任务时间。
        std::mutex mu;
        std::condition_variable cv;
        int ready = 0;
        bool started = false;
        bool abort_work = false;
        int affinity_errors = 0;
        std::vector<std::thread> pool;
        pool.reserve(threads);

        try {
          for (int tid = 0; tid < threads; ++tid) {
            pool.emplace_back([&, tid]() {
                const bool bound = cpus.empty() || bind_worker(cpus[tid]);
                {
                    std::unique_lock<std::mutex> lock(mu);
                    if (!bound) ++affinity_errors;
                    ++ready;
                    cv.notify_all();
                    cv.wait(lock, [&]() { return started; });
                    if (inspect || abort_work) return;
                }

                const std::uint64_t work = events / static_cast<std::uint64_t>(threads)
                    + (static_cast<std::uint64_t>(tid) < events % static_cast<std::uint64_t>(threads));
                std::uint64_t state = 0x9e3779b97f4a7c15ULL
                    ^ (static_cast<std::uint64_t>(tid + 1) * 0xbf58476d1ce4e5b9ULL);

                if (mode == "global") {
                    for (std::uint64_t i = 0; i < work; ++i) {
                        state = step(state);
                        global_count.fetch_add(1, std::memory_order_relaxed);
                    }
                } else if (mode == "compact") {
                    // 每 8 个计数器确定处于同一 64 B 行，8 个以上使用多行。
                    auto& own = compact_at(tid);
                    for (std::uint64_t i = 0; i < work; ++i) {
                        state = step(state);
                        own.fetch_add(1, std::memory_order_relaxed);
                    }
                } else if (mode == "padded") {
                    // 与 compact 完全一样的计数操作，只改变每个元素的物理间距。
                    auto& own = padded[tid].value;
                    for (std::uint64_t i = 0; i < work; ++i) {
                        state = step(state);
                        own.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    // 保留单个全局计数器，但每 K 次请求合并为一次原子更新。
                    std::uint64_t pending = 0;
                    for (std::uint64_t i = 0; i < work; ++i) {
                        state = step(state);
                        ++pending;
                        if (pending == batch_size) {
                            global_count.fetch_add(pending, std::memory_order_relaxed);
                            pending = 0;
                        }
                    }
                    if (pending != 0) {
                        global_count.fetch_add(pending, std::memory_order_relaxed);
                    }
                }
                final_states[tid].value.store(state, std::memory_order_relaxed);
            });
          }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mu);
                abort_work = true;
                started = true;
            }
            cv.notify_all();
            for (auto& worker : pool) worker.join();
            throw;
        }

        using clock = std::chrono::steady_clock;
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&]() { return ready == threads; });
        abort_work = affinity_errors != 0;
        const auto start = clock::now();
        started = true;
        lock.unlock();
        cv.notify_all();
        for (auto& worker : pool) worker.join();
        const auto finish = clock::now();
        if (affinity_errors) throw std::runtime_error("Worker CPU binding verification failed");
        if (inspect) {
            std::cout << "cache_line_bytes=64 atomic_bytes=8 compact_stride=8 padded_stride=64"
                      << " lock_free=1 binding=" << (cpus.empty() ? "none" : "verified")
                      << " workers=" << threads << " counting_loops_executed=0\n";
            std::cout << "tid,cpu,compact_address,compact_line,compact_offset,padded_address,padded_line\n";
            for (int i = 0; i < threads; ++i) {
                const auto c = reinterpret_cast<std::uintptr_t>(&compact_at(i));
                const auto p = reinterpret_cast<std::uintptr_t>(&padded[i].value);
                std::cout << i << ',' << (cpus.empty() ? -1 : cpus[i]) << ','
                          << c << ',' << (c - compact_base) / CACHE_LINE << ','
                          << c % CACHE_LINE << ',' << p << ','
                          << (p - padded_base) / CACHE_LINE << '\n';
            }
            return 0;
        }

        std::uint64_t actual = 0;
        if (mode == "global" || mode == "batch") {
            actual = global_count.load(std::memory_order_relaxed);
        } else if (mode == "compact") {
            for (int i = 0; i < threads; ++i)
                actual += compact_at(i).load(std::memory_order_relaxed);
        } else {
            for (const auto& x : padded) actual += x.value.load(std::memory_order_relaxed);
        }
        std::uint64_t checksum = 0;
        for (int i = 0; i < threads; ++i) {
            checksum ^= final_states[i].value.load(std::memory_order_relaxed) +
                static_cast<std::uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL;
        }
        const double seconds = std::chrono::duration<double>(finish - start).count();
        const double throughput_mops = static_cast<double>(events) / seconds / 1e6;
        const bool correct = (actual == events);

        // 每次运行一行 CSV：mode,threads,events,batch,seconds,mops,actual,correct,checksum
        std::cout << mode << ',' << threads << ',' << events << ',' << batch_size << ','
                  << std::fixed << std::setprecision(9) << seconds << ','
                  << std::setprecision(5) << throughput_mops << ','
                  << actual << ',' << (correct ? 1 : 0) << ',' << checksum << '\n';
        return correct ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 2;
    }
}
