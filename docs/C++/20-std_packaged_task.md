# std::packaged_task
`std::packaged_task` 是 C++11 引入的一个 **线程任务包装器**，主要用来把一个可调用对象（函数、lambda、函数对象、成员函数等）包装成一个 **可以异步执行，并能返回结果的任务**。

它和 `std::future`、`std::promise` 是一套组合拳，核心作用是：

> **把一个任务和它的返回值绑定到一起**，执行任务时结果会自动存到 `std::future` 里，方便在其他地方获取。

---

## 1. 头文件

```cpp
#include <future>
```

---

## 2. 基本语法

```cpp
std::packaged_task<ReturnType(Args...)> task(可调用对象);
```

* **`ReturnType`**：任务的返回值类型
* **`Args...`**：任务的参数类型
* 构造时传入一个可调用对象（普通函数、lambda 等）

---

## 3. 使用流程

1. **用 `std::packaged_task` 包装任务**
2. **通过 `get_future()` 获取一个 `std::future` 对象**
3. **执行任务（直接调用或放到线程里执行）**
4. **通过 `future.get()` 获取任务结果**

---

## 4. 示例

### (1) 基本用法

```cpp
#include <iostream>
#include <future>
#include <thread>

int add(int a, int b) {
    return a + b;
}

int main() {
    // 1. 包装任务
    std::packaged_task<int(int, int)> task(add);

    // 2. 获取 future
    std::future<int> result = task.get_future();

    // 3. 执行任务
    task(10, 20);

    // 4. 获取结果
    std::cout << "result = " << result.get() << "\n"; // 输出 30
}
```

---

### (2) 放在线程里执行

```cpp
#include <iostream>
#include <future>
#include <thread>

int slow_add(int a, int b) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return a + b;
}

int main() {
    std::packaged_task<int(int, int)> task(slow_add);
    std::future<int> result = task.get_future();

    // 把任务丢到线程里执行
    std::thread t(std::move(task), 5, 6);
    t.detach(); // 分离线程

    std::cout << "等待结果中...\n";
    std::cout << "结果: " << result.get() << "\n"; // 阻塞直到任务完成
}
```

---

## 5. 特点

* **一次性**：`std::packaged_task` 不能重复执行同一个任务（除非重新绑定）。
* **支持任意可调用对象**（函数、lambda、`std::bind` 结果等）。
* 任务执行和结果获取可以在不同线程进行，适合异步编程。

---

## 6. 对比

| 工具                   | 作用                           |
| -------------------- | ---------------------------- |
| `std::promise`       | 手动设置一个值供 `future` 获取         |
| `std::future`        | 获取异步结果                       |
| `std::packaged_task` | 把一个函数和 `future` 绑定，执行时自动设置结果 |

---

✅ **一句话总结**
`std::packaged_task` = “帮你自动把函数执行结果送进 future 的快递员”。你只要把任务交给它，它会负责打包、发货（执行）、送达（结果获取）。

