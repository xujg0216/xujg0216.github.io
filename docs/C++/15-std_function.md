
# `std::function<void()>`
`std::function<void()>` 是 C++ 标准库里非常重要且实用的一个模板类，用于封装任何**符合“无参数、无返回值”函数签名的可调用对象**。它使得你可以用统一的类型存储、传递和调用不同类型的函数、函数对象或 lambda 表达式。

---

## 1. 定义

* `std::function<R(Args...)>` 是一个模板类，表示可以存储任意可调用对象，只要它的函数签名是 `R(Args...)`。
* 它可以存储普通函数指针、Lambda、函数对象（重载了`operator()`的类对象）、绑定表达式（`std::bind`结果）等。
* 它的底层实现会做类型擦除（type erasure），让不同的可调用对象表现为统一的接口。

## 2. 封装函数类型

* 这是 `std::function` 的一种特化，代表封装**无参数、返回void**的函数类型。
* 换句话说，`std::function<void()>`可以存储任何能被调用且签名为`void f()`的可调用对象。

## 3. 用途

* 在需要统一处理不同任务、回调或者异步操作时非常方便。
* 你不需要关心具体任务是什么，只要调用`operator()`即可执行。
* 它的灵活性适合放入任务队列、事件系统、信号槽等场景。

---

## 4. 举例说明

```cpp
#include <iostream>
#include <functional>

void foo() {
    std::cout << "foo called\n";
}

int main() {
    // 存普通函数
    std::function<void()> f = foo;
    f(); // 输出: foo called

    // 存Lambda表达式
    f = []() {
        std::cout << "lambda called\n";
    };
    f(); // 输出: lambda called

    // 存函数对象
    struct Bar {
        void operator()() const {
            std::cout << "Bar called\n";
        }
    };
    f = Bar();
    f(); // 输出: Bar called

    return 0;
}
```

---

## 5. 线程池中 `std::function<void()>` 的作用

* 线程池任务队列通常存储`std::function<void()>`，这样无论提交的任务是函数指针、Lambda还是绑定的成员函数，都可以统一存放并由线程调用。
* 线程取出任务后只需调用`task()`即可执行任务，无需知道任务具体类型。

---

## 6. 其他签名函数

`std::function` 不仅能包装无参无返回的函数（如 `std::function<void()>`），还能包装**任意签名的函数**，包括带参数和返回值的各种函数。


它的模板参数是函数签名格式：

```cpp
std::function<ReturnType(ArgTypes...)>
```

* `ReturnType` 是返回类型（可以是`void`、`int`、`std::string`等）
* `ArgTypes...` 是参数类型列表（可以是0个或多个参数）


```cpp
#include <iostream>
#include <functional>
#include <string>

// 无参无返回
std::function<void()> f1 = []() {
    std::cout << "Hello World\n";
};

// 有参数无返回
std::function<void(int)> f2 = [](int x) {
    std::cout << "Number: " << x << "\n";
};

// 有参数有返回
std::function<int(int,int)> f3 = [](int a, int b) {
    return a + b;
};

// 带复杂类型参数和返回
std::function<std::string(const std::string&, int)> f4 = [](const std::string& s, int n) {
    return s + " number " + std::to_string(n);
};

int main() {
    f1();              // 输出 Hello World
    f2(42);            // 输出 Number: 42
    std::cout << f3(1, 2) << "\n"; // 输出 3
    std::cout << f4("Test", 7) << "\n"; // 输出 Test number 7
}
```

`std::function` 能包装：

* 普通函数指针
* 函数对象（重载了`operator()`的类实例）
* Lambda表达式
* `std::bind`生成的绑定表达式
* 成员函数指针（需要配合对象实例调用）



