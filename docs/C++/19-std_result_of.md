# std::result_of

`std::result_of<Func(Ts...)>::type` 是 **C++11/14** 里用来**推导某个可调用对象在给定参数下的返回类型**的模板工具。


## 1. 基本含义

* **`Func`**：可调用对象类型（函数类型、函数指针、lambda 类型、成员函数指针等）
* **`Ts...`**：调用时的参数类型列表
* **`std::result_of<...>::type`**：表示 **调用 `Func` 传入 `Ts...` 类型的参数后，返回值的类型**


## 2. 举例

### (1) 普通函数

```cpp
#include <iostream>
#include <type_traits>

int add(int a, double b) {
    return a + b;
}

int main() {
    using type = std::result_of<decltype(add)*(int, double)>::type;
    // decltype(add)* 表示函数指针类型
    std::cout << std::is_same<type, int>::value << "\n"; // 输出 1 (true)
}
```

---

### (2) Lambda

```cpp
#include <iostream>
#include <type_traits>

int main() {
    auto lambda = [](int a, int b) { return a + b; };

    using type = std::result_of<decltype(lambda)(int, int)>::type;
    std::cout << std::is_same<type, int>::value << "\n"; // 输出 1
}
```

---

### (3) 成员函数

```cpp
#include <iostream>
#include <type_traits>

struct Foo {
    double mul(int a, float b) { return a * b; }
};

int main() {
    using type = std::result_of<decltype(&Foo::mul)(Foo, int, float)>::type;
    std::cout << std::is_same<type, double>::value << "\n"; // 输出 1
}
```

---

## 3. 为什么有它

在泛型编程里，你有一个模板，里面会调用一个未知类型的函数/对象，你需要在**编译期**知道返回值类型，就可以用 `std::result_of`。

例子：

```cpp
template <typename Func, typename... Args>
void call_and_print(Func f, Args... args) {
    using ReturnType = typename std::result_of<Func(Args...)>::type;
    ReturnType ret = f(args...);
    std::cout << ret << "\n";
}
```

---

## 4. 注意

* `std::result_of` 在 **C++17 被弃用**，推荐使用 `std::invoke_result`。
* C++14 可以省略 `::type`，用 `std::result_of_t<...>`。
* C++17 可以用：

```cpp
using ReturnType = std::invoke_result_t<Func, Args...>;
```

---

✅ **一句话总结**
`std::result_of<Func(Ts...)>::type` = “**如果我用类型 `Ts...` 调用类型 `Func`，它的返回值类型是什么？**”

