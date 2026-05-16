# `std::bind`
>C++11 引入的一个**函数适配器**，主要作用是：**“把一个可调用对象（函数、成员函数、函数对象、lambda 等）绑定部分参数，生成一个新的可调用对象。”**


## 1. 头文件

```cpp
#include <functional>
```


## 2. 基本语法

```cpp
auto new_func = std::bind(可调用对象, 参数1, 参数2, ...);
```

* **可调用对象**：普通函数、成员函数、lambda、函数对象等。
* **参数**：

  * 可以是具体值（直接绑定这个值）
  * 可以是占位符 `std::placeholders::_1`、`_2`... 表示在调用新函数时由外部传入。


## 3. 示例

### **(1) 绑定普通函数**

```cpp
#include <iostream>
#include <functional>

void func(int a, int b) {
    std::cout << "a = " << a << ", b = " << b << "\n";
}

int main() {
    using namespace std::placeholders;

    auto f1 = std::bind(func, 10, 20); // 两个参数都固定
    f1(); // 输出: a = 10, b = 20

    auto f2 = std::bind(func, _1, 100); // 第一个参数外部传
    f2(5); // 输出: a = 5, b = 100

    auto f3 = std::bind(func, _2, _1); // 调换参数顺序
    f3(1, 2); // 输出: a = 2, b = 1
}
```

---

### **(2) 绑定成员函数**

```cpp
#include <iostream>
#include <functional>

struct Foo {
    void print_sum(int x, int y) {
        std::cout << "sum = " << x + y << "\n";
    }
};

int main() {
    using namespace std::placeholders;

    Foo foo;
    auto f = std::bind(&Foo::print_sum, &foo, _1, 20);
    f(5); // 输出: sum = 25
}
```

注意：

* 成员函数绑定时，第一个参数要传对象地址（或引用）。

---

### **(3) 绑定 lambda**

```cpp
#include <iostream>
#include <functional>

int main() {
    using namespace std::placeholders;

    auto lambda = [](int a, int b){ std::cout << a + b << "\n"; };

    auto f = std::bind(lambda, _1, 10);
    f(5); // 输出: 15
}
```

---

## 4. 占位符说明

占位符来自 `std::placeholders` 命名空间：

* `_1` 表示调用新函数时的第一个实参
* `_2` 表示第二个实参
* 以此类推...

---

## 5. 使用场景

* **回调函数参数不匹配时的适配**
* **部分参数预先固定**
* **调整参数顺序**
* **将成员函数转换为普通可调用对象**

---

## 6. 注意事项

* `std::bind` 在 C++17 之后很多场景可以用 **lambda** 替代，lambda 更直观。
* 如果绑定的是引用参数，要用 `std::ref` 或 `std::cref`，否则会拷贝。

```cpp
#include <iostream>
#include <functional>

void print(int& x) { std::cout << x << "\n"; }

int main() {
    int a = 10;
    auto f = std::bind(print, std::ref(a)); // 绑定引用
    f(); // 输出 10
}
```


