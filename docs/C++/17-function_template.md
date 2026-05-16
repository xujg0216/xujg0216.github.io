
# 函数模板


## 1. **普通函数模板 (Function Template)**
```cpp
template <typename T>
返回类型 函数名(T 参数) {
    // 函数体
}
```

```cpp
template<class T>
T add(T a, T b) { return a + b; }

add(1, 2);      // T 推导为 int
add(1.5, 2.5);  // T 推导为 double
```

---

## 2. **可变参数模板 (Variadic Template)**

`class... Args` 这种写法是 **模板参数包**，表示这里有一个数量可变的模板参数列表。

* `Args...` 是一个类型列表，比如 `(int, double, std::string)`。
* 和普通模板参数不同，它的数量可以是 0 个、1 个或多个。
* 对应的**函数形参**部分 `Args&&... args` 就是**函数参数包**。

**例子**

```cpp
template<class... Types>
void printAll(Types... args) {
    // 展开参数包（C++17 可以用 fold expression）
    int dummy[] = {0, ((std::cout << args << " "), 0)...};
}

printAll(1, "hi", 3.14); // Types 推导为 (int, const char*, double)
```

---









## 3. **转发引用 (Forwarding Reference / Universal Reference)**

在模板里，如果你写

```cpp
F&& f
Args&&... args
```

**并且** `F` / `Args` 是模板参数，`&&` 就不是单纯的右值引用，而是**转发引用**（C++11 引入）。

转发引用的特点：

* 既可以绑定到左值，也可以绑定到右值。
* 和 `std::forward` 配合，可以保持实参的值类别（左值保持左值、右值保持右值）。
* 用于泛型代码中避免不必要的拷贝。

**例子**

```cpp
template<class T>
void wrapper(T&& x) {
    func(std::forward<T>(x)); // 如果是右值，移动过去；如果是左值，按引用传过去
}

int a = 10;
wrapper(a);    // T 推导为 int&，T&& -> int& && -> int& 绑定左值
wrapper(10);   // T 推导为 int，T&& -> int&& 绑定右值
```

在 `enqueue` 中：

```cpp
F&& f
Args&&... args
```

这样设计是为了支持：

* `pool.enqueue(myFunction, 1, 2)`
* `pool.enqueue([]{ /* lambda */ })`
* `pool.enqueue(std::move(heavyObj))`

并且尽可能减少拷贝。

---

## 4. **模板类型推导 (Template Argument Deduction)**

`F` 和 `Args...` 的类型是在调用 `enqueue` 时由编译器推导出来的。

```cpp
pool.enqueue([](int x){ return x + 1; }, 42);
```

这里推导过程：

* `F` → `lambda` 的具体类型（匿名类型，只有编译器知道）。
* `Args...` → `int`。

---

## 5. **参数包展开 (Parameter Pack Expansion)**

在函数体里经常会看到：

```cpp
std::forward<Args>(args)...
```

这里的 `...` 是**包展开运算符**，它会把参数包 `Args` 和 `args` 按一一对应展开。

展开过程类似：

```cpp
std::forward<Arg1>(arg1), std::forward<Arg2>(arg2), ...
```

这也是为什么你能把不定个数的参数直接原封不动传给 `std::bind` 或 `std::invoke`。

---

## 6. **组合起来的意义**

综合来看：

```cpp
template<class F, class... Args>
```

*

```cpp
auto enqueue(F&& f, Args&&... args)
```

就是：

* 泛型：可支持任意可调用对象类型。
* 可变参数：支持任意数量和类型的参数传递给该可调用对象。
* 完美转发：保留参数的值类别（避免不必要的拷贝）。
* 高扩展性：可以支持普通函数、lambda、`std::function`、仿函数等，且参数类型灵活。

---

如果你愿意，我可以帮你画一张**调用时模板推导的流程图**，把 `F`、`Args...` 在 `enqueue` 调用时是怎么一步步被推导出来的，以及为什么 `&&` 会变成转发引用。
这个图会比文字更直观。

你想让我画吗？
