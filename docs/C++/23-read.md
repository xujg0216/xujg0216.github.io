# 读取文件

### 1. C语言读取文件

C主要使用`fopen`， `fread`， `fgets`， `fscanf`等函数进行文件读取

#### 1.1 读取二进制文件

```c
#include <stdio.h>
#include <stdlib.h>

unsigned char* read_binary_file(const char* filename, size_t* file_size)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        printf("Failed to open file: %s\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END); // 移动到文件末尾
    *file_size = ftell(fp); // 获取文件大小
    rewind(fp); // 回到文件开头

    unsigned char* buffer = (unsinged char* )malloc(*file_size);

    if(!buffer)
    {
        printf("Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }

    if (fread(buffer, 1, *file_size, fp) != *file_size)
    {
        printf("failed to read file\n");
        free(buffer);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    return buffer; //返回文件内容
}
```

`size_t fread(void *ptr, size_t size, size_t count, FILE *stream);`
参数说明
- `ptr`：指向要存储读取数据的缓冲区（通常是 `char*` 或 `unsigned char*`）。
- `size`：每个数据项的大小（以字节为单位）。
- `count`：要读取的数据项个数。
- `stream`：指向 `FILE*` 文件流。
返回值
返回成功读取的数据项个数（count）。
如果 读取数量小于 count，可能是 文件结束 或 读取出错。

#### 1.2 逐行读取文本文件

```c
#include <stdio.h>
#include <stdlib.h>

void read_text_file(const char* filename)
{
    FILE* fp = fopen(filename, "r");
    if (fp ==NULL)
    {
        printf("Failed to open file:%s\n", filename);
        return;
    }
    char buffer[256]; // 缓冲区
    while(fgets(buffer, sizeof(buffer), fp))
    {
        printf("%s", buffer);
    }
    fclose(fp);
}

```


#### 1.3 逐字符读取

```c
#include <stdio.h>

void read_file_char_by_char(const char* filename)
{
    FILE* fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("Failed to open file: %s\n", filename);
        return;
    }
    int ch;
    while((ch = fgetc(fp)) != EOF){
        putchar(ch);
    }
}

```