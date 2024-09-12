#pragma once
#ifndef TIME_KEEPER
#define TIME_KEEPER

#include <functional>
#include <chrono>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
void WithTime(function<void()> func, const char* PreInfo);
class TimeKeeper
{
public:
    TimeKeeper(); // 构造函数
    ~TimeKeeper(); // 析构函数

    void Init(); // 初始化计时器
    void dur() const; // 输出执行时间
    void dur_withPre(const char*);
private:
    std::chrono::high_resolution_clock::time_point startTime; // 存储开始时间
};
#endif
