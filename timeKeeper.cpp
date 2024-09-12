#include "timeKeeper.h"
#include <chrono>
#include <functional>
#include <iostream>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
void WithTime(function<void()> func, const char* PreInfo) {
    auto beginTime = std::chrono::high_resolution_clock::now();
    func(); // 调用传入的函数
    auto endTime = std::chrono::high_resolution_clock::now();
    auto timeInterval = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime);
    cout << PreInfo << "Execution time: " << timeInterval.count() << " ms" << endl;
}
TimeKeeper::TimeKeeper() : startTime(std::chrono::high_resolution_clock::now()) {}

TimeKeeper::~TimeKeeper() {}

void TimeKeeper::Init()
{
    startTime = std::chrono::high_resolution_clock::now();
}

void TimeKeeper::dur() const
{
    auto endTime = std::chrono::high_resolution_clock::now();
    auto timeInterval = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "TimeKeeper dur time: " << timeInterval.count() << " ms" << std::endl;
}

void TimeKeeper::dur_withPre(const char* Pre) {
    std::cout << Pre;
    TimeKeeper::dur();
}