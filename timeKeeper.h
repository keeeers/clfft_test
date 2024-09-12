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
    TimeKeeper(); // ���캯��
    ~TimeKeeper(); // ��������

    void Init(); // ��ʼ����ʱ��
    void dur() const; // ���ִ��ʱ��
    void dur_withPre(const char*);
private:
    std::chrono::high_resolution_clock::time_point startTime; // �洢��ʼʱ��
};
#endif
