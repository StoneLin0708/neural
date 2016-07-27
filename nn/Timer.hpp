#pragma once
#include <chrono>
#include <iostream>

#define TIMER_MEASURE_MACRO( fun , text ) {\
        do{\
            Timer timer;\
            timer.start();\
            fun\
            auto t = timer.countMS();\
            std::cout << text <<  t << "ms" << std::endl;\
        }while(false);\
    }

class Timer{
public:
  typedef std::chrono::high_resolution_clock Clock;
  void start()
  {
      epoch = Clock::now();
  }

  double countMS()
  {
      return std::chrono::duration_cast<std::chrono::microseconds>(time_elapsed()).count()/1000.0;
  }

  Clock::duration time_elapsed() const
  {
      return Clock::now() - epoch;
  }

private:
  Clock::time_point epoch;
};

