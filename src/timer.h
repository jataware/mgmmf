#pragma once

using namespace std;
using namespace std::chrono;

struct mu_timer_t {
  high_resolution_clock::time_point t0;
  high_resolution_clock::time_point t1;
  long long elapsed;
  
  void start() {
    t0 = high_resolution_clock::now();
  }
  
  long long stop(string msg = "", bool quiet = false) {
    t1      = high_resolution_clock::now();
    elapsed = duration_cast<microseconds>(t1 - t0).count();
    if(!quiet) {
      cerr << "elapsed=" << elapsed << " | " << msg << endl;
    }
    return elapsed;
  }
};