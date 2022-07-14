#pragma once

#include <queue>

class sort_fn {
    public:
        bool operator()(pair<cost_t, uint_t> &p1, pair<cost_t, uint_t> &p2) { 
          if(p1.first > p2.first) {
            return true;
          } else if(p1.first == p2.first) {
            return (p1.second < p2.second);
            // return rand() % 2 == 0;
          } else {
            return false;
          }
        }
};

void row_argsort_heap(size_t* sel, cost_t* x, uint_t& u, const uint_t n, const uint_t m, int_t* w_p) {
    priority_queue<pair<cost_t, uint_t>, vector<pair<cost_t, uint_t>>, sort_fn> q;

    for(uint_t ii = 0; ii < n; ii++) {
      cost_t* _x = x + ii * m;

      for(uint_t i = 0; i < n; i++) {
        uint_t j = w_p[i];
        q.push(make_pair(_x[j], i));
      }

      for(uint_t i = n; i < m; i++) {
        uint_t j = w_p[i];
        if(_x[j] > q.top().first) {
          q.push(make_pair(_x[j], i));
          q.pop();
        }
      }

      for(uint_t j = 0; j < n; j++) {
        sel[ii * n + j] = w_p[q.top().second];
        q.pop();
      }
    }

    std::sort(sel, sel + n * n);
    auto it = std::unique(sel, sel + n * n);
    u = std::distance(sel, it);
}

void row_argsort(size_t* sel, cost_t* x, uint_t& u, const uint_t n, const uint_t m) {
    
    for(uint_t i = 0; i < n; i++) {
      
      size_t* tmp = (size_t*)malloc(m * sizeof(size_t));
      
      for(uint_t j = 0; j < m; j++) tmp[j] = j;
      
      cost_t* _x = x + i * m;
#ifdef LAP__FULLSORT
      std::stable_sort(
        tmp,
        tmp + m,
        [&_x](int left, int right) -> bool {
            return _x[left] > _x[right];
        });
#else
      std::nth_element(
        tmp,
        tmp + n,
        tmp + m,
        [&_x](int left, int right) -> bool {
            return _x[left] > _x[right];
        });
#endif

        for(uint_t j = 0; j < n; j++) {
          sel[i * n + j] = tmp[j];
        }
        
        free(tmp);
    }
    
    std::sort(sel, sel + n * n);
    auto it = std::unique(sel, sel + n * n);
    u = std::distance(sel, it);
}


void rect_lap(const uint_t n, const uint_t m, cost_t* cost, cost_t* pcost, int_t* ind, int_t* w_p) {
    uint_t u;
    
    size_t* sel = (size_t*)malloc(n * n * sizeof(size_t));
    
    // <<
#ifdef LAP__HEAPSORT
  if(w_p != nullptr) {
    row_argsort_heap(sel, cost, u, n, m, w_p);
  } else {
    row_argsort(sel, cost, u, n, m);
  }
#else
  row_argsort(sel, pcost, u, n, m);
  // // COMPATIBILITY STUFF
  // if(w_p != nullptr) {
  //   for(uint_t i = 0; i < u; i++) {
  //     sel[i] = w_p[sel[i]];
  //   }
  // }
  // std::sort(sel, sel + u);  
#endif
    
    std::random_shuffle(sel, sel + u); // increase randomness
    cost_t* cost_sub = (cost_t*)malloc(n * u * sizeof(cost_t));
    
    cost_t cost_sub_max = -1;
    for(uint_t i = 0; i < n; i++) {
      for(uint_t j = 0; j < u; j++) {
        cost_t val = cost[i * m + sel[j]];
        cost_sub[i * u + j] = val;
        if(val > cost_sub_max) {
          cost_sub_max = val;
        }
      }
    }
    
    cost_t cost_sub_max2 = -1;
    for(uint_t i = 0; i < n * u; i++) {
      cost_sub[i] = (1 + cost_sub_max) - cost_sub[i];
      if(cost_sub[i] > cost_sub_max2) {
        cost_sub_max2 = cost_sub[i];
      }
    }
    
    uint_t k         = n + u;
    cost_t* cost_pad = (cost_t*)malloc(k * k * sizeof(cost_t));
    for(uint_t r = 0; r < k; r++) {
      for(uint_t c = 0; c < k; c++) {
        if((r < u) && (c < n)) {
          cost_pad[r * k + c] = cost_sub[c * u + r];
        } else if(r >= u && c >= n) {
          cost_pad[r * k + c] = 0;
        } else {
          cost_pad[r * k + c] = 1 + cost_sub_max2;
        }
      }
    }

    int_t* x = (int_t*)malloc(k * sizeof(int_t));
    int_t* y = (int_t*)malloc(k * sizeof(int_t));
    lapjv_internal(k, cost_pad, x, y);

    for(uint_t i = 0; i < n; i++)
      ind[i] = sel[y[i]];
    
    free(sel);
    free(cost_sub);
    free(cost_pad);
    free(x);
    free(y);
}