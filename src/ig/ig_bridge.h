#ifndef IG_BRIDGE_H
#define IG_BRIDGE_H

#include <string>

// Thin C-style bridge around the author's Iterated Greedy (Corrêa et al.) so the
// CUDA solver can use it as its local search without touching IG headers.
struct IGHandle;

IGHandle* ig_create(const char* instance_path, double alpha_constr, double alpha_ig);
void ig_destroy(IGHandle* h);

// Improves one ant solution (perm of customers 1..n + delivery mode 0=home/1=locker).
// Splits perm into routes, runs RVND + Iterated Greedy from it, and returns the
// improved solution as perm/mode plus the IG-computed cost and vehicle count.
// Returns false if the input could not be turned into an IG solution.
bool ig_improve(IGHandle* h, const int* perm, const int* mode, int n,
                int capacity, unsigned seed,
                int* out_perm, int* out_mode, double* out_cost, int* out_vehicles,
                std::string* out_routes = nullptr);

#endif
