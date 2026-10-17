#include "ig_bridge.h"
#include "Search.h"
#include "Utils.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <set>
#include <map>

struct IGHandle {
    Instance* instance;
    Config* config;
};

IGHandle* ig_create(const char* instance_path, double alpha_constr, double alpha_ig) {
    // The IG parser splits on ' ' while the benchmark files use tabs: normalise first.
    std::ifstream in(instance_path);
    if (!in.is_open()) return nullptr;
    std::stringstream buf;
    buf << in.rdbuf();
    std::string text = buf.str();
    for (char& c : text) if (c == '\t' || c == '\r') c = ' ';
    // strip trailing spaces so tokenize does not produce empty trailing tokens
    std::stringstream ss(text);
    std::string line, fixed;
    while (std::getline(ss, line)) {
        while (!line.empty() && line.back() == ' ') line.pop_back();
        fixed += line + "\n";
    }

    char tmpl[] = "/tmp/ig_instance_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return nullptr;
    FILE* f = fdopen(fd, "w");
    fputs(fixed.c_str(), f);
    fclose(f);

    std::string name = instance_path;
    Instance* inst = Utils::buildInstance(tmpl);
    inst->name = name.substr(name.find_last_of('/') + 1);
    remove(tmpl);

    IGHandle* h = new IGHandle();
    h->instance = inst;
    h->config = new Config(1, alpha_constr, alpha_ig, false);
    return h;
}

void ig_destroy(IGHandle* h) {
    if (!h) return;
    delete h->config;
    delete h->instance;
    delete h;
}

bool ig_improve(IGHandle* h, const int* perm, const int* mode, int n,
                int capacity, unsigned seed,
                int* out_perm, int* out_mode, double* out_cost, int* out_vehicles,
                std::string* out_routes) {
    Instance* inst = h->instance;

    // Split perm into routes with the same rule as the GPU evaluator
    // (new vehicle on capacity / time-window / repeated-locker violation).
    std::vector<std::vector<std::string>> routes;
    std::vector<std::string> cur;
    double load = 0.0, time = 0.0;
    Node* curr = &inst->nodes.at(0);
    std::set<Node*> visited_lockers;
    std::string open_locker_id;                    // id of the locker sequence being built
    std::vector<std::string> open_locker_customers;
    Node* open_locker = nullptr;

    auto flush_locker = [&]() {
        if (open_locker) {
            std::string s = open_locker->id + "(";
            for (size_t i = 0; i < open_locker_customers.size(); ++i)
                s += (i ? "," : "") + open_locker_customers[i];
            s += ")";
            cur.push_back(s);
            open_locker = nullptr;
            open_locker_customers.clear();
        }
    };
    auto close_route = [&]() {
        flush_locker();
        if (!cur.empty()) {
            cur.insert(cur.begin(), "D0");
            cur.push_back("Dt");
            routes.push_back(cur);
            cur.clear();
        }
        load = 0.0; time = 0.0; curr = &inst->nodes.at(0);
        visited_lockers.clear();
    };

    for (int idx = 0; idx < n; ++idx) {
        Node* cust = inst->find_node_per_id("C" + std::to_string(perm[idx] - 1));
        if (!cust) { if (getenv("IG_DEBUG")) fprintf(stderr, "IG: no node C%d\n", perm[idx]-1); return false; }
        Node* target = cust;
        bool at_locker = (mode[idx] == 1);
        if (at_locker) {
            target = cust->designated_locker;
            if (!target) { if (getenv("IG_DEBUG")) fprintf(stderr, "IG: C%d has no designated locker\n", perm[idx]-1); return false; }
        }
        for (int attempt = 0; attempt < 2; ++attempt) {
            double arr = time + inst->distances[curr->index][target->index];
            bool dup = at_locker && visited_lockers.count(target) && target != curr;
            bool viol = (load + cust->load_demand > capacity) || (arr > target->time_window[1] + 1e-9) || dup;
            if (viol && attempt == 0) { close_route(); continue; }
            double e = target->time_window[0];
            bool stay = at_locker && target == curr;
            time = (arr > e ? arr : e) + (stay ? 0.0 : target->service_time);
            load += cust->load_demand;
            if (at_locker) {
                if (open_locker != target) { flush_locker(); open_locker = target; }
                open_locker_customers.push_back(cust->id);
                visited_lockers.insert(target);
            } else {
                flush_locker();
                cur.push_back(cust->id);
            }
            curr = target;
            break;
        }
    }
    close_route();

    if (getenv("IG_DEBUG")) fprintf(stderr, "IG: input split into %zu routes\n", routes.size());
    srand(seed);
    Search* search = new Search(inst, h->config);
    auto count = [&](Solution* sl) {
        int c = 0;
        for (int r = 0; r < sl->used_routes; ++r)
            for (size_t q = 1; q + 1 < sl->routes.at(r)->sequences.size(); ++q)
                c += (int)sl->routes.at(r)->sequences[q].customers.size();
        return c;
    };
    const bool dbg = getenv("IG_DEBUG") != nullptr;
    search->build_predefined_solution(routes);
    // build_predefined_solution does not maintain the per-route bookkeeping that the
    // construction heuristic keeps (locker visit counters, smallest demand on the route).
    for (int r = 0; r < search->solution->used_routes; ++r) {
        Route* route = search->solution->routes.at(r);
        double min_demand = -1.0;
        for (size_t q = 1; q + 1 < route->sequences.size(); ++q) {
            Sequence& sq = route->sequences[q];
            if (sq.node->type == "p") route->visited_lockers[sq.node] += 1;
            for (Node* c : sq.customers)
                if (min_demand < 0 || c->load_demand < min_demand) min_demand = c->load_demand;
        }
        route->minimun_route_load = min_demand < 0 ? 0.0 : min_demand;
    }
    if (dbg) fprintf(stderr, "IG: built %d cost %.1f\n", count(search->solution), search->solution->cost);
    search->rvnd_inter();
    if (dbg) fprintf(stderr, "IG: rvnd %d cost %.1f\n", count(search->solution), search->solution->cost);
    search->iterated_greedy();
    if (dbg) fprintf(stderr, "IG: ig %d cost %.1f\n", count(search->solution), search->solution->cost);

    Solution* sol = search->solution;
    int pos = 0, veh = 0;
    for (int r = 0; r < sol->used_routes; ++r) {
        std::vector<Sequence>& seqs = sol->routes.at(r)->sequences;
        if (seqs.size() <= 2) continue;
        ++veh;
        for (size_t s = 1; s + 1 < seqs.size(); ++s) {
            bool is_p = (seqs[s].node->type == "p");
            for (Node* c : seqs[s].customers) {
                if (pos >= n) { if (getenv("IG_DEBUG")) fprintf(stderr, "IG: output has too many customers\n"); delete search; return false; }
                out_perm[pos] = c->index - inst->customer_indexes[0] + 1;
                out_mode[pos] = is_p ? 1 : 0;
                ++pos;
            }
        }
    }
    if (out_routes) {
        out_routes->clear();
        for (int r = 0; r < sol->used_routes; ++r) {
            std::vector<Sequence>& seqs = sol->routes.at(r)->sequences;
            if (seqs.size() <= 2) continue;
            for (size_t s = 1; s + 1 < seqs.size(); ++s) {
                if (seqs[s].node->type == "p") {
                    *out_routes += seqs[s].node->id + "(";
                    for (size_t i = 0; i < seqs[s].customers.size(); ++i)
                        *out_routes += (i ? "," : "") + seqs[s].customers[i]->id;
                    *out_routes += ") ";
                } else {
                    *out_routes += seqs[s].node->id + " ";
                }
            }
            *out_routes += "| ";
        }
    }
    *out_cost = sol->cost;
    *out_vehicles = veh;
    delete search;
    if (pos != n && getenv("IG_DEBUG")) fprintf(stderr, "IG: output has %d/%d customers\n", pos, n);
    return pos == n;
}
