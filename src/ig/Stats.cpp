//
// Created by Rafael on 12/01/2025.
//

#include "Stats.h"

#include "Solution.h"

Stats::Stats(Instance* instance, Config* config) {
    this->instance = instance;
    this->config = config;
}

void Stats::set_result(Solution* solution, double time) {
    this->costs.push_back(solution->cost);
    this->times.push_back(time);

    if(this->best_solution == nullptr || solution->cost < this->best_solution->cost) {
        this->best_solution = solution;
    }

    if(this->best_time == -1.0 || time < this->best_time) {
        this->best_time = time;
    }

    this->avg_cost+=solution->cost;
    this->avg_time+=time;
}

void Stats::finish_stats() {
    this->avg_cost/=(double)this->config->runs;
    this->avg_time/=(double)this->config->runs;
}
