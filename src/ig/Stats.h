//
// Created by Rafael on 12/01/2025.
//

#ifndef STATS_H
#define STATS_H
#include "Config.h"
#include "Instance.h"
#include "Solution.h"


class Stats {
public:
    Stats(Instance* instance, Config* config);
    void set_result(Solution *solution, double time);
    void finish_stats();

    Instance* instance;
    Config* config;
    vector<double> costs;
    vector<double> times;

    Solution* best_solution = nullptr;
    double best_time = 0.0;
    double avg_cost = 0.0;
    double avg_time = 0.0;


};



#endif //STATS_H
