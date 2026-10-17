//
// Created by Rafael on 24/11/2024.
//

#ifndef UTILS_H
#define UTILS_H
#include "Instance.h"
#include <vector>
#include "Search.h"
#include "Stats.h"

#define BIG_M 1000000

class Utils {
public:
    //Instance builder
    static vector<Instance*> buildInstances();
    static Instance* buildInstance(string fileName);
    static void defineNodeIndexes(Instance* instance);

    static void fix_instance(Instance* instance);

    static void print_result_file(Search *search, Instance *instance, int run, double time, long seed);
    static vector<string> tookenize(string str, string symbol);


    static void print_final_stats(Stats* stats);
    static void print_screen_run(Stats* stats);
    static void test_print_viability(Solution *solution, long seed);
    static void test_cost(Solution* solution);
    static void print_candidate_list(vector<tuple<int, int, Sequence, double>> *cand_list);
    static void print_gurobi_variables(Solution* solution);

};



#endif //UTILS_H
