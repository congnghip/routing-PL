//
// Created by Rafael on 27/01/2025.
//

#ifndef SOLUTION_H
#define SOLUTION_H
#include "Instance.h"
#include "Route.h"
#include "Sequence.h"
#include <algorithm>


class Solution {
public:
    Solution(Instance* instance);
    ~Solution();

    void initialize_routes();
    void calculate_total_cost();
    void print();

    void clone_this_to(Solution *s);

    Instance* instance;
    vector<Route*> routes;
    vector<bool> served;
    double cost;
    int used_routes;
};



#endif //SOLUTION_H
