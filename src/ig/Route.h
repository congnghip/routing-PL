//
// Created by Rafael on 06/03/2025.
//

#ifndef ROUTE_H
#define ROUTE_H
#include <vector>
#include "Instance.h"
#include "Sequence.h"

using namespace std;
class Route {
public:
    Route(Instance* instance);

    string get_route_string(int route_label);
    void clone_this_to(Route* r);



    Instance* instance;
    vector<Sequence> sequences;
    unordered_map<Node*,int> visited_lockers; //<locker, contador>
    double traveled_distance = 0.0;
    double load = 0.0;
    double minimun_route_load = 0.0;


};



#endif //ROUTE_H
