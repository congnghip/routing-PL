//
// Created by Rafael on 24/11/2024.
//

#ifndef INSTANCE_H
#define INSTANCE_H
#include <string>
#include <vector>
#include "Node.h"
#include <math.h>

using namespace std;
class Instance {
public:
    string name;
    int n_node;
    vector<Node> nodes;
    int customer_indexes[2]; //[i,j)
    int locker_indexes[2]; //[i,j)
    int customers_qty;
    int locker_qty;
    double load_capacity; //CV
    double avg_speed; //v
    double** distances;
    int max_vehicle;


    explicit Instance();
    ~Instance();

    void calculate_distances();
    void print();
    Node* find_node_per_id(string id);
};



#endif //INSTANCE_H
