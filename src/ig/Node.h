//
// Created by Rafael on 24/11/2024.
//

#ifndef NODE_H
#define NODE_H
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;
class Node {
public:
    string id;
    string type;
    int index;
    double coord[2]; //{x,y}
    double time_window[2]; // [e_i,l_i]
    double load_demand;
    double service_time;
    Node* designated_locker;
    Node* nearest_station;
    vector<Node*> designated_customers;
    int customers_qty;

    Node();
    ~Node();
    void print();

};



#endif //NODE_H
