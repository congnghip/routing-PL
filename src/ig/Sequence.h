//
// Created by Rafael on 11/01/2025.
//

#ifndef SEQUENCE_H
#define SEQUENCE_H
#include "Node.h"
#include "vector"
#include <unordered_map>


class Sequence {

public:
    Node* node = nullptr;
    vector<Node*> customers;
    char method = 'u';
    double time_off = 0.0;
    double current_time = 0.0;

    void clone_this_to(Sequence* clone) {
        clone->node = this->node;
        clone->customers = this->customers; //TODO testar
        clone->method = this->method;
        clone->time_off = this->time_off;
        clone->current_time = this->current_time;
    }

    /*void reset_values() {
        this->method = 'u';
        this->time_off = 0.0;
        this->current_time = 0.0;
        //this->sequence_demand = 0.0;
    }*/
};


#endif //SEQUENCE_H
