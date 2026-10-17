//
// Created by Rafael on 08/03/2025.
//
#include "Search.h"


bool Search::swap_1_1_broke_load(Route *route_a, Sequence* seq_a, Route *route_b, Sequence* seq_b) {

    double seq_a_demand = 0.0;
    double seq_b_demand = 0.0;

    for(Node* n: seq_a->customers) {
        seq_a_demand += n->load_demand;
    }

    for(Node* n: seq_b->customers) {
        seq_b_demand += n->load_demand;
    }

    bool broke_a = (route_a->load  - seq_a_demand + seq_b_demand) > this->instance->load_capacity;
    bool broke_b = (route_b->load  - seq_b_demand + seq_a_demand) > this->instance->load_capacity;
    return broke_a || broke_b;

}

bool Search::swap_2_2_broke_load(Route *route_a, Sequence *seq_a_1, Sequence *seq_a_2,
                                 Route *route_b, Sequence *seq_b_1, Sequence *seq_b_2) {

    double seq_a_demand = 0.0;

    for(Node* n: seq_a_1->customers) {
        seq_a_demand += n->load_demand;
    }

    for(Node* n: seq_a_2->customers) {
        seq_a_demand += n->load_demand;
    }

    double seq_b_demand = 0.0;

    for(Node* n: seq_b_1->customers) {
        seq_b_demand += n->load_demand;
    }

    for(Node* n: seq_b_2->customers) {
        seq_b_demand += n->load_demand;
    }



    bool broke_a = (route_a->load  - seq_a_demand + seq_b_demand) > this->instance->load_capacity;
    bool broke_b = (route_b->load  - seq_b_demand + seq_a_demand) > this->instance->load_capacity;
    return broke_a || broke_b;
}

bool Search::shift_1_0_broke_load(Sequence *seq_a, Route *route_b) {

    double seq_a_demand = 0.0;

    for(Node* n: seq_a->customers) {
        seq_a_demand += n->load_demand;
    }

    return  (route_b->load + seq_a_demand) > this->instance->load_capacity;
}

bool Search::swap_2_0_broke_load(Sequence *seq_a_1, Sequence* seq_a_2, Route *route_b) {

    double seq_a_demand = 0.0;

    for(Node* n: seq_a_1->customers) {
        seq_a_demand += n->load_demand;
    }

    for(Node* n: seq_a_2->customers) {
        seq_a_demand += n->load_demand;
    }

    return  (route_b->load + seq_a_demand) > this->instance->load_capacity;
}

bool Search::swap_2_1_broke_load(Route *route_a, Sequence* seq_a_1, Sequence *seq_a_2, Route *route_b, Sequence* seq_b) {
    double seq_a_demand = 0.0;

    for(Node* n: seq_a_1->customers) {
        seq_a_demand += n->load_demand;
    }

    for(Node* n: seq_a_2->customers) {
        seq_a_demand += n->load_demand;
    }

    double seq_b_demand = 0.0;

    for(Node* n: seq_b->customers) {
        seq_b_demand += n->load_demand;
    }


    bool broke_a = (route_a->load - seq_a_demand + seq_b_demand) > this->instance->load_capacity;
    bool broke_b = (route_b->load - seq_b_demand + seq_a_demand) > this->instance->load_capacity;
    return broke_a || broke_b;

}

bool Search::broke_time_window() {

    return this->virtual_sequence->current_time < this->virtual_sequence->node->time_window[0] ||
        this->virtual_sequence->current_time > this->virtual_sequence->node->time_window[1];
}