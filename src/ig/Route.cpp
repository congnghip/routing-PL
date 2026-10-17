//
// Created by Rafael on 06/03/2025.
//

#include "Route.h"

Route::Route(Instance* instance) {
    this->instance = instance;
    this->sequences.reserve(this->instance->n_node);

    Sequence depot_0, depot_t;
    depot_0.node = &this->instance->nodes.at(0);
    depot_0.customers = {};
    depot_t.node = &this->instance->nodes.at(1);
    depot_t.customers = {};
    this->sequences = {depot_0,depot_t};

    for(int i=this->instance->locker_indexes[0]; i<this->instance->locker_indexes[1];i++) {
        Node* locker = &this->instance->nodes[i];
        this->visited_lockers[locker] = 0;
    }
}

string Route::get_route_string(int route_label) {
    string s= "V_"+to_string(route_label)+": ";
    for (int i_seq=0; i_seq < (int)this->sequences.size(); i_seq++) {
        Sequence* sequence = &this->sequences.at(i_seq);
        s+=sequence->node->id;
        if (sequence->node->type == "p") {
            s+="(";

            for(int i_customer=0; i_customer < (int)sequence->customers.size();i_customer++) {
                Node* customer = sequence->customers.at(i_customer);
                s+= customer->id;
                if(i_customer < (int)sequence->customers.size() -1) {
                    s+=", ";
                }
            }

            s+=")";
        }
        s+=" -> ";
    }

    return s;
}

void Route::clone_this_to(Route *r) {
    r->instance = this->instance;
    r->sequences = this->sequences;
    r->visited_lockers = this->visited_lockers;
    r->traveled_distance = this->traveled_distance;
    r->load = this->load;
    r->minimun_route_load = this->minimun_route_load;
}
