//
// Created by Rafael on 27/01/2025.
//

#include "Solution.h"

#include "Counter.h"


Solution::Solution(Instance* instance) {
    this->instance = instance;
    this->used_routes = 0;
    this->cost = 0.0;

    for (int i=0;i<this->instance->n_node;i++) {
        this->served.push_back(false);
    }

    for(int i=0;i<this->instance->max_vehicle;i++) {
        Route* r = new Route(instance);
        this->routes.push_back(r);
    }

}
//TODO testar
Solution::~Solution() {
    for(Route * r: this->routes) {
        delete r;
    }
}

void Solution::calculate_total_cost() {
    this->cost = 0.0;
    for (Route* r : this->routes) {
        this->cost += r->traveled_distance;

        if((int)r->sequences.size() == 2) {
            break;
        }

    }
}

// TODO testar
void Solution::print() {
    vector<string> route_strings;

    for (int i=0; i<this->instance->max_vehicle; i++) {
        if((int)this->routes.at(i)->sequences.size()>2) {
            route_strings.push_back(this->routes.at(i)->get_route_string(i));
        }else {
            break;
        }
    }

    cout<<"TOTAL COST: "<<this->cost<<endl;
    cout<<"PATHS: "<<endl;

    for(const string& s: route_strings) {
        cout<<s<<endl;
    }
}

void Solution::clone_this_to(Solution *s) {

    s->instance = this->instance;

    for(int i_route = 0; i_route < this->instance->max_vehicle; i_route++) {
        Route* this_route = this->routes.at(i_route);
        Route* s_route = s->routes.at(i_route);
        this_route->clone_this_to(s_route);
    }

    //s->routes = this->routes;
    s->served = this->served;
    s->cost = this->cost;
    s->used_routes = this->used_routes;

}
