//
// Created by Rafael on 24/11/2024.
//

#include "Instance.h"

#include <algorithm>

Instance::Instance() {
    this->name="";
    this->n_node = 0;
    this->nodes = vector<Node>();
    this->customer_indexes[0] = 0;
    this->customer_indexes[1] = 0;
    this->locker_indexes[0] = 0;
    this->locker_indexes[1] = 0;

    this->customers_qty = 0;
    this->locker_qty = 0;

    this->load_capacity=0;
    this->avg_speed=0;
}

Instance::~Instance() {

    if(this->nodes.empty()) {
        for(int i=0;i<this->n_node;i++) {
            delete this->distances[i];
        }
        delete [] this->distances;
    }

}
void Instance::calculate_distances() {
    this->distances = new double*[this->n_node];
    for(int i=0; i<this->n_node; i++) {
        this->distances[i] = new double[this->n_node];
    }

    for(int i=0; i<this->n_node; i++) {
        for(int j=i; j<this->n_node;j++) {
            if(i==j) {
                this->distances[i][j] = 0.0;
            }else {
                double distance = sqrt(
                        pow(this->nodes.at(i).coord[0]-this->nodes.at(j).coord[0],2)
                        +
                        pow(this->nodes.at(i).coord[1]-this->nodes.at(j).coord[1],2)
                        );
                        distance = trunc(distance*10)/10;
                // distance = ceil(distance*10);
                this->distances[i][j] = distance;

                this->distances[j][i] = this->distances[i][j];
            }
        }
    }
}

void Instance::print() {
    cout<<"Params: "<<endl;
    cout<<"- CV: "+to_string(this->load_capacity)<<endl;
    cout<<"- V: "+to_string(this->avg_speed)<<endl;

    cout<<endl<<"Nodes:"<<endl;

    for(int i=0;i<this->n_node;i++) {
        this->nodes.at(i).print();
    }

    cout<<endl<<"Distances:"<<endl;

    for(int i=0;i<this->n_node;i++) {
        for(int j=0;j<this->n_node;j++) {
            cout<<this->distances[i][j]<<" ";
        }
        cout<<endl;
    }
}

Node * Instance::find_node_per_id(string id) {
    for(int i=0; i<(int)this->nodes.size(); i++) {
        if(this->nodes.at(i).id == id) {
            return &this->nodes.at(i);
        }
    }
    return nullptr;
}
