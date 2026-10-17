//
// Created by Rafael on 24/11/2024.
//

#include "Node.h"


Node::Node() {
    this->id="";
    this->type="";
    this->index = -1;
    this->coord[0]=0;
    this->coord[1]=0;
    this->time_window[0]=0;
    this->time_window[1]=0;
    this->load_demand=0;
    this->service_time=0;
    this->designated_locker=nullptr;
    this->nearest_station=nullptr;
    this->designated_customers = vector<Node*>();
    this->customers_qty = 0;
}

Node::~Node() {

}

void Node::print() {
    cout<<setprecision(2)<<fixed;
    cout<<"ID: "+this->id+" - TP:"+this->type + " - XY("+to_string(this->coord[0])
    +","+to_string(this->coord[1])+")"+ " - TW("+to_string(this->time_window[0])
    +","+to_string(this->time_window[1])+") - Servtime: "+to_string(this->service_time)<<endl;
}
