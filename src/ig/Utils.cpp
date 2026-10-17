//
// Created by Rafael on 24/11/2024.
//

#include "Utils.h"


void Utils::fix_instance(Instance* instance) {

    vector<string> instances_to_fix{
        "C104_co_25.txt",
        "C108_co_25.txt",
        "C201_co_25.txt",
        "C202_co_25.txt",
        "C203_co_25.txt",
        "C204_co_25.txt",
        "C205_co_25.txt",
        "C206_co_25.txt",
        "C207_co_25.txt"
    };

    for(string s: instances_to_fix) {
        if(s == instance->name) {
            cout<<"Fixing "<<instance->name<<endl;
            ifstream input_file;
            input_file.open("instances/vrppl/25/" + instance->name);

            ofstream output_file;
            output_file.open("instances/fixed_instances/fixed_"+instance->name,ofstream::out);
            output_file.close();
            output_file.open("instances/fixed_instances/fixed_"+instance->name,ofstream::app);
            if(input_file.is_open()) {
                string line;
                for(int i=0;i<55;i++) {
                    getline(input_file,line);
                    output_file<<line<<endl;
                }

                for(int i_cus = instance->customer_indexes[0]; i_cus < instance->customer_indexes[1]; i_cus++) {
                    Node* n = &instance->nodes.at(i_cus);
                    if(n->type == "c3" || n->type == "c2") {

                        Node* p0 = &instance->nodes.at(instance->locker_indexes[0]);
                        Node* p1 = &instance->nodes.at(instance->locker_indexes[0]+1);

                        if(instance->distances[n->index][p0->index] < instance->distances[n->index][p1->index]) {
                            output_file<<"1 0"<<endl;
                        }else {
                            output_file<<"0 1"<<endl;
                        }
                    }else{
                        output_file<<"0 0"<<endl;
                    }

                }
            }
            output_file.close();
            input_file.close();
        }
    }

}
vector<Instance *> Utils::buildInstances() {
    ifstream file;
    file.open("instances/instances.txt");
    string line;
    vector<Instance*> instances;

    if(file.is_open()) {
        while(getline(file,line)) {
            Instance* new_instance = nullptr;
            new_instance = Utils::buildInstance(line);
            instances.push_back(new_instance);
        }
    }else {
        cout<<"File not opened"<<endl;
    }

    file.close();
    return instances;
}


Instance * Utils::buildInstance(string fileName) {

    ifstream file;
    file.open(fileName);
    string line;

    if(!file.is_open()) {
        cout<<"File not found"<<endl;
        exit(1);
    }

    vector<Node> nodes;
    vector<string> splited_line;

    Instance* instance = new Instance();
    instance->avg_speed = 1.0;
    vector<string> instanceName = Utils::tookenize(fileName,"/");
    instance->name = instanceName.at(instanceName.size()-1);

    //Numero de clientes e número de lockers
    getline(file,line);
    splited_line = Utils::tookenize(line," ");
    instance->customers_qty = stoi(splited_line.at(0));
    instance->locker_qty = stoi(splited_line.at(1));
    instance->nodes.reserve(instance->n_node);

    //Numero de veíclos e capacidade de carga
    getline(file,line);
    splited_line = Utils::tookenize(line," ");
    instance->max_vehicle = stoi(splited_line.at(0));
    instance->load_capacity = stod(splited_line.at(1));

    Node node_d0,node_dt;
    node_d0.id = "D0";
    node_d0.index = 0;
    node_dt.id = "Dt";
    node_dt.index = 1;
    nodes.push_back(node_d0);
    nodes.push_back(node_dt);

    //Calculando load, index e ids
    for(int i=0; i<instance->customers_qty; i++) {
        getline(file,line);

        Node node;
        node.id = "C" + to_string(i);
        node.load_demand = stoi(line);
        node.index = nodes.size();
        nodes.push_back(node);
    }

    for(int i=0; i<instance->locker_qty; i++) {
        Node node;
        node.id = "P" + to_string(i);
        node.load_demand = 0;
        node.index = nodes.size();
        nodes.push_back(node);
    }

    instance->n_node = nodes.size();

    //Definindo coordenadas, TW, ST e Customer type
    for(int i=0;i<instance->n_node;i++) {
        if(i!=1) {
            getline(file,line);
        }
        splited_line = Utils::tookenize(line," ");
        Node* node = &nodes.at(i);
        node->coord[0] = stod(splited_line.at(0));
        node->coord[1] = stod(splited_line.at(1));


        switch (stoi(splited_line.at(5))) {
            case 0:
                nodes.at(i).type = "d";
            break;
            case 1:
                nodes.at(i).type = "c1";
            break;
            case 2:
                nodes.at(i).type = "c2";
            break;
            case 3:
                nodes.at(i).type = "c3";
            break;
            case 4:
                nodes.at(i).type = "p";
            break;
            default:
                nodes.at(i).type = "unknown";
            break;
        }

        node->service_time = stod(splited_line.at(4));

        if(node->type != "d" && node->type != "p") {
            node->time_window[0] = stod(splited_line.at(2));
            node->time_window[1] = stod(splited_line.at(3));
        }else {
            node->time_window[0] = 0.0;
            node->time_window[1] = BIG_M;
        }

    }

    instance->nodes = nodes;
    defineNodeIndexes(instance);
    for(int i=instance->customer_indexes[0];i<instance->customer_indexes[1];i++) {
        getline(file,line);
        if(instance->nodes.at(i).type != "c1") {
            splited_line = Utils::tookenize(line," ");
            for(int j=0;j<splited_line.size();j++) {
                if(splited_line.at(j) == "1") {
                    Node* locker = &instance->nodes.at(instance->locker_indexes[0] + j);
                    instance->nodes.at(i).designated_locker = locker;
                    locker->designated_customers.push_back(&instance->nodes.at(i));
                    break;  //TODO VER O QUE FAZER COM A INSTÂNCIA
                }
            }
        }
    }

    instance->calculate_distances();

    return instance;
}

void Utils::defineNodeIndexes(Instance *instance) {

    instance->customer_indexes[0] = 2;

    for(int i=2; i<instance->n_node; i++) {

        if(instance->nodes.at(i).type == "p") {
            instance->customer_indexes[1]=i;
            instance->locker_indexes[0]=i;
            instance->locker_indexes[1]=instance->n_node;
            break;
        }

    }
}


void Utils::print_result_file(Search *search, Instance *instance, int run, double time, long seed) {
    ofstream file;

    if(search == nullptr) {
        file.open("Output/"+instance->name+"_stats", ofstream::out);
        if (!file.is_open()) {
            cout<<"Output file not opened"<<endl;
        }
        file<<"run,cost,#_vehicles,time,seed"<<endl;
    }else{
        file.open("Output/"+search->instance->name+"_stats", ofstream::app);
        if (!file.is_open()) {
            cout<<"Output file not opened"<<endl;
        }
        file<<"--- "<<to_string(run)<<","<<to_string(search->solution->cost)<<","<<to_string(search->solution->used_routes)<<","<<to_string(time)<<","<<to_string(seed)<<endl;

        for (Route* route: search->solution->routes) {
            vector<Sequence>* route_sequences = &route->sequences;
            if(route_sequences->size()>2) {
                string route_string = "";

                for (Sequence sequence: *route_sequences) {
                    route_string+=sequence.node->id;

                    if (sequence.node->id != "Dt") {
                        if(sequence.node->type == "p") {
                            route_string+="(";

                            for(int i_customer=0; i_customer < (int)sequence.customers.size();i_customer++) {
                                Node* customer = sequence.customers.at(i_customer);
                                route_string+= customer->id;
                                if(i_customer < (int)sequence.customers.size() -1) {
                                    route_string+=",";
                                }
                            }

                            route_string+=")";
                        }
                        route_string+=" ";
                    }
                }
                file<<route_string<<endl;
            }

        }
        file<<endl;

    }


    file.close();

}


vector<string> Utils::tookenize(string str, string symbol) {

    stringstream ss(str);

    vector<string> tokens;
    string temp_str;

    while(getline(ss,temp_str,symbol[0])) {
        tokens.push_back(temp_str);
    }

    return tokens;

}


void Utils::print_final_stats(Stats *stats) {
    ofstream file;

    if(stats->instance->name == "C101_co_25.txt") {
        file.open("Output/0_final",ios::out);
        file<<"instance,avg_time,avg_cost,#_vehicles,best_cost"<<endl;
        file.close();
    }

    file.open("Output/0_final",ios::app);

    file<<stats->instance->name<<","<<to_string(stats->avg_time)<<","<<to_string(stats->avg_cost)<<","<<to_string(stats->best_solution->used_routes)<<","<<to_string(stats->best_solution->cost)<<endl;

    file.close();
}

void Utils::print_screen_run(Stats *stats) {
    cout<<"AVG_COST: "<<stats->avg_cost<<" - AVG_TIME: "<<stats->avg_time<<" - BEST_COST: "<<stats->best_solution->cost<<" - BEST_TIME: "<<stats->best_time<<endl;
}
//TODO testar
void Utils::test_print_viability(Solution *solution, long seed) {
    int inviabilities = 0;
    bool customer_viable = true;
    bool load_viable = true;
    bool load_sum_viable = true;
    bool time_window_viable = true;
    bool locker_attrib_viable = true;
    int used_vehicles = 0;

    vector<string> inviable_customers;
    vector<string> inviable_tws;
    vector<string> inviable_load_sum;
    vector<string> wrong_locker;

    //Verificando viabilidade de cliente atendido
    for(int i=solution->instance->customer_indexes[0];
        i<solution->instance->customer_indexes[1];
        i++) {

        if(!solution->served[i]) {
            customer_viable = false;
            Node* customer = &solution->instance->nodes.at(i);
            inviable_customers.push_back(customer->id);
            //inviabilities++;
        }

    }

    //Verificando viabilidade de load e tw
    for(Route* r: solution->routes ) {

        //Verificando viabilidade de load
        if(r->load > solution->instance->load_capacity) {

            load_viable = false;
            inviabilities++;
        }

        //Verificando se o somatorio de loads está certo
        double load_sum = 0.0;
        for(const Sequence& s: r->sequences) {
            for(Node* n: s.customers) {
                load_sum += n->load_demand;
            }
        }

        if(load_sum != r->load) {
            load_sum_viable = false;
            inviabilities++;
        }

        //Verificando viabilidade de tw
        for(const Sequence& s: r->sequences) {
            if((s.current_time < s.node->time_window[0] && Count::differs(s.current_time,s.node->time_window[0]))
                ||
                (s.current_time > s.node->time_window[1] && Count::differs(s.current_time,s.node->time_window[1]))) {
                Node* node = s.node;
                time_window_viable = false;
                inviable_tws.push_back(node->id);
                inviabilities++;
            }
        }

        //Contando veículos usados
        if((int)r->sequences.size() > 2) {
            used_vehicles++;
        }

        //Verificando se os clientes estão no locker certo
        for(const Sequence& s: r->sequences) {
            if(s.node->type == "p") {
                for(Node* n: s.customers) {
                    if(n->designated_locker != s.node) {
                        locker_attrib_viable = false;
                        wrong_locker.push_back(n->id);
                    }
                }

            }
        }


    }

    //Imprimindo resultados
    if(!customer_viable) {
        cout<<"Customer inviability ("<<(int)inviable_customers.size()<<"): (";
        for(const string& customer: inviable_customers) {
            cout<< customer<<", ";
        }
        cout<<endl;
    }

    if(!load_sum_viable) {
        cout<<"Load sum!"<<endl;
    }

    if(!load_viable) {
        cout<<"Load inviability ("<<endl;
    }

    if(!time_window_viable) {
        cout<<"Time window inviability ("<<(int)inviable_tws.size()<<"): (";
        for(string time: inviable_tws) {
            cout<< time;
            if(time != *(inviable_tws.end()-1)) {
                cout<<", ";
            }
        }
        cout<<")"<<endl;
        solution->print();
    }

    if(used_vehicles != solution->used_routes) {
        inviabilities++;
        cout<<"# Vehicle differs: "<<used_vehicles<<" x "<<solution->used_routes<<endl;
    }
    if(!locker_attrib_viable) {
        inviabilities++;
        cout<<"# Wrong locker: (";

        for(string s: wrong_locker) {
            cout<<s<<",";
        }
        cout<<")"<<endl;
    }

    //Saindo se houver inviabilidade
    if(inviabilities>0) {
        cout<<to_string(inviabilities)<<" inviabilities"<<endl;
        cout<<"SEED "<<to_string(seed)<<endl;
        exit(10);
    }



}
//TODO testar
void Utils::test_cost(Solution *solution) {
    double cost = 0.0;

    for(Route* route: solution->routes) {
        double travelled_dist = 0.0;
        for(int i=1;i<(int)route->sequences.size();i++) {
            Sequence prev_sequence = route->sequences.at(i-1);
            Sequence sequence = route->sequences.at(i);
            travelled_dist+= solution->instance->distances[prev_sequence.node->index][sequence.node->index];
        }
        if(travelled_dist != route->traveled_distance && Count::differs(travelled_dist,route->traveled_distance)) {
            cout<<"Erro de distancia da rota (E:"<<travelled_dist<<" G:"<<route->traveled_distance<<")"<< endl;
            exit(8);
        }
        cost += travelled_dist;
    }

    if( Count::differs(cost,solution->cost)) {
        cout<<"Erro de custo (E:"<<cost<<" G:"<<solution->cost<<")"<< endl;
        exit(8);
    }

}
//TODO testar
void Utils::print_candidate_list(vector<tuple<int, int, Sequence, double>> *cand_list) {
    for(tuple<int,int,Sequence, double> cand: *cand_list) {
        int cand_route = get<0>(cand);
        int cand_index = get<1>(cand);
        Sequence *cand_sequence = &get<2>(cand);

        if(cand_sequence->node!=cand_sequence->customers.at(0)) {
            cout<<cand_route<<" - "<<cand_index<<" - (";
            for(int i=0; i<(int)(cand_sequence->customers.size())-1;i++) {
                cout<<cand_sequence->customers.at(i)->id<<", ";
            }
            cout<<cand_sequence->customers.at((int)(cand_sequence->customers.size())-1)->id<<")"<<cand_sequence->node->id<<" - "<<get<3>(cand)<<endl;;
        }else {
            cout<<cand_route<<" - "<<cand_index<<" - "<<cand_sequence->node->id<<" - "<<get<3>(cand)<<endl;;
        }
    }
    cout<<endl;
}

void Utils::print_gurobi_variables(Solution *solution) {

    Instance* instance = solution->instance;
    vector<vector<int>> x_ijk;

    //inicializando vector
    for (int k = 0; k < solution->used_routes; k++) {
        x_ijk.push_back({});
        for (int i=0; i<instance->n_node - 1; i++) {

        }
    }
}


