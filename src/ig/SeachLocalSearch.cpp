//
// Created by Rafael on 09/03/2025.
//

#include "Search.h"
#include "Utils.h"
//LS INTRA ROUTE
void Search::ls_intra_exchange(bool *improved) {

    for(int i_route=0; i_route<(int)this->solution->routes.size();i_route++) {
        Route* route = this->solution->routes.at(i_route);
        vector<Sequence> *route_sequences = &route->sequences;
        if(route->sequences.size()>4) {

            double best_delta = 0.0;
            int coordinates[2] = {-1,-1}; //i_seq_a,i_seq_b

            for(int i_seq_a=1;i_seq_a<(int)(route_sequences->size()-2);i_seq_a++) {

                Sequence* seq_a = &route_sequences->at(i_seq_a);

                for(int i_seq_b = i_seq_a+1; i_seq_b<(int)(route_sequences->size()-1);i_seq_b++) {
                    Sequence* seq_b = &route_sequences->at(i_seq_b);

                    if(seq_a->node->id != seq_b->node->id) {
                        double delta = calculate_delta_exchange(route_sequences, i_seq_a,i_seq_b);

                        if(Count::improves(0.0,delta) &&
                        Count::improves(best_delta,delta)) {
                            //Testar viabilidade
                            if(propagate_virtual_exchange(i_route,i_seq_a,i_seq_b)) {
                                best_delta = delta;
                                coordinates[0] = i_seq_a;
                                coordinates[1] = i_seq_b;
                            }

                        }
                    }

                }

            }

            if(best_delta<0.0) {
                *improved = true;
                swap_sequence_intraroute(i_route,coordinates[0],coordinates[1], best_delta);
                propagate(i_route,coordinates[0]-1);
                this->solution->cost+=best_delta;
                i_route--;
            }
        }
    }
}


void Search::ls_intra_2opt(bool *improved) {

    for (int i_route = 0;i_route<(int)this->solution->routes.size();i_route++) {

        double best_delta = 0.0;
        int coordinates[2] = {-1,-1}; //i_seq_a,i_seq_b

        Route* route = this->solution->routes.at(i_route);
        vector<Sequence>* route_sequences = &route->sequences;
           for(int i_seq_a = 0; i_seq_a<((int)(route_sequences->size())-3);i_seq_a++) {
               Sequence* seq_a = &route_sequences->at(i_seq_a);

               for(int i_seq_b = i_seq_a+5; i_seq_b<(int)route_sequences->size();i_seq_b++) {
                   Sequence* seq_b = &route_sequences->at(i_seq_b);

                   //Inverter a rota inteira não faz efeito algum
                   if(seq_a->node->id =="D0" && seq_b->node->id == "Dt") {break;}

                   double delta = calculate_delta_2opt(route_sequences,i_seq_a, i_seq_b);

                   if(Count::improves(0.0,delta) &&
                       Count::improves(best_delta,delta)) {
                        //Testar viabilidade
                       if(propagate_virtual_2opt(i_route,i_seq_a,i_seq_b)) {
                           best_delta = delta;
                           coordinates[0] = i_seq_a;
                           coordinates[1] = i_seq_b;
                       }
                   }

               }


           }

            if(best_delta<0.0) {
                *improved = true;
                reverse(route_sequences->begin()+coordinates[0]+1,route_sequences->begin()+coordinates[1]);
                propagate(i_route,coordinates[0]);
                route->traveled_distance += best_delta;
                this->solution->cost += best_delta;
                i_route--;
            }

    }

}

void Search::ls_intra_or_opt_1(bool *improved) {
    bool is_reduction = false;
    for (int i_route = 0;i_route<(int)this->solution->routes.size();i_route++) {

        double best_delta = 0.0;
        int coordinates[2] = {-1,-1}; //i_seq_a,i_seq_b

        Route* route = this->solution->routes.at(i_route);
        vector<Sequence>* route_sequences = &route->sequences;
        for(int i_seq_a = 1; i_seq_a<((int)(route_sequences->size())-2);i_seq_a++) {
            Sequence* seq_a = &route_sequences->at(i_seq_a);
            Node* node_a = seq_a->node;

            double delta;
            //Verificando no se o nó é um cliente flex e se existe o seu locker na rota
            if(node_a->type == "c3" && route->visited_lockers[node_a->designated_locker] > 0) {
                int i_locker = 0;
                while(route->sequences[++i_locker].node != node_a->designated_locker);

                delta = this->calculate_delta_destruction(route_sequences,i_seq_a);
                if(Count::improves(0.0,delta) &&
                                        Count::improves(best_delta,delta)) {
                    best_delta = delta;
                    coordinates[0] = i_seq_a;
                    coordinates[1] = i_locker;
                    is_reduction = true;
                }
            }else {

                Sequence* seq_b = nullptr;
                //Seq_b é o indice do cara que vem depois do novo destino do seq_a. Seq_b eh empurrado

                for(int i_seq_b = 1; i_seq_b<(int)(route_sequences->size());i_seq_b++) {
                    //Não faz sentido testar com i=j ou com o vizinho mais proximo (Vizinhança ja eh contemplada pelo exchange)
                    if (abs(i_seq_a-i_seq_b)>1) {
                        seq_b = &route_sequences->at(i_seq_b);
                        delta = calculate_delta_or_opt_1(route_sequences,i_seq_a,i_seq_b);
                        if(Count::improves(0.0,delta) &&
                            Count::improves(best_delta,delta)) {
                            if (i_seq_a<i_seq_b) {
                                if (propagate_virtual_or_opt_1_up(i_route,i_seq_a,i_seq_b)) {
                                    best_delta = delta;
                                    coordinates[0] = i_seq_a;
                                    coordinates[1] = i_seq_b;
                                    is_reduction = false;
                                }
                            }else {
                                if (propagate_virtual_or_opt_1_down(i_route,i_seq_a,i_seq_b)) {
                                    best_delta = delta;
                                    coordinates[0] = i_seq_a;
                                    coordinates[1] = i_seq_b;
                                    is_reduction = false;
                                }
                            }

                        }
                    }
                }
            }



        }


        if (best_delta < 0.0) {
            *improved = true;
            if(is_reduction) {
                Sequence* seq_a = &route->sequences.at(coordinates[0]);
                Sequence* seq_b = &route->sequences.at(coordinates[1]);
                seq_b->customers.push_back(seq_a->node);
                route_sequences->erase(route_sequences->begin() + coordinates[0],route_sequences->begin() + coordinates[0] + 1);

                propagate(i_route,coordinates[0]-1);

            }else {
                shift(route_sequences,coordinates[0],coordinates[1]);
                propagate(i_route,min(coordinates[0],coordinates[1])-1);
            }

            route->traveled_distance += best_delta;
            this->solution->cost += best_delta;
            i_route--;
        }

    }
}

void Search::ls_intra_or_opt_k(int k, bool *improved) {

    for (int i_route = 0;i_route<(int)this->solution->routes.size();i_route++) {

        Route* route = this->solution->routes.at(i_route);
        vector<Sequence>* route_sequences = &route->sequences;

        if (route_sequences->size()>(k+2)) {

            double best_delta = 0.0;
            int coordinates[2] = {-1,-1}; //i_seq_a,i_seq_b

            for(int i_seq_a = 1; i_seq_a<((int)(route_sequences->size())-k);i_seq_a++) {
                Sequence* seq_a = &route_sequences->at(i_seq_a);
                Sequence* seq_b = nullptr;
                //Pra traz
                for (int i_seq_b = i_seq_a-1; i_seq_b>0;i_seq_b--) {
                    seq_b = &route_sequences->at(i_seq_b);

                    double delta = calculate_delta_or_opt_k(k,route_sequences,i_seq_a,i_seq_b);

                    if(Count::improves(0.0,delta) &&
                        Count::improves(best_delta,delta)) {
                        if (propagate_virtual_or_opt_k_down(k,i_route,i_seq_a,i_seq_b)) {
                            best_delta = delta;
                            coordinates[0] = i_seq_a;
                            coordinates[1] = i_seq_b;
                        }

                    }
                }

                //Pra frente
                for(int i_seq_b = i_seq_a+k+1; i_seq_b<(int)route_sequences->size();i_seq_b++) {
                    seq_b = &route_sequences->at(i_seq_b);

                    double delta = calculate_delta_or_opt_k(k,route_sequences,i_seq_a,i_seq_b);

                    if(Count::improves(0.0,delta) &&
                        Count::improves(best_delta,delta)) {
                        if (propagate_virtual_or_opt_k_up(k,i_route,i_seq_a,i_seq_b)) {
                            best_delta = delta;
                            coordinates[0] = i_seq_a;
                            coordinates[1] = i_seq_b;
                        }
                    }

                }
            }
            if (best_delta < 0.0) {
                *improved = true;
                shift_k(k,route_sequences,coordinates[0],coordinates[1]);
                propagate(i_route,min(coordinates[0],coordinates[1])-1);
                route->traveled_distance += best_delta;
                this->solution->cost+= best_delta;
                i_route--;
            }
        }


    }
}
//LS INTER ROUTE

void Search::ls_inter_swap_1_1(bool *improved) {
    double best_delta = 0.0;
    int coordinates[4] = {-1,-1,-1,-1}; //i_route_a,i_seq_a,i_route_b,i_seq_b
    Route* route_a = nullptr;
    Route* route_b = nullptr;
    vector<Sequence> * route_a_sequences = nullptr;
    vector<Sequence> * route_b_sequences = nullptr;
    Sequence * seq_a = nullptr;
    Sequence * seq_b = nullptr;
    for (int i_route_a = 0;i_route_a<((int)this->solution->routes.size()-1);i_route_a++) {
        route_a = this->solution->routes.at(i_route_a);
        route_a_sequences = &route_a->sequences;
        if(route_a_sequences->size()>2) {
            for (int i_route_b = i_route_a+1;i_route_b<(int)this->solution->routes.size();i_route_b++) {
                route_b = this->solution->routes.at(i_route_b);
                route_b_sequences = &route_b->sequences;

                if(route_b_sequences->size()>2) {

                    for(int i_seq_a=1; i_seq_a<((int)route_a_sequences->size()-1);i_seq_a++) {
                        seq_a = &route_a_sequences->at(i_seq_a);

                        for(int i_seq_b = 1; i_seq_b<((int)route_b_sequences->size()-1);i_seq_b++) {
                            seq_b = &route_b_sequences->at(i_seq_b);

                            //Trocar locker com locker não faz efeito
                            if(seq_a->node->id != seq_b->node->id) {

                                if(!swap_1_1_broke_load(route_a,seq_a,route_b,seq_b)) {
                                    double delta = calculate_delta_swap_1_1(route_a_sequences,i_seq_a,route_b_sequences,i_seq_b,BOTH);
                                    if(Count::improves(0.0,delta) &&
                                        Count::improves(best_delta,delta)) {
                                        if(propagate_virtual_swap_1_1(i_route_b,i_seq_b-1,seq_a) &&
                                            propagate_virtual_swap_1_1(i_route_a,i_seq_a-1,seq_b)) {
                                            best_delta = delta;
                                            coordinates[0] = i_route_a;
                                            coordinates[1] = i_seq_a;
                                            coordinates[2] = i_route_b;
                                            coordinates[3] = i_seq_b;
                                        }
                                    }

                                }

                            }

                        }
                    }
                }else {
                    break;
                }
            }
        }else {
            break;
        }
    }

    if(best_delta<0.0) {
        *improved = true;
        persist_swap_1_1(coordinates,best_delta);
    }

}

void Search::ls_inter_swap_2_2(bool *improved) {
    double best_delta = 0.0;
    int coordinates[4] = {-1,-1,-1,-1}; //i_route_a,i_seq_a,i_route_b,i_seq_b
    Route * route_a = nullptr;
    Route * route_b = nullptr;
    vector<Sequence> * route_a_sequences = nullptr;
    vector<Sequence> * route_b_sequences = nullptr;
    Sequence * seq_a_1 = nullptr;
    Sequence * seq_a_2 = nullptr;
    Sequence * seq_b_1 = nullptr;
    Sequence * seq_b_2 = nullptr;

    for (int i_route_a = 0;i_route_a<(int)this->solution->routes.size();i_route_a++) {
        route_a = this->solution->routes.at(i_route_a);
        route_a_sequences = &route_a->sequences;
        if(route_a_sequences->size()>3) {
            for (int i_route_b = i_route_a + 1;i_route_b<(int)this->solution->routes.size();i_route_b++) {

                if(i_route_b != i_route_a) {
                    route_b = this->solution->routes.at(i_route_b);
                    route_b_sequences = &route_b->sequences;
                    if(route_b_sequences->size()>3) {
                        for(int i_seq_a=1; i_seq_a<((int)route_a_sequences->size()-2);i_seq_a++) {
                            seq_a_1 = &route_a_sequences->at(i_seq_a);
                            seq_a_2 = &route_a_sequences->at(i_seq_a+1);

                            for(int i_seq_b = 1; i_seq_b<((int)route_b_sequences->size()-2);i_seq_b++) {
                                seq_b_1 = &route_b_sequences->at(i_seq_b);
                                seq_b_2 = &route_b_sequences->at(i_seq_b+1);

                                //Trocar locker com locker não faz efeito TODO testar depois
                                    if(!swap_2_2_broke_load(route_a,seq_a_1,seq_a_2,route_b,seq_b_1,seq_b_2)) {
                                        double delta = calculate_delta_swap_2_2(route_a_sequences,i_seq_a,route_b_sequences,i_seq_b,BOTH);
                                        if(Count::improves(0.0,delta) &&
                                            Count::improves(best_delta,delta)) {
                                            if(propagate_virtual_swap_2_2(i_route_a,i_seq_a-1,seq_b_1,seq_b_2) &&
                                                propagate_virtual_swap_2_2(i_route_b,i_seq_b-1,seq_a_1,seq_a_2)) {
                                                best_delta = delta;
                                                coordinates[0] = i_route_a;
                                                coordinates[1] = i_seq_a;
                                                coordinates[2] = i_route_b;
                                                coordinates[3] = i_seq_b;
                                                }
                                            }

                                    }

                            }
                        }
                    }
                }
            }
        }
    }
    if(best_delta<0.0) {
        *improved = true;
        this->persist_swap_2_2(coordinates,best_delta);
    }

}


void Search::ls_inter_split(bool *improved) {
    double best_delta = 0.0;
    int coordinates[3] = {-1,-1,-1};
    Route * route_a = nullptr;
    Route * route_b = this->solution->routes.at(this->solution->used_routes);
    vector<Sequence> * route_a_sequences = nullptr;
    vector<Sequence> * route_b_sequences = &route_b->sequences;
    Sequence * seq_a_1 = nullptr;
    Sequence * seq_a_2 = nullptr;
    Sequence * seq_b = nullptr;

    for (int i_route_a = 0; i_route_a < this->solution->used_routes ;i_route_a++) {
        route_a = this->solution->routes.at(i_route_a);
        route_a_sequences = &route_a->sequences;

        if ((int)route_a_sequences->size()>3) {

            for (int i_seq_a_1 = 1; i_seq_a_1<((int)route_a_sequences->size() - 1);i_seq_a_1++) {
                seq_a_1 = &route_a_sequences->at(i_seq_a_1);

                for (int i_seq_a_2 = i_seq_a_1; i_seq_a_2 < ((int)route_a_sequences->size()-1);i_seq_a_2++) {
                    if (i_seq_a_1 == 1 && i_seq_a_2 == (int)route_a_sequences->size()-2) {break;}
                    seq_a_2 = &route_a_sequences->at(i_seq_a_2);

                    double delta = this->calculate_delta_split(route_a_sequences,i_seq_a_1, i_seq_a_2, 'c');
                    if (Count::improves(best_delta,delta)) {
                        coordinates[0] = i_route_a;
                        coordinates[1] = i_seq_a_1;
                        coordinates[2] = i_seq_a_2;
                        best_delta = delta;
                    }
                }
            }
        }
    }

    if (best_delta < 0.0) {
        *improved = true;
        this->persist_split(coordinates,best_delta);
    }
}

void Search::ls_inter_shift_1_0(bool *improved) {
    double best_delta = 0.0;
    int coordinates[4] = {-1,-1,-1,-1}; //i_route_a,i_seq_a,i_route_b,i_seq_b
    Route* route_a = nullptr;
    Route* route_b = nullptr;
    vector<Sequence> * route_a_sequences = nullptr;
    vector<Sequence> * route_b_sequences = nullptr;
    Sequence * seq_a = nullptr;
    Sequence * seq_b = nullptr;
    bool is_reduction = false;

    for (int i_route_a = 0;i_route_a<this->solution->used_routes;i_route_a++) {
        route_a = this->solution->routes.at(i_route_a);
        route_a_sequences = &route_a->sequences;

        for (int i_route_b = 0;i_route_b<this->solution->used_routes;i_route_b++) {

            if(i_route_a!=i_route_b) {
                route_b = this->solution->routes.at(i_route_b);
                route_b_sequences = &route_b->sequences;

                //Verificando se pelo menos um cliente da rota A cabe na rota B
                if((route_b->load + route_a->minimun_route_load) <= this->instance->load_capacity) {
                    //Aponta pro nó de A que vai
                    for(int i_seq_a = 1; i_seq_a<((int)route_a_sequences->size()-1);i_seq_a++) {
                        seq_a = &route_a_sequences->at(i_seq_a);

                        //Verificando se o nó da seq A cabe na rota B
                        if(!this->shift_1_0_broke_load(seq_a,route_b)) {

                            Node* node_a = seq_a->node;
                            double delta;

                            //Verificando no se o nó é um cliente flex e se existe o seu locker na rota de destino
                            if(node_a->type == "c3" && route_b->visited_lockers[node_a->designated_locker] > 0) {
                                int i_locker = 0;
                                while(route_b->sequences[++i_locker].node != node_a->designated_locker);

                                delta = this->calculate_delta_destruction(route_a_sequences,i_seq_a);

                                if(Count::improves(0.0,delta) &&
                                    Count::improves(best_delta,delta)) {
                                    best_delta = delta;
                                    coordinates[0] = i_route_a;
                                    coordinates[1] = i_seq_a;
                                    coordinates[2] = i_route_b;
                                    coordinates[3] = i_locker;
                                    is_reduction = true;
                                }


                            }else {

                                //Aponta pro nó de B que vai vir antes
                                for(int i_seq_b = 0; i_seq_b<((int)route_b_sequences->size()-1);i_seq_b++) {
                                    seq_b = &route_b_sequences->at(i_seq_b); //TODO depois remover

                                    delta = calculate_delta_shift_1_0(route_a_sequences,i_seq_a,route_b_sequences,i_seq_b,BOTH);
                                    if(Count::improves(0.0,delta) &&
                                        Count::improves(best_delta,delta)) {
                                        if(propagate_virtual(i_route_b,i_seq_b,seq_a)) {
                                            best_delta = delta;
                                            coordinates[0] = i_route_a;
                                            coordinates[1] = i_seq_a;
                                            coordinates[2] = i_route_b;
                                            coordinates[3] = i_seq_b;
                                            is_reduction = false;
                                        }
                                    }
                                }

                            }


                        }
                    }
                }
            }
        }
    }

    if(best_delta<0.0) {
        *improved = true;
        this->persist_shift_1_0(coordinates,best_delta, is_reduction);
    }
}


void Search::ls_inter_shift_2_0(bool *improved) {

    double best_delta = 0.0;
    int coordinates[4] = {-1,-1,-1,-1}; //i_route_a,i_seq_a,i_route_b,i_seq_b
    Route* route_a = nullptr;
    Route* route_b = nullptr;
    vector<Sequence> * route_a_sequences = nullptr;
    vector<Sequence> * route_b_sequences = nullptr;
    Sequence * seq_a_1 = nullptr;
    Sequence * seq_a_2 = nullptr;
    Sequence * seq_b = nullptr;
    for (int i_route_a = 0;i_route_a<this->solution->used_routes;i_route_a++) {
        route_a = this->solution->routes.at(i_route_a);
        route_a_sequences = &route_a->sequences;
        if((int)route_a_sequences->size() > 3) {
            for (int i_route_b = 0;i_route_b<this->solution->used_routes;i_route_b++) {

                if(i_route_a!=i_route_b) {
                    route_b = this->solution->routes.at(i_route_b);
                    route_b_sequences = &route_b->sequences;

                    //Verificando se pelo menos um cliente da rota A cabe na rota B
                    if((route_b->load + route_a->minimun_route_load) <= this->instance->load_capacity) {
                        //Aponta pro nó de A que vai
                        for(int i_seq_a = 1; i_seq_a<((int)route_a_sequences->size()-2);i_seq_a++) {
                            seq_a_1 = &route_a_sequences->at(i_seq_a);
                            seq_a_2 = &route_a_sequences->at(i_seq_a+1);

                            //Verificando se os nós da seq A cabe na rota B
                            if(!this->swap_2_0_broke_load(seq_a_1,seq_a_2,route_b)) {

                                //Aponta pro nó de B que vai vir antes
                                for(int i_seq_b = 0; i_seq_b<((int)route_b_sequences->size()-1);i_seq_b++) {
                                    seq_b = &route_b_sequences->at(i_seq_b); //TODO depois remover

                                    double delta = calculate_delta_shift_2_0(route_a_sequences,i_seq_a,route_b_sequences,i_seq_b, BOTH);
                                    if(Count::improves(0.0,delta) &&
                                        Count::improves(best_delta,delta)) {

                                        if(propagate_virtual_shift_2_0(i_route_b,i_seq_b,seq_a_1,seq_a_2)) {
                                            best_delta = delta;
                                            coordinates[0] = i_route_a;
                                            coordinates[1] = i_seq_a;
                                            coordinates[2] = i_route_b;
                                            coordinates[3] = i_seq_b;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if(best_delta<0.0) {
        *improved = true;
        this->persist_shift_2_0(coordinates,best_delta);
    }
}

void Search::ls_inter_swap_2_1(bool *improved) {
    double best_delta = 0.0;
    int coordinates[4] = {-1,-1,-1,-1}; //i_route_a,i_seq_a,i_route_b,i_seq_b
    Route * route_a = nullptr;
    vector<Sequence> * route_a_sequences = nullptr;
    Sequence * seq_a_1 = nullptr;
    Sequence * seq_a_2 = nullptr;
    Route * route_b = nullptr;
    vector<Sequence> * route_b_sequences = nullptr;
    Sequence * seq_b = nullptr;

    for (int i_route_a = 0;i_route_a<(int)this->solution->routes.size();i_route_a++) {
        route_a = this->solution->routes.at(i_route_a);
        route_a_sequences = &route_a->sequences;
        if((int)route_a_sequences->size()>3) {
            for (int i_route_b = 0;i_route_b<(int)this->solution->routes.size();i_route_b++) {

                if(i_route_b != i_route_a) {
                    route_b = this->solution->routes.at(i_route_b);
                    route_b_sequences = &route_b->sequences;
                    if((int)route_b_sequences->size()>2) {
                        for(int i_seq_a=1; i_seq_a<((int)route_a_sequences->size()-2);i_seq_a++) {
                            seq_a_1 = &route_a_sequences->at(i_seq_a);
                            seq_a_2 = &route_a_sequences->at(i_seq_a+1);

                            for(int i_seq_b = 1; i_seq_b<((int)route_b_sequences->size()-1);i_seq_b++) {
                                seq_b = &route_b_sequences->at(i_seq_b);

                                //Trocar locker com locker não faz efeito TODO testar depois
                                    if(!swap_2_1_broke_load(route_a,seq_a_1,seq_a_2,route_b,seq_b)) {
                                        double delta = calculate_delta_swap_2_1(route_a_sequences,i_seq_a,route_b_sequences,i_seq_b, BOTH);
                                        if(Count::improves(0.0,delta) &&
                                            Count::improves(best_delta,delta)) {
                                            if(propagate_virtual_swap_2_1(i_route_a,i_seq_a-1,seq_b) &&
                                                propagate_virtual_swap_2_1(i_route_b,i_seq_b-1,seq_a_1,seq_a_2)) {
                                                best_delta = delta;
                                                coordinates[0] = i_route_a;
                                                coordinates[1] = i_seq_a;
                                                coordinates[2] = i_route_b;
                                                coordinates[3] = i_seq_b;
                                                }
                                            }
                                    }
                            }
                        }
                    }
                }
            }
        }
    }

    if(best_delta<0.0) {
        *improved = true;
        this->persist_swap_2_1(coordinates,best_delta);
    }
}



void Search::ls_locker_reducer(bool *improved) {

    for(int i_route = 0; i_route < this->solution->used_routes; i_route++) {
        Route * route = this->solution->routes.at(i_route);

        for(int i_locker = this->instance->locker_indexes[0]; i_locker < this->instance->locker_indexes[1]; i_locker++) {
            Node* locker = &this->instance->nodes.at(i_locker);

            if(route->visited_lockers[locker] > 1) {
                reduce_double_locker(i_route,route, locker);
                if(improved != nullptr) {
                    *improved = true;
                }
            }
        }

    }


}

void Search::reduce_double_locker(int i_route, Route *route, Node *locker) {

    vector<Sequence>* route_sequences = &route->sequences;

    //Encontrando lockers
    int i_locker_a = 1;
    while(route_sequences->at(i_locker_a).node != locker) {i_locker_a++;}

    int i_locker_b = i_locker_a + 1;
    while(route_sequences->at(i_locker_b).node != locker) {i_locker_b++;}

    double delta_a_to_b = this->calculate_delta_locker_reduce(route_sequences,i_locker_a,i_locker_b,locker,'b');
    double delta_b_to_a = this->calculate_delta_locker_reduce(route_sequences,i_locker_a,i_locker_b,locker,'a');

    Sequence* main_locker_sequence = nullptr;
    Sequence* reduced_locker_sequence = nullptr;
    int i_reduced = -1;
    double choosed_delta = 0.0;

    if(delta_a_to_b < delta_b_to_a) {
        main_locker_sequence = &route_sequences->at(i_locker_b);
        i_reduced = i_locker_a;
        choosed_delta = delta_a_to_b;
    }else {
        main_locker_sequence = &route_sequences->at(i_locker_a);
        i_reduced = i_locker_b;
        choosed_delta = delta_b_to_a;
    }

    reduced_locker_sequence = &route_sequences->at(i_reduced);

    for(Node* customer: reduced_locker_sequence->customers) {
        main_locker_sequence->customers.push_back(customer);
    }


    route_sequences->erase(route_sequences->begin() + i_reduced,route_sequences->begin() + i_reduced + 1);


    route->traveled_distance += choosed_delta;
    this->solution->cost += choosed_delta;
    route->visited_lockers[locker] --;
    this->propagate(i_route,i_locker_a-1);


}

void Search::ls_hybrid_reducer(bool *improved) {

    for(int i_route = 0; i_route < this->solution->used_routes; i_route++) {
        Route * route = this->solution->routes.at(i_route);
        vector<Sequence>* sequences = &route->sequences;
        for(int i_locker = this->instance->locker_indexes[0]; i_locker < this->instance->locker_indexes[1]; i_locker++) {
            Node* locker = &this->instance->nodes.at(i_locker);

            if(route->visited_lockers[locker] > 0) {

                for(int i_seq=1; i_seq < (int)(sequences->size()-1); i_seq++) {
                    Sequence* seq = &sequences->at(i_seq);
                    Node *node = seq->node;
                    if(node->type == "c3" && node->designated_locker->id == locker->id) {
                        reduce_hybrid(i_route,route, i_seq, seq, improved);
                    }

                }


            }
        }

    }
}

void Search::reduce_hybrid(int i_route, Route *route, int i_seq, Sequence *seq, bool *improved) {

    double delta = calculate_delta_destruction(&route->sequences,i_seq);
    if(delta < 0) {

        vector<Sequence> *route_sequences = &route->sequences;
        Node* n = seq->node;
        for(int i=1;i<(int)(route_sequences->size()-1); i++) {
            Sequence* s = &route_sequences->at(i);
            if(s->node->id == n->designated_locker->id) {
                s->customers.push_back(n);
                route_sequences->erase(route_sequences->begin() + i_seq,route_sequences->begin() + i_seq + 1);
                propagate(i_route,i_seq-1);
                this->solution->cost += delta;
                route->traveled_distance += delta;
                if(improved != nullptr) {
                    *improved = true;
                }
                break;
            }
        }
    }

}
