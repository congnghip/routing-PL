//
// Created by Rafael on 08/03/2025.
//

#include "Search.h"
#include "Utils.h"

void Search::persist_swap_1_1(int *coordinates, double delta)
{
    Route *route_a = this->solution->routes.at(coordinates[0]);
    vector<Sequence> *route_a_sequences = &route_a->sequences;
    Sequence *seq_a = &route_a_sequences->at(coordinates[1]);
    Route *route_b = this->solution->routes.at(coordinates[2]);
    vector<Sequence> *route_b_sequences = &route_b->sequences;
    Sequence *seq_b = &route_b_sequences->at(coordinates[3]);

    double delta_a = calculate_delta_swap_1_1(route_a_sequences, coordinates[1], route_b_sequences, coordinates[3], 'a');
    double delta_b = calculate_delta_swap_1_1(route_a_sequences, coordinates[1], route_b_sequences, coordinates[3], 'b');

    route_a->traveled_distance += delta_a;
    route_b->traveled_distance += delta_b;
    this->solution->cost += delta;

    // Atualizando loads total e calcula o menor load das sequences
    double seq_a_load = 0.0;
    double min_seq_a_demand = -1.0;
    for (Node *n : seq_a->customers)
    {
        seq_a_load += n->load_demand;
        if (min_seq_a_demand == -1.0 || n->load_demand < min_seq_a_demand)
        {
            min_seq_a_demand = n->load_demand;
        }
    }

    double seq_b_load = 0.0;
    double min_seq_b_demand = -1.0;
    for (Node *n : seq_b->customers)
    {
        seq_b_load += n->load_demand;
        if (min_seq_b_demand == -1.0 || n->load_demand < min_seq_b_demand)
        {
            min_seq_b_demand = n->load_demand;
        }
    }

    route_a->load += (seq_b_load - seq_a_load);
    route_b->load += (seq_a_load - seq_b_load);

    // Atualizando a demanda mínima das rotas a, se for necessario
    if (min_seq_b_demand <= route_a->minimun_route_load)
    {
        route_a->minimun_route_load = min_seq_b_demand;
    }
    else if (route_a->minimun_route_load == min_seq_a_demand)
    {
        route_a->minimun_route_load = -1.0;
        for (int i = 1; i < (int)route_a_sequences->size() - 1; i++)
        {
            Sequence *s = &route_a_sequences->at(i);
            if (s != seq_a)
            {
                for (Node *n : s->customers)
                {
                    if (route_a->minimun_route_load == -1.0 || n->load_demand < route_a->minimun_route_load)
                    {
                        route_a->minimun_route_load = n->load_demand;
                    }
                }
            }
        }
    }

    if (min_seq_a_demand <= route_b->minimun_route_load)
    {
        route_b->minimun_route_load = min_seq_a_demand;
    }
    else if (route_b->minimun_route_load == min_seq_b_demand)
    {
        route_b->minimun_route_load = -1.0;
        for (int i = 1; i < (int)route_b_sequences->size() - 1; i++)
        {
            Sequence *s = &route_b_sequences->at(i);
            if (s != seq_b)
            {
                for (Node *n : s->customers)
                {
                    if (route_b->minimun_route_load == -1.0 || n->load_demand < route_b->minimun_route_load)
                    {
                        route_b->minimun_route_load = n->load_demand;
                    }
                }
            }
        }
    }

    // Atualizando contador de lockers por rota
    if (seq_a->node->type == "p")
    {
        route_a->visited_lockers[seq_a->node]--;
        route_b->visited_lockers[seq_a->node]++;
    }
    if (seq_b->node->type == "p")
    {
        route_b->visited_lockers[seq_b->node]--;
        route_a->visited_lockers[seq_b->node]++;
    }

    route_a_sequences->insert(route_a_sequences->begin() + coordinates[1] + 1, 1, *seq_b);
    route_b_sequences->insert(route_b_sequences->begin() + coordinates[3] + 1, 1, *seq_a);

    route_a_sequences->erase(route_a_sequences->begin() + coordinates[1], route_a_sequences->begin() + coordinates[1] + 1);
    route_b_sequences->erase(route_b_sequences->begin() + coordinates[3], route_b_sequences->begin() + coordinates[3] + 1);

    propagate(coordinates[2], coordinates[3] - 1);
    propagate(coordinates[0], coordinates[1] - 1);
}

void Search::persist_swap_2_2(int *coordinates, double delta)
{
    Route *route_a = this->solution->routes.at(coordinates[0]);
    vector<Sequence> *route_a_sequences = &route_a->sequences;
    Sequence *seq_a_1 = &route_a_sequences->at(coordinates[1]);
    Sequence *seq_a_2 = &route_a_sequences->at(coordinates[1] + 1);
    Route *route_b = this->solution->routes.at(coordinates[2]);
    vector<Sequence> *route_b_sequences = &route_b->sequences;
    Sequence *seq_b_1 = &route_b_sequences->at(coordinates[3]);
    Sequence *seq_b_2 = &route_b_sequences->at(coordinates[3] + 1);

    double delta_a = calculate_delta_swap_2_2(route_a_sequences, coordinates[1], route_b_sequences, coordinates[3], 'a');
    double delta_b = calculate_delta_swap_2_2(route_a_sequences, coordinates[1], route_b_sequences, coordinates[3], 'b');

    route_a->traveled_distance += delta_a;
    route_b->traveled_distance += delta_b;
    this->solution->cost += delta;

    // Atualizando loads total e calcula o menor load das sequences
    double seq_a_load = 0.0;
    double min_seq_a_demand = -1.0;
    for (Node *n : seq_a_1->customers)
    {
        seq_a_load += n->load_demand;
        if (min_seq_a_demand == -1.0 || n->load_demand < min_seq_a_demand)
        {
            min_seq_a_demand = n->load_demand;
        }
    }
    for (Node *n : seq_a_2->customers)
    {
        seq_a_load += n->load_demand;
        if (n->load_demand < min_seq_a_demand)
        {
            min_seq_a_demand = n->load_demand;
        }
    }

    double seq_b_load = 0.0;
    double min_seq_b_demand = -1.0;
    for (Node *n : seq_b_1->customers)
    {
        seq_b_load += n->load_demand;
        if (min_seq_b_demand == -1.0 || n->load_demand < min_seq_b_demand)
        {
            min_seq_b_demand = n->load_demand;
        }
    }
    for (Node *n : seq_b_2->customers)
    {
        seq_b_load += n->load_demand;
        if (min_seq_b_demand == -1.0 || n->load_demand < min_seq_b_demand)
        {
            min_seq_b_demand = n->load_demand;
        }
    }

    route_a->load += (seq_b_load - seq_a_load);
    route_b->load += (seq_a_load - seq_b_load);

    // Atualizando a demanda mínima para as rotas, se for necessario
    if (min_seq_b_demand <= route_a->minimun_route_load)
    {
        route_a->minimun_route_load = min_seq_b_demand;
    }
    else if (route_a->minimun_route_load == min_seq_a_demand)
    {
        route_a->minimun_route_load = -1.0;
        for (int i = 1; i < (int)route_a_sequences->size() - 1; i++)
        {
            Sequence *s = &route_a_sequences->at(i);
            if (s != seq_a_1 && s != seq_a_2)
            {

                for (Node *n : s->customers)
                {
                    if (route_a->minimun_route_load == -1.0 || n->load_demand < route_a->minimun_route_load)
                    {
                        route_a->minimun_route_load = n->load_demand;
                    }
                }
            }
        }
    }

    if (min_seq_a_demand <= route_b->minimun_route_load)
    {
        route_b->minimun_route_load = min_seq_a_demand;
    }
    else if (route_b->minimun_route_load == min_seq_b_demand)
    {
        route_b->minimun_route_load = -1.0;
        for (int i = 1; i < (int)route_b_sequences->size() - 1; i++)
        {
            Sequence *s = &route_b_sequences->at(i);
            if (s != seq_b_1 && s != seq_b_2)
            {

                for (Node *n : s->customers)
                {
                    if (route_b->minimun_route_load == -1.0 || n->load_demand < route_b->minimun_route_load)
                    {
                        route_b->minimun_route_load = n->load_demand;
                    }
                }
            }
        }
    }

    // Atualizando contador de lockers por rota
    if (seq_a_1->node->type == "p")
    {
        route_a->visited_lockers[seq_a_1->node]--;
        route_b->visited_lockers[seq_a_1->node]++;
    }
    if (seq_a_2->node->type == "p")
    {
        route_a->visited_lockers[seq_a_2->node]--;
        route_b->visited_lockers[seq_a_2->node]++;
    }

    if (seq_b_1->node->type == "p")
    {
        route_b->visited_lockers[seq_b_1->node]--;
        route_a->visited_lockers[seq_b_1->node]++;
    }
    if (seq_b_2->node->type == "p")
    {
        route_b->visited_lockers[seq_b_2->node]--;
        route_a->visited_lockers[seq_b_2->node]++;
    }

    route_a_sequences->insert(route_a_sequences->begin() + coordinates[1] + 2, route_b_sequences->begin() + coordinates[3], route_b_sequences->begin() + coordinates[3] + 2);
    route_b_sequences->insert(route_b_sequences->begin() + coordinates[3] + 2, route_a_sequences->begin() + coordinates[1], route_a_sequences->begin() + coordinates[1] + 2);

    route_a_sequences->erase(route_a_sequences->begin() + coordinates[1], route_a_sequences->begin() + coordinates[1] + 2);
    route_b_sequences->erase(route_b_sequences->begin() + coordinates[3], route_b_sequences->begin() + coordinates[3] + 2);

    propagate(coordinates[0], coordinates[1] - 1);
    propagate(coordinates[2], coordinates[3] - 1);
}

void Search::persist_shift_1_0(int *coordinates, double delta, bool is_reduction)
{
    Route *route_a = this->solution->routes.at(coordinates[0]);
    vector<Sequence> *route_a_sequences = &route_a->sequences;
    Sequence *seq_a = &route_a_sequences->at(coordinates[1]);
    Node* node_a = seq_a->node;
    Route *route_b = this->solution->routes.at(coordinates[2]);
    vector<Sequence> *route_b_sequences = &route_b->sequences;
    Sequence *seq_b = &route_b_sequences->at(coordinates[3]);
    Node* node_b = seq_b->node;


    if(is_reduction) {
        route_a->traveled_distance += delta;

    }else {
        double delta_a = calculate_delta_shift_1_0(route_a_sequences, coordinates[1], route_b_sequences, coordinates[3], 'a');
        double delta_b = calculate_delta_shift_1_0(route_a_sequences, coordinates[1], route_b_sequences, coordinates[3], 'b');
        route_a->traveled_distance += delta_a;
        route_b->traveled_distance += delta_b;

    }


    this->solution->cost += delta;

    // Atualizando loads das rotas
    double seq_a_load = 0.0;
    double min_seq_a_demand = -1.0;
    for (Node *n : seq_a->customers)
    {
        seq_a_load += n->load_demand;
        if (min_seq_a_demand == -1.0 || n->load_demand < min_seq_a_demand)
        {
            min_seq_a_demand = n->load_demand;
        }
    }

    route_a->load -= seq_a_load;
    route_b->load += seq_a_load;

    // Atualizando a demanda mínima para a rota b, se for necessario
    if (min_seq_a_demand < route_b->minimun_route_load)
    {
        route_b->minimun_route_load = min_seq_a_demand;
    }

    // Reajustando a demanda mínima da rota a, se o cara que saiu detém a menor demanda
    if (route_a->minimun_route_load == min_seq_a_demand)
    {
        route_a->minimun_route_load = -1.0;
        for (int i = 1; i < (int)route_a_sequences->size() - 1; i++)
        {
            Sequence *s = &route_a_sequences->at(i);

            if (s != seq_a)
            {
                for (Node *n : s->customers)
                {
                    if (route_a->minimun_route_load == -1.0 || n->load_demand < route_a->minimun_route_load)
                    {
                        route_a->minimun_route_load = n->load_demand;
                    }
                }
            }
        }
    }

    // Atualizando contador de lockers por rota
    if (seq_a->node->type == "p")
    {
        route_a->visited_lockers[seq_a->node]--;
        route_b->visited_lockers[seq_a->node]++;
    }

    //Se for reduction, incluir o nó na lista de clientes da seq b, ao invez de inclui-lo na rota
    if(is_reduction) {
        seq_b->customers.push_back(node_a);
    }else {
        route_b_sequences->insert(route_b_sequences->begin() + coordinates[3] + 1, 1, *seq_a);
    }

    route_a_sequences->erase(route_a_sequences->begin() + coordinates[1], route_a_sequences->begin() + coordinates[1] + 1);


    propagate(coordinates[0], coordinates[1] - 1);
    //Se for reduction, a rota b não precisa de propagate
    if(!is_reduction) {
        propagate(coordinates[2], coordinates[3]);
    }

    // Tratando esvaziamento da rota a
    if ((int)route_a_sequences->size() == 2)
    {
        route_a->traveled_distance = 0.0;
        route_a->load = 0.0;
        route_a->minimun_route_load = 0.0;
        iter_swap(this->solution->routes.begin() + coordinates[0], this->solution->routes.begin() + this->solution->used_routes - 1);
        this->solution->used_routes--;
    }

    // Tratando o caso de criação de uma rota nova
    if (!is_reduction && (int)route_b_sequences->size() == 3)
    {
        this->solution->used_routes++;
    }
}

void Search::persist_shift_2_0(int *coordinates, double delta)
{
    Route *route_a = this->solution->routes.at(coordinates[0]);
    vector<Sequence> *route_a_sequences = &route_a->sequences;
    Sequence *seq_a_1 = &route_a_sequences->at(coordinates[1]);
    Sequence *seq_a_2 = &route_a_sequences->at(coordinates[1] + 1);
    Route *route_b = this->solution->routes.at(coordinates[2]);
    vector<Sequence> *route_b_sequences = &route_b->sequences;
    Sequence *seq_b = &route_b_sequences->at(coordinates[3]);

    double delta_a = calculate_delta_shift_2_0(route_a_sequences, coordinates[1], route_b_sequences, coordinates[3], 'a');
    double delta_b = calculate_delta_shift_2_0(route_a_sequences, coordinates[1], route_b_sequences, coordinates[3], 'b');

    route_a->traveled_distance += delta_a;
    route_b->traveled_distance += delta_b;
    this->solution->cost += delta;

    // Atualizando loads das rotas
    double seq_a_load = 0.0;
    double min_seq_a_demand = -1.0;
    for (Node *n : seq_a_1->customers)
    {
        seq_a_load += n->load_demand;
        if (min_seq_a_demand == -1.0 || n->load_demand < min_seq_a_demand)
        {
            min_seq_a_demand = n->load_demand;
        }
    }

    for (Node *n : seq_a_2->customers)
    {
        seq_a_load += n->load_demand;
        if (n->load_demand < min_seq_a_demand)
        {
            min_seq_a_demand = n->load_demand;
        }
    }

    route_a->load -= seq_a_load;
    route_b->load += seq_a_load;

    // Atualizando a demanda mínima para a rota b, se for necessario
    if (min_seq_a_demand < route_b->minimun_route_load)
    {
        route_b->minimun_route_load = min_seq_a_demand;
    }

    // Reajustando a demanda mínima da rota a, se o cara que saiu detém a menor demanda
    if (route_a->minimun_route_load == min_seq_a_demand)
    {
        route_a->minimun_route_load = -1.0;
        for (int i = 1; i < (int)route_a_sequences->size() - 1; i++)
        {
            Sequence *s = &route_a_sequences->at(i);

            if (s != seq_a_1 && s != seq_a_2)
            {
                for (Node *n : s->customers)
                {
                    if (route_a->minimun_route_load == -1.0 || n->load_demand < route_a->minimun_route_load)
                    {
                        route_a->minimun_route_load = n->load_demand;
                    }
                }
            }
        }
    }

    // Atualizando contador de lockers por rota
    if (seq_a_1->node->type == "p")
    {
        route_a->visited_lockers[seq_a_1->node]--;
        route_b->visited_lockers[seq_a_1->node]++;
    }

    if (seq_a_2->node->type == "p")
    {
        route_a->visited_lockers[seq_a_2->node]--;
        route_b->visited_lockers[seq_a_2->node]++;
    }

    route_b_sequences->insert(route_b_sequences->begin() + coordinates[3] + 1, route_a_sequences->begin() + coordinates[1], route_a_sequences->begin() + coordinates[1] + 2);
    route_a_sequences->erase(route_a_sequences->begin() + coordinates[1], route_a_sequences->begin() + coordinates[1] + 2);

    propagate(coordinates[2], coordinates[3]);
    propagate(coordinates[0], coordinates[1] - 1);

    // Tratando esvaziamento da rota a
    if ((int)route_a_sequences->size() == 2)
    {
        route_a->traveled_distance = 0.0;
        route_a->load = 0.0;
        route_a->minimun_route_load = 0.0;
        iter_swap(this->solution->routes.begin() + coordinates[0], this->solution->routes.begin() + this->solution->used_routes - 1);
        this->solution->used_routes--;
    }

    // Tratando o caso de criação de uma rota nova em b
    if ((int)route_b_sequences->size() == 4)
    {
        this->solution->used_routes++;
    }
}

void Search::persist_swap_2_1(int *coordinates, double delta)
{
    Route* route_a = this->solution->routes.at(coordinates[0]);
    vector<Sequence> *route_a_sequences = &route_a->sequences;
    Sequence *seq_a_1 = &route_a_sequences->at(coordinates[1]);
    Sequence *seq_a_2 = &route_a_sequences->at(coordinates[1] + 1);
    Route *route_b = this->solution->routes.at(coordinates[2]);
    vector<Sequence> *route_b_sequences = &route_b->sequences;
    Sequence *seq_b = &route_b_sequences->at(coordinates[3]);

    double delta_a = calculate_delta_swap_2_1(route_a_sequences, coordinates[1], route_b_sequences, coordinates[3], 'a');
    double delta_b = calculate_delta_swap_2_1(route_a_sequences, coordinates[1], route_b_sequences, coordinates[3], 'b');

    route_a->traveled_distance += delta_a;
    route_b->traveled_distance += delta_b;
    this->solution->cost += delta;

    // Atualizando loads total e calcula o menor load das sequences
    double seq_a_load = 0.0;
    double min_seq_a_demand = -1.0;
    for (Node *n : seq_a_1->customers)
    {
        seq_a_load += n->load_demand;
        if (min_seq_a_demand == -1.0 || n->load_demand < min_seq_a_demand)
        {
            min_seq_a_demand = n->load_demand;
        }
    }
    for (Node *n : seq_a_2->customers)
    {
        seq_a_load += n->load_demand;
        if (n->load_demand < min_seq_a_demand)
        {
            min_seq_a_demand = n->load_demand;
        }
    }

    double seq_b_load = 0.0;
    double min_seq_b_demand = -1.0;
    for (Node *n : seq_b->customers)
    {
        seq_b_load += n->load_demand;
        if (min_seq_b_demand == -1.0 || n->load_demand < min_seq_b_demand)
        {
            min_seq_b_demand = n->load_demand;
        }
    }


    route_a->load += (seq_b_load - seq_a_load);
    route_b->load += (seq_a_load - seq_b_load);

    // Atualizando a demanda mínima para as rotas, se for necessario
    if (min_seq_b_demand <= route_a->minimun_route_load)
    {
        route_a->minimun_route_load = min_seq_b_demand;
    }
    else if (route_a->minimun_route_load == min_seq_a_demand)
    {
        route_a->minimun_route_load = -1.0;
        for (int i = 1; i < (int)route_a_sequences->size() - 1; i++)
        {
            Sequence *s = &route_a_sequences->at(i);
            if (s != seq_a_1 && s != seq_a_2)
            {

                for (Node *n : s->customers)
                {
                    if (route_a->minimun_route_load == -1.0 || n->load_demand < route_a->minimun_route_load)
                    {
                        route_a->minimun_route_load = n->load_demand;
                    }
                }
            }
        }
    }

    if (min_seq_a_demand <= route_b->minimun_route_load)
    {
        route_b->minimun_route_load = min_seq_a_demand;
    }
    else if (route_b->minimun_route_load == min_seq_b_demand)
    {
        route_b->minimun_route_load = -1.0;
        for (int i = 1; i < (int)route_b_sequences->size() - 1; i++)
        {
            Sequence *s = &route_b_sequences->at(i);
            if (s != seq_b)
            {

                for (Node *n : s->customers)
                {
                    if (route_b->minimun_route_load == -1.0 || n->load_demand < route_b->minimun_route_load)
                    {
                        route_b->minimun_route_load = n->load_demand;
                    }
                }
            }
        }
    }

    // Atualizando contador de lockers por rota
    if (seq_a_1->node->type == "p")
    {
        route_a->visited_lockers[seq_a_1->node]--;
        route_b->visited_lockers[seq_a_1->node]++;
    }
    if (seq_a_2->node->type == "p")
    {
        route_a->visited_lockers[seq_a_2->node]--;
        route_b->visited_lockers[seq_a_2->node]++;
    }

    if (seq_b->node->type == "p")
    {
        route_b->visited_lockers[seq_b->node]--;
        route_a->visited_lockers[seq_b->node]++;
    }



    route_b_sequences->insert(route_b_sequences->begin() + coordinates[3] + 1, route_a_sequences->begin() + coordinates[1], route_a_sequences->begin() + coordinates[1] + 2);
    route_a_sequences->insert(route_a_sequences->begin() + coordinates[1] + 2, 1, *seq_b);

    route_a_sequences->erase(route_a_sequences->begin() + coordinates[1], route_a_sequences->begin() + coordinates[1] + 2);
    route_b_sequences->erase(route_b_sequences->begin() + coordinates[3], route_b_sequences->begin() + coordinates[3] + 1);

    propagate(coordinates[0], coordinates[1] - 1);
    propagate(coordinates[2], coordinates[3] - 1);

}


void Search::persist_split(int *coordinates, double delta) {
    int i_route_a = coordinates[0];
    int i_seq_a_1 = coordinates[1];
    int i_seq_a_2 = coordinates[2];
    Route* route_a = this->solution->routes.at(i_route_a);
    vector<Sequence>* route_a_sequences = &route_a->sequences;
    Sequence *seq_a_1 = &route_a_sequences->at(i_seq_a_1);
    Sequence *seq_a_2 = &route_a_sequences->at(i_seq_a_2);
    Route* route_b = this->solution->routes.at(this->solution->used_routes);
    vector<Sequence>* route_b_sequences = &route_b->sequences;

    //Atualizando o load das rotas
    // Tambem ajustando o contador de lockers da rota
    double seq_a_load = 0.0;
    double min_seq_a_load = -1.0;

    for (int i=i_seq_a_1; i<=i_seq_a_2; i++) {
        Sequence *s = &route_a_sequences->at(i);

        // Ajustando o contador de lockers da rota
        if (s->node->type == "p") {
            route_a->visited_lockers[s->node]--;
            route_b->visited_lockers[s->node]++;
        }

        for (Node* n: s->customers) {
            seq_a_load += n->load_demand;
            if (n->load_demand < min_seq_a_load || min_seq_a_load == -1.0) {
                min_seq_a_load = n->load_demand;
            }
        }
    }

    route_a->load -= seq_a_load;
    route_b->load += seq_a_load;

    // Atualizando a demanda mínima para a rota b
    route_b->minimun_route_load = min_seq_a_load;

    // Reajustando a demanda mínima da rota a, se o cara que saiu detém a menor demanda
    if (route_a->minimun_route_load == min_seq_a_load) {
        route_a->minimun_route_load = -1.0;
        for (int i=1; i < (int)route_a_sequences->size() - 1;i++) {
            Sequence *s = &route_a_sequences->at(i);

            if (i == i_seq_a_1) {
                i = i_seq_a_2;
            }else {
                for (Node* n: s->customers) {
                    if (n->load_demand < route_a->minimun_route_load || route_a->minimun_route_load == -1.0) {
                        route_a->minimun_route_load = n->load_demand;
                    }
                }
            }
        }
    }

    route_b_sequences->insert(route_b_sequences->begin() + 1,(route_a_sequences->begin() + i_seq_a_1),(route_a_sequences->begin() + i_seq_a_2 + 1));
    route_a_sequences->erase(route_a_sequences->begin() + i_seq_a_1, route_a_sequences->begin() + i_seq_a_2 + 1);

    propagate(i_route_a, i_seq_a_1 - 1);
    propagate(this->solution->used_routes, 0);

    this->solution->used_routes++;
    this->solution->cost += delta;

    //Atualizando distância da rota a
    route_a->traveled_distance = 0;
    for (int i=1; i < (int)route_a_sequences->size();i++) {
        Node* n_i = route_a_sequences->at(i).node;
        Node* n_j = route_a_sequences->at(i-1).node;

        route_a->traveled_distance += this->instance->distances[n_i->index][n_j->index];
    }

    //Atualizando distância da rota b
    route_b->traveled_distance = 0;
    for (int i=1; i < (int)route_b_sequences->size();i++) {
        Node* n_i = route_b_sequences->at(i).node;
        Node* n_j = route_b_sequences->at(i-1).node;

        route_b->traveled_distance += this->instance->distances[n_i->index][n_j->index];
    }
}

