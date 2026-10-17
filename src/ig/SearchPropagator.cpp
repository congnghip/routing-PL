//
// Created by Rafael on 08/03/2025.
//

#include "Search.h"

//TODO testar
bool Search::propagate_virtual_or_opt_1_up(int route_index, int i_seq_a, int i_seq_b) {
    Route* route = this->solution->routes.at(route_index);
    vector<Sequence> *route_sequences = &route->sequences;

    route_sequences->at(i_seq_a-1).clone_this_to(this->virtual_sequence);

    Sequence* previous_sequence = &route_sequences->at(i_seq_a-1);
    Sequence* current_sequence = &route_sequences->at(i_seq_a+1);
    fill_forward_virtual(previous_sequence,current_sequence);
    if(broke_time_window()) {
        return false;
    }

    for (int i=i_seq_a+2; i<i_seq_b; i++) {
        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);

        fill_forward_virtual(previous_sequence,current_sequence);
        if(broke_time_window()) {
            return false;
        }

    }

    previous_sequence = current_sequence;
    current_sequence = &route_sequences->at(i_seq_a);

    fill_forward_virtual(previous_sequence,current_sequence);
    if(broke_time_window()) {
        return false;
    }

    for (int i= i_seq_b; i<(int)route_sequences->size(); i++) {
        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);

        fill_forward_virtual(previous_sequence,current_sequence);
        if(broke_time_window()) {
            return false;
        }
    }

    return true;

}

//TODO testar
bool Search::propagate_virtual_or_opt_k_up(int k, int route_index, int i_seq_a, int i_seq_b) {
    Route* route = this->solution->routes.at(route_index);
    vector<Sequence> *route_sequences = &route->sequences;

    route_sequences->at(i_seq_a-1).clone_this_to(this->virtual_sequence);

    Sequence* previous_sequence = &route_sequences->at(i_seq_a-1);
    Sequence* current_sequence = &route_sequences->at(i_seq_a+k);
    fill_forward_virtual(previous_sequence,current_sequence);
    if(broke_time_window()) {
        return false;
    }

    for (int i=i_seq_a+k+1; i<i_seq_b; i++) {
        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);

        fill_forward_virtual(previous_sequence,current_sequence);
        if(broke_time_window()) {
            return false;
        }

    }

    for (int i=i_seq_a; i<(i_seq_a+k+1);i++) {
        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);

        fill_forward_virtual(previous_sequence,current_sequence);
        if(broke_time_window()) {
            return false;
        }
    }

    for (int i= i_seq_b; i<(int)route_sequences->size(); i++) {
        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);

        fill_forward_virtual(previous_sequence,current_sequence);
        if(broke_time_window()) {
            return false;
        }
    }

    return true;
}
//TODO testar
bool Search::propagate_virtual_or_opt_1_down(int route_index, int i_seq_a, int i_seq_b) {
    Route* route = this->solution->routes.at(route_index);
    vector<Sequence> *route_sequences = &route->sequences;

    route_sequences->at(i_seq_b-1).clone_this_to(this->virtual_sequence);

    Sequence* previous_sequence = &route_sequences->at(i_seq_b-1);
    Sequence* current_sequence = &route_sequences->at(i_seq_a);
    fill_forward_virtual(previous_sequence,current_sequence);
    if(broke_time_window()) {
        return false;
    }

    for (int i = i_seq_b; i<i_seq_a; i++) {

        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);
        fill_forward_virtual(previous_sequence,current_sequence);
        if(broke_time_window()) {
            return false;
        }

    }

    previous_sequence = current_sequence;
    current_sequence = &route_sequences->at(i_seq_a + 1);
    fill_forward_virtual(previous_sequence,current_sequence);
    if(broke_time_window()) {
        return false;
    }

    for (int i = i_seq_a+2; i<(int)route_sequences->size(); i++) {

        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);
        fill_forward_virtual(previous_sequence,current_sequence);
        if(broke_time_window()) {
            return false;
        }
    }

    return true;
}
//TODO testar
bool Search::propagate_virtual_or_opt_k_down(int k, int route_index, int i_seq_a, int i_seq_b) {
    Route* route = this->solution->routes.at(route_index);
    vector<Sequence> *route_sequences = &route->sequences;

    route_sequences->at(i_seq_b-1).clone_this_to(this->virtual_sequence);

    Sequence* previous_sequence = &route_sequences->at(i_seq_b-1);
    Sequence* current_sequence = &route_sequences->at(i_seq_a);
    fill_forward_virtual(previous_sequence,current_sequence);
    if(broke_time_window()) {
        return false;
    }

    for (int i = i_seq_a+1; i<i_seq_a+k; i++) {

        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);
        fill_forward_virtual(previous_sequence,current_sequence);
        if(broke_time_window()) {
            return false;
        }

    }

    for (int i = i_seq_b; i<i_seq_a; i++) {

        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);
        fill_forward_virtual(previous_sequence,current_sequence);
        if(broke_time_window()) {
            return false;
        }

    }

    for (int i = i_seq_a + k; i<(int)route_sequences->size(); i++) {

        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);
        fill_forward_virtual(previous_sequence,current_sequence);
        if(broke_time_window()) {
            return false;
        }

    }

    return true;
}

bool Search::propagate_virtual_swap_1_1(int route_index, int previous_sequence_index, Sequence *cand_sequence) {
    Route* route = this->solution->routes.at(route_index);
    vector<Sequence>* route_sequences = &route->sequences;

    route_sequences->at(previous_sequence_index).clone_this_to(this->virtual_sequence);

    Sequence* previous_sequence = &route_sequences->at(previous_sequence_index);
    Sequence* current_sequence = cand_sequence;
    fill_forward_virtual(previous_sequence, current_sequence);

    if(broke_time_window()) {
        return false;
    }

    for(int i=previous_sequence_index+2; i<(int)route_sequences->size(); i++) {

        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);

        fill_forward_virtual(previous_sequence, current_sequence);
        if(broke_time_window()) {
            return false;
        }

    }

    return true;
}

bool Search::propagate_virtual_swap_2_2(int route_index, int previous_sequence_index, Sequence *cand_sequence_1,
    Sequence *cand_sequence_2) {
    Route* route = this->solution->routes.at(route_index);
    vector<Sequence>* route_sequences = &route->sequences;

    route_sequences->at(previous_sequence_index).clone_this_to(this->virtual_sequence);

    //Propagando primeiro nó
    Sequence* previous_sequence = &route_sequences->at(previous_sequence_index);
    Sequence* current_sequence = cand_sequence_1;
    fill_forward_virtual(previous_sequence, current_sequence);

    if(broke_time_window()) {
        return false;
    }

    //Propagando segundo nó
    previous_sequence = current_sequence;
    current_sequence = cand_sequence_2;
    fill_forward_virtual(previous_sequence, current_sequence);
    if(broke_time_window()) {
        return false;
    }
    //TODO testar
    for(int i=previous_sequence_index+3; i<(int)route_sequences->size(); i++) {

        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);

        fill_forward_virtual(previous_sequence, current_sequence);
        if(broke_time_window()) {
            return false;
        }

    }

    return true;
}


bool Search::propagate_virtual_shift_2_0(int route_index, int previous_sequence_index, Sequence* cand_sequence_1, Sequence* cand_sequence_2) {
    Route* route = this->solution->routes.at(route_index);
    vector<Sequence>* route_sequences = &route->sequences;

    route_sequences->at(previous_sequence_index).clone_this_to(this->virtual_sequence);
    //Propagando primeiro nó
    Sequence* previous_sequence = &route_sequences->at(previous_sequence_index);
    Sequence* current_sequence = cand_sequence_1;

    fill_forward_virtual(previous_sequence, current_sequence);

    if(broke_time_window()) {
        return false;
    }

    //Propagando segundo nó
    previous_sequence = current_sequence;
    current_sequence = cand_sequence_2;

    fill_forward_virtual(previous_sequence, current_sequence);
    if(broke_time_window()) {
        return false;
    }

    for(int i=previous_sequence_index+1; i<(int)route_sequences->size(); i++) {

        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);

        fill_forward_virtual(previous_sequence, current_sequence);
        if(broke_time_window()) {
            return false;
        }

    }

    return true;
}


bool Search::propagate_virtual_swap_2_1(int route_index, int previous_sequence_index, Sequence *cand_sequence) {
    Route* route = this->solution->routes.at(route_index);
    vector<Sequence>* route_sequence = &route->sequences;

    route_sequence->at(previous_sequence_index).clone_this_to(this->virtual_sequence);

    //Propagando nó de a
    Sequence* previous_sequence = &route_sequence->at(previous_sequence_index);
    Sequence* current_sequence = cand_sequence;
    fill_forward_virtual(previous_sequence, current_sequence);
    if(broke_time_window()) {
        return false;
    }

    //TODO testar
    for(int i=previous_sequence_index+3; i<(int)route_sequence->size(); i++) {

        previous_sequence = current_sequence;
        current_sequence = &route_sequence->at(i);

        fill_forward_virtual(previous_sequence, current_sequence);
        if(broke_time_window()) {
            return false;
        }

    }

    return true;
}


bool Search::propagate_virtual_swap_2_1(int route_index, int previous_sequence_index, Sequence *cand_sequence_1,
                                        Sequence *cand_sequence_2) {
    Route* route = this->solution->routes.at(route_index);
    vector<Sequence>* route_sequences = &route->sequences;

    route_sequences->at(previous_sequence_index).clone_this_to(this->virtual_sequence);

    //Propagando primeiro nó
    Sequence* previous_sequence = &route_sequences->at(previous_sequence_index);
    Sequence* current_sequence = cand_sequence_1;
    fill_forward_virtual(previous_sequence, current_sequence);
    if(broke_time_window()) {
        return false;
    }

    //Propagando segundo nó
    previous_sequence = current_sequence;
    current_sequence = cand_sequence_2;
    fill_forward_virtual(previous_sequence, current_sequence);
    if(broke_time_window()) {
        return false;
    }
    //TODO testar
    for(int i=previous_sequence_index+2; i<(int)route_sequences->size(); i++) {

        previous_sequence = current_sequence;
        current_sequence = &route_sequences->at(i);

        fill_forward_virtual(previous_sequence, current_sequence);
        if(broke_time_window()) {
            return false;
        }

    }

    return true;
}
