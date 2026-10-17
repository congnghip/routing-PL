//
// Created by Rafael on 12/01/2025.
//

#ifndef CONFIG_H
#define CONFIG_H
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
class Config {
public:
    int runs;
    vector<long> seeds;
    double alpha_constr;
    double alpha_ig;
    int run;

    Config(int runs, double alpha_constr, double alpha_ig, bool pred_seed) {
        this->runs = runs;
        this->alpha_constr = alpha_constr;
        this->alpha_ig = alpha_ig;
        //srand(clock());
        srand(time(NULL));
        if(pred_seed) {
            build_predefined_seeds();
        }else {
            for(int i=0;i<runs;i++) {
                this->seeds.push_back(rand());
            }
        }
    };

    void print() {
        ofstream file;
        file.open("Output/0_config",ofstream::out);
        if (!file.is_open()) {
            cout<<"Output file not opened"<<endl;
        }
        file<<"runs: "<<to_string(this->runs)<<endl;
        file<<"alpha_constr: "<<to_string(this->alpha_constr)<<endl;
        file<<"alpha_ig: "<<to_string(this->alpha_ig)<<endl;
        file<<"seeds:"<<endl;
        for(int i=0;i<this->runs;i++) {
            file<<to_string(i)<<"-"<<this->seeds.at(i)<<endl;
        }

        file.close();
    }

    void build_predefined_seeds() {
        ifstream file;
        file.open("pred_seeds.txt");

        if(!file.is_open()) {
            cout<<"File not found"<<endl;
            exit(1);
        }
        string line;
        for(int i=0;i<this->runs;i++) {
            getline(file,line);
            this->seeds.push_back(stol(line));

        }
    }

};


#endif //CONFIG_H
