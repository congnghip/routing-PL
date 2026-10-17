//
// Created by Rafael on 06/03/2025.
//

#ifndef COUNT_H
#define COUNT_H

using namespace std;
class Count {
public:
    static bool differs(double a, double b, double epsilon = 0.001) {return abs(a-b) > epsilon;};
    static bool improves(double best, double current) {return current < best && differs(best,current);};
    static double mult_trunc(double a, double b) {return trunc(a*b*10)/10;}
    static double trun(double a) {return trunc(a*10)/10;}

};

#endif //COUNT_H
