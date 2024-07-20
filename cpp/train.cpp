#include "utils.h"

int main() {
    Series s(15);

    double values[] = {1.0, 2.0, 3.0, 4.0, 5.0,
                       6.0, 7.0, 8.0, 9.0, 10.0,
                       11.0, 12.0, 13.0, 14.0, 15.0};
    s.setValues(values);

    s.print(2);

    DataFrame df(60000, 784);

    df.read_csv("data/X_train.csv");

    df.print(2);

    return 0;
}
