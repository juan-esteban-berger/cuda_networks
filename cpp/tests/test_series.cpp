#include <gtest/gtest.h>
#include "utils.h"  // Include your Series class definition

TEST(SeriesTest, Initialization) {
    Series s(10);  // Create a Series of length 10
    EXPECT_EQ(s.length, 10);  // Check if the length is correctly set
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
