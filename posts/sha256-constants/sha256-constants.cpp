#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <bitset>

// transform double to bitset<64> format
// input: double
// output: std::bitset<64>
std::bitset<64> double2bit(double x)
{
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    std::bitset<64> b(u);
    return b;
}

// get first 32bits of the fractional part of double in bitset<32> format
// input: double
// output: std::bitset<32>
std::bitset<32> get_fraction(double x)
{
    std::bitset<64> b = double2bit(x);
    std::bitset<32> res;
    for (int i = 0; i < 32; i++) res[31 - i] = b[63 - (i + 12)];
    return res;
}

// calculate the square root using Newton method
// input: double
// output: double
// stop condition: first 32bits of fractional part of square root becomes consistent
double sqrt(double x)
{
    double res = x;
    double res_prev;
    while (true)
    {
        res_prev = res;
        res = (res + x / res) / 2;
        std::bitset<64> b1 = double2bit(res_prev);
        std::bitset<64> b2 = double2bit(res);
        bool is_consistent = true;
        for (int i = 0; i < 44; i++)
        {
            if (b1[63 - i] != b2[63 - i])
            {
                is_consistent = false;
                break;
            }
        }
        if (is_consistent) break;
    }
    return res;
}

int main()
{
    // get first 8 prime numbers
    std::vector<int> primes;
    int n = 2;
    while (primes.size() < 8)
    {
        bool is_prime = true;
        for (int i = 2; i < n; i++)
        {
            if (n % i == 0)
            {
                is_prime = false;
                break;
            }
        }
        if (is_prime) primes.push_back(n);
        n++;
    }

    // calculate the square root
    std::vector<double> sqrt_primes;
    for (auto p: primes) sqrt_primes.push_back(sqrt(p));

    // print out square roots in hex forms
    for (int i=0; i<8; i++)
    {
        std::cout << "h" << i << " := " << std::hex << "0x" << get_fraction(sqrt_primes[i]).to_ulong() << std::endl;
    }
}