#include "Sample.h"

std::ostream& operator<<(std::ostream&os, const Sample&sample)
{
    // TODO: �ڴ˴����� return ���
    os << "eigen vetor as follow:\n";
    for (const auto& x : sample.feature)os << x << ' ';
    os << "\nlable as follow:\n"; 
    for (const auto& x : sample.label)os << x << ' ';
    os << "\n\n";
    return os;
}
