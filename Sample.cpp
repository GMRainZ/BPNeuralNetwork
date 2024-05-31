#include "Sample.h"

std::ostream& operator<<(std::ostream&os, const Sample&sample)
{
    // TODO: 在此处插入 return 语句
    os << "eigen vetor as follow:\n";
    for (const auto& x : sample.feature)os << x << ' ';
    os << "\nlable as follow:\n"; 
    for (const auto& x : sample.label)os << x << ' ';
    os << "\n\n";
    return os;
}
