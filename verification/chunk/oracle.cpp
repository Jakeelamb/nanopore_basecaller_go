// Links the untouched pinned Dorado read_pipeline/base/chunk.cpp.
#include "read_pipeline/base/chunk.h"
#include <exception>
#include <iostream>
#include <string>
int main() {
 std::string mode;
 std::size_t n,size,stride,overlap,granularity;
 while(std::cin >> mode >> n >> size >> stride >> overlap >> granularity) {
  try {
   if(mode=="fixed") {
    auto out=dorado::utils::generate_chunks(n,size,stride,overlap);
    std::cout << "ok";
    for(auto x:out) std::cout << ' ' << x;
   } else {
    auto out=mode=="variable" ? dorado::utils::generate_variable_chunks(n,size,stride,overlap)
       : dorado::utils::generate_variable_chunks_tx(n,size,stride,granularity,overlap);
    std::cout << "ok";
    for(auto x:out) std::cout << ' ' << x.first << ',' << x.second;
   }
  } catch(const std::exception&) { std::cout << "error"; }
  std::cout << '\n';
 }
}
