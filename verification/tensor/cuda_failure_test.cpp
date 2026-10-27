#include "drain_ownership.h"
#include "storage_byte_interval.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

void check_storage_byte_intervals() {
 auto interval = [](uint64_t offset, uint64_t end, uint64_t bytes) {
  StorageByteInterval out;
  if (!checked_storage_byte_interval(offset, end, bytes, out)) throw std::logic_error("valid byte interval rejected");
  return out;
 };
 auto input = interval(0, 215040, 2);
 if (input.offset != 0 || input.end != 430080) throw std::logic_error("F16 endpoints were not converted to bytes");
 for (uint64_t output_offset : {uint64_t(65536), uint64_t(262144), uint64_t(430079)}) {
  auto output = interval(output_offset, output_offset + 2097152, 1);
  if (storage_byte_intervals_disjoint(input, output) || storage_byte_intervals_disjoint(output, input)) {
   throw std::logic_error("mixed F16/I8 byte overlap accepted");
  }
 }
 auto adjacent = interval(430080, 430080 + 2097152, 1);
 if (!storage_byte_intervals_disjoint(input, adjacent) || !storage_byte_intervals_disjoint(adjacent, input)) {
  throw std::logic_error("adjacent mixed F16/I8 intervals rejected");
 }
 auto f32 = interval(107520, 107522, 4);
 if (!storage_byte_intervals_disjoint(input, f32)) throw std::logic_error("adjacent F32 interval rejected");
 if (storage_byte_intervals_disjoint(adjacent, f32)) throw std::logic_error("I8/F32 byte overlap accepted");
 auto empty = interval(1, 1, 4);
 if (!storage_byte_intervals_disjoint(input, empty)) throw std::logic_error("empty interval overlaps storage");
 const auto max = std::numeric_limits<uint64_t>::max();
 auto largest = interval(max / 4 - 1, max / 4, 4);
 if (largest.end != (max / 4) * 4) throw std::logic_error("largest checked endpoint changed");
 StorageByteInterval invalid;
 if (checked_storage_byte_interval(0, max / 2 + 1, 2, invalid) ||
     checked_storage_byte_interval(max / 2 + 1, max / 2 + 2, 2, invalid) ||
     checked_storage_byte_interval(2, 1, 1, invalid) ||
     checked_storage_byte_interval(0, 1, 0, invalid)) {
  throw std::logic_error("invalid or overflowing byte interval accepted");
 }
 std::cout << "checked byte intervals reject shifted F16/I8 overlap, accept adjacency, and reject overflow\n";
}

void check_storage_reinterpret_offsets() {
 StorageByteInterval parent;
 if (!checked_storage_byte_interval(10, 30, 2, parent)) throw std::logic_error("valid parent interval rejected");
 uint64_t start = 0;
 if (!checked_storage_reinterpret_offset(parent, 4, 3, start) || start != 8 ||
     !checked_storage_reinterpret_offset(parent, 1, 0, start) || start != 20 ||
     !checked_storage_reinterpret_offset(parent, 4, 10, start) || start != 15) {
  throw std::logic_error("reinterpret offset did not preserve parent byte bounds");
 }
 const auto max = std::numeric_limits<uint64_t>::max();
 if (checked_storage_reinterpret_offset(parent, 2, max, start) ||
     checked_storage_reinterpret_offset(parent, 2, uint64_t(1) << 63, start) ||
     checked_storage_reinterpret_offset(parent, 4, 11, start) ||
     checked_storage_reinterpret_offset({max - 3, max}, 4, 1, start) ||
     checked_storage_reinterpret_offset({1, 16}, 2, 0, start) ||
     checked_storage_reinterpret_offset({0, max}, 1, 0, start) ||
     checked_storage_reinterpret_offset({2, 1}, 1, 0, start) ||
     checked_storage_reinterpret_offset(parent, 0, 0, start)) {
  throw std::logic_error("overflowing, unaligned, or out-of-parent reinterpret offset accepted");
 }
 std::cout << "checked reinterpret offsets reject huge offsets before arithmetic and preserve parent byte bounds\n";
}
int main(){
 try{
 check_storage_byte_intervals();
 check_storage_reinterpret_offsets();
 int destroyed=0;struct Owner {int* destroyed;~Owner(){++*destroyed;}};
 auto owner=std::make_shared<Owner>();owner->destroyed=&destroyed;std::vector<std::shared_ptr<Owner>> held;
 bool poisoned=false;int drains=0;
 try{retain_until_drained([]{},[&]{++drains;throw std::runtime_error("injected drain failure");},[&]{poisoned=true;},[&]{held.push_back(owner);});throw std::logic_error("drain failure accepted");}catch(const std::runtime_error&){}
 owner.reset();if(!poisoned||drains!=2||destroyed||held.size()!=1)throw std::logic_error("failed sync dropped owner");
 try{release_after_drain([]{throw std::runtime_error("injected close drain failure");},[&]{held.clear();});throw std::logic_error("close failure accepted");}catch(const std::runtime_error&){}
 if(destroyed||held.empty())throw std::logic_error("failed close dropped owner");
 release_after_drain([]{},[&]{held.clear();});if(destroyed!=1)throw std::logic_error("successful close did not release owner once");
 bool retained=false;drains=0;
 try{retain_until_drained([]{throw std::runtime_error("injected enqueue failure");},[&]{++drains;},[]{},[&]{retained=true;});}catch(const std::runtime_error&){}
 if(retained||drains!=1)throw std::logic_error("successful recovery drain retained owner");
 std::cout<<"injected sync and close failure retain owners; successful retry releases exactly once\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
