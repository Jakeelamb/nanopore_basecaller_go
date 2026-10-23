#include "drain_ownership.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
int main(){
 try{
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
