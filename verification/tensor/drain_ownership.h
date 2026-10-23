#pragma once
#include <exception>
template<class Enqueue,class Drain,class Poison,class Retain>
void retain_until_drained(Enqueue enqueue,Drain drain,Poison poison,Retain retain) {
 try {enqueue();drain();}
 catch(...) {
  auto original=std::current_exception();poison();
  try {drain();}catch(...){retain();}
  std::rethrow_exception(original);
 }
}
template<class Drain,class Release> void release_after_drain(Drain drain,Release release) {drain();release();}
