#include <cerrno>
#include <unistd.h>

namespace d01_gate {
unsigned sequence=0;
void phase(const std::string&name){
    const auto n=++sequence;
    const auto line="{\"schema\":\"nanopore-supgpu-loader-gate/v1\",\"kind\":\"phase\",\"sequence\":"+std::to_string(n)+",\"cookie\":0,\"path\":\""+name+"\",\"namespace\":0,\"pid\":"+std::to_string(getpid())+"}\n";
    size_t sent=0;
    while(sent<line.size()){const auto rc=::write(5,line.data()+sent,line.size()-sent);if(rc<0 && errno==EINTR)continue;if(rc<=0)throw std::runtime_error("D01 phase request unavailable");sent+=static_cast<size_t>(rc);}
    std::string reply;char c=0;
    while(reply.size()<128){const auto rc=::read(6,&c,1);if(rc<0 && errno==EINTR)continue;if(rc!=1)throw std::runtime_error("D01 phase approval unavailable");reply+=c;if(c=='\n')break;}
    if(reply!="allow phase "+std::to_string(n)+"\n")throw std::runtime_error("D01 runtime phase denied: "+name);
}
}
