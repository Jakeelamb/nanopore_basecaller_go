//go:build ignore

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include "demux/AdapterDetector.h"
#include "adapter_primer_kits.h"
#include "models/kits.h"
#include "utils/sequence_utils.h"
#include "utils/log_utils.h"
#include "utils/string_utils.h"

namespace {
#include "reverse-impl.slice.cpp"
}
#define NVTX3_FUNC_RANGE() ((void)0)
namespace dorado::utils {
#include "reverse.slice.cpp"
}
namespace dorado::adapter_primer_kits {
namespace {
#include "tables.slice.cpp"
}
#include "manager-constructor.slice.cpp"
#include "candidates.slice.cpp"
AdapterPrimerManager::AdapterPrimerManager(const std::string&) { throw std::runtime_error("custom primers outside oracle scope"); }
}
#include "source/dorado/demux/AdapterDetector.cpp"
namespace dorado::demux {
class Trimmer { public: static std::pair<int,int> determine_trim_interval(AdapterScoreResult&,int); };
#include "trim.slice.cpp"
}
namespace utils = dorado::utils;
struct ReadShim { struct { std::string read_id; std::pair<int,int> adapter_trim_interval; } read_common; };
void intersect(std::pair<int,int> adapter_trim_interval, std::pair<int,int> primer_trim_interval, ReadShim& read, int& m_num_untrimmed_short_reads) {
#include "intersection.slice.cpp"
}

void end_json(const dorado::SingleEndResult& end) {
    std::cout << "{\"Score\":" << end.score << ",\"Name\":\"" << end.name << "\",\"Start\":" << end.position.first << ",\"End\":" << end.position.second << '}';
}
void pair_json(const dorado::AdapterScoreResult& pair) {
    std::cout << "{\"Front\":"; end_json(pair.front); std::cout << ",\"Rear\":"; end_json(pair.rear); std::cout << '}';
}
void interval_json(std::pair<int,int> interval) { std::cout << "{\"Start\":" << interval.first << ",\"End\":" << interval.second << '}'; }
void detection_json(const dorado::AdapterScoreResult& raw,const dorado::AdapterScoreResult& accepted,std::pair<int,int> interval) {
    std::cout << "{\"Raw\":"; pair_json(raw); std::cout << ",\"Accepted\":"; pair_json(accepted); std::cout << ",\"Retained\":"; interval_json(interval); std::cout << '}';
}
std::string decode(std::string s) { return s == "-" ? "" : s; }

int main() {
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    try {
        std::string line;
        while (std::getline(std::cin,line)) {
            std::istringstream in(line); std::string op; in >> op;
            if (op == "ALIGN") {
                std::string q,t; int equality; in >> q >> t >> equality; q=decode(q); t=decode(t);
                auto config = dorado::demux::init_edlib_config_for_adapters();
                if (equality == 0) { config.additionalEqualities=nullptr; config.additionalEqualitiesLength=0; }
                auto result=edlibAlign(q.data(),int(q.size()),t.data(),int(t.size()),config);
                bool valid=result.startLocations && result.endLocations;
                std::cout << "{\"Distance\":" << result.editDistance << ",\"Start\":" << (valid?result.startLocations[0]:-1) << ",\"End\":" << (valid?result.endLocations[0]:-1) << ",\"Valid\":" << (valid?"true":"false") << "}\n";
                edlibFreeAlignResult(result);
            } else if (op == "BEST") {
                int count; in >> count; std::vector<dorado::SingleEndResult> results;
                for(int i=0;i<count;++i) { uint32_t bits; dorado::SingleEndResult end; in>>bits>>end.position.first>>end.position.second>>end.name; std::memcpy(&end.score,&bits,4); results.push_back(end); }
                end_json(dorado::demux::get_best_result(results)); std::cout << '\n';
            } else if (op == "TRIM") {
                int length; in >> length; dorado::AdapterScoreResult pair;
                for(auto* end:{&pair.front,&pair.rear}) { uint32_t bits; in >> bits >> end->position.first >> end->position.second >> end->name; std::memcpy(&end->score,&bits,4); }
                auto interval=dorado::demux::Trimmer::determine_trim_interval(pair,length);
                detection_json(pair,pair,interval); std::cout << '\n';
            } else if (op == "DETECT") {
                std::string kit,seq; int adapters,primers; in >> kit >> seq >> adapters >> primers; seq=decode(seq);
                dorado::demux::AdapterDetector detector(std::nullopt);
                dorado::AdapterScoreResult rawA,rawP;
                if(adapters) rawA=detector.find_adapters(seq,kit);
                if(primers) rawP=detector.find_primers(seq,kit,dorado::demux::PrimerAux::DEFAULT);
                auto a=rawA,p=rawP;
                auto ai=dorado::demux::Trimmer::determine_trim_interval(a,int(seq.size()));
                auto pi=dorado::demux::Trimmer::determine_trim_interval(p,int(seq.size()));
                dorado::PrimerClassification classification;
                if(primers) classification=detector.classify_primers(p,pi,seq);
                ReadShim read; read.read_common.adapter_trim_interval={0,int(seq.size())}; int conflicts=0;
                if(adapters || primers) intersect(ai,pi,read,conflicts);
                std::cout << "{\"Kit\":\"" << kit << "\",\"Admission\":\"" << (kit=="ANY"?"missing_no_stock_candidates":"supported") << "\",\"Adapters\":";
                detection_json(rawA,a,ai); std::cout << ",\"Primers\":"; detection_json(rawP,p,pi);
                std::cout << ",\"PrimerClassification\":{\"Name\":\"" << classification.primer_name << "\",\"Orientation\":" << int(classification.orientation) << "},\"Retained\":";
                interval_json(read.read_common.adapter_trim_interval); std::cout << ",\"Conflict\":" << (conflicts?"true":"false") << "}\n";
            } else if (op == "CANDIDATES") {
                dorado::demux::AdapterDetector detector(std::nullopt);
                auto& a=detector.get_adapter_sequences("SQK-LSK114"); auto& p=detector.get_primer_sequences("SQK-LSK114",dorado::demux::PrimerAux::DEFAULT);
                std::cout << "["; bool first=true; for(auto* list:{&a,&p}) for(const auto& q:*list) { if(!first) std::cout << ','; first=false; std::cout << "[\"" << q.name << "\",\"" << q.front_sequence << "\",\"" << q.rear_sequence << "\"]"; } std::cout << "]\n";
            } else throw std::runtime_error("unsupported oracle operation");
            if(!in) throw std::runtime_error("malformed oracle input");
        }
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
