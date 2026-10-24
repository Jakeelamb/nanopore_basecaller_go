#include "abi.h"
#include "runtime.h"
#include <ATen/ATen.h>
#include <torch/serialize.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <memory>
#include "cuda_records.h"
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
std::mutex mutex;
uint64_t next_id = 1;
using Tensors = std::unordered_map<uint64_t, at::Tensor>;
struct EngineRecord {
 Tensors tensors;
 std::unordered_map<uint64_t,std::shared_ptr<CUDADeviceRecord>> devices;
 std::unordered_map<uint64_t,std::shared_ptr<CUDAStreamRecord>> streams;
 std::unordered_map<uint64_t,std::shared_ptr<CUDAHalfRecord>> halves;
 std::unordered_map<uint64_t,std::shared_ptr<CUDAStorageRecord>> storages;
};
std::unordered_map<uint64_t, EngineRecord> engines;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
uint64_t fresh() {
    require(next_id != std::numeric_limits<uint64_t>::max(), "handle IDs exhausted");
    return next_id++;
}
Tensors& engine(uint64_t id) {
    auto it = engines.find(id);
    require(it != engines.end(), "invalid or closed engine handle");
    return it->second.tensors;
}
at::Tensor get(Tensors& ts, uint64_t id) {
    auto it = ts.find(id);
    require(it != ts.end(), "invalid, foreign or released tensor handle");
    return it->second;
}
uint64_t put(Tensors& ts, at::Tensor tensor) {
    require(tensor.device().is_cpu() && tensor.scalar_type() == at::kFloat,
            "only CPU float32 tensors are supported");
    require(tensor.dim()<=32,"tensor rank exceeds ABI limit");
    auto id = fresh();
    ts.emplace(id, std::move(tensor));
    return id;
}
uint64_t put_indices(Tensors& ts, at::Tensor tensor) {
    require(tensor.device().is_cpu() && tensor.scalar_type()==at::kLong,"indices must be CPU int64");
    auto id=fresh();ts.emplace(id,std::move(tensor));return id;
}
template<class F> int boundary(char* error, size_t cap, F f) noexcept {
    if (!error || cap == 0) return 1;
    error[0] = 0;
    try { std::lock_guard<std::mutex> lock(mutex); c10::InferenceMode guard; f(); return 0; }
    catch (const std::exception& ex) {
        auto n = std::min(cap-1, std::strlen(ex.what()));
        std::memcpy(error, ex.what(), n); error[n] = 0;
    } catch (...) {
        const char* text = "unknown native exception";
        auto n = std::min(cap-1, std::strlen(text));
        std::memcpy(error, text, n); error[n] = 0;
    }
    return 1;
}
size_t elements(const int64_t* dims, size_t rank) {
    require(rank <= 32 && (rank == 0 || dims), "invalid shape rank or pointer");
    size_t count = 1;
    for (size_t i=0; i<rank; ++i) {
        require(dims[i] >= 0, "negative tensor dimension");
        auto dim = static_cast<size_t>(dims[i]);
        require(dim == 0 || count <= static_cast<size_t>(INT64_MAX)/dim, "tensor shape overflow");
        count *= dim;
    }
    require(count <= SIZE_MAX/sizeof(float), "tensor byte count overflow");
    return count;
}
}
#include "cuda_backend.h"
#include "cuda_storage.inc"
#include "hac_smallconv.inc"
extern "C" {
uint32_t nt_version() { return NT_ABI_VERSION; }
int nt_open(uint64_t* out, char* err, size_t cap) {
    return boundary(err,cap,[&]{ require(out, "null output"); initialize_tensor_runtime(); auto id=fresh(); engines.emplace(id,EngineRecord{}); *out=id; });
}
int nt_close(uint64_t e, char* err, size_t cap) {
    return boundary(err,cap,[&]{ engine(e); close_engine_cuda(e); engines.erase(e); });
}
int nt_new(uint64_t e, const int64_t* dims, size_t rank, const float* data, size_t count, uint64_t* out, char* err, size_t cap) {
    return boundary(err,cap,[&]{ auto& ts=engine(e); require(out,"null output");
        require(elements(dims,rank)==count,"shape and data length differ");
        require(count==0 || data,"null data");
        auto t=at::empty(at::IntArrayRef(dims,rank),at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
        if(count) std::memcpy(t.data_ptr<float>(),data,count*sizeof(float));
        *out=put(ts,std::move(t)); });
}
int nt_new_int64(uint64_t e,const int64_t* dims,size_t rank,const int64_t* data,size_t count,uint64_t* out,char* err,size_t cap) {
    return boundary(err,cap,[&]{auto& ts=engine(e);require(out,"null output");
        require(elements(dims,rank)==count,"shape and data length differ");require(count==0 || data,"null data");require(count<=SIZE_MAX/sizeof(int64_t),"index byte count overflow");
        auto t=at::empty(at::IntArrayRef(dims,rank),at::TensorOptions().dtype(at::kLong).device(at::kCPU));
        if(count)std::memcpy(t.data_ptr<int64_t>(),data,count*sizeof(int64_t));*out=put_indices(ts,std::move(t));});
}
int nt_load(uint64_t e, const char* path, uint64_t* out, char* err, size_t cap) {
    return boundary(err,cap,[&]{ auto& ts=engine(e); require(path && out,"null path or output");
        std::vector<at::Tensor> weights; torch::load(weights,path,at::Device(at::kCPU));
        require(weights.size()==1,"weight archive must contain exactly one tensor"); *out=put(ts,std::move(weights[0])); });
}
int nt_release(uint64_t e,uint64_t id,char* err,size_t cap) {
    return boundary(err,cap,[&]{ auto& ts=engine(e); get(ts,id); ts.erase(id); });
}
int nt_shape(uint64_t e,uint64_t id,int64_t* dims,size_t cap_dims,size_t* rank,char* err,size_t cap) {
    return boundary(err,cap,[&]{ auto t=get(engine(e),id); require(rank,"null rank output");
        require(cap_dims>=static_cast<size_t>(t.dim()) && (t.dim()==0 || dims),"shape buffer too small");
        *rank=t.dim(); std::copy(t.sizes().begin(),t.sizes().end(),dims); });
}
int nt_data(uint64_t e,uint64_t id,float* data,size_t count,char* err,size_t cap) {
    return boundary(err,cap,[&]{ auto t=get(engine(e),id).contiguous();
        require(t.scalar_type()==at::kFloat,"float32 copy requires a float tensor");
        require(count==static_cast<size_t>(t.numel()) && (count==0 || data),"data buffer length mismatch");
        if(count) std::memcpy(data,t.data_ptr<float>(),count*sizeof(float)); });
}
int nt_data_int64(uint64_t e,uint64_t id,int64_t* data,size_t count,char* err,size_t cap) {
    return boundary(err,cap,[&]{auto t=get(engine(e),id).contiguous();require(t.scalar_type()==at::kLong,"int64 copy requires an index tensor");
        require(count==static_cast<size_t>(t.numel()) && (count==0 || data),"index buffer length mismatch");if(count)std::memcpy(data,t.data_ptr<int64_t>(),count*sizeof(int64_t));});
}
int nt_run(uint64_t e,int op,const uint64_t* ids,size_t count,const int64_t* p,size_t pn,const double* f,size_t fn,uint64_t* out,char* err,size_t cap) {
    return boundary(err,cap,[&]{ auto& ts=engine(e); require(out && (count==0 || ids),"missing tensor input or output");
        require((pn==0 || p) && (fn==0 || f),"null operation parameters");
        std::vector<at::Tensor> t; for(size_t i=0;i<count;++i) {
            auto value=get(ts,ids[i]);
            auto dtype=((op==NT_INDEX_SELECT && i==1) || op==NT_ARGSORT) ? at::kLong : at::kFloat;
            require(value.scalar_type()==dtype,"operation input dtype mismatch");
            t.push_back(std::move(value));
        }
        auto arity=[&](size_t n,size_t ints,size_t floats){require(count==n && pn==ints && fn==floats,"invalid operation arity");};
        at::Tensor r;
        switch(op) {
        case NT_ADD: arity(2,0,0); r=t[0]+t[1]; break;
        case NT_MUL: arity(2,0,0); r=t[0]*t[1]; break;
        case NT_MATMUL: arity(2,0,0); r=at::matmul(t[0],t[1]); break;
        case NT_CLAMP: arity(1,0,2); require(f[0]<=f[1],"invalid clamp bounds"); r=at::clamp(t[0],f[0],f[1]); break;
        case NT_TANH: arity(1,0,0); r=at::tanh(t[0]); break;
        case NT_SIGMOID: arity(1,0,0); r=at::sigmoid(t[0]); break;
        case NT_SILU: arity(1,0,0); r=at::silu(t[0]); break;
        case NT_TRANSPOSE: arity(1,2,0); r=t[0].transpose(p[0],p[1]); break;
        case NT_FLIP: require(count==1 && fn==0,"invalid flip arity"); r=at::flip(t[0],at::IntArrayRef(p,pn)); break;
        case NT_RESHAPE: require(count==1 && fn==0 && pn<=32,"invalid reshape arity"); r=t[0].reshape(at::IntArrayRef(p,pn)); break;
        case NT_NARROW: arity(1,3,0); r=t[0].narrow(p[0],p[1],p[2]); break;
        case NT_CONCAT: require(pn==1 && fn==0,"invalid concat arity"); r=at::cat(t,p[0]); break;
        case NT_STACK: require(pn==1 && fn==0,"invalid stack arity"); r=at::stack(t,p[0]); break;
        case NT_CONV1D: require((count==2 || count==3) && pn==4 && fn==0,"invalid conv1d arity");
            r=at::conv1d(t[0],t[1],count==3 ? std::optional<at::Tensor>(t[2]) : std::nullopt,{p[0]},{p[1]},{p[2]},p[3]); break;
        case NT_LINEAR: require((count==2 || count==3) && pn==0 && fn==0,"invalid linear arity");
            r=at::linear(t[0],t[1],count==3 ? std::optional<at::Tensor>(t[2]) : std::nullopt); break;
        case NT_ADD_SCALAR: arity(1,0,1); r=t[0]+f[0]; break;
        case NT_MUL_SCALAR: arity(1,0,1); r=t[0]*f[0]; break;
        case NT_PERMUTE: require(count==1 && fn==0,"invalid permute arity"); r=t[0].permute(at::IntArrayRef(p,pn)); break;
        case NT_CONTIGUOUS: arity(1,0,0); r=t[0].contiguous(); break;
        case NT_ZEROS: require(count==0 && fn==0,"invalid zeros arity"); elements(p,pn); r=at::zeros(at::IntArrayRef(p,pn),at::TensorOptions().dtype(at::kFloat).device(at::kCPU)); break;
        case NT_INDEX_SELECT: arity(2,1,0); require(t[0].scalar_type()==at::kFloat && t[1].scalar_type()==at::kLong,"index_select requires float32 input and int64 indices"); r=at::index_select(t[0],p[0],t[1]); break;
        case NT_LOGSUMEXP: arity(1,2,0); require(p[1]==0 || p[1]==1,"invalid keepdim flag"); r=at::logsumexp(t[0],{p[0]},p[1]!=0); break;
        case NT_SOFTMAX: arity(1,1,0); r=at::softmax(t[0],p[0]); break;
        case NT_UNSQUEEZE: arity(1,1,0); r=t[0].unsqueeze(p[0]); break;
        case NT_ARGSORT: arity(1,1,0); r=at::argsort(t[0],p[0],false); break;
        case NT_MEAN: arity(1,2,0); require(p[1]==0 || p[1]==1,"invalid keepdim flag"); r=at::mean(t[0],{p[0]},p[1]!=0); break;
        case NT_RSQRT: arity(1,0,0); r=at::rsqrt(t[0]); break;
        case NT_RECIPROCAL: arity(1,0,0); r=at::reciprocal(t[0]); break;
        case NT_SIN: arity(1,0,0); r=at::sin(t[0]); break;
        case NT_COS: arity(1,0,0); r=at::cos(t[0]); break;
        case NT_SUB: arity(2,0,0); r=t[0]-t[1]; break;
        case NT_SQUARE: arity(1,0,0); r=t[0].square(); break;
        default: throw std::runtime_error("unknown operation ID");
        }
        *out=op==NT_ARGSORT ? put_indices(ts,std::move(r)) : put(ts,std::move(r)); });
}
int nt_sdpa(uint64_t e,uint64_t q_id,uint64_t k_id,uint64_t v_id,const int64_t* dims,size_t rank,const uint8_t* mask,size_t count,uint64_t* out,char* err,size_t cap) {
    return boundary(err,cap,[&]{auto& ts=engine(e);require(out,"null output");
        auto q=get(ts,q_id),k=get(ts,k_id),v=get(ts,v_id);
        for(const auto& t : {q,k,v}) require(t.scalar_type()==at::kFloat,"attention inputs must be CPU float32");
        std::optional<at::Tensor> opt_mask;
        if(dims || rank || mask || count) {
            require(elements(dims,rank)==count,"mask shape and data length differ");
            require(count==0 || mask,"null mask data");
            auto m=at::empty(at::IntArrayRef(dims,rank),at::TensorOptions().dtype(at::kBool).device(at::kCPU));
            auto dst=m.data_ptr<bool>();
            for(size_t i=0;i<count;++i){require(mask[i]<=1,"mask bytes must be 0 or 1");dst[i]=mask[i]!=0;}
            opt_mask=std::move(m);
        }
        *out=put(ts,at::scaled_dot_product_attention(q,k,v,opt_mask));
    });
}
}
