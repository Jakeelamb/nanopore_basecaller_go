/* CPU-only glibc loader control. No CUDA, model or tensor code. */
extern void *dlopen(const char *, int);
extern int dlclose(void *);
static long call3(long n, long a, long b, long c) {
    long r; __asm__ volatile("syscall" : "=a"(r) : "a"(n), "D"(a), "S"(b), "d"(c) : "rcx", "r11", "memory"); return r;
}
static long open_file(const char *p, long flags, long mode) {
    register long r10 __asm__("r10") = mode;
    long r; __asm__ volatile("syscall" : "=a"(r) : "a"(257L), "D"(-100L), "S"(p), "d"(flags), "r"(r10) : "rcx", "r11", "memory"); return r;
}
static unsigned long length(const char *s) { unsigned long n = 0; while (s[n]) ++n; return n; }
static void stop(int code) { call3(231,code,0,0); for (;;) {} }
static void pre_model(void) {
    char b[512], reply[64], digits[32]; unsigned long n=0, k=0;
    const char *prefix="{\"schema\":\"nanopore-supgpu-loader-gate/v1\",\"kind\":\"phase\",\"sequence\":1,\"cookie\":0,\"path\":\"pre-model\",\"namespace\":0,\"pid\":";
    while (*prefix) b[n++]=*prefix++;
    unsigned long pid=(unsigned long)call3(39,0,0,0);
    do { digits[k++]='0'+pid%10; pid/=10; } while(pid);
    while(k) b[n++]=digits[--k]; b[n++]='}'; b[n++]='\n';
    unsigned long done=0;
    while(done<n) { long r=call3(1,5,(long)(b+done),n-done); if(r==-4) continue; if(r<=0) stop(80); done+=r; }
    n=0; while(n<sizeof(reply)) { long r=call3(0,6,(long)(reply+n),1); if(r==-4) continue; if(r!=1) stop(81); if(reply[n++]=='\n') break; }
    const char *expected="allow phase 1\n";
    if(n!=length(expected)) stop(82);
    for(unsigned long i=0;i<n;i++) if(reply[i]!=expected[i]) stop(83);
}
int main(int argc, char **argv) {
    if(argc!=4) return 84;
    pre_model();
    void *object=dlopen(argv[1],2); /* RTLD_NOW | RTLD_LOCAL */
    if(!object) return 85;
    long fd=open_file(argv[3],193,0600); /* O_WRONLY | O_CREAT | O_EXCL */
    if(fd<0) return 86;
    const char *message="dlopen returned\n";
    long written=call3(1,fd,(long)message,length(message)); call3(3,fd,0,0);
    if(written!=(long)length(message)) return 87;
    return dlclose(object)==0 ? 0 : 88;
}
