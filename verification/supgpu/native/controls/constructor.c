/* Marker is argv[2] from /proc/self/cmdline; no environment or libc imports. */
#ifndef CONTROL_VARIANT
#error CONTROL_VARIANT must be fixed by the control plan
#endif
static long call3(long n, long a, long b, long c) {
    long r; __asm__ volatile("syscall" : "=a"(r) : "a"(n), "D"(a), "S"(b), "d"(c) : "rcx", "r11", "memory"); return r;
}
static long open_file(const char *p, long flags, long mode) {
    register long r10 __asm__("r10") = mode;
    long r; __asm__ volatile("syscall" : "=a"(r) : "a"(257L), "D"(-100L), "S"(p), "d"(flags), "r"(r10) : "rcx", "r11", "memory"); return r;
}
static void stop(void) { call3(231,89,0,0); for (;;) {} }
__attribute__((constructor)) static void control_constructor(void) {
    char args[4096]; long fd=open_file("/proc/self/cmdline",0,0);
    if(fd<0) stop(); long n=call3(0,fd,(long)args,sizeof(args)); call3(3,fd,0,0);
    if(n<=0 || n>=(long)sizeof(args) || args[n-1]!=0) stop();
    long offset=0;
    for(int i=0;i<2;i++) { while(offset<n && args[offset]) ++offset; ++offset; }
    if(offset>=n || args[offset]!='/') stop();
    fd=open_file(args+offset,193,0600);
    if(fd<0) stop();
#if CONTROL_VARIANT == 1
    const char marker[]="allowed constructor\n";
#elif CONTROL_VARIANT == 2
    const char marker[]="wrong-tier constructor\n";
#elif CONTROL_VARIANT == 3
    const char marker[]="unknown constructor\n";
#else
#error invalid CONTROL_VARIANT
#endif
    long written=call3(1,fd,(long)marker,sizeof(marker)-1); call3(3,fd,0,0);
    if(written!=(long)sizeof(marker)-1) stop();
}
