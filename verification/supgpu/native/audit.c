/* Verification loader observer for the pinned Linux x86-64/glibc target.
 * No libc, constructors, arithmetic hooks or symbol interposition. Every loader
 * object rendezvous precedes its initializers and dlopen returning to callers.
 * FDs 3/4 are private request/approval pipes supplied by the Go controller. */
typedef unsigned long word;
struct link_map { word address; char *name; void *dynamic; struct link_map *next, *previous; };
static volatile int busy;
static word sequence;

static long call3(long n, long a, long b, long c) {
    long r;
    __asm__ volatile("syscall" : "=a"(r) : "a"(n), "D"(a), "S"(b), "d"(c) : "rcx", "r11", "memory");
    return r;
}
static void stop(void) { call3(1, 2, (long)"supgpu loader gate denied or unavailable\n", sizeof("supgpu loader gate denied or unavailable\n")-1); call3(231, 124, 0, 0); for (;;) {} }
static void put(char *b, word *n, char c) { if (*n >= 4094) stop(); b[(*n)++] = c; }
static void text(char *b, word *n, const char *s) { while (*s) put(b, n, *s++); }
static void number(char *b, word *n, word v) { char d[32]; int k = 0; do { d[k++] = '0' + v % 10; v /= 10; } while (v); while (k) put(b, n, d[--k]); }
static void string(char *b, word *n, const char *s) {
    const char *hex = "0123456789abcdef";
    put(b, n, '"');
    while (*s) { unsigned char c = *s++; if (c == '"' || c == '\\') { put(b,n,'\\'); put(b,n,c); }
        else if (c < 32) { text(b,n,"\\u00"); put(b,n,hex[c >> 4]); put(b,n,hex[c & 15]); } else put(b,n,c); }
    put(b,n,'"');
}
static void send(const char *kind, word cookie, const char *path, long ns) {
    char b[4096], expected[128], reply[128]; word n = 0, en = 0, rn = 0, seq;
    while (__sync_lock_test_and_set(&busy, 1)) {}
    seq = ++sequence;
    text(b,&n,"{\"schema\":\"nanopore-supgpu-loader-gate/v1\",\"kind\":"); string(b,&n,kind);
    text(b,&n,",\"sequence\":"); number(b,&n,seq); text(b,&n,",\"cookie\":"); number(b,&n,cookie);
    text(b,&n,",\"path\":"); string(b,&n,path); text(b,&n,",\"namespace\":");
    if (ns < 0) { put(b,&n,'-'); number(b,&n,(word)(-ns)); } else number(b,&n,(word)ns);
    text(b,&n,",\"pid\":"); number(b,&n,(word)call3(39,0,0,0)); text(b,&n,"}\n");
    word sent = 0; while (sent < n) { long r = call3(1,3,(long)(b+sent),(long)(n-sent)); if (r == -4) continue; if (r <= 0) stop(); sent += (word)r; }
    text(expected,&en,"allow "); text(expected,&en,kind); put(expected,&en,' '); number(expected,&en,seq); put(expected,&en,'\n');
    while (rn < sizeof(reply)) { long r = call3(0,4,(long)(reply+rn),1); if (r == -4) continue; if (r != 1) stop(); if (reply[rn++] == '\n') break; }
    if (rn != en) stop(); for (word i = 0; i < en; i++) if (reply[i] != expected[i]) stop();
    __sync_lock_release(&busy);
}
unsigned int la_version(unsigned int v) { if (v < 2) stop(); send("hello",0,"",0); return 2; }
unsigned int la_objopen(struct link_map *map, long ns, word *cookie) {
    /* Cookie identity is the immutable link-map address, never dereferenced by Go. */
    *cookie = (word)map; send("open",*cookie,map->name ? map->name : "",ns); return 0;
}
unsigned int la_objclose(word *cookie) { send("close",*cookie,"",0); return 0; }
