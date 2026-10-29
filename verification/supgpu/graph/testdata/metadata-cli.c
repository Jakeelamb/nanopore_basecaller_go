#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#ifndef SELECT_PROVIDER_PATH
#error "controller must select an absolute provider path"
#endif

int main(int argc, char **argv) {
    if (argc != 5 || strcmp(argv[1], "-i") || strcmp(argv[2], "0") ||
        strcmp(argv[3], "--query-gpu=index,uuid,pci.bus_id,driver_version") ||
        strcmp(argv[4], "--format=csv,noheader,nounits")) return 61;
    void *provider = dlopen(SELECT_PROVIDER_PATH, RTLD_LAZY | RTLD_LOCAL);
    if (!provider) return 62;
    const char *(*csv)(void) = (const char *(*)(void))dlsym(provider, "metadata_csv");
    if (!csv) return 63;
    if (fputs(csv(), stdout) < 0) return 64;
    return dlclose(provider) ? 65 : 0;
}
