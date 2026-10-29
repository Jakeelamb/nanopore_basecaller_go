#include <fcntl.h>
#include <unistd.h>

#ifndef CONSTRUCTOR_MARKER
#error "controller must select an absolute fresh CONSTRUCTOR_MARKER"
#endif

__attribute__((constructor)) static void record_constructor(void) {
    int fd = open(CONSTRUCTOR_MARKER, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd >= 0) { (void)write(fd, "constructed\n", 12); (void)close(fd); }
}

const char *metadata_csv(void) {
    return "0, GPU-00112233-4455-6677-8899-aabbccddeeff, 00000000:01:00.0, 580.1\n";
}
