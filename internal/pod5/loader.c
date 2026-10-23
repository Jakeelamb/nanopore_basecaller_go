//go:build cgo && linux

#include "loader.h"
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define API_FUNCTIONS(X) \
 X(pod5_init) X(pod5_get_error_no) X(pod5_get_error_string) \
 X(pod5_open_file) X(pod5_close_and_free_reader) X(pod5_get_read_count) \
 X(pod5_get_read_batch_count) X(pod5_get_read_batch) X(pod5_free_read_batch) \
 X(pod5_get_read_batch_row_count) X(pod5_get_read_batch_row_info_data) \
 X(pod5_get_run_info) X(pod5_free_run_info) X(pod5_get_end_reason) \
 X(pod5_get_pore_type) X(pod5_get_read_complete_sample_count) X(pod5_get_read_complete_signal) \
 X(pod5_get_file_info) X(pod5_get_file_run_info_count) X(pod5_get_file_run_info)
#define DECLARE(name) static __typeof__(&name) api_##name;
API_FUNCTIONS(DECLARE)
static pthread_mutex_t init_lock = PTHREAD_MUTEX_INITIALIZER;
static void *library_handle;
static char library_path[PATH_MAX];
static char initialization_error[P5_ERROR_CAP];
struct p5_reader { Pod5FileReader_t *file; };

static int fail(char *error, const char *operation) {
    snprintf(error, P5_ERROR_CAP, "%s: POD5 error %d: %s", operation,
             (int)api_pod5_get_error_no(), api_pod5_get_error_string());
    return -1;
}
static int load_library(const char *path, char *error) {
    char canonical[PATH_MAX];
    if (!realpath(path, canonical)) {
        snprintf(error, P5_ERROR_CAP, "cannot resolve POD5 library: %s", path);
        return -1;
    }
    pthread_mutex_lock(&init_lock);
    if (library_handle) {
        int same = strcmp(library_path, canonical) == 0;
        int initialized = initialization_error[0] == '\0';
        if (same && !initialized) snprintf(error, P5_ERROR_CAP, "%s", initialization_error);
        pthread_mutex_unlock(&init_lock);
        if (!same) snprintf(error, P5_ERROR_CAP, "POD5 process already initialized with a different library");
        return same && initialized ? 0 : -1;
    }
    void *handle = dlopen(canonical, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        snprintf(error, P5_ERROR_CAP, "dlopen POD5: %s", dlerror());
        pthread_mutex_unlock(&init_lock);
        return -1;
    }
#define RESOLVE(name) \
    *(void **)(&api_##name) = dlsym(handle, #name); \
    if (!api_##name) { snprintf(error, P5_ERROR_CAP, "missing POD5 symbol %s", #name); \
        dlclose(handle); pthread_mutex_unlock(&init_lock); return -1; }
    API_FUNCTIONS(RESOLVE)
    // Even a failed init may have registered process-global Arrow extension types.
    library_handle = handle;
    strcpy(library_path, canonical);
    if (api_pod5_init() != POD5_OK) {
        fail(initialization_error, "pod5_init");
        snprintf(error, P5_ERROR_CAP, "%s", initialization_error);
        pthread_mutex_unlock(&init_lock);
        return -1;
    }
    // Arrow extension registration is process global. Readers never terminate or unload it.
    pthread_mutex_unlock(&init_lock);
    return 0;
}
struct p5_reader *p5_open(const char *library, const char *file, size_t max_reads, size_t *count, size_t *batches, char *error) {
    if (load_library(library, error)) return NULL;
    Pod5FileReader_t *f = api_pod5_open_file(file);
    if (!f) { fail(error, "pod5_open_file"); return NULL; }
    if (api_pod5_get_read_count(f, count) != POD5_OK) {
        fail(error, "pod5_get_read_count"); api_pod5_close_and_free_reader(f); return NULL;
    }
    if (*count > max_reads) {
        snprintf(error, P5_ERROR_CAP, "POD5 read count %zu exceeds MaxReads %zu", *count, max_reads);
        api_pod5_close_and_free_reader(f); return NULL;
    }
    if (api_pod5_get_read_batch_count(batches, f) != POD5_OK) {
        fail(error, "pod5_get_read_batch_count"); api_pod5_close_and_free_reader(f); return NULL;
    }
    struct p5_reader *r = malloc(sizeof(*r));
    if (!r) { snprintf(error, P5_ERROR_CAP, "allocate POD5 reader"); api_pod5_close_and_free_reader(f); return NULL; }
    r->file = f;
    return r;
}
int p5_close(struct p5_reader *r, char *error) {
    int rc = api_pod5_close_and_free_reader(r->file);
    if (rc != POD5_OK) fail(error, "pod5_close_and_free_reader");
    free(r);
    return rc == POD5_OK ? 0 : -1;
}
int p5_batch_rows(struct p5_reader *r, size_t batch, size_t *rows, char *error) {
    Pod5ReadRecordBatch_t *b = NULL;
    if (api_pod5_get_read_batch(&b, r->file, batch) != POD5_OK) return fail(error, "pod5_get_read_batch");
    int rc = api_pod5_get_read_batch_row_count(rows, b);
    if (rc != POD5_OK) fail(error, "pod5_get_read_batch_row_count");
    int freed = api_pod5_free_read_batch(b);
    if (rc != POD5_OK) return -1;
    return freed == POD5_OK ? 0 : fail(error, "pod5_free_read_batch");
}
int p5_row(struct p5_reader *r, size_t batch, size_t row, ReadBatchRowInfo_t *info, char *error) {
    Pod5ReadRecordBatch_t *b = NULL;
    if (api_pod5_get_read_batch(&b, r->file, batch) != POD5_OK) return fail(error, "pod5_get_read_batch");
    uint16_t version = 0;
    int rc = api_pod5_get_read_batch_row_info_data(b, row, READ_BATCH_ROW_INFO_VERSION, info, &version);
    if (rc != POD5_OK) fail(error, "pod5_get_read_batch_row_info_data");
    int freed = api_pod5_free_read_batch(b);
    if (rc != POD5_OK) return -1;
    return freed == POD5_OK ? 0 : fail(error, "pod5_free_read_batch");
}
int p5_result_free(struct p5_result *s, char *error) {
    int rc = 0;
    free(s->signal);
    if (s->run && api_pod5_free_run_info(s->run) != POD5_OK) rc = fail(error, "pod5_free_run_info");
    if (s->batch && api_pod5_free_read_batch(s->batch) != POD5_OK && rc == 0) rc = fail(error, "pod5_free_read_batch");
    free(s);
    return rc;
}
int p5_file_info(struct p5_reader *r, FileInfo_t *info, run_info_index_t *runs, char *error) {
    if (api_pod5_get_file_info(r->file, info) != POD5_OK) return fail(error, "pod5_get_file_info");
    if (api_pod5_get_file_run_info_count(r->file, runs) != POD5_OK) return fail(error, "pod5_get_file_run_info_count");
    return 0;
}
struct p5_result *p5_file_run(struct p5_reader *r, run_info_index_t index, char *error) {
    struct p5_result *s = calloc(1, sizeof(*s));
    if (!s) { snprintf(error, P5_ERROR_CAP, "allocate POD5 run info"); return NULL; }
    if (api_pod5_get_file_run_info(r->file, index, &s->run) != POD5_OK) {
        fail(error, "pod5_get_file_run_info");
        free(s);
        return NULL;
    }
    return s;
}
struct p5_result *p5_read(struct p5_reader *r, size_t batch, size_t row, uint64_t max_samples, char *error) {
    struct p5_result *s = calloc(1, sizeof(*s));
    if (!s) { snprintf(error, P5_ERROR_CAP, "allocate POD5 result"); return NULL; }
#define CHECK(call) if ((call) != POD5_OK) { fail(error, #call); goto failed; }
    CHECK(api_pod5_get_read_batch(&s->batch, r->file, batch));
    CHECK(api_pod5_get_read_batch_row_info_data(s->batch, row, READ_BATCH_ROW_INFO_VERSION, &s->row, &s->table_version));
    size_t count = 0;
    CHECK(api_pod5_get_read_complete_sample_count(r->file, s->batch, row, &count));
    if (count != s->row.num_samples || count > max_samples || count > SIZE_MAX / sizeof(int16_t)) {
        snprintf(error, P5_ERROR_CAP, "POD5 signal count mismatch or MaxSamples exceeded: %zu / %llu, limit %llu", count,
                 (unsigned long long)s->row.num_samples, (unsigned long long)max_samples);
        goto failed;
    }
    s->signal = malloc(count ? count * sizeof(int16_t) : 1);
    if (!s->signal) { snprintf(error, P5_ERROR_CAP, "allocate POD5 signal"); goto failed; }
    CHECK(api_pod5_get_read_complete_signal(r->file, s->batch, row, count, s->signal));
    CHECK(api_pod5_get_run_info(s->batch, s->row.run_info, &s->run));
    size_t size = sizeof(s->end_reason_name);
    CHECK(api_pod5_get_end_reason(s->batch, s->row.end_reason, &s->end_reason, s->end_reason_name, &size));
    size = sizeof(s->pore_type);
    CHECK(api_pod5_get_pore_type(s->batch, s->row.pore_type, s->pore_type, &size));
    return s;
failed:;
    char cleanup_error[P5_ERROR_CAP];
    p5_result_free(s, cleanup_error);
    return NULL;
}
