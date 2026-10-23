#ifndef NANOPORE_POD5_LOADER_H
#define NANOPORE_POD5_LOADER_H
#include "pod5_format/c_api.h"
#define P5_ERROR_CAP 2048
struct p5_reader;
struct p5_result {
    ReadBatchRowInfo_t row;
    uint16_t table_version;
    RunInfoDictData_t *run;
    Pod5ReadRecordBatch_t *batch;
    int16_t *signal;
    pod5_end_reason_t end_reason;
    char end_reason_name[1024];
    char pore_type[1024];
};
struct p5_reader *p5_open(const char *library, const char *file, size_t max_reads, size_t *count, size_t *batches, char *error);
int p5_close(struct p5_reader *, char *error);
int p5_batch_rows(struct p5_reader *, size_t batch, size_t *rows, char *error);
int p5_row(struct p5_reader *, size_t batch, size_t row, ReadBatchRowInfo_t *info, char *error);
int p5_file_info(struct p5_reader *, FileInfo_t *, run_info_index_t *, char *error);
struct p5_result *p5_file_run(struct p5_reader *, run_info_index_t index, char *error);
struct p5_result *p5_read(struct p5_reader *, size_t batch, size_t row, uint64_t max_samples, char *error);
int p5_result_free(struct p5_result *, char *error);
#endif
