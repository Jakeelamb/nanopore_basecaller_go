//go:build cgo && linux

package pod5

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -ldl -lpthread
#include "loader.h"
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"errors"
	"unsafe"
)

type nativeReader struct{ ptr *C.struct_p5_reader }

func errorBuffer(b *[C.P5_ERROR_CAP]C.char) error { return errors.New(C.GoString(&b[0])) }
func openNative(library, file string, maxReads uint64) (*nativeReader, uint64, uint64, error) {
	l, f := C.CString(library), C.CString(file)
	defer C.free(unsafe.Pointer(l))
	defer C.free(unsafe.Pointer(f))
	var count, batches C.size_t
	var b [C.P5_ERROR_CAP]C.char
	p := C.p5_open(l, f, C.size_t(maxReads), &count, &batches, &b[0])
	if p == nil {
		return nil, 0, 0, errorBuffer(&b)
	}
	return &nativeReader{p}, uint64(count), uint64(batches), nil
}
func (r *nativeReader) close() error {
	var b [C.P5_ERROR_CAP]C.char
	if C.p5_close(r.ptr, &b[0]) != 0 {
		return errorBuffer(&b)
	}
	return nil
}
func (r *nativeReader) batchRows(batch uint64) (uint64, error) {
	var count C.size_t
	var b [C.P5_ERROR_CAP]C.char
	if C.p5_batch_rows(r.ptr, C.size_t(batch), &count, &b[0]) != 0 {
		return 0, errorBuffer(&b)
	}
	return uint64(count), nil
}
func (r *nativeReader) rowID(batch, row uint64) (ID, error) {
	var info C.ReadBatchRowInfo_t
	var b [C.P5_ERROR_CAP]C.char
	if C.p5_row(r.ptr, C.size_t(batch), C.size_t(row), &info, &b[0]) != 0 {
		return ID{}, errorBuffer(&b)
	}
	var id ID
	for i := range id {
		id[i] = byte(info.read_id[i])
	}
	return id, nil
}
func copyMap(kv C.struct_KeyValueData) map[string]string {
	m := make(map[string]string, int(kv.size))
	keys := unsafe.Slice(kv.keys, int(kv.size))
	values := unsafe.Slice(kv.values, int(kv.size))
	for i := range keys {
		m[C.GoString(keys[i])] = C.GoString(values[i])
	}
	return m
}
func (r *nativeReader) read(loc location, maxSamples uint64) (out Read, err error) {
	var b [C.P5_ERROR_CAP]C.char
	s := C.p5_read(r.ptr, C.size_t(loc.batch), C.size_t(loc.row), C.uint64_t(maxSamples), &b[0])
	if s == nil {
		return Read{}, errorBuffer(&b)
	}
	defer func() {
		if C.p5_result_free(s, &b[0]) != 0 {
			err = errors.Join(err, errorBuffer(&b))
		}
	}()
	row := s.row
	for i := range out.ID {
		out.ID[i] = byte(row.read_id[i])
	}
	out.Signal = append([]int16{}, unsafe.Slice((*int16)(unsafe.Pointer(s.signal)), int(row.num_samples))...)
	out.NumSamples = uint64(row.num_samples)
	out.CalibrationScale = float32(row.calibration_scale)
	out.CalibrationOffset = float32(row.calibration_offset)
	out.OpenPoreLevel = float32(row.open_pore_level)
	out.MedianBefore = float32(row.median_before)
	out.ReadNumber = uint32(row.read_number)
	out.StartSample = uint64(row.start_sample)
	out.Channel = uint16(row.channel)
	out.Well = uint8(row.well)
	out.NumMinknowEvents = uint64(row.num_minknow_events)
	out.EndReason = uint8(s.end_reason)
	out.EndReasonForced = row.end_reason_forced != 0
	out.EndReasonName = C.GoString(&s.end_reason_name[0])
	out.PoreType = C.GoString(&s.pore_type[0])
	out.ReadTableVersion = uint16(s.table_version)
	out.RunInfo = copyRunInfo(s.run)
	return out, nil
}

func copyRunInfo(run *C.RunInfoDictData_t) (out RunInfo) {
	out.SampleRate = uint64(run.sample_rate)
	out.RunID = C.GoString(run.protocol_run_id)
	out.AcquisitionID = C.GoString(run.acquisition_id)
	out.AcquisitionStartTimeMS = int64(run.acquisition_start_time_ms)
	out.ProtocolStartTimeMS = int64(run.protocol_start_time_ms)
	out.FlowCellID = C.GoString(run.flow_cell_id)
	out.FlowCellProductCode = C.GoString(run.flow_cell_product_code)
	out.SequencingKit = C.GoString(run.sequencing_kit)
	out.PositionID = C.GoString(run.sequencer_position)
	out.PositionType = C.GoString(run.sequencer_position_type)
	out.SampleID = C.GoString(run.sample_id)
	out.ExperimentID = C.GoString(run.experiment_name)
	out.ProtocolName = C.GoString(run.protocol_name)
	out.Software = C.GoString(run.software)
	out.SystemName = C.GoString(run.system_name)
	out.SystemType = C.GoString(run.system_type)
	out.ADCMin = int16(run.adc_min)
	out.ADCMax = int16(run.adc_max)
	out.ContextTags = copyMap(run.context_tags)
	out.TrackingID = copyMap(run.tracking_id)
	if out.ExperimentID == "" {
		out.ExperimentID = out.TrackingID["protocol_group_id"]
	}
	return out
}

func (r *nativeReader) metadata(ctx context.Context) (FileMetadata, error) {
	var info C.FileInfo_t
	var runs C.run_info_index_t
	var b [C.P5_ERROR_CAP]C.char
	if C.p5_file_info(r.ptr, &info, &runs, &b[0]) != 0 {
		return FileMetadata{}, errorBuffer(&b)
	}
	out := FileMetadata{OriginalVersion: [3]uint16{uint16(info.version.major), uint16(info.version.minor), uint16(info.version.revision)}, Runs: make([]RunInfo, 0, int(runs))}
	for i := range out.ID {
		out.ID[i] = byte(info.file_identifier[i])
	}
	for i := uint32(0); i < uint32(runs); i++ {
		if err := ctx.Err(); err != nil {
			return FileMetadata{}, err
		}
		s := C.p5_file_run(r.ptr, C.run_info_index_t(i), &b[0])
		if s == nil {
			return FileMetadata{}, errorBuffer(&b)
		}
		out.Runs = append(out.Runs, copyRunInfo(s.run))
		if C.p5_result_free(s, &b[0]) != 0 {
			return FileMetadata{}, errorBuffer(&b)
		}
	}
	if err := ctx.Err(); err != nil {
		return FileMetadata{}, err
	}
	return out, nil
}
