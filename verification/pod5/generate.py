"""Produce independent POD5 reader fixtures. Run only through the stock uv project."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import sys
from datetime import datetime, timezone

import pod5


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def milliseconds(value):
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = value - epoch
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


def extract(read, output, name):
    run = read.run_info
    signal = read.signal.astype("<i2", copy=False).tobytes()
    filename = name + ".signal.i16le"
    (output / filename).write_bytes(signal)
    metadata = {
        "ID": list(read.read_id.bytes),
        "NumSamples": read.num_samples,
        "ReadNumber": read.read_number,
        "StartSample": read.start_sample,
        "EndSample": read.start_sample + read.num_samples,
        "Channel": read.pore.channel,
        "Well": read.pore.well,
        "NumMinknowEvents": read.num_minknow_events,
        "EndReason": read.end_reason.reason.value,
        "EndReasonForced": read.end_reason.forced,
        "EndReasonName": read.end_reason.reason.name.lower(),
        "PoreType": read.pore.pore_type,
        "SampleRate": run.sample_rate,
        "RunID": run.protocol_run_id,
        "AcquisitionID": run.acquisition_id,
        "AcquisitionStartTimeMS": milliseconds(run.acquisition_start_time),
        "ProtocolStartTimeMS": milliseconds(run.protocol_start_time),
        "FlowCellID": run.flow_cell_id,
        "FlowCellProductCode": run.flow_cell_product_code,
        "SequencingKit": run.sequencing_kit,
        "PositionID": run.sequencer_position,
        "PositionType": run.sequencer_position_type,
        "SampleID": run.sample_id,
        "ExperimentID": run.experiment_name or run.tracking_id.get("protocol_group_id", ""),
        "ProtocolName": run.protocol_name,
        "Software": run.software,
        "SystemName": run.system_name,
        "SystemType": run.system_type,
        "ADCMin": run.adc_min,
        "ADCMax": run.adc_max,
        "ContextTags": run.context_tags,
        "TrackingID": run.tracking_id,
    }
    return {
        "id": str(read.read_id),
        "signal_file": filename,
        "signal_sha256": hashlib.sha256(signal).hexdigest(),
        "metadata": metadata,
        "float_bits": {
            "CalibrationScale": bits(read.calibration.scale),
            "CalibrationOffset": bits(read.calibration.offset),
            "OpenPoreLevel": bits(read.open_pore_level),
            "MedianBefore": bits(read.median_before),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if pod5.__version__ != "0.3.49":
        raise RuntimeError("This oracle requires pod5==0.3.49")
    if importlib.metadata.version("lib_pod5") != "0.3.49":
        raise RuntimeError("This oracle requires lib_pod5==0.3.49")
    if args.output.resolve().is_relative_to(args.stock.resolve()):
        raise RuntimeError("Keep generated fixtures outside the stock workspace")
    args.output.mkdir(parents=True, exist_ok=False)
    selected = (args.stock / "data/splits/smoke.txt").read_text().splitlines()[0]
    inputs = [
        ("single_split_read", args.stock / "software/dorado-src/tests/data/single_split_read/single_split_read.pod5", None),
        ("hg002_smoke", args.stock / "data/raw/HG002/PAW70337/PAW70337_66b2eea5_de8117b1_33.pod5", [selected]),
    ]
    manifest = {
        "schema": 1,
        "producer": "pod5==0.3.49",
        "lib_pod5_version": importlib.metadata.version("lib_pod5"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "generator_sha256": sha256(Path(__file__)),
        "fixtures": [],
    }
    for name, path, selection in inputs:
        with pod5.Reader(path) as reader:
            fixture = {
                "name": name,
                "source_path": str(path.resolve()),
                "source_bytes": path.stat().st_size,
                "source_sha256": sha256(path),
                "file_read_count": reader.num_reads,
                "file_id": str(reader.file_identifier),
                "file_original_version": [int(value) for value in str(reader.original_file_version).split(".")],
                "producer_logical_file_version": str(reader.logical_file_version),
                "file_ids": reader.read_ids,
                "selection": selection,
                "reads": [extract(r, args.output, name) for r in reader.reads(selection=selection, missing_ok=False)],
            }
            if len(fixture["reads"]) != 1:
                raise RuntimeError("The bounded oracle must extract exactly one read per fixture")
            manifest["fixtures"].append(fixture)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["exit_code"] = 0
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.output / "manifest.json"), "reads": 2}))


if __name__ == "__main__":
    main()
