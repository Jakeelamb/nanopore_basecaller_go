#!/usr/bin/env bash
set -euo pipefail
mode=producer
if [[ $# == 4 && $1 == --consumer-only ]]; then
    mode=consumer
    build=$(realpath "$2")
    build_sha=$3
    out=$4
elif [[ $# == 2 ]]; then
    prepared=$(realpath "$1")
    prepared_sha=$(sha256sum "$prepared" | cut -d' ' -f1)
    out=$2
else
    echo 'usage: host-controls.sh PREPARED_JSON NEW_OUTPUT | --consumer-only BUILD_JSON BUILD_SHA NEW_OUTPUT' >&2
    exit 2
fi
mkdir "$out"
out=$(realpath "$out")
export GOMAXPROCS=2 GOMEMLIMIT=3GiB CGO_ENABLED=0 GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off GOENV=off GOFLAGS= GOWORK=off GOOS=linux GOARCH=amd64 GOAMD64=v1
export SUPGPU_GUARD_TEST_EVIDENCE="$out/descendant-controls"
go_binary=$(command -v go)

run() {
    local label=$1
    shift
    printf '%s\0' "$@" > "$out/$label.argv.nul"
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$out/$label.started"
    set +e
    "$@" > "$out/$label.stdout" 2> "$out/$label.stderr"
    local result=$?
    set -e
    printf '%s\n' "$result" > "$out/$label.exit"
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$out/$label.finished"
    sha256sum "$out/$label.stdout" "$out/$label.stderr" > "$out/$label.logs.sha256"
    return "$result"
}

run host-tests "$go_binary" test -json ./verification/supgpu/...
run go-environment "$go_binary" env -json GOROOT GOTOOLDIR GOVERSION GOOS GOARCH GOAMD64 CGO_ENABLED GOFLAGS GOWORK GOENV
run go-dependencies "$go_binary" list -deps -json ./verification/supgpu/cmd
jq -r '.Dir as $dir | ((.GoFiles // []) + (.CgoFiles // []) + (.SFiles // []) + (.HFiles // []) + (.SysoFiles // []) + (.EmbedFiles // []))[] | $dir + "/" + .' "$out/go-dependencies.stdout" > "$out/go-selected.paths"
go_root=$(jq -r .GOROOT "$out/go-environment.stdout")
go_tool_dir=$(jq -r .GOTOOLDIR "$out/go-environment.stdout")
rg --files "$go_tool_dir" "$go_root/pkg/include" >> "$out/go-selected.paths"
printf '%s\n' "$go_binary" "$go_root/bin/go" "$PWD/go.mod" "$PWD/verification/supgpu/host-controls.sh" >> "$out/go-selected.paths"
sort -u "$out/go-selected.paths" > "$out/go-selected-sorted.paths"
while IFS= read -r path; do
    sha256sum "$path"
done < "$out/go-selected-sorted.paths" > "$out/before-go-build.sha256"
run host-build "$go_binary" build -buildvcs=false -o "$out/supgpu-host" ./verification/supgpu/cmd
run freeze-controller "$out/supgpu-host" -unit freeze-controller -input "$out"
controller_sha=$(sha256sum "$out/controller.json" | cut -d' ' -f1)
if [[ $mode == consumer ]]; then
    run old-build-new-consumer "$out/supgpu-host" -unit check-runtime-consumer -build "$build" -build-sha256 "$build_sha" -controller "$out/controller.json" -controller-sha256 "$controller_sha"
    jq '.snapshots |= map(select(.original | endswith("/verification/supgpu/runtime.go") | not))' "$out/controller.json" > "$out/omitted-consumer-source.json"
    mutant_sha=$(sha256sum "$out/omitted-consumer-source.json" | cut -d' ' -f1)
    if run reject-consumer-source "$out/supgpu-host" -unit check-runtime-consumer -build "$build" -build-sha256 "$build_sha" -controller "$out/omitted-consumer-source.json" -controller-sha256 "$mutant_sha"; then
        echo 'omitted consumer source accepted' >&2
        exit 1
    fi
    cp "$out/supgpu-host" "$out/modified-controller"
    printf 'X' >> "$out/modified-controller"
    if run reject-consumer-binary "$out/modified-controller" -unit check-runtime-consumer -build "$build" -build-sha256 "$build_sha" -controller "$out/controller.json" -controller-sha256 "$controller_sha"; then
        echo 'modified consumer binary accepted' >&2
        exit 1
    fi
    printf 'consumer host checks passed; accepted NN build reused, no NN generation/compilation or runtime plan generation\n' > "$out/result.txt"
    exit 0
fi
run genuine-preflight "$out/supgpu-host" -unit preflight -prepared "$prepared" -prepared-sha256 "$prepared_sha"
run source-generation "$out/supgpu-host" -unit generate -prepared "$prepared" -prepared-sha256 "$prepared_sha" -out "$out/generated"
run build-plan "$out/supgpu-host" -unit plan-build -prepared "$prepared" -prepared-sha256 "$prepared_sha" -generated "$out/generated" -build-dir /home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/build-gcc15 -controller "$out/controller.json" -controller-sha256 "$controller_sha" -out "$out/build"
plan_sha=$(sha256sum "$out/build/plan.json" | cut -d' ' -f1)
run check-plan "$out/supgpu-host" -unit check-plan -plan "$out/build/plan.json" -plan-sha256 "$plan_sha"
for kind in quoted-header unfrozen-link omitted-go-source compiler-dso compiler-closure; do
    case "$kind" in
        quoted-header) jq '.commands[4].argv |= map(select(. != "-iquote" and . != "/home/jake/Projects/nanopore_basecaller/software/source-build-qualification-20261023/source-input-gcc15/dorado/basecall/model"))' "$out/build/plan.json" > "$out/$kind.json" ;;
        unfrozen-link) jq '.inputs |= map(select(.path | endswith("/libtorch_cpu.so") | not))' "$out/build/plan.json" > "$out/$kind.json" ;;
        omitted-go-source) jq '.snapshots |= map(select(.original | endswith("/verification/tensor/runner/admission.go") | not))' "$out/build/plan.json" > "$out/$kind.json" ;;
        compiler-dso) jq '.inputs |= map(select(.path | contains("/libisl.so.") | not))' "$out/build/plan.json" > "$out/$kind.json" ;;
        compiler-closure) jq '.inputs |= map(select(.path | contains("/libisl.so.") | not)) | .compiler_runtime.inputs |= map(select(.path | contains("/libisl.so.") | not)) | .compiler_runtime.elf_nodes |= map(select(.object.path | contains("/libisl.so.") | not))' "$out/build/plan.json" > "$out/$kind.json" ;;
    esac
    mutant_sha=$(sha256sum "$out/$kind.json" | cut -d' ' -f1)
    if run "reject-$kind" "$out/supgpu-host" -unit check-plan -plan "$out/$kind.json" -plan-sha256 "$mutant_sha"; then
        echo "accepted $kind plan mutation" >&2
        exit 1
    fi
done
cp "$out/supgpu-host" "$out/modified-controller"
printf 'X' >> "$out/modified-controller"
if run reject-controller-binary "$out/modified-controller" -unit check-plan -plan "$out/build/plan.json" -plan-sha256 "$plan_sha"; then
    echo 'modified controller executable accepted' >&2
    exit 1
fi
for kind in fp8 geometry calibration; do
    case "$kind" in
        fp8) jq '.use_f8 = false' "$prepared" > "$out/$kind.json" ;;
        geometry) jq '.geometry.chunk = 192' "$prepared" > "$out/$kind.json" ;;
        calibration) jq '.calibration_offset_bits += 1' "$prepared" > "$out/$kind.json" ;;
    esac
    mutant_sha=$(sha256sum "$out/$kind.json" | cut -d' ' -f1)
    if run "reject-$kind" "$out/supgpu-host" -unit preflight -prepared "$out/$kind.json" -prepared-sha256 "$mutant_sha"; then
        echo "accepted $kind mutation" >&2
        exit 1
    fi
done
if run reject-native-build "$out/supgpu-host" -unit build -plan /must/not/open; then
    echo 'native build ran without separate admission' >&2
    exit 1
fi
if run reject-native-capture "$out/supgpu-host" -unit source-graph -out "$out/must-not-exist"; then
    echo 'native capture ran without separate admission' >&2
    exit 1
fi
if run reject-loader-attestation "$out/supgpu-host" -unit attest-runtime -out "$out/must-not-exist"; then
    echo 'loader attestation ran without separate admission' >&2
    exit 1
fi
test ! -e "$out/must-not-exist"
printf 'host checks passed; no C++ compiler, native library, GPU query or host lock was used\n' > "$out/result.txt"
