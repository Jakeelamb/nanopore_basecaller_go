#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
    echo 'usage: audit-controls.sh ACCEPTED_RUNTIME_JSON RUNTIME_SHA NEW_HOST_OUTPUT' >&2
    exit 2
fi
runtime=$(realpath "$1")
runtime_sha=$2
mkdir "$3"
out=$(realpath "$3")
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
run host-tests "$go_binary" test -count=1 -json -run ^TestAuditControl ./verification/supgpu
run go-environment "$go_binary" env -json GOROOT GOTOOLDIR GOVERSION GOOS GOARCH GOAMD64 CGO_ENABLED GOFLAGS GOWORK GOENV
run go-dependencies "$go_binary" list -deps -test -json ./verification/supgpu
jq -r '.Dir as $dir | ((.GoFiles // []) + (.CgoFiles // []) + (.SFiles // []) + (.HFiles // []) + (.SysoFiles // []) + (.EmbedFiles // []))[] | if startswith("/") then . else $dir + "/" + . end' "$out/go-dependencies.stdout" > "$out/go-selected.paths"
go_root=$(jq -r .GOROOT "$out/go-environment.stdout")
go_tool_dir=$(jq -r .GOTOOLDIR "$out/go-environment.stdout")
rg --files "$go_tool_dir" "$go_root/pkg/include" >> "$out/go-selected.paths"
printf '%s\n' "$go_binary" "$go_root/bin/go" "$PWD/go.mod" "$PWD/verification/supgpu/audit-controls.sh" "$PWD/verification/supgpu/native/controls/loader.c" "$PWD/verification/supgpu/native/controls/constructor.c" >> "$out/go-selected.paths"
sort -u "$out/go-selected.paths" > "$out/go-selected-sorted.paths"
while IFS= read -r path; do sha256sum "$path"; done < "$out/go-selected-sorted.paths" > "$out/before-go-build.sha256"
run host-build "$go_binary" test -c -buildvcs=false -o "$out/audit-controls" ./verification/supgpu
run freeze-controller "$out/audit-controls" supgpu-audit-control freeze -dir "$out"
controller_sha=$(sha256sum "$out/controller.json" | cut -d' ' -f1)
run auditor-proof "$out/audit-controls" supgpu-audit-control proof -controller "$out/controller.json" -controller-sha256 "$controller_sha" -runtime-plan "$runtime" -runtime-plan-sha256 "$runtime_sha" -out "$out/auditor-proof.json"
proof_sha=$(sha256sum "$out/auditor-proof.json" | cut -d' ' -f1)
run control-plan "$out/audit-controls" supgpu-audit-control plan -controller "$out/controller.json" -controller-sha256 "$controller_sha" -proof "$out/auditor-proof.json" -proof-sha256 "$proof_sha" -out "$out/native"
plan_sha=$(sha256sum "$out/native/plan.json" | cut -d' ' -f1)
run check-plan "$out/audit-controls" supgpu-audit-control check-plan -plan "$out/native/plan.json" -plan-sha256 "$plan_sha"
for kind in compiler-dso fixture-source; do
    case "$kind" in
        compiler-dso) jq '.inputs |= map(select(.path|contains("/libisl.so.")|not))' "$out/native/plan.json" > "$out/$kind.json" ;;
        fixture-source) jq '.commands[1].argv |= map(if . == "-DCONTROL_VARIANT=1" then "-DCONTROL_VARIANT=2" else . end)' "$out/native/plan.json" > "$out/$kind.json" ;;
    esac
    mutant_sha=$(sha256sum "$out/$kind.json" | cut -d' ' -f1)
    if run "reject-$kind" "$out/audit-controls" supgpu-audit-control check-plan -plan "$out/$kind.json" -plan-sha256 "$mutant_sha"; then
        echo "accepted $kind mutation" >&2
        exit 1
    fi
done
jq '.snapshots |= map(select(.original|endswith("/verification/supgpu/audit_controls_test.go")|not))' "$out/controller.json" > "$out/omitted-controller-source.json"
mutant_sha=$(sha256sum "$out/omitted-controller-source.json" | cut -d' ' -f1)
# Plan's controller pin size must describe the independently rehashed mutant.
mutant_bytes=$(stat -c%s "$out/omitted-controller-source.json")
jq --arg path "$out/omitted-controller-source.json" --arg sha "$mutant_sha" --argjson bytes "$mutant_bytes" '.controller = {path:$path,sha256:$sha,bytes:$bytes}' "$out/native/plan.json" > "$out/omitted-source-plan.json"
mutant_sha=$(sha256sum "$out/omitted-source-plan.json" | cut -d' ' -f1)
if run reject-controller-source "$out/audit-controls" supgpu-audit-control check-plan -plan "$out/omitted-source-plan.json" -plan-sha256 "$mutant_sha"; then
    echo 'omitted audit controller source accepted' >&2
    exit 1
fi
jq '.selected_cpu_providers += [{object:{path:"/unconsumed/libtorch_cuda.so",sha256:"0000000000000000000000000000000000000000000000000000000000000000",bytes:0},soname:"libtorch_cuda.so",needed:[],tier:"wrong",startup:true}]' "$out/auditor-proof.json" > "$out/wrong-cpu-proof.json"
mutant_sha=$(sha256sum "$out/wrong-cpu-proof.json" | cut -d' ' -f1)
mutant_bytes=$(stat -c%s "$out/wrong-cpu-proof.json")
jq --arg path "$out/wrong-cpu-proof.json" --arg sha "$mutant_sha" --argjson bytes "$mutant_bytes" '.auditor_producer_proof = {path:$path,sha256:$sha,bytes:$bytes}' "$out/native/plan.json" > "$out/wrong-cpu-plan.json"
mutant_sha=$(sha256sum "$out/wrong-cpu-plan.json" | cut -d' ' -f1)
if run reject-unconsumed-nn-runtime "$out/audit-controls" supgpu-audit-control check-plan -plan "$out/wrong-cpu-plan.json" -plan-sha256 "$mutant_sha"; then
    echo 'unconsumed NN image entered CPU fixture runtime' >&2
    exit 1
fi
cp "$out/audit-controls" "$out/modified-controller"
printf 'X' >> "$out/modified-controller"
if run reject-controller-binary "$out/modified-controller" supgpu-audit-control check-plan -plan "$out/native/plan.json" -plan-sha256 "$plan_sha"; then
    echo 'modified audit controller accepted' >&2
    exit 1
fi
for phase in compile run; do
    if run "reject-unadmitted-$phase" "$out/audit-controls" supgpu-audit-control "$phase" -plan /must/not/open -build /must/not/open -out "$out/must-not-exist"; then
        echo "unadmitted native audit $phase accepted" >&2
        exit 1
    fi
done
test ! -e "$out/must-not-exist"
printf 'audit controls source/host ready; no native compiler query/preprocess/compile, DSO/target load, device/GPU/model work\n' > "$out/result.txt"
