def text: type == "string" and length > 0;
def texts: type == "array" and length > 0 and all(.[]; text);
def require($ok; $message): if $ok then . else error($message) end;
def acyclic($cards; $id; $trail):
  if ($trail | index($id)) != null then false
  else all(($cards[] | select(.id == $id) | .depends_on[]);
    acyclic($cards; .; $trail + [$id])) end;

. as $plan
| require(.schema == "go-gpu-caller-cards/v1"; "unknown card schema")
| require((.base_commit | test("^[0-9a-f]{40}$")); "base_commit must be an exact commit")
| require((.cards | type == "array" and length > 0); "cards must be nonempty")
| .cards as $cards
| [$cards[].id] as $ids
| require(($ids | length == (unique | length)); "duplicate card id")
| require(all($cards[];
    (.id | test("^[A-Z][0-9]{2}$")) and
    (.title | text) and (.area | text) and (.owner_role | text) and
    (.status | IN("planned", "source-ready", "accepted")) and
    (.depends_on | type == "array" and all(.[]; text)) and
    (.owns | texts) and (.interface | text) and (.source | texts) and
    (.build | texts) and (.acceptance | texts) and (.negative_controls | texts) and
    (.reuse | texts) and (.resources | texts) and (.review | text) and
    (.commands | type == "array" and length > 0 and
      all(.[]; (.kind | IN("existing", "to-implement")) and (.command | text)))
  ); "missing or invalid card field")
| require(all($cards[]; all(.depends_on[]; . as $id | $ids | index($id) != null));
    "dependency names an unknown card")
| require(all($ids[]; acyclic($cards; .; [])); "dependency cycle")
| [$cards[].owns[]] as $paths
| require(($paths | length == (unique | length)); "two cards own the same file")
| require(all($cards[]; all(.resources[];
    IN("source", "native-build", "gpu-capture", "cpu-heavy", "full-cohort", "timed")));
    "unknown resource class")
| require(all($cards[] | select(.status == "source-ready"); (.evidence | texts));
    "source-ready card lacks evidence")
| require(all($cards[] | select(.status == "accepted");
    (.evidence | texts) and (.verdict_sha256 | test("^[0-9a-f]{64}$")));
    "accepted card lacks evidence and pinned independent verdict")
| {
    result: "PASS",
    cards: ($cards | length),
    accepted: ([$cards[] | select(.status == "accepted")] | length),
    source_ready: ([$cards[] | select(.status == "source-ready")] | length),
    ready_for_source_assignment: [$cards[] |
      select(all(.depends_on[]; . as $id |
        any($cards[]; .id == $id and .status == "accepted"))) | .id],
    note: "Read the operational hold and dependency contracts before dispatch. Counts are not a completion percentage."
  }
