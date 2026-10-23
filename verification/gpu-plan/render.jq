def bullets: map("- " + .) | join("\n");
"# Go GPU caller feature cards\n\nGenerated from `cards.json`. Read [the dispatch rules](README.md) before assigning work.\n\nThe commands marked to implement are deliverables, not existing tools. A source-ready card still needs native acceptance.\n",
"| Card | Work | State | Dependencies |\n| --- | --- | --- | --- |",
(.cards[] | "| [\(.id)](#\(.id | ascii_downcase)) | \(.title) | \(.status) | \(.depends_on | if length == 0 then "None" else join(", ") end) |"),
(.cards[] |
  "\n<a id=\"\(.id | ascii_downcase)\"></a>\n\n## \(.id). \(.title)\n\n" +
  "State is **\(.status)**. Owner role is `\(.owner_role)`. Dependencies are " +
  (.depends_on | if length == 0 then "none." else (join(", ") + ".") end) +
  " Resources are \(.resources | join(", ")).\n\n" +
  "### Own these files\n\n" + (.owns | map("`" + . + "`") | bullets) +
  "\n\n### Implement this boundary\n\n\(.interface)\n\n" + (.build | bullets) +
  "\n\n### Read and reuse\n\n" + (.source | bullets) + "\n\n" + (.reuse | bullets) +
  "\n\n### Pass these gates\n\n" + (.acceptance | bullets) +
  "\n\n### Reject these failures\n\n" + (.negative_controls | bullets) +
  "\n\n### Run or supply these commands\n\n" +
  (.commands | map("- " + .kind + ". `" + .command + "`") | join("\n")) +
  "\n\nIndependent review. \(.review)\n" +
  (if .evidence then "\nCurrent evidence.\n\n" + (.evidence | bullets) + "\n" else "" end) | rtrimstr("\n"))
